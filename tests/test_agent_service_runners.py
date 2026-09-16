import asyncio
import os
import sys
import threading
import time
from types import ModuleType

import numpy as np
import pytest

import qwen_asr_vllm.agent as agent_package
import qwen_asr_vllm.agent.nano_llm as nano_llm
import qwen_asr_vllm.agent.service_runners as service_runners
from qwen_asr_vllm.agent.nano_llm import NanoVllmStepBatchingBackend
from qwen_asr_vllm.agent.async_coordinator import AsyncVoiceAgentCoordinator
from qwen_asr_vllm.agent.service_runners import (
    ThreadedAsrEngine,
    ProcessAsrEngine,
    ProcessConcurrentTtsBackend,
    UnifiedTtsStreamBatchScheduler,
    ProcessNanoLlmBackend,
    ProcessTtsBackend,
    MultiplexedThreadedLlmBackend,
    MultiplexedThreadedTtsBackend,
)
from qwen_asr_vllm.engine.streaming import StreamEvent


class RecordingAsrSession:
    def __init__(self):
        self.created_on = threading.get_ident()
        self.feed_threads = []
        self.close_threads = []
        self.calls = 0

    def feed(self, pcm):
        self.calls += 1
        self.feed_threads.append(threading.get_ident())
        if self.calls == 1:
            return [
                StreamEvent(
                    kind="committed",
                    text="status",
                    committed_text="status",
                    audio_seconds=0.4,
                )
            ]
        time.sleep(0.4)
        return []

    def close(self, reuse_last=True):
        self.close_threads.append(threading.get_ident())
        return []


class RecordingAsrEngine:
    def __init__(self):
        self.session = None

    def open_stream(self, **kwargs):
        self.session = RecordingAsrSession()
        return self.session


class RecordingLlm:
    def __init__(self):
        self.thread_ids = []
        self.finished_at = None

    def chat_stream(self, text):
        for chunk in ["First.", " Second."]:
            self.thread_ids.append(threading.get_ident())
            time.sleep(0.03)
            yield chunk
        self.finished_at = time.perf_counter()


class ResetRecordingLlm(RecordingLlm):
    def __init__(self):
        super().__init__()
        self.resets = 0

    def reset(self):
        self.resets += 1


class RecordingTts:
    def __init__(self):
        self.thread_ids = []
        self.starts = []

    def synthesize(self, text):
        self.thread_ids.append(threading.get_ident())
        self.starts.append((text, time.perf_counter()))
        time.sleep(0.08)
        return ("wav:" + text).encode()


class BatchRecordingTts:
    def __init__(self):
        self.batches = []

    def synthesize_batch(self, texts):
        batch = list(texts)
        self.batches.append(batch)
        time.sleep(0.02)
        return [("wav:" + text).encode() for text in batch]


class StepRecordingLlm:
    def __init__(self):
        self.active = 0
        self.max_active = 0

    def chat_stream(self, text):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            yield text + ":1"
            time.sleep(0.03)
            yield text + ":2"
        finally:
            self.active -= 1


class FakeNanoTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(token_ids)


class FakeNanoSeq:
    def __init__(self, seq_id, prompt, tokens):
        self.seq_id = seq_id
        self.prompt = prompt
        self.tokens = list(tokens)
        self.completion_token_ids = []


class FakeNanoEngine:
    def __init__(self):
        self.tokenizer = FakeNanoTokenizer()
        self.scheduler = type("Scheduler", (), {"waiting": [], "running": []})()
        self._next_seq_id = 0
        self.step_batch_sizes = []
        self.added_prompts = []

    def add_request(self, prompt, sampling_params):
        tokens = {
            "first": ["A", "B"],
            "second": ["X", "Y"],
        }[prompt]
        seq = FakeNanoSeq(self._next_seq_id, prompt, tokens)
        self._next_seq_id += 1
        self.scheduler.waiting.append(seq)
        self.added_prompts.append(prompt)

    def step(self):
        if self.scheduler.waiting:
            self.scheduler.running.extend(self.scheduler.waiting)
            self.scheduler.waiting.clear()
        self.step_batch_sizes.append(len(self.scheduler.running))
        finished = []
        still_running = []
        for seq in self.scheduler.running:
            seq.completion_token_ids.append(seq.tokens[len(seq.completion_token_ids)])
            if len(seq.completion_token_ids) == len(seq.tokens):
                finished.append((seq.seq_id, list(seq.completion_token_ids), None))
            else:
                still_running.append(seq)
        self.scheduler.running = still_running
        return finished, -len(self.step_batch_sizes)

    def is_finished(self):
        return not self.scheduler.waiting and not self.scheduler.running


class HistoryNanoTokenizer(FakeNanoTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        return "|".join(message["content"] for message in messages)


class HistoryNanoEngine(FakeNanoEngine):
    def __init__(self):
        super().__init__()
        self.tokenizer = HistoryNanoTokenizer()

    def add_request(self, prompt, sampling_params):
        seq = FakeNanoSeq(self._next_seq_id, prompt, ["R"])
        self._next_seq_id += 1
        self.scheduler.waiting.append(seq)
        self.added_prompts.append(prompt)


class FailingNanoEngine(FakeNanoEngine):
    def add_request(self, prompt, sampling_params):
        raise RuntimeError("add failed")


class WarmupNanoEngine(FakeNanoEngine):
    def add_request(self, prompt, sampling_params):
        seq = FakeNanoSeq(self._next_seq_id, prompt, ["W"])
        self._next_seq_id += 1
        self.scheduler.waiting.append(seq)
        self.added_prompts.append(prompt)

    def step(self):
        if self.scheduler.waiting:
            self.scheduler.running.extend(self.scheduler.waiting)
            self.scheduler.waiting.clear()
        batch_size = len(self.scheduler.running)
        self.step_batch_sizes.append(batch_size)
        finished = []
        for seq in self.scheduler.running:
            seq.completion_token_ids.append("W")
            finished.append((seq.seq_id, list(seq.completion_token_ids), None))
        self.scheduler.running = []
        return finished, -batch_size


def test_normalize_warmup_batch_sizes_clips_and_deduplicates():
    assert nano_llm.normalize_warmup_batch_sizes(
        (1, 2, 4, 4, 8),
        max_num_seqs=4,
    ) == (1, 2, 4)


def test_normalize_warmup_batch_sizes_rejects_non_positive_values():
    with pytest.raises(ValueError, match="warmup batch sizes must be positive"):
        nano_llm.normalize_warmup_batch_sizes((0, 2), max_num_seqs=4)


def test_nano_vllm_backend_precaptures_startup_batch_shapes():
    engine = WarmupNanoEngine()
    backend = NanoVllmStepBatchingBackend(
        engine=engine,
        sampling_cls=lambda **kwargs: kwargs,
        system_prompt="",
        max_num_seqs=4,
        warmup=True,
        warmup_batch_sizes=(1, 2, 4),
    )
    try:
        assert engine.step_batch_sizes == [1, 2, 4]
        assert backend.warmup_stats == {
            "batch_sizes": [1, 2, 4],
            "step_calls": 3,
            "max_step_batch_size": 4,
        }
        assert backend.stats == {"step_calls": 0, "max_step_batch_size": 0}
    finally:
        backend.close()


def test_nano_vllm_backend_propagates_fixed_kv_blocks(monkeypatch):
    captured = {}
    fake_package = ModuleType("nanovllm")
    fake_package.__path__ = []
    fake_llm_module = ModuleType("nanovllm.llm")
    fake_sampling_module = ModuleType("nanovllm.sampling_params")

    def build_engine(model_path, **kwargs):
        captured.update(model_path=model_path, **kwargs)
        return FakeNanoEngine()

    fake_llm_module.LLM = build_engine
    fake_sampling_module.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "nanovllm", fake_package)
    monkeypatch.setitem(sys.modules, "nanovllm.llm", fake_llm_module)
    monkeypatch.setitem(sys.modules, "nanovllm.sampling_params", fake_sampling_module)

    backend = NanoVllmStepBatchingBackend(
        model_path="/model",
        nanovllm_root="",
        num_kvcache_blocks=64,
        warmup=False,
    )
    try:
        assert captured["num_kvcache_blocks"] == 64
    finally:
        backend.close()


class ProcessRecordingAsrSession:
    def __init__(self):
        self.calls = 0

    def feed(self, pcm):
        self.calls += 1
        if self.calls == 1:
            return [
                StreamEvent(
                    kind="committed",
                    text=str(os.getpid()),
                    committed_text=str(os.getpid()),
                    audio_seconds=0.4,
                )
            ]
        time.sleep(0.4)
        return []

    def close(self, reuse_last=True):
        return []


class ProcessRecordingAsrEngine:
    def open_stream(self, **kwargs):
        return ProcessRecordingAsrSession()


class ProcessRecordingLlm:
    def chat_stream(self, text):
        for chunk in ["First.", " Second."]:
            time.sleep(0.03)
            yield chunk


class ProcessSlowLlm:
    def chat_stream(self, text):
        time.sleep(0.2)
        yield f"{text}:1"
        time.sleep(0.2)
        yield f"{text}:2"


class ProcessStreamingTts:
    def synthesize(self, text):
        return b"fallback:" + text.encode()

    def synthesize_stream(self, text):
        yield b"stream-1:" + text.encode()
        yield b"stream-2:" + text.encode()


class ProcessSlowStreamingTts:
    def synthesize(self, text):
        time.sleep(0.3)
        return b"fallback:" + text.encode()

    def synthesize_stream(self, text):
        time.sleep(0.2)
        yield b"stream-1:" + text.encode()
        time.sleep(0.2)
        yield b"stream-2:" + text.encode()


class ProcessBatchStreamingTts:
    def synthesize(self, text):
        return b"fallback:" + text.encode()

    def synthesize_stream(self, text):
        time.sleep(0.4)
        yield b"single:" + text.encode()

    def synthesize_stream_batch(self, texts):
        time.sleep(0.2)
        for index, text in enumerate(texts):
            yield index, b"batch:" + text.encode()


class ProcessCancelableLlm:
    def __init__(self):
        self.cancelled = False

    def chat_stream(self, text):
        yield text + ":first"
        while not self.cancelled:
            time.sleep(0.01)

    def cancel(self, request_id=None):
        self.cancelled = True


class ProcessHistoryLlm:
    supports_session_history = True

    def __init__(self):
        self.histories = {}

    def chat_stream(self, text, *, session_id=None):
        history = self.histories.setdefault(session_id, [])
        yield "|".join(history + [text])
        history.append(text)

    def reset(self, *, session_id=None):
        if session_id is None:
            self.histories.clear()
        else:
            self.histories.pop(session_id, None)


class ProcessRecordingTts:
    def synthesize(self, text):
        time.sleep(0.08)
        return f"{os.getpid()}:{text}".encode()


def make_process_asr_engine():
    return ProcessRecordingAsrEngine()


def make_process_llm():
    return ProcessRecordingLlm()


def make_process_slow_llm():
    return ProcessSlowLlm()


def make_process_tts():
    return ProcessRecordingTts()


async def collect_until_done(coordinator, timeout=2.0):
    events = []
    while True:
        event = await coordinator.next_event(timeout=timeout)
        events.append(event.as_dict())
        if event.type == "done":
            return events


def make_process_streaming_tts():
    return ProcessStreamingTts()


def make_process_slow_streaming_tts():
    return ProcessSlowStreamingTts()


def make_process_batch_streaming_tts():
    return ProcessBatchStreamingTts()


def make_process_cancelable_llm():
    return ProcessCancelableLlm()


def make_process_history_llm():
    return ProcessHistoryLlm()


def test_threaded_asr_engine_runs_session_lifecycle_on_one_service_thread():
    engine = RecordingAsrEngine()
    threaded = ThreadedAsrEngine(engine)
    session = threaded.open_stream()

    try:
        main_thread = threading.get_ident()
        assert session.feed(np.zeros(1600, dtype=np.float32))[0].kind == "committed"
        assert session.close() == []

        raw_session = engine.session
        assert raw_session is not None
        assert raw_session.created_on != main_thread
        assert set(raw_session.feed_threads) == {raw_session.created_on}
        assert set(raw_session.close_threads) == {raw_session.created_on}
    finally:
        threaded.close()


def test_multiplexed_threaded_tts_batches_concurrent_requests():
    raw = BatchRecordingTts()
    tts = MultiplexedThreadedTtsBackend(raw, batch_window_ms=20)
    results = []

    def synthesize(text):
        results.append(tts.synthesize(text))

    first = threading.Thread(target=synthesize, args=("one",))
    second = threading.Thread(target=synthesize, args=("two",))

    try:
        first.start()
        second.start()
        first.join(timeout=1)
        second.join(timeout=1)

        assert sorted(results) == [b"wav:one", b"wav:two"]
        assert raw.batches == [["one", "two"]]
    finally:
        tts.close()


def test_multiplexed_threaded_llm_interleaves_concurrent_streams():
    raw = StepRecordingLlm()
    llm = MultiplexedThreadedLlmBackend(raw)
    outputs = {}

    def collect(name):
        outputs[name] = list(llm.chat_stream(name))

    first = threading.Thread(target=collect, args=("first",))
    second = threading.Thread(target=collect, args=("second",))

    try:
        first.start()
        second.start()
        first.join(timeout=1)
        second.join(timeout=1)

        assert outputs == {
            "first": ["first:1", "first:2"],
            "second": ["second:1", "second:2"],
        }
        assert raw.max_active == 2
    finally:
        llm.close()


def test_nano_vllm_step_batching_backend_batches_concurrent_requests():
    engine = FakeNanoEngine()
    llm = NanoVllmStepBatchingBackend(
        engine=engine,
        sampling_cls=lambda **kwargs: kwargs,
        system_prompt="",
        admission_window_ms=20,
        warmup=False,
    )
    outputs = {}

    def collect(name):
        outputs[name] = list(llm.chat_stream(name))

    first = threading.Thread(target=collect, args=("first",))
    second = threading.Thread(target=collect, args=("second",))

    try:
        first.start()
        second.start()
        first.join(timeout=1)
        second.join(timeout=1)

        assert outputs == {
            "first": ["A", "B"],
            "second": ["X", "Y"],
        }
        assert engine.added_prompts == ["first", "second"]
        assert max(engine.step_batch_sizes) == 2
        assert llm.stats["max_step_batch_size"] == 2
    finally:
        llm.close()


def test_nano_vllm_backend_keeps_history_isolated_by_session_id():
    engine = HistoryNanoEngine()
    llm = NanoVllmStepBatchingBackend(
        engine=engine,
        sampling_cls=lambda **kwargs: kwargs,
        system_prompt="",
        warmup=False,
    )
    try:
        assert "".join(llm.chat_stream("alpha", session_id="a")) == "R"
        assert "".join(llm.chat_stream("beta", session_id="b")) == "R"
        assert "".join(llm.chat_stream("next-a", session_id="a")) == "R"
        assert "".join(llm.chat_stream("next-b", session_id="b")) == "R"
        assert engine.added_prompts == [
            "alpha",
            "beta",
            "alpha|R|next-a",
            "beta|R|next-b",
        ]

        llm.reset(session_id="a")
        assert "".join(llm.chat_stream("fresh-a", session_id="a")) == "R"
        assert engine.added_prompts[-1] == "fresh-a"
    finally:
        llm.close()


def test_nano_vllm_backend_runtime_metrics_include_warmup_and_online_stats():
    llm = NanoVllmStepBatchingBackend(
        engine=WarmupNanoEngine(),
        sampling_cls=lambda **kwargs: kwargs,
        system_prompt="",
        max_num_seqs=2,
        warmup=True,
        warmup_batch_sizes=(1, 2),
    )
    try:
        assert llm.runtime_metrics() == {
            "online": {"step_calls": 0, "max_step_batch_size": 0},
            "warmup": {
                "batch_sizes": [1, 2],
                "step_calls": 2,
                "max_step_batch_size": 2,
            },
        }
    finally:
        llm.close()


def test_nano_vllm_step_batching_backend_propagates_add_request_errors():
    llm = NanoVllmStepBatchingBackend(
        engine=FailingNanoEngine(),
        sampling_cls=lambda **kwargs: kwargs,
        system_prompt="",
        warmup=False,
    )

    try:
        try:
            list(llm.chat_stream("first"))
        except RuntimeError as exc:
            assert "add failed" in str(exc)
        else:
            raise AssertionError("expected add_request error")
    finally:
        llm.close()


def test_async_coordinator_overlaps_with_threaded_service_runners():
    asyncio.run(_async_coordinator_overlaps_with_threaded_service_runners())


async def _async_coordinator_overlaps_with_threaded_service_runners():
    raw_asr = RecordingAsrEngine()
    raw_llm = RecordingLlm()
    raw_tts = RecordingTts()
    asr = ThreadedAsrEngine(raw_asr)
    llm = MultiplexedThreadedLlmBackend(raw_llm)
    tts = MultiplexedThreadedTtsBackend(raw_tts)
    coordinator = AsyncVoiceAgentCoordinator(
        asr,
        llm,
        tts,
        llm_trigger="committed",
        tts_concurrency=2,
    )

    try:
        started = time.perf_counter()
        await coordinator.feed(np.zeros(1600, dtype=np.float32))
        blocking_feed = asyncio.create_task(
            coordinator.feed(np.zeros(1600, dtype=np.float32))
        )
        events = await collect_until_done(coordinator)
        elapsed = time.perf_counter() - started

        core_types = [
            event["type"]
            for event in events
            if event["type"] in {
                "asr_committed",
                "llm_start",
                "llm_chunk",
                "llm_done",
                "tts_chunk",
                "done",
            }
        ]
        assert core_types == [
            "asr_committed",
            "llm_start",
            "llm_chunk",
            "llm_chunk",
            "llm_done",
            "tts_chunk",
            "tts_chunk",
            "done",
        ]
        assert raw_tts.starts[0][1] < raw_llm.finished_at
        assert not blocking_feed.done()
        await blocking_feed
        assert elapsed < 0.3
    finally:
        await coordinator.stop()
        asr.close()
        llm.close()
        tts.close()


def test_async_coordinator_resets_llm_when_session_starts():
    asr = RecordingAsrEngine()
    llm = ResetRecordingLlm()
    tts = RecordingTts()

    coordinator = AsyncVoiceAgentCoordinator(
        asr,
        llm,
        tts,
        llm_trigger="committed",
    )

    assert coordinator.llm is llm
    assert llm.resets == 1


def test_async_coordinator_can_share_a_pre_reset_llm_service():
    asr = RecordingAsrEngine()
    llm = ResetRecordingLlm()
    tts = RecordingTts()

    coordinator = AsyncVoiceAgentCoordinator(
        asr,
        llm,
        tts,
        llm_trigger="committed",
        reset_llm_on_start=False,
    )

    assert coordinator.llm is llm
    assert llm.resets == 0


def test_process_service_runners_execute_backends_outside_parent_process():
    parent_pid = os.getpid()
    asr = ProcessAsrEngine(make_process_asr_engine, context="fork")
    llm = ProcessNanoLlmBackend(make_process_llm, context="fork")
    tts = ProcessTtsBackend(make_process_tts, context="fork")

    try:
        session = asr.open_stream()
        event = session.feed(np.zeros(1600, dtype=np.float32))[0]
        llm_text = "".join(llm.chat_stream("hello"))
        audio = tts.synthesize("First.")

        assert int(event.text) != parent_pid
        assert llm_text == "First. Second."
        assert int(audio.split(b":", 1)[0]) != parent_pid
    finally:
        asr.close()
        llm.close()
        tts.close()


def test_serial_process_llm_backend_is_removed_from_agent_api():
    removed_name = "Process" + "LlmBackend"
    assert not hasattr(service_runners, removed_name)
    assert not hasattr(agent_package, removed_name)
    assert removed_name not in agent_package.__all__


def test_process_nano_llm_backend_overlaps_concurrent_streams_in_one_process():
    llm = ProcessNanoLlmBackend(make_process_slow_llm, context="fork")
    outputs = {}

    def collect(name):
        outputs[name] = list(llm.chat_stream(name))

    first = threading.Thread(target=collect, args=("first",))
    second = threading.Thread(target=collect, args=("second",))

    try:
        started = time.perf_counter()
        first.start()
        second.start()
        first.join(timeout=2)
        second.join(timeout=2)
        elapsed = time.perf_counter() - started

        assert outputs == {
            "first": ["first:1", "first:2"],
            "second": ["second:1", "second:2"],
        }
        assert elapsed < 0.65
    finally:
        llm.close()


def test_async_coordinator_overlaps_with_process_service_runners():
    asyncio.run(_async_coordinator_overlaps_with_process_service_runners())


async def _async_coordinator_overlaps_with_process_service_runners():
    asr = ProcessAsrEngine(make_process_asr_engine, context="fork")
    llm = ProcessNanoLlmBackend(make_process_llm, context="fork")
    tts = ProcessTtsBackend(make_process_tts, context="fork")
    coordinator = AsyncVoiceAgentCoordinator(
        asr,
        llm,
        tts,
        llm_trigger="committed",
        tts_concurrency=2,
    )

    try:
        started = time.perf_counter()
        await coordinator.feed(np.zeros(1600, dtype=np.float32))
        blocking_feed = asyncio.create_task(
            coordinator.feed(np.zeros(1600, dtype=np.float32))
        )
        events = await collect_until_done(coordinator)
        elapsed = time.perf_counter() - started

        core_types = [
            event["type"]
            for event in events
            if event["type"] in {
                "asr_committed",
                "llm_start",
                "llm_chunk",
                "llm_done",
                "tts_chunk",
                "done",
            }
        ]
        assert core_types == [
            "asr_committed",
            "llm_start",
            "llm_chunk",
            "llm_chunk",
            "llm_done",
            "tts_chunk",
            "tts_chunk",
            "done",
        ]
        assert not blocking_feed.done()
        await blocking_feed
        assert elapsed < 0.3
    finally:
        await coordinator.stop()
        asr.close()
        llm.close()
        tts.close()



def test_process_tts_backend_streams_chunks_when_backend_supports_streaming():
    tts = ProcessTtsBackend(make_process_streaming_tts, context="fork")
    try:
        assert list(tts.synthesize_stream("hello")) == [
            b"stream-1:hello",
            b"stream-2:hello",
        ]
    finally:
        tts.close()


def test_process_tts_backend_uses_shared_memory_for_large_audio_payloads():
    tts = ProcessTtsBackend(
        make_process_streaming_tts,
        context="spawn",
        shared_memory_threshold=8,
    )
    try:
        chunks = list(tts.synthesize_stream("shared"))
        assert chunks == [b"stream-1:shared", b"stream-2:shared"]
        assert tts.runtime_metrics()["shared_memory"]["received_segments"] == 2
    finally:
        tts.close()


def test_process_concurrent_tts_backend_overlaps_streaming_requests():
    tts = ProcessConcurrentTtsBackend(
        make_process_slow_streaming_tts,
        context="fork",
        timeout=3,
        max_workers=2,
    )
    outputs = {}

    def collect(name):
        outputs[name] = list(tts.synthesize_stream(name))

    first = threading.Thread(target=collect, args=("first",))
    second = threading.Thread(target=collect, args=("second",))

    try:
        started = time.perf_counter()
        first.start()
        second.start()
        first.join(timeout=2)
        second.join(timeout=2)
        elapsed = time.perf_counter() - started

        assert outputs == {
            "first": [b"stream-1:first", b"stream-2:first"],
            "second": [b"stream-1:second", b"stream-2:second"],
        }
        assert elapsed < 0.65
    finally:
        tts.close()


def test_process_concurrent_tts_backend_batches_streaming_requests_by_explicit_legacy_scheduler(
    monkeypatch,
):
    monkeypatch.setenv("VOICE_TTS_REQUEST_STEP_SCHEDULER", "0")
    monkeypatch.setenv("VOICE_TTS_LEGACY_STREAM_BATCH", "1")
    tts = ProcessConcurrentTtsBackend(
        make_process_batch_streaming_tts,
        context="fork",
        timeout=3,
        max_workers=1,
        batch_window_ms=50,
        max_batch_size=4,
    )
    outputs = {}

    def collect(name):
        outputs[name] = list(tts.synthesize_stream(name))

    first = threading.Thread(target=collect, args=("first",))
    second = threading.Thread(target=collect, args=("second",))

    try:
        started = time.perf_counter()
        first.start()
        second.start()
        first.join(timeout=2)
        second.join(timeout=2)
        elapsed = time.perf_counter() - started

        assert outputs == {
            "first": [b"batch:first"],
            "second": [b"batch:second"],
        }
        assert elapsed < 0.35
        scheduler_metrics = tts.runtime_metrics()["scheduler"]
        assert scheduler_metrics["batch_sizes"] == [2]
        assert scheduler_metrics["max_active_slots"] == 2
    finally:
        tts.close()


def test_process_concurrent_tts_backend_rejects_non_qwen_backend_in_strict_mode(
    monkeypatch,
):
    monkeypatch.setenv("VOICE_TTS_REQUEST_STEP_SCHEDULER", "1")
    monkeypatch.setenv("VOICE_TTS_LEGACY_STREAM_BATCH", "0")

    with pytest.raises(RuntimeError, match="request-id codec scheduler"):
        ProcessConcurrentTtsBackend(
            make_process_batch_streaming_tts,
            context="fork",
            timeout=3,
            max_workers=2,
            batch_window_ms=10,
            max_batch_size=2,
        )


def test_process_concurrent_tts_backend_returns_shared_memory_chunks_as_bytes():
    tts = ProcessConcurrentTtsBackend(
        make_process_streaming_tts,
        context="spawn",
        timeout=15,
        max_workers=1,
        shared_memory_threshold=8,
    )
    try:
        chunks = list(tts.synthesize_stream("shared"))
        assert chunks == [b"stream-1:shared", b"stream-2:shared"]
        metrics = tts.runtime_metrics()["shared_memory"]
        assert metrics["received_segments"] == 2
        assert metrics["received_bytes"] == sum(map(len, chunks))
        assert metrics["outstanding_segments"] == 0
    finally:
        tts.close()


def test_unified_tts_stream_batch_scheduler_serializes_batches_and_demuxes_ids():
    calls = []
    active = 0
    max_active = 0
    state_lock = threading.Lock()
    completed = threading.Event()
    events = []

    def synthesize_stream_batch(texts):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
            calls.append(list(texts))
        try:
            time.sleep(0.02)
            for index, text in enumerate(texts):
                yield index, (f"batch:{text}").encode()
        finally:
            with state_lock:
                active -= 1

    def emit(kind, request_id, payload):
        events.append((kind, request_id, payload))
        if kind == "done" and sum(item[0] == "done" for item in events) == 4:
            completed.set()

    scheduler = UnifiedTtsStreamBatchScheduler(
        synthesize_stream_batch,
        emit,
        batch_window_ms=1,
        max_batch_size=2,
    )
    try:
        for request_id, text in enumerate(("a", "b", "c", "d"), start=1):
            scheduler.submit(request_id, text)
        assert completed.wait(timeout=1)
    finally:
        scheduler.close()

    assert calls == [["a", "b"], ["c", "d"]]
    assert max_active == 1
    assert [(kind, request_id) for kind, request_id, _ in events] == [
        ("chunk", 1),
        ("chunk", 2),
        ("done", 1),
        ("done", 2),
        ("chunk", 3),
        ("chunk", 4),
        ("done", 3),
        ("done", 4),
    ]


def test_unified_tts_scheduler_refills_backlog_without_another_window():
    calls = []
    completed = threading.Event()
    events = []

    def synthesize_stream_batch(texts):
        calls.append((list(texts), time.perf_counter()))
        time.sleep(0.02)
        for index, text in enumerate(texts):
            yield index, text.encode()

    def emit(kind, request_id, payload):
        events.append((kind, request_id, payload))
        if sum(item[0] == "done" for item in events) == 3:
            completed.set()

    scheduler = UnifiedTtsStreamBatchScheduler(
        synthesize_stream_batch,
        emit,
        batch_window_ms=80,
        max_batch_size=2,
    )
    try:
        submitted_at = time.perf_counter()
        scheduler.submit(1, "a")
        scheduler.submit(2, "b")
        scheduler.submit(3, "c")
        assert completed.wait(timeout=1)
        metrics = scheduler.metrics()
    finally:
        scheduler.close()

    assert [texts for texts, _started in calls] == [["a", "b"], ["c"]]
    assert calls[0][1] - submitted_at < 0.06
    assert calls[1][1] - calls[0][1] < 0.06
    assert metrics["refill_count"] == 1
    assert metrics["active_slots"] == 3
    assert metrics["padded_slots"] == 1
    assert metrics["max_active_slots"] == 2
    assert len(metrics["queue_wait_ms"]) == 3


def test_unified_tts_scheduler_groups_compatible_requests_and_skips_cancelled():
    calls = []
    events = []
    cancelled = {4}
    completed = threading.Event()

    def synthesize_stream_batch(texts):
        calls.append(list(texts))
        for index, text in enumerate(texts):
            yield index, text.encode()

    def emit(kind, request_id, payload):
        events.append((kind, request_id, payload))
        if sum(item[0] == "done" for item in events) == 4:
            completed.set()

    scheduler = UnifiedTtsStreamBatchScheduler(
        synthesize_stream_batch,
        emit,
        is_cancelled=cancelled.__contains__,
        compatibility_key=lambda text: text.split(":", 1)[0],
        batch_window_ms=20,
        max_batch_size=3,
    )
    try:
        scheduler.submit(1, "a:1")
        scheduler.submit(2, "b:1")
        scheduler.submit(3, "a:2")
        scheduler.submit(4, "a:cancel")
        assert completed.wait(timeout=1)
        metrics = scheduler.metrics()
    finally:
        scheduler.close()

    assert calls == [["a:1", "a:2"], ["b:1"]]
    assert metrics["cancellations"] == 1
    assert metrics["batch_sizes"] == [2, 1]
    assert ("done", 4, None) in events


def test_unified_tts_scheduler_counts_cancellation_during_active_batch():
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    cancelled = set()
    events = []

    def synthesize_stream_batch(texts):
        started.set()
        assert release.wait(timeout=1)
        yield 0, texts[0].encode()

    def emit(kind, request_id, payload):
        events.append((kind, request_id, payload))
        if kind == "done":
            completed.set()

    scheduler = UnifiedTtsStreamBatchScheduler(
        synthesize_stream_batch,
        emit,
        is_cancelled=cancelled.__contains__,
        batch_window_ms=0,
        max_batch_size=1,
    )
    try:
        scheduler.submit(1, "cancel-me")
        assert started.wait(timeout=1)
        cancelled.add(1)
        release.set()
        assert completed.wait(timeout=1)
        metrics = scheduler.metrics()
    finally:
        scheduler.close()

    assert metrics["cancellations"] == 1
    assert events == [("done", 1, None)]


def test_process_nano_llm_backend_cancel_stops_open_stream():
    llm = ProcessNanoLlmBackend(make_process_cancelable_llm, context="fork", timeout=2)
    try:
        stream = llm.chat_stream("hello")
        assert next(stream) == "hello:first"
        llm.cancel()
        try:
            next(stream)
        except StopIteration:
            pass
        else:
            raise AssertionError("expected cancelled stream to finish")
    finally:
        llm.close()


def test_process_nano_llm_backend_reset_clears_child_history():
    llm = ProcessNanoLlmBackend(make_process_history_llm, context="fork", timeout=2)
    try:
        assert "".join(llm.chat_stream("first")) == "first"
        assert "".join(llm.chat_stream("second")) == "first|second"

        llm.reset()

        assert "".join(llm.chat_stream("third")) == "third"
    finally:
        llm.close()


def test_process_nano_llm_backend_preserves_session_history_isolation():
    llm = ProcessNanoLlmBackend(make_process_history_llm, context="fork", timeout=2)
    try:
        assert "".join(llm.chat_stream("alpha", session_id="a")) == "alpha"
        assert "".join(llm.chat_stream("beta", session_id="b")) == "beta"
        assert "".join(llm.chat_stream("next", session_id="a")) == "alpha|next"
        assert "".join(llm.chat_stream("next", session_id="b")) == "beta|next"

        llm.reset(session_id="a")
        assert "".join(llm.chat_stream("fresh", session_id="a")) == "fresh"
        assert "".join(llm.chat_stream("again", session_id="b")) == "beta|next|again"
    finally:
        llm.close()



def test_process_tts_backend_reports_streaming_capability_from_child_backend():
    plain = ProcessTtsBackend(make_process_tts, context="fork")
    streaming = ProcessTtsBackend(make_process_streaming_tts, context="fork")
    try:
        assert plain.supports_streaming_tts is False
        assert streaming.supports_streaming_tts is True
    finally:
        plain.close()
        streaming.close()
