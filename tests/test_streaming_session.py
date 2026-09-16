"""Streaming session: commit protocol and growing-prefix re-transcribe."""

from __future__ import annotations

import time

import numpy as np
import pytest

from qwen_asr_vllm.engine.async_engine import AsyncAsrEngine, RequestCancelled
from qwen_asr_vllm.engine.engine import AsrOutput
from qwen_asr_vllm.engine.request import RequestTimings
from qwen_asr_vllm.engine.scheduler import SchedulerStats
from qwen_asr_vllm.engine.streaming import (
    advance_committed,
    commit_candidate,
    join_words,
    split_words,
)
from qwen_asr_vllm.metrics import Metrics


class TestCommitHelpers:
    def test_commit_candidate_drops_lag(self):
        assert commit_candidate("one two three four", 2) == "one two"

    def test_commit_candidate_lag_zero_is_full(self):
        assert commit_candidate("one two", 0) == "one two"

    def test_commit_candidate_lag_past_end_is_empty(self):
        assert commit_candidate("one two", 5) == ""

    def test_advance_extends_prefix(self):
        text, violation = advance_committed("one two", "one two three")
        assert text == "one two three" and not violation

    def test_advance_conflict_keeps_previous(self):
        text, violation = advance_committed("one two", "one dos three")
        assert text == "one two" and violation

    def test_split_join_roundtrip(self):
        assert join_words(split_words("a  b")) == "a b"


class StreamingStubEngine:
    """AsyncAsrEngine stand-in: text grows with buffered audio length."""

    def __init__(self, step_seconds: float = 0.001):
        self.metrics = Metrics()
        self.config = type("Config", (), {"model": "stub"})()
        self.num_kvcache_blocks = 64
        self.block_manager = type("BlockManager", (), {"num_free_blocks": 64})()
        self.scheduler = self
        self.stats = SchedulerStats()
        self.step_seconds = step_seconds
        self._next_id = 0
        self._queue: list = []
        self._cancelled: set[int] = set()
        self.stall = False
        self.submit_count = 0
        self.cancel_count = 0

    def prepare_request(self, waveform, sample_rate, context, language, sampling):
        seconds = len(waveform) / sample_rate
        n_words = max(1, int(seconds))
        text = " ".join(f"w{i}" for i in range(n_words))
        request = type(
            "Request",
            (),
            {
                "request_id": self._next_id,
                "features": type("Features", (), {"audio_seconds": seconds})(),
                "text": text,
                "token_ids": list(range(1000, 1000 + n_words)),
            },
        )()
        self._next_id += 1
        return request

    def admit(self, request):
        self._queue.append(request)
        self.submit_count += 1
        self.metrics.record_received(request.features.audio_seconds)
        return request

    def cancel(self, request_id):
        if any(r.request_id == request_id for r in self._queue):
            self._cancelled.add(request_id)
            self.cancel_count += 1
            return True
        return False

    @property
    def has_work(self):
        return bool(self._queue)

    def step(self):
        time.sleep(self.step_seconds)
        if not self._queue or (self.stall and not self._cancelled):
            return []
        request = self._queue.pop(0)
        cancelled = request.request_id in self._cancelled
        output = AsrOutput(
            request_id=request.request_id,
            text="" if cancelled else request.text,
            language="en",
            raw_text="",
            audio_seconds=request.features.audio_seconds,
            num_prompt_tokens=16,
            num_audio_tokens=13,
            num_output_tokens=0 if cancelled else len(request.token_ids),
            finish_reason="cancelled" if cancelled else "stop",
            timings=RequestTimings(finish=time.perf_counter()),
            output_token_ids=[] if cancelled else list(request.token_ids),
        )
        self.metrics.record_finished(output)
        return [output]

    def transcribe_with_draft(
        self,
        waveform,
        draft_token_ids=None,
        sample_rate=16000,
        context="",
        language=None,
        sampling=None,
    ):
        """Loop-thread stub for the default ``speculate`` policy."""
        seconds = len(waveform) / sample_rate
        n_words = max(1, int(seconds))
        text = " ".join(f"w{i}" for i in range(n_words))
        tokens = list(range(1000, 1000 + n_words))
        draft = list(draft_token_ids or [])
        accepted = 0
        for left, right in zip(draft, tokens):
            if left != right:
                break
            accepted += 1
        return AsrOutput(
            request_id=self._next_id,
            text=text,
            language="en",
            raw_text=text,
            audio_seconds=seconds,
            num_prompt_tokens=16,
            num_audio_tokens=13,
            num_output_tokens=len(tokens),
            finish_reason="stop",
            timings=RequestTimings(finish=time.perf_counter()),
            output_token_ids=tokens,
            draft_tokens=len(draft),
            draft_accepted=accepted,
        )


@pytest.fixture
def streaming_engine():
    engines = []

    def build(**kwargs):
        stub = StreamingStubEngine(**kwargs)
        engine = AsyncAsrEngine(engine=stub, frontend_workers=2)
        engines.append(engine)
        return engine, stub

    yield build
    for engine in engines:
        engine.close()


class TestStreamingSession:
    def test_default_policy_is_speculate(self, streaming_engine):
        engine, _ = streaming_engine()
        session = engine.open_stream()
        assert session.chunk_policy == "speculate"

    def test_growing_feeds_emit_partials(self, streaming_engine):
        engine, _ = streaming_engine()
        session = engine.open_stream(language="en", chunk_policy="retranscribe")
        events1 = session.feed(np.zeros(16000, dtype=np.float32))
        events2 = session.feed(np.zeros(16000, dtype=np.float32))
        assert [e.kind for e in events1] == ["partial"]
        assert events1[0].text == "w0"
        assert [e.kind for e in events2] == ["partial"]
        assert events2[0].text == "w0 w1"
        assert events2[0].chunk_index == 1

    def test_close_reuses_last_partial(self, streaming_engine):
        engine, stub = streaming_engine()
        session = engine.open_stream(chunk_policy="retranscribe")
        session.feed(np.zeros(16000, dtype=np.float32))
        submits_before = stub.submit_count
        finals = session.close(reuse_last=True)
        assert stub.submit_count == submits_before
        assert len(finals) == 1 and finals[0].kind == "final"
        assert finals[0].text == "w0"

    def test_feed_after_close_raises(self, streaming_engine):
        engine, _ = streaming_engine()
        session = engine.open_stream(chunk_policy="retranscribe")
        session.feed(np.zeros(16000, dtype=np.float32))
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            session.feed(np.zeros(16000, dtype=np.float32))

    def test_commit_lag_emits_committed(self, streaming_engine):
        engine, _ = streaming_engine()
        session = engine.open_stream(commit_lag_words=1, chunk_policy="retranscribe")
        # 3s → words w0 w1 w2; lag 1 → commit w0 w1
        events = session.feed(np.zeros(48000, dtype=np.float32))
        kinds = [e.kind for e in events]
        assert kinds == ["partial", "committed"]
        assert events[1].text == "w0 w1"
        assert session.committed_text == "w0 w1"

    def test_commit_violation_keeps_history(self, streaming_engine):
        engine, stub = streaming_engine()
        session = engine.open_stream(commit_lag_words=1, chunk_policy="retranscribe")

        # Force a stable commit first.
        session.feed(np.zeros(48000, dtype=np.float32))
        assert session.committed_text == "w0 w1"

        # Next prepare returns a conflicting hypothesis for the same length.
        original = stub.prepare_request

        def conflicting(waveform, sample_rate, context, language, sampling):
            request = original(waveform, sample_rate, context, language, sampling)
            request.text = "x0 x1 x2 x3"
            request.token_ids = [1, 2, 3, 4]
            return request

        stub.prepare_request = conflicting
        events = session.feed(np.zeros(16000, dtype=np.float32))
        committed = [e for e in events if e.kind == "committed"]
        assert committed and committed[0].commit_violation
        assert session.committed_text == "w0 w1"

    def test_second_feed_cancels_in_flight(self, streaming_engine):
        engine, stub = streaming_engine(step_seconds=0.01)
        stub.stall = True
        session = engine.open_stream(chunk_policy="retranscribe")
        handle = engine.submit(np.zeros(16000, dtype=np.float32))
        # Wait until admitted into the stub queue so cancel can find it.
        deadline = time.time() + 2
        while not stub._queue and time.time() < deadline:
            time.sleep(0.005)
        assert stub._queue and not handle.done()
        session._in_flight = handle
        session._cancel_in_flight()
        assert stub.cancel_count >= 1
        with pytest.raises(RequestCancelled):
            handle.result(timeout=2)

    def test_rejects_unknown_policy(self, streaming_engine):
        engine, _ = streaming_engine()
        with pytest.raises(ValueError, match="chunk_policy"):
            engine.open_stream(chunk_policy="bogus")

    def test_speculate_policy_accepted_by_constructor(self, streaming_engine):
        engine, _ = streaming_engine()
        session = engine.open_stream(chunk_policy="speculate")
        assert session.chunk_policy == "speculate"

    def test_short_audio_emits_nothing(self, streaming_engine):
        engine, stub = streaming_engine()
        session = engine.open_stream()
        assert session.feed(np.zeros(80, dtype=np.float32)) == []
        assert stub.submit_count == 0


@pytest.mark.gpu
@pytest.mark.checkpoint
@pytest.mark.slow
class TestStreamingSessionGpu:
    def test_final_matches_oneshot_transcribe(self, model_dir):
        from bench.data import load_librispeech
        from bench.metrics import normalize_text

        sample = load_librispeech(split="test-clean", num_samples=8, seed=1)
        sample = max(sample, key=lambda s: s.duration)
        # Use ~4s so two 2s feeds stay short for the test budget.
        audio = sample.audio[: int(4.0 * 16000)]
        engine = AsyncAsrEngine(
            model=model_dir,
            max_model_len=2048,
            max_num_batched_tokens=2048,
            max_num_seqs=1,
            gpu_memory_utilization=0.35,
        )
        try:
            oneshot = engine.transcribe(audio, language="en")
            session = engine.open_stream(language="en")
            chunk = int(2.0 * 16000)
            for start in range(0, len(audio), chunk):
                session.feed(audio[start : start + chunk])
            finals = session.close(reuse_last=True)
            assert len(finals) == 1 and finals[0].kind == "final"
            assert normalize_text(finals[0].text) == normalize_text(oneshot.text)
            # Let the loop drain any cancel leftovers.
            deadline = time.time() + 5
            while engine.num_in_flight and time.time() < deadline:
                time.sleep(0.05)
            assert engine.num_in_flight == 0
        finally:
            engine.close()

    def test_speculate_session_matches_retranscribe(self, model_dir):
        from bench.data import load_librispeech
        from bench.metrics import normalize_text

        sample = load_librispeech(split="test-clean", num_samples=8, seed=4)
        sample = max(sample, key=lambda s: s.duration)
        audio = sample.audio[: int(6.0 * 16000)]
        chunk = int(2.0 * 16000)
        engine = AsyncAsrEngine(
            model=model_dir,
            max_model_len=2048,
            max_num_batched_tokens=2048,
            max_num_seqs=1,
            gpu_memory_utilization=0.35,
        )
        try:
            base = engine.open_stream(language="en", chunk_policy="retranscribe")
            for start in range(0, len(audio), chunk):
                base.feed(audio[start : start + chunk])
            base_final = base.close(reuse_last=True)[0]

            spec = engine.open_stream(language="en", chunk_policy="speculate")
            for start in range(0, len(audio), chunk):
                spec.feed(audio[start : start + chunk])
            spec_final = spec.close(reuse_last=True)[0]

            assert normalize_text(spec_final.text) == normalize_text(base_final.text)
            assert spec.total_draft_tokens > 0
            assert 0 <= spec.total_draft_accepted <= spec.total_draft_tokens
        finally:
            engine.close()
