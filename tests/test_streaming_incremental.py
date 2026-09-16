"""Incremental streaming: tail decode + bounded window recompute."""

from __future__ import annotations

import time

import numpy as np
import pytest

from qwen_asr_vllm.engine.async_engine import AsyncAsrEngine
from qwen_asr_vllm.engine.engine import AsrOutput
from qwen_asr_vllm.engine.request import RequestTimings
from qwen_asr_vllm.engine.scheduler import SchedulerStats
from qwen_asr_vllm.engine.streaming import join_committed_tail
from qwen_asr_vllm.metrics import Metrics


class IncrementalStubEngine:
    """Returns ``w0..w{n-1}`` with n = max(1, round(seconds * words_per_second))."""

    def __init__(self, words_per_second: float = 2.0, step_seconds: float = 0.001):
        self.metrics = Metrics()
        self.config = type("Config", (), {"model": "stub"})()
        self.num_kvcache_blocks = 64
        self.block_manager = type("BlockManager", (), {"num_free_blocks": 64})()
        self.scheduler = self
        self.stats = SchedulerStats()
        self.step_seconds = step_seconds
        self.words_per_second = words_per_second
        self._next_id = 0
        self._queue: list = []
        self._cancelled: set[int] = set()
        self.waveform_lengths: list[int] = []

    def prepare_request(self, waveform, sample_rate, context, language, sampling):
        self.waveform_lengths.append(len(waveform))
        seconds = len(waveform) / sample_rate
        n_words = max(1, round(seconds * self.words_per_second))
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
        self.metrics.record_received(request.features.audio_seconds)
        return request

    def cancel(self, request_id):
        if any(r.request_id == request_id for r in self._queue):
            self._cancelled.add(request_id)
            return True
        return False

    @property
    def has_work(self):
        return bool(self._queue)

    def step(self):
        time.sleep(self.step_seconds)
        if not self._queue:
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


@pytest.fixture
def inc_engine():
    engines = []

    def build(**kwargs):
        stub = IncrementalStubEngine(**kwargs)
        engine = AsyncAsrEngine(engine=stub, frontend_workers=2)
        engines.append(engine)
        return engine, stub

    yield build
    for engine in engines:
        engine.close()


class TestIncrementalHelpers:
    def test_join_committed_tail(self):
        assert join_committed_tail("a b", "c") == "a b c"
        assert join_committed_tail("", "c") == "c"


class TestIncrementalSession:
    def test_tail_transcribes_less_than_full_retranscribe(self, inc_engine):
        engine, stub = inc_engine()
        # 10s of audio in 1s feeds; recompute every 100s so mostly tail-only.
        session = engine.open_stream(
            chunk_policy="incremental",
            commit_lag_words=1,
            recompute_seconds=100.0,
            recompute_overlap_seconds=0.0,
            tail_min_seconds=0.5,
        )
        for _ in range(10):
            session.feed(np.zeros(16000, dtype=np.float32))
        session.close(reuse_last=True)

        # Growing-prefix baseline would send 1+2+...+10 = 55s of audio.
        assert session.total_transcribed_seconds < 25.0
        assert session._committed_audio_end > 0
        assert sum(stub.waveform_lengths) == session.total_transcribed_samples

    def test_window_recompute_runs(self, inc_engine):
        engine, _ = inc_engine()
        session = engine.open_stream(
            chunk_policy="incremental",
            commit_lag_words=2,
            recompute_seconds=3.0,
            recompute_overlap_seconds=1.0,
            tail_min_seconds=0.5,
        )
        for _ in range(6):
            session.feed(np.zeros(16000, dtype=np.float32))
        assert session.window_recomputes >= 1
        session.close()

    def test_partial_joins_committed_and_tail(self, inc_engine):
        engine, _ = inc_engine(words_per_second=2.0)
        session = engine.open_stream(
            chunk_policy="incremental",
            commit_lag_words=2,
            recompute_seconds=100.0,
            tail_min_seconds=0.5,
        )
        # 3s → ~6 words from stub if full; first feed is tail from 0.
        events = session.feed(np.zeros(3 * 16000, dtype=np.float32))
        partials = [e for e in events if e.kind == "partial"]
        assert partials
        assert partials[0].text.startswith("w0")
        # After commit lag 2, committed should be non-empty for 6-word hyp.
        if session.committed_text:
            events2 = session.feed(np.zeros(16000, dtype=np.float32))
            p2 = next(e for e in events2 if e.kind == "partial")
            assert p2.text.startswith(session.committed_text) or session.committed_text in p2.text
        session.close()

    def test_default_lag_for_incremental(self, inc_engine):
        engine, _ = inc_engine()
        session = engine.open_stream(chunk_policy="incremental")
        assert session.commit_lag_words == 16


@pytest.mark.gpu
@pytest.mark.checkpoint
@pytest.mark.slow
class TestIncrementalGpu:
    def test_incremental_sends_less_audio_than_retranscribe(self, model_dir):
        from bench.data import load_librispeech

        sample = load_librispeech(split="test-clean", num_samples=8, seed=5)
        sample = max(sample, key=lambda s: s.duration)
        audio = sample.audio[: int(12.0 * 16000)]
        chunk = int(2.0 * 16000)

        engine = AsyncAsrEngine(
            model=model_dir,
            max_model_len=4096,
            max_num_batched_tokens=4096,
            max_num_seqs=1,
            gpu_memory_utilization=0.4,
        )
        try:
            full = engine.open_stream(language="en", chunk_policy="retranscribe")
            for start in range(0, len(audio), chunk):
                full.feed(audio[start : start + chunk])
            full.close(reuse_last=True)

            inc = engine.open_stream(
                language="en",
                chunk_policy="incremental",
                commit_lag_words=8,
                recompute_seconds=6.0,
                recompute_overlap_seconds=2.0,
                tail_min_seconds=0.5,
            )
            for start in range(0, len(audio), chunk):
                inc.feed(audio[start : start + chunk])
            finals = inc.close(reuse_last=True)
            assert finals and finals[0].kind == "final"
            assert finals[0].text.strip()
            # Retranscribe growing prefix ≈ 2+4+...+12 = 42s; incremental should be lower.
            assert inc.total_transcribed_seconds < full.total_transcribed_seconds
            assert inc.total_transcribed_seconds < 30.0
        finally:
            engine.close()
