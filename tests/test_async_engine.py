"""Concurrent submission, cancellation and timeouts.

Driven against a stub engine rather than a real one: what needs pinning here is the
threading contract -- that engine state is only ever touched by the loop thread,
that futures resolve exactly once, that a cancelled request never resolves with a
transcription -- and none of that involves a GPU.
"""
import threading
import time

import numpy as np
import pytest

from qwen_asr_vllm.engine.async_engine import AsyncAsrEngine, RequestCancelled
from qwen_asr_vllm.engine.engine import AsrOutput
from qwen_asr_vllm.engine.request import RequestTimings
from qwen_asr_vllm.engine.scheduler import SchedulerStats
from qwen_asr_vllm.metrics import Metrics


class StubEngine:
    """Stands in for AsrEngine: same surface, counts thread affinity violations."""

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
        self.mutating_threads: set[str] = set()
        self.prepare_threads: set[str] = set()
        self.reject = None
        # When set, step() leaves the queue alone: a request admitted while stalled
        # never completes, which is how the timeout path is reached deterministically.
        self.stall = False

    # ---- prepare runs on worker threads
    def prepare_request(self, waveform, sample_rate, context, language, sampling):
        self.prepare_threads.add(threading.current_thread().name)
        if self.reject is not None:
            raise self.reject
        request = type(
            "Request",
            (),
            {
                "request_id": self._next_id,
                "features": type("Features", (), {"audio_seconds": 1.0})(),
            },
        )()
        self._next_id += 1
        return request

    # ---- everything below must only ever run on the loop thread
    def admit(self, request):
        self.mutating_threads.add(threading.current_thread().name)
        self._queue.append(request)
        self.metrics.record_received(request.features.audio_seconds)
        return request

    def cancel(self, request_id):
        self.mutating_threads.add(threading.current_thread().name)
        if any(r.request_id == request_id for r in self._queue):
            self._cancelled.add(request_id)
            return True
        return False

    @property
    def has_work(self):
        return bool(self._queue)

    def step(self):
        self.mutating_threads.add(threading.current_thread().name)
        time.sleep(self.step_seconds)
        if not self._queue or (self.stall and not self._cancelled):
            return []
        request = self._queue.pop(0)
        cancelled = request.request_id in self._cancelled
        output = AsrOutput(
            request_id=request.request_id,
            text="" if cancelled else f"text-{request.request_id}",
            language="en",
            raw_text="",
            audio_seconds=1.0,
            num_prompt_tokens=16,
            num_audio_tokens=13,
            num_output_tokens=0 if cancelled else 5,
            finish_reason="cancelled" if cancelled else "stop",
            timings=RequestTimings(finish=time.perf_counter()),
        )
        self.metrics.record_finished(output)
        return [output]


@pytest.fixture
def async_engine():
    engines = []

    def build(**kwargs):
        stub = StubEngine(**kwargs)
        engine = AsyncAsrEngine(engine=stub, frontend_workers=4)
        engines.append(engine)
        return engine, stub

    yield build
    for engine in engines:
        engine.close()


WAVEFORM = np.zeros(16000, dtype=np.float32)


def test_submit_returns_before_the_result_is_ready(async_engine):
    engine, _ = async_engine(step_seconds=0.05)
    handle = engine.submit(WAVEFORM)
    assert not handle.done()
    assert handle.result(timeout=5).text.startswith("text-")


def test_many_concurrent_submissions_all_resolve_once(async_engine):
    engine, _ = async_engine()
    handles = [engine.submit(WAVEFORM) for _ in range(32)]
    texts = [handle.result(timeout=10).text for handle in handles]

    assert len(set(texts)) == 32
    assert all(handle.done() for handle in handles)


def test_engine_state_is_touched_by_exactly_one_thread(async_engine):
    engine, stub = async_engine()
    for _ in range(16):
        engine.submit(WAVEFORM)
    time.sleep(0.3)

    assert stub.mutating_threads == {"asr-engine"}
    # The whole point of the split: mel work did not happen on the loop thread.
    assert stub.prepare_threads
    assert "asr-engine" not in stub.prepare_threads


def test_frontend_rejection_reaches_the_caller(async_engine):
    engine, stub = async_engine()
    stub.reject = ValueError("audio is too long")

    with pytest.raises(ValueError, match="too long"):
        engine.submit(WAVEFORM).result(timeout=5)
    assert engine.metrics.requests_failed == 1


def test_cancel_before_completion_raises_instead_of_returning_text(async_engine):
    engine, _ = async_engine(step_seconds=0.2)
    first = engine.submit(WAVEFORM)
    second = engine.submit(WAVEFORM)

    assert second.cancel() is True
    with pytest.raises(RequestCancelled):
        second.result(timeout=5)
    # Cancelling one request must not disturb its neighbour.
    assert first.result(timeout=5).text == "text-0"
    assert engine.metrics.requests_cancelled == 1


def test_cancelling_a_finished_request_reports_false(async_engine):
    engine, _ = async_engine()
    handle = engine.submit(WAVEFORM)
    handle.result(timeout=5)

    assert handle.cancel() is False


def test_timeout_fails_the_future_and_is_counted(async_engine):
    engine, _ = async_engine(step_seconds=0.3)
    engine.submit(WAVEFORM)  # occupies the loop
    slow = engine.submit(WAVEFORM, timeout=0.05)

    with pytest.raises(RequestCancelled, match="timed out"):
        slow.result(timeout=5)
    assert engine.metrics.requests_timed_out == 1


def test_health_reports_liveness_and_depth(async_engine):
    engine, _ = async_engine(step_seconds=0.2)
    engine.submit(WAVEFORM)
    time.sleep(0.05)

    health = engine.health()
    assert health["status"] == "ok"
    assert health["kv_cache_blocks"] == 64
    assert health["in_flight"] >= 1


def test_close_fails_outstanding_requests_rather_than_hanging(async_engine):
    engine, _ = async_engine(step_seconds=1.0)
    handles = [engine.submit(WAVEFORM) for _ in range(4)]
    time.sleep(0.05)
    engine.close()

    outstanding = [h for h in handles if not h.done()]
    for handle in outstanding:
        with pytest.raises(RequestCancelled, match="shutdown"):
            handle.result(timeout=5)


def test_submitting_after_close_is_refused(async_engine):
    engine, _ = async_engine()
    engine.close()
    with pytest.raises(RuntimeError, match="shutting down"):
        engine.submit(WAVEFORM)


def test_idle_engine_does_not_spin(async_engine):
    """An idle loop must park, not busy-wait; otherwise it burns a core per engine."""
    _, stub = async_engine()
    time.sleep(0.2)
    steps_seen = len(stub.mutating_threads)
    time.sleep(0.2)

    # No work submitted, so step() should never have been reached with an empty queue.
    assert not stub.has_work
    assert len(stub.mutating_threads) == steps_seen
