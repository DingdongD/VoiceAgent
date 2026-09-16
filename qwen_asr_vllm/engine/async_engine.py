"""Concurrent front door for AsrEngine.

``AsrEngine`` is single-threaded by construction: the scheduler, block manager and
KV cache are all plain mutable state with no locking, which is the right trade for
an inference loop. Serving needs many callers at once, so this wraps it rather than
making it thread-safe -- exactly one thread ever touches engine state.

    caller threads          worker pool             loop thread
    submit(waveform) --> mel + prompt build --> admit, step, deliver

Splitting the frontend out is not only about lock discipline. Mel extraction is
pure CPU, and running it inline meant the GPU sat idle through it; here it overlaps
with the encode and decode steps of requests already in flight.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from qwen_asr_vllm.config import EngineConfig
from qwen_asr_vllm.engine.engine import AsrEngine, AsrOutput
from qwen_asr_vllm.engine.request import AsrRequest, SamplingParams

logger = logging.getLogger(__name__)

# How long the loop thread parks when there is nothing to do. Short enough that a
# submission is picked up promptly, long enough not to spin a core while idle. The
# submit path also sets an event, so this is only a backstop for deadline checks.
IDLE_POLL_SECONDS = 0.005


class RequestCancelled(Exception):
    """Raised from ``handle.result()`` when a request was cancelled or timed out."""


class RequestHandle:
    """A submitted request: the future carrying its transcription, plus cancellation.

    The request id is assigned on a worker thread when the prompt is built, so it is
    not available the instant ``submit`` returns. ``cancel`` waits for it rather than
    exposing that race to callers.
    """

    def __init__(self, engine: AsyncAsrEngine):
        self._engine = engine
        self._id_ready = threading.Event()
        self._request_id: int | None = None
        self.future: Future = Future()

    def _set_request_id(self, request_id: int) -> None:
        self._request_id = request_id
        self._id_ready.set()

    @property
    def request_id(self) -> int | None:
        return self._request_id

    def result(self, timeout: float | None = None) -> AsrOutput:
        return self.future.result(timeout=timeout)

    def done(self) -> bool:
        return self.future.done()

    def cancel(self, timeout: float = 1.0) -> bool:
        """Drop this request. Returns whether it was still in flight."""
        if not self._id_ready.wait(timeout=timeout):
            return False
        return self._engine.cancel(self._request_id)


class AsyncAsrEngine:
    """Thread-safe wrapper driving the engine loop in the background."""

    def __init__(
        self,
        model: str | None = None,
        config: EngineConfig | None = None,
        engine: AsrEngine | None = None,
        frontend_workers: int = 4,
        **kwargs,
    ):
        self.engine = engine or AsrEngine(model=model, config=config, **kwargs)
        self.metrics = self.engine.metrics
        self.config = self.engine.config

        self._submissions: queue.SimpleQueue = queue.SimpleQueue()
        # Callables run on the loop thread against ``self.engine`` (draft verify +
        # residual decode must touch the block manager / KV cache there).
        self._engine_jobs: queue.SimpleQueue = queue.SimpleQueue()
        self._futures: dict[int, Future] = {}
        self._deadlines: dict[int, float] = {}
        self._cancel_requests: set[int] = set()
        self._state_lock = threading.Lock()
        self._work = threading.Event()
        self._closing = threading.Event()

        self._frontend = ThreadPoolExecutor(
            max_workers=frontend_workers, thread_name_prefix="asr-frontend"
        )
        self._loop = threading.Thread(target=self._run_loop, name="asr-engine", daemon=True)
        self._loop.start()

    # ------------------------------------------------------------------ submit

    def submit(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        context: str = "",
        language: str | None = None,
        sampling: SamplingParams | None = None,
        timeout: float | None = None,
    ) -> RequestHandle:
        """Queue one transcription. Returns immediately.

        The returned future resolves with an ``AsrOutput``, or raises: whatever
        ``prepare_request`` rejected (audio too long, too short), or
        ``RequestCancelled`` if the request was cancelled or outlived ``timeout``.
        """
        if self._closing.is_set():
            raise RuntimeError("engine is shutting down")

        handle = RequestHandle(self)
        future = handle.future
        # A monotonic deadline, so the timeout covers queueing as well as compute --
        # under load that is where the time actually goes.
        deadline = time.monotonic() + timeout if timeout is not None else None

        def prepare() -> None:
            try:
                request = self.engine.prepare_request(
                    waveform, sample_rate, context, language, sampling
                )
            except BaseException as exc:  # noqa: BLE001 - delivered to the caller
                self.metrics.record_failed()
                future.set_exception(exc)
                handle._set_request_id(-1)
                return
            with self._state_lock:
                self._futures[request.request_id] = future
                if deadline is not None:
                    self._deadlines[request.request_id] = deadline
            handle._set_request_id(request.request_id)
            self._submissions.put(request)
            self._work.set()

        self._frontend.submit(prepare)
        return handle

    def transcribe(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        context: str = "",
        language: str | None = None,
        sampling: SamplingParams | None = None,
        timeout: float | None = None,
    ) -> AsrOutput:
        """Blocking convenience wrapper around ``submit``."""
        return self.submit(
            waveform, sample_rate, context, language, sampling, timeout
        ).result()

    def open_stream(
        self,
        language: str | None = None,
        context: str = "",
        commit_lag_words: int | None = None,
        chunk_policy: str = "speculate",
        sample_rate: int = 16000,
        recompute_seconds: float = 6.0,
        recompute_overlap_seconds: float = 2.0,
        tail_min_seconds: float = 0.5,
        on_violation: str = "keep",
    ):
        """Open a streaming session.

        Default ``chunk_policy`` is ``speculate`` (WER-preserving draft acceleration).
        ``retranscribe`` is the slow baseline; ``incremental`` trades WER for latency
        and logs a warning. When ``commit_lag_words`` is omitted, incremental
        defaults to 16 and the others to 0.
        """
        from qwen_asr_vllm.engine.streaming import StreamingSession

        if chunk_policy == "incremental":
            logger.warning(
                "chunk_policy='incremental' reduces audio work but regresses WER "
                "(~0.03→~0.33 under default knobs); use 'speculate' for "
                "WER-preserving streaming acceleration"
            )
        if commit_lag_words is None:
            commit_lag_words = 16 if chunk_policy == "incremental" else 0

        return StreamingSession(
            self,
            language=language,
            context=context,
            commit_lag_words=commit_lag_words,
            chunk_policy=chunk_policy,
            sample_rate=sample_rate,
            recompute_seconds=recompute_seconds,
            recompute_overlap_seconds=recompute_overlap_seconds,
            tail_min_seconds=tail_min_seconds,
            on_violation=on_violation,
        )

    def transcribe_with_draft(
        self,
        waveform: np.ndarray,
        draft_token_ids: list[int] | None = None,
        sample_rate: int = 16000,
        context: str = "",
        language: str | None = None,
        sampling: SamplingParams | None = None,
    ) -> AsrOutput:
        """Run ``AsrEngine.transcribe_with_draft`` on the loop thread."""
        if self._closing.is_set():
            raise RuntimeError("engine is shutting down")
        future: Future = Future()

        def job(engine: AsrEngine) -> AsrOutput:
            return engine.transcribe_with_draft(
                waveform,
                draft_token_ids=draft_token_ids,
                sample_rate=sample_rate,
                context=context,
                language=language,
                sampling=sampling,
            )

        self._engine_jobs.put((job, future))
        self._work.set()
        return future.result()

    def cancel(self, request_id: int | None) -> bool:
        """Ask the loop thread to drop a request. Returns whether it was in flight.

        Cancellation is queued rather than applied here: touching the scheduler from
        a caller thread is exactly what this class exists to prevent.
        """
        if request_id is None:
            return False
        with self._state_lock:
            if request_id not in self._futures:
                return False
            self._cancel_requests.add(request_id)
        self._work.set()
        return True

    # -------------------------------------------------------------------- loop

    def _run_loop(self) -> None:
        while not self._closing.is_set():
            admitted = self._drain_submissions()
            cancelled = self._apply_cancellations()
            expired = self._expire_overdue()
            jobs = self._drain_engine_jobs()

            outputs = []
            if self.engine.scheduler.has_work:
                outputs = self.engine.step()
            self._deliver(outputs)

            if not (admitted or cancelled or expired or outputs or jobs) and not (
                self.engine.scheduler.has_work
            ):
                # Nothing pending: wait to be woken rather than burning a core.
                self._work.wait(timeout=IDLE_POLL_SECONDS)
                self._work.clear()

    def _drain_engine_jobs(self) -> int:
        ran = 0
        while True:
            try:
                job, future = self._engine_jobs.get_nowait()
            except queue.Empty:
                return ran
            if future.done():
                continue
            try:
                future.set_result(job(self.engine))
            except BaseException as exc:  # noqa: BLE001 - delivered to the caller
                future.set_exception(exc)
            ran += 1

    def _drain_submissions(self) -> int:
        admitted = 0
        while True:
            try:
                request: AsrRequest = self._submissions.get_nowait()
            except queue.Empty:
                return admitted
            self.engine.admit(request)
            admitted += 1

    def _apply_cancellations(self) -> int:
        with self._state_lock:
            pending, self._cancel_requests = self._cancel_requests, set()
        for request_id in pending:
            if self.engine.cancel(request_id):
                self.metrics.record_cancelled()
            else:
                # Finished between the ask and now; its output is already on its way.
                logger.debug("cancellation for request %d arrived too late", request_id)
        return len(pending)

    def _expire_overdue(self) -> int:
        now = time.monotonic()
        with self._state_lock:
            overdue = [rid for rid, deadline in self._deadlines.items() if deadline <= now]
        for request_id in overdue:
            with self._state_lock:
                self._deadlines.pop(request_id, None)
            if self.engine.cancel(request_id):
                self.metrics.record_timed_out()
                self._fail(request_id, RequestCancelled(f"request {request_id} timed out"))
        return len(overdue)

    def _deliver(self, outputs: list[AsrOutput]) -> None:
        for output in outputs:
            with self._state_lock:
                future = self._futures.pop(output.request_id, None)
                self._deadlines.pop(output.request_id, None)
            if future is None or future.done():
                continue
            if output.finish_reason == "cancelled":
                future.set_exception(
                    RequestCancelled(f"request {output.request_id} was cancelled")
                )
            else:
                future.set_result(output)

    def _fail(self, request_id: int, exc: BaseException) -> None:
        with self._state_lock:
            future = self._futures.pop(request_id, None)
        if future is not None and not future.done():
            future.set_exception(exc)

    # ---------------------------------------------------------------- lifecycle

    @property
    def num_in_flight(self) -> int:
        with self._state_lock:
            return len(self._futures)

    def health(self) -> dict:
        return {
            "status": "ok" if self._loop.is_alive() else "loop_thread_dead",
            "in_flight": self.num_in_flight,
            "kv_cache_blocks": self.engine.num_kvcache_blocks,
            "kv_cache_blocks_free": self.engine.block_manager.num_free_blocks,
            "model": self.config.model,
        }

    def close(self, timeout: float = 10.0) -> None:
        """Stop the loop thread, failing anything still in flight."""
        if self._closing.is_set():
            return
        self._closing.set()
        self._work.set()
        self._loop.join(timeout=timeout)
        self._frontend.shutdown(wait=False, cancel_futures=True)
        with self._state_lock:
            pending, self._futures = self._futures, {}
        for request_id, future in pending.items():
            if not future.done():
                future.set_exception(
                    RequestCancelled(f"request {request_id} dropped at shutdown")
                )

    def __enter__(self) -> AsyncAsrEngine:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
