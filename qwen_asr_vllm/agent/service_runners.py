from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Generator
from itertools import count
from typing import Any, TypeVar

from qwen_asr_vllm.agent.shared_bytes import SharedBytesTransport

T = TypeVar("T")


def cuda_memory_snapshot(device=None, *, torch_module=None) -> dict[str, Any]:
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return {"available": False, "initialized": False}
    cuda = torch_module.cuda
    if not cuda.is_available():
        return {"available": False, "initialized": False}
    if not cuda.is_initialized():
        return {"available": True, "initialized": False}
    resolved_device = device if device is not None else f"cuda:{cuda.current_device()}"
    free, total = cuda.mem_get_info(resolved_device)
    return {
        "available": True,
        "initialized": True,
        "device": str(resolved_device),
        "allocated_bytes": int(cuda.memory_allocated(resolved_device)),
        "reserved_bytes": int(cuda.memory_reserved(resolved_device)),
        "peak_allocated_bytes": int(cuda.max_memory_allocated(resolved_device)),
        "device_used_bytes": int(total - free),
        "device_free_bytes": int(free),
        "device_total_bytes": int(total),
    }


def _backend_startup_metrics(backend: Any, **extra) -> dict[str, Any]:
    runtime_metrics = getattr(backend, "runtime_metrics", None)
    backend_metrics = runtime_metrics() if callable(runtime_metrics) else {}
    return {
        "pid": os.getpid(),
        "cuda_memory": cuda_memory_snapshot(),
        "backend": backend_metrics,
        **extra,
    }


def _normalize_ready_metrics(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, tuple):
        metrics = {"pid": payload[0]}
        if len(payload) > 1:
            metrics["supports_streaming"] = bool(payload[1])
        return metrics
    return {"pid": payload}


class _ThreadService:
    """Run a backend object on one long-lived thread and communicate by queues."""

    def __init__(
        self,
        backend: Any | Callable[[], Any],
        *,
        name: str,
        load_in_runner: bool = False,
    ):
        self._backend_or_factory = backend
        self._load_in_runner = load_in_runner
        self._requests: queue.Queue[Any] = queue.Queue()
        self._ready = threading.Event()
        self._closed = False
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise self._startup_error

    def call(self, fn: Callable[[Any], T]) -> T:
        if self._closed:
            raise RuntimeError("service runner is closed")
        result_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._requests.put(("call", fn, result_queue))
        ok, value = result_queue.get()
        if ok:
            return value
        raise value

    def stream(self, fn: Callable[[Any], Generator[T, None, None]]) -> Generator[T, None, None]:
        if self._closed:
            raise RuntimeError("service runner is closed")
        result_queue: queue.Queue[Any] = queue.Queue()
        sentinel = object()
        self._requests.put(("stream", fn, result_queue, sentinel))
        while True:
            ok, value = result_queue.get()
            if value is sentinel:
                return
            if ok:
                yield value
            else:
                raise value

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._requests.put(("close",))
        self._thread.join(timeout=5)

    def _run(self) -> None:
        try:
            backend = (
                self._backend_or_factory()
                if self._load_in_runner
                else self._backend_or_factory
            )
        except BaseException as exc:  # noqa: BLE001 - propagate startup errors
            self._startup_error = exc
            self._ready.set()
            return

        self._ready.set()
        while True:
            request = self._requests.get()
            kind = request[0]
            if kind == "close":
                close = getattr(backend, "close", None)
                if callable(close):
                    close()
                return
            if kind == "call":
                _, fn, result_queue = request
                try:
                    result_queue.put((True, fn(backend)))
                except BaseException as exc:  # noqa: BLE001 - propagate backend errors
                    result_queue.put((False, exc))
                continue
            if kind == "stream":
                _, fn, result_queue, sentinel = request
                try:
                    for item in fn(backend):
                        result_queue.put((True, item))
                except BaseException as exc:  # noqa: BLE001 - propagate backend errors
                    result_queue.put((False, exc))
                finally:
                    result_queue.put((True, sentinel))


class ThreadedAsrSession:
    def __init__(self, service: _ThreadService, session_id: int):
        self._service = service
        self._session_id = session_id

    def feed(self, pcm):
        return self._service.call(
            lambda state: state["sessions"][self._session_id].feed(pcm)
        )

    def close(self, reuse_last: bool = True):
        return self._service.call(
            lambda state: state["sessions"][self._session_id].close(
                reuse_last=reuse_last
            )
        )


class ThreadedAsrEngine:
    """ASR engine wrapper whose sessions live and run on one service thread."""

    def __init__(
        self,
        engine: Any | Callable[[], Any],
        *,
        load_in_runner: bool = False,
        name: str = "asr-service-runner",
    ):
        def create_state():
            backend = engine() if load_in_runner else engine
            return {"engine": backend, "sessions": {}, "next_id": 0}

        self._service = _ThreadService(create_state, name=name, load_in_runner=True)

    def open_stream(self, **kwargs) -> ThreadedAsrSession:
        session_id = self._service.call(self._open_stream(kwargs))
        return ThreadedAsrSession(self._service, session_id)

    def close(self) -> None:
        if not self._service._closed:
            self._service.call(self._close_state)
        self._service.close()

    @staticmethod
    def _open_stream(kwargs: dict[str, Any]):
        def open_stream(state):
            session_id = state["next_id"]
            state["next_id"] += 1
            state["sessions"][session_id] = state["engine"].open_stream(**kwargs)
            return session_id

        return open_stream

    @staticmethod
    def _close_state(state) -> None:
        for session in state["sessions"].values():
            close = getattr(session, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        close_engine = getattr(state["engine"], "close", None)
        if callable(close_engine):
            try:
                close_engine()
            except Exception:
                pass


class MultiplexedThreadedLlmBackend:
    """Single-engine LLM runner with request-id streams and round-robin stepping."""

    def __init__(
        self,
        llm: Any | Callable[[], Any],
        *,
        load_in_runner: bool = False,
        admission_window_ms: float = 5.0,
        name: str = "llm-mux-runner",
    ):
        self._llm_or_factory = llm
        self._load_in_runner = load_in_runner
        self._admission_window = max(0.0, admission_window_ms / 1000.0)
        self._requests: queue.Queue[Any] = queue.Queue()
        self._ready = threading.Event()
        self._closed = False
        self._startup_error: BaseException | None = None
        self._ids = count(1)
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise self._startup_error

    def chat_stream(self, text: str):
        if self._closed:
            raise RuntimeError("multiplexed LLM runner is closed")
        out: queue.Queue[Any] = queue.Queue()
        self._requests.put(("stream", next(self._ids), text, out))
        while True:
            ok, value = out.get()
            if ok == "chunk":
                yield value
                continue
            if ok == "done":
                return
            raise value

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._requests.put(("close",))
        self._thread.join(timeout=5)

    def _run(self) -> None:
        try:
            llm = self._llm_or_factory() if self._load_in_runner else self._llm_or_factory
        except BaseException as exc:  # noqa: BLE001
            self._startup_error = exc
            self._ready.set()
            return

        self._ready.set()
        active: dict[int, tuple[Any, queue.Queue[Any]]] = {}
        closed = False
        while not closed or active:
            if not active:
                command = self._requests.get()
                closed = self._handle_llm_mux_command(llm, command, active)
                closed = (
                    self._drain_llm_mux_requests(
                        llm, active, wait=self._admission_window
                    )
                    or closed
                )
            else:
                closed = (
                    self._drain_llm_mux_requests(llm, active, wait=0.0)
                    or closed
                )

            for request_id, (stream, out) in list(active.items()):
                try:
                    chunk = next(stream)
                    if chunk:
                        out.put(("chunk", chunk))
                except StopIteration:
                    out.put(("done", None))
                    active.pop(request_id, None)
                except BaseException as exc:  # noqa: BLE001
                    out.put(("error", exc))
                    active.pop(request_id, None)

        _safe_close(llm)

    def _drain_llm_mux_requests(self, llm, active, *, wait: float) -> bool:
        closed = False
        while True:
            try:
                command = self._requests.get(timeout=wait)
            except queue.Empty:
                return closed
            if self._handle_llm_mux_command(llm, command, active):
                closed = True
            wait = 0.0

    @staticmethod
    def _handle_llm_mux_command(llm, command, active) -> bool:
        kind = command[0]
        if kind == "close":
            return True
        if kind == "stream":
            _, request_id, text, out = command
            active[request_id] = (iter(llm.chat_stream(text)), out)
        return False


class MultiplexedThreadedTtsBackend:
    """Single-engine TTS runner with request-id micro-batching."""

    def __init__(
        self,
        tts: Any | Callable[[], Any],
        *,
        load_in_runner: bool = False,
        batch_window_ms: float = 10.0,
        max_batch_size: int = 8,
        name: str = "tts-mux-runner",
    ):
        self._tts_or_factory = tts
        self._load_in_runner = load_in_runner
        self._batch_window = max(0.0, batch_window_ms / 1000.0)
        self._max_batch_size = max(1, int(max_batch_size))
        self._requests: queue.Queue[Any] = queue.Queue()
        self._ready = threading.Event()
        self._closed = False
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise self._startup_error

    def synthesize(self, text: str) -> bytes | None:
        if self._closed:
            raise RuntimeError("multiplexed TTS runner is closed")
        out: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._requests.put(("synthesize", text, out))
        ok, value = out.get()
        if ok:
            return value
        raise value

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._requests.put(("close",))
        self._thread.join(timeout=5)

    def _run(self) -> None:
        try:
            tts = self._tts_or_factory() if self._load_in_runner else self._tts_or_factory
        except BaseException as exc:  # noqa: BLE001
            self._startup_error = exc
            self._ready.set()
            return

        self._ready.set()
        while True:
            command = self._requests.get()
            if command[0] == "close":
                _safe_close(tts)
                return
            batch = [command]
            deadline = threading.Event()
            deadline.wait(self._batch_window)
            while len(batch) < self._max_batch_size:
                try:
                    item = self._requests.get_nowait()
                except queue.Empty:
                    break
                if item[0] == "close":
                    self._requests.put(item)
                    break
                batch.append(item)
            self._run_tts_batch(tts, batch)

    @staticmethod
    def _run_tts_batch(tts, batch) -> None:
        texts = [item[1] for item in batch]
        outs = [item[2] for item in batch]
        try:
            synthesize_batch = getattr(tts, "synthesize_batch", None)
            if callable(synthesize_batch):
                audios = list(synthesize_batch(texts))
            else:
                audios = [tts.synthesize(text) for text in texts]
            if len(audios) != len(outs):
                raise RuntimeError("synthesize_batch returned the wrong number of items")
            for out, audio in zip(outs, audios):
                out.put((True, audio))
        except BaseException as exc:  # noqa: BLE001
            for out in outs:
                out.put((False, exc))


def _safe_close(obj: Any) -> None:
    close = getattr(obj, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _process_service_main(
    stage: str,
    factory,
    requests,
    responses,
    ready,
    shared_memory_threshold: int,
) -> None:
    payload_transport = SharedBytesTransport(threshold=shared_memory_threshold)
    try:
        backend = factory()
        state = {"engine": backend, "sessions": {}, "next_id": 0}
        ready.put(("ready", _backend_startup_metrics(backend)))
    except BaseException:  # noqa: BLE001 - report startup failure to parent
        ready.put(("error", traceback.format_exc()))
        return

    while True:
        command = requests.get()
        op = command[0]
        try:
            if op == "shutdown":
                for session in state["sessions"].values():
                    _safe_close(session)
                _safe_close(state["engine"])
                payload_transport.close()
                responses.put(("done", None))
                return
            if stage == "asr":
                _handle_asr_process_command(state, command, responses)
            elif stage == "tts":
                _handle_tts_process_command(
                    state["engine"], command, responses, payload_transport
                )
            else:
                responses.put(("error", f"unknown process runner stage: {stage}"))
        except BaseException:  # noqa: BLE001 - propagate backend error text
            responses.put(("error", traceback.format_exc()))


def _handle_asr_process_command(state, command, responses) -> None:
    op = command[0]
    if op == "open_stream":
        _, kwargs = command
        session_id = state["next_id"]
        state["next_id"] += 1
        state["sessions"][session_id] = state["engine"].open_stream(**kwargs)
        responses.put(("result", session_id))
        return
    if op == "feed":
        _, session_id, pcm = command
        responses.put(("result", state["sessions"][session_id].feed(pcm)))
        return
    if op == "close_session":
        _, session_id, reuse_last = command
        session = state["sessions"].get(session_id)
        if session is None:
            responses.put(("result", []))
            return
        responses.put(("result", session.close(reuse_last=reuse_last)))
        return
    responses.put(("error", f"unknown ASR command: {op}"))


def _handle_tts_process_command(tts, command, responses, payload_transport) -> None:
    op = command[0]
    if op == "supports_streaming":
        supports = getattr(tts, "supports_streaming_tts", None)
        if supports is None:
            supports = callable(getattr(tts, "synthesize_stream", None))
        responses.put(("result", bool(supports)))
        return
    if op == "synthesize":
        _, text = command
        responses.put(("result", payload_transport.encode(tts.synthesize(text))))
        return
    if op == "synthesize_stream":
        _, text = command
        synthesize_stream = getattr(tts, "synthesize_stream", None)
        try:
            if callable(synthesize_stream):
                for chunk in synthesize_stream(text):
                    if chunk:
                        responses.put(("chunk", payload_transport.encode(chunk)))
            else:
                audio = tts.synthesize(text)
                if audio:
                    responses.put(("chunk", payload_transport.encode(audio)))
            responses.put(("done", None))
        except BaseException:  # noqa: BLE001 - propagate backend error text
            responses.put(("error", traceback.format_exc()))
        return
    responses.put(("error", f"unknown TTS command: {op}"))


def _process_concurrent_llm_service_main(factory, requests, responses, ready) -> None:
    try:
        llm = factory()
        ready.put(("ready", _backend_startup_metrics(llm)))
    except BaseException:  # noqa: BLE001 - report startup failure to parent
        ready.put(("error", traceback.format_exc()))
        return

    active: set[int] = set()
    cancelled: set[int] = set()
    state_lock = threading.Lock()

    def is_cancelled(request_id: int) -> bool:
        with state_lock:
            return request_id in cancelled

    def run_stream(request_id: int, text: str, session_id: Any = None) -> None:
        with state_lock:
            active.add(request_id)
        try:
            stream = (
                llm.chat_stream(text, session_id=session_id)
                if getattr(llm, "supports_session_history", False)
                else llm.chat_stream(text)
            )
            for chunk in stream:
                if is_cancelled(request_id):
                    break
                if chunk:
                    responses.put(("chunk", request_id, chunk))
            responses.put(("done", request_id, None))
        except BaseException:  # noqa: BLE001 - propagate backend error text
            if not is_cancelled(request_id):
                responses.put(("error", request_id, traceback.format_exc()))
            else:
                responses.put(("done", request_id, None))
        finally:
            with state_lock:
                active.discard(request_id)
                cancelled.discard(request_id)

    def cancel_request(request_id: int | None) -> None:
        with state_lock:
            targets = set(active) if request_id is None else {int(request_id)}
            cancelled.update(targets)
        cancel = getattr(llm, "cancel", None)
        if callable(cancel):
            try:
                cancel(request_id)
            except TypeError:
                cancel()

    while True:
        command = requests.get()
        op = command[0]
        if op == "shutdown":
            _safe_close(llm)
            responses.put(("shutdown_done", None, None))
            return
        if op == "cancel":
            _, request_id = command
            cancel_request(request_id)
            continue
        if op == "reset":
            _, request_id, session_id = command
            reset = getattr(llm, "reset", None)
            try:
                if callable(reset):
                    if getattr(llm, "supports_session_history", False):
                        reset(session_id=session_id)
                    else:
                        reset()
                responses.put(("done", request_id, None))
            except BaseException:  # noqa: BLE001 - propagate backend error text
                responses.put(("error", request_id, traceback.format_exc()))
            continue
        if op == "chat_stream":
            _, request_id, text, session_id = command
            thread = threading.Thread(
                target=run_stream,
                args=(request_id, text, session_id),
                name=f"llm-process-stream-{request_id}",
                daemon=True,
            )
            thread.start()
            continue
        responses.put(
            (
                "error",
                command[1] if len(command) > 1 else -1,
                f"unknown LLM command: {op}",
            )
        )


class UnifiedTtsStreamBatchScheduler:
    """Legacy compatibility scheduler for explicit ``compat`` callers.

    The production profile never constructs this class. Current Qwen-TTS
    serving uses ``RequestIdCodecScheduler`` so request admission, codec steps,
    and outer active-prefix execution share one service-owned loop.
    """

    def __init__(
        self,
        synthesize_stream_batch: Callable[[list[str]], Generator[Any, None, None]],
        emit: Callable[[str, int, Any], None],
        *,
        is_cancelled: Callable[[int], bool] | None = None,
        compatibility_key: Callable[[str], Any] | None = None,
        batch_window_ms: float = 10.0,
        max_batch_size: int = 8,
        max_pending_requests: int = 256,
        on_metrics: Callable[[dict[str, Any]], None] | None = None,
        name: str = "tts-unified-stream-batcher",
    ):
        self._synthesize_stream_batch = synthesize_stream_batch
        self._emit = emit
        self._is_cancelled = is_cancelled or (lambda _request_id: False)
        self._compatibility_key = compatibility_key or (lambda _text: None)
        self._batch_window = max(0.0, float(batch_window_ms) / 1000.0)
        self._max_batch_size = max(1, int(max_batch_size))
        self._requests: queue.Queue[tuple[int, str, float] | None] = queue.Queue(
            maxsize=max(1, int(max_pending_requests))
        )
        self._pending: deque[tuple[int, str, float]] = deque()
        self._on_metrics = on_metrics
        self._metrics_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.batch_sizes: list[int] = []
        self.queue_wait_ms: list[float] = []
        self.refill_count = 0
        self.active_slots = 0
        self.padded_slots = 0
        self.max_active_slots = 0
        self.cancellations = 0
        self._thread.start()

    def submit(self, request_id: int, text: str) -> None:
        if self._closed:
            raise RuntimeError("unified TTS stream batch scheduler is closed")
        try:
            self._requests.put_nowait(
                (int(request_id), str(text), time.monotonic())
            )
        except queue.Full as exc:
            raise RuntimeError("unified TTS pending queue is full") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._requests.put(None)
        self._thread.join(timeout=5)

    def metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            return {
                "batch_sizes": list(self.batch_sizes),
                "batches": len(self.batch_sizes),
                "batch_window_ms": self._batch_window * 1000.0,
                "max_batch_size": self._max_batch_size,
                "queue_wait_ms": list(self.queue_wait_ms),
                "refill_count": self.refill_count,
                "active_slots": self.active_slots,
                "padded_slots": self.padded_slots,
                "max_active_slots": self.max_active_slots,
                "cancellations": self.cancellations,
                "pending": len(self._pending) + self._requests.qsize(),
                "closed": bool(self._closed),
            }

    def _run(self) -> None:
        while True:
            refill = bool(self._pending)
            if not self._pending:
                first = self._requests.get()
                if first is None:
                    return
                self._pending.append(first)
            if not refill and self._batch_window > 0.0:
                self._collect_until(time.monotonic() + self._batch_window)
            else:
                self._drain_available()
            batch = self._pop_compatible_batch()
            self._execute(batch, refill=refill)
            self._drain_available()

    def _collect_until(self, deadline: float) -> None:
        compatibility_key = self._compatibility_key(self._pending[0][1])
        compatible_count = sum(
            self._compatibility_key(item[1]) == compatibility_key
            for item in self._pending
        )
        while compatible_count < self._max_batch_size:
            timeout = deadline - time.monotonic()
            if timeout <= 0.0:
                return
            try:
                item = self._requests.get(timeout=timeout)
            except queue.Empty:
                return
            if item is None:
                self._requests.put_nowait(None)
                return
            self._pending.append(item)
            compatible_count += int(
                self._compatibility_key(item[1]) == compatibility_key
            )

    def _drain_available(self) -> None:
        while True:
            try:
                item = self._requests.get_nowait()
            except queue.Empty:
                return
            if item is None:
                self._requests.put_nowait(None)
                return
            self._pending.append(item)

    def _pop_compatible_batch(self) -> list[tuple[int, str, float]]:
        first_key = self._compatibility_key(self._pending[0][1])
        selected: list[tuple[int, str, float]] = []
        deferred: deque[tuple[int, str, float]] = deque()
        while self._pending:
            item = self._pending.popleft()
            if (
                len(selected) < self._max_batch_size
                and self._compatibility_key(item[1]) == first_key
            ):
                selected.append(item)
            else:
                deferred.append(item)
        self._pending = deferred
        return selected

    def _execute(
        self, batch: list[tuple[int, str, float]], *, refill: bool = False
    ) -> None:
        now = time.monotonic()
        active_batch = []
        cancelled_ids = []
        for request_id, text, submitted_at in batch:
            if self._is_cancelled(request_id):
                cancelled_ids.append(request_id)
            else:
                active_batch.append((request_id, text, submitted_at))
        for request_id in cancelled_ids:
            self._emit("done", request_id, None)
        if not active_batch:
            with self._metrics_lock:
                self.cancellations += len(cancelled_ids)
            return
        request_ids = [request_id for request_id, _text, _time in active_batch]
        texts = [text for _request_id, text, _time in active_batch]
        failure: BaseException | None = None
        try:
            for index, chunk in self._synthesize_stream_batch(texts):
                index = int(index)
                if index < 0 or index >= len(request_ids):
                    raise RuntimeError(
                        f"TTS batch returned invalid request index {index}"
                    )
                request_id = request_ids[index]
                if chunk and not self._is_cancelled(request_id):
                    self._emit("chunk", request_id, chunk)
        except BaseException as exc:  # noqa: BLE001 - propagate to request owners
            failure = exc
        finally:
            cancelled_during_batch = sum(
                self._is_cancelled(request_id) for request_id in request_ids
            )
            with self._metrics_lock:
                batch_size = len(active_batch)
                self.batch_sizes.append(batch_size)
                self.queue_wait_ms.extend(
                    (now - submitted_at) * 1000.0
                    for _request_id, _text, submitted_at in active_batch
                )
                self.refill_count += int(refill)
                self.active_slots += batch_size
                self.padded_slots += self._max_batch_size - batch_size
                self.max_active_slots = max(self.max_active_slots, batch_size)
                self.cancellations += len(cancelled_ids) + cancelled_during_batch
            if self._on_metrics is not None:
                self._on_metrics(self.metrics())
        for request_id in request_ids:
            if failure is not None and not self._is_cancelled(request_id):
                self._emit("error", request_id, str(failure))
            else:
                self._emit("done", request_id, None)


def _process_concurrent_tts_service_main(
    factory,
    requests,
    responses,
    ready,
    max_workers: int,
    batch_window_ms: float,
    max_batch_size: int,
    shared_memory_threshold: int,
) -> None:
    payload_transport = SharedBytesTransport(threshold=shared_memory_threshold)
    try:
        tts = factory()
        supports_streaming = getattr(tts, "supports_streaming_tts", None)
        if supports_streaming is None:
            supports_streaming = callable(getattr(tts, "synthesize_stream", None))
    except BaseException:  # noqa: BLE001 - report startup failure to parent
        ready.put(("error", traceback.format_exc()))
        payload_transport.close()
        return

    active: set[int] = set()
    cancelled: set[int] = set()
    state_lock = threading.Lock()
    slots = threading.Semaphore(max(1, int(max_workers)))
    batch_window = max(0.0, float(batch_window_ms) / 1000.0)
    max_batch_size = max(1, int(max_batch_size))

    def is_cancelled(request_id: int) -> bool:
        with state_lock:
            return request_id in cancelled

    def run_synthesize(request_id: int, text: str) -> None:
        with slots:
            with state_lock:
                active.add(request_id)
            try:
                if is_cancelled(request_id):
                    responses.put(("done", request_id, None))
                    return
                responses.put(
                    ("result", request_id, payload_transport.encode(tts.synthesize(text)))
                )
            except BaseException:  # noqa: BLE001 - propagate backend error text
                if not is_cancelled(request_id):
                    responses.put(("error", request_id, traceback.format_exc()))
            finally:
                with state_lock:
                    active.discard(request_id)
                    cancelled.discard(request_id)

    def run_stream(request_id: int, text: str) -> None:
        with slots:
            with state_lock:
                active.add(request_id)
            try:
                synthesize_stream = getattr(tts, "synthesize_stream", None)
                if callable(synthesize_stream):
                    for chunk in synthesize_stream(text):
                        if is_cancelled(request_id):
                            break
                        if chunk:
                            responses.put(
                                ("chunk", request_id, payload_transport.encode(chunk))
                            )
                else:
                    audio = tts.synthesize(text)
                    if audio and not is_cancelled(request_id):
                        responses.put(
                            ("chunk", request_id, payload_transport.encode(audio))
                        )
                responses.put(("done", request_id, None))
            except BaseException:  # noqa: BLE001 - propagate backend error text
                if not is_cancelled(request_id):
                    responses.put(("error", request_id, traceback.format_exc()))
                else:
                    responses.put(("done", request_id, None))
            finally:
                with state_lock:
                    active.discard(request_id)
                    cancelled.discard(request_id)

    def cancel_request(request_id: int | None) -> None:
        with state_lock:
            targets = set(active) if request_id is None else {int(request_id)}
            cancelled.update(targets)
        cancel = getattr(tts, "cancel", None)
        if callable(cancel):
            try:
                cancel(request_id)
            except TypeError:
                cancel()

    request_step_enabled = os.environ.get("VOICE_TTS_REQUEST_STEP_SCHEDULER") == "1"
    legacy_batch_enabled = os.environ.get("VOICE_TTS_LEGACY_STREAM_BATCH") == "1"
    if request_step_enabled and legacy_batch_enabled:
        ready.put(
            (
                "error",
                "request-id codec scheduler and legacy stream batch cannot both be enabled",
            )
        )
        _safe_close(tts)
        payload_transport.close()
        return

    stream_batch_scheduler = None

    def emit_batch_event(kind: str, request_id: int, payload: Any) -> None:
        if kind == "chunk" and is_cancelled(request_id):
            return
        responses.put((kind, request_id, payload_transport.encode(payload)))
        if kind in {"done", "error"}:
            with state_lock:
                active.discard(request_id)
                cancelled.discard(request_id)

    try:
        if request_step_enabled:
            from qwen_asr_vllm.agent.qwen_tts_request_scheduler import (
                RequestIdCodecScheduler,
            )

            stream_batch_scheduler = RequestIdCodecScheduler(
                tts,
                emit_batch_event,
                is_cancelled=is_cancelled,
                max_batch_size=max_batch_size,
                on_metrics=lambda metrics: responses.put(("metrics", None, metrics)),
            )
            scheduler_mode = "request-id-step"
        elif legacy_batch_enabled and callable(
            getattr(tts, "synthesize_stream_batch", None)
        ):
            stream_batch_scheduler = UnifiedTtsStreamBatchScheduler(
                tts.synthesize_stream_batch,
                emit_batch_event,
                is_cancelled=is_cancelled,
                compatibility_key=getattr(tts, "stream_batch_compatibility_key", None),
                batch_window_ms=batch_window * 1000.0,
                max_batch_size=max_batch_size,
                on_metrics=lambda metrics: responses.put(("metrics", None, metrics)),
            )
            scheduler_mode = "legacy-stream-batch"
        else:
            scheduler_mode = "scalar-stream"
    except BaseException as exc:  # noqa: BLE001 - fail startup, never downgrade
        ready.put(
            (
                "error",
                "request-id codec scheduler startup failed: "
                + "".join(traceback.format_exception(exc)),
            )
        )
        _safe_close(tts)
        payload_transport.close()
        return

    ready.put(
        (
            "ready",
            _backend_startup_metrics(
                tts,
                supports_streaming=bool(supports_streaming),
                tts_scheduler=scheduler_mode,
            ),
        )
    )

    while True:
        command = requests.get()
        op = command[0]
        if op == "shutdown":
            cancel_request(None)
            if stream_batch_scheduler is not None:
                stream_batch_scheduler.close()
            _safe_close(tts)
            payload_transport.close()
            responses.put(("shutdown_done", None, None))
            return
        if op == "cancel":
            _, request_id = command
            cancel_request(request_id)
            continue
        if op == "synthesize":
            _, request_id, text = command
            if request_step_enabled:
                responses.put(("error", request_id,
                               "request step scheduler requires streaming requests"))
                continue
            thread = threading.Thread(
                target=run_synthesize,
                args=(request_id, text),
                name=f"tts-process-synthesize-{request_id}",
                daemon=True,
            )
            thread.start()
            continue
        if op == "synthesize_stream":
            _, request_id, text = command
            if stream_batch_scheduler is not None:
                with state_lock:
                    active.add(request_id)
                try:
                    stream_batch_scheduler.submit(request_id, text)
                except Exception as exc:
                    emit_batch_event("error", request_id, str(exc))
            else:
                thread = threading.Thread(
                    target=run_stream,
                    args=(request_id, text),
                    name=f"tts-process-stream-{request_id}",
                    daemon=True,
                )
                thread.start()
            continue
        responses.put(
            (
                "error",
                command[1] if len(command) > 1 else -1,
                f"unknown TTS command: {op}",
            )
        )


class _ProcessService:
    def __init__(
        self,
        stage: str,
        factory: Callable[[], Any],
        *,
        context: str = "spawn",
        timeout: float = 300.0,
        shared_memory_threshold: int = 64 * 1024,
    ):
        self._timeout = float(timeout)
        self._ctx = mp.get_context(context)
        self._requests = self._ctx.Queue()
        self._responses = self._ctx.Queue()
        ready = self._ctx.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._closed = False
        self._payload_transport = SharedBytesTransport(
            threshold=shared_memory_threshold
        )
        self._process = self._ctx.Process(
            target=_process_service_main,
            args=(
                stage,
                factory,
                self._requests,
                self._responses,
                ready,
                max(1, int(shared_memory_threshold)),
            ),
            daemon=True,
        )
        self._process.start()
        kind, payload = ready.get(timeout=self._timeout)
        if kind == "error":
            self.close()
            raise RuntimeError(payload)
        self.startup_metrics = _normalize_ready_metrics(payload)

    def request(self, command: tuple[Any, ...]) -> Any:
        with self._lock:
            self._send(command)
            kind, payload = self._recv()
            if kind == "error":
                raise RuntimeError(payload)
            if kind != "result":
                raise RuntimeError(f"unexpected process response: {kind}")
            return payload

    def stream(self, command: tuple[Any, ...]):
        with self._lock:
            self._send(command)
            while True:
                kind, payload = self._recv()
                if kind == "chunk":
                    yield payload
                    continue
                if kind == "done":
                    return
                if kind == "error":
                    raise RuntimeError(payload)
                raise RuntimeError(f"unexpected process response: {kind}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.is_alive():
            try:
                with self._lock:
                    self._requests.put(("shutdown",))
                    self._responses.get(timeout=5)
            except Exception:
                pass
            self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._payload_transport.close()

    def _send(self, command: tuple[Any, ...]) -> None:
        if self._closed:
            raise RuntimeError("process service runner is closed")
        if not self._process.is_alive():
            raise RuntimeError("process service runner is not alive")
        self._requests.put(command)

    def _recv(self) -> tuple[str, Any]:
        try:
            kind, payload = self._responses.get(timeout=self._timeout)
            if kind in {"chunk", "result"}:
                payload = self._payload_transport.decode(payload)
            return kind, payload
        except queue.Empty as exc:
            raise TimeoutError("process service runner timed out") from exc

    def shared_memory_metrics(self) -> dict[str, Any]:
        return self._payload_transport.metrics()


class ProcessAsrSession:
    def __init__(self, service: _ProcessService, session_id: int):
        self._service = service
        self._session_id = session_id

    def feed(self, pcm):
        return self._service.request(("feed", self._session_id, pcm))

    def close(self, reuse_last: bool = True):
        return self._service.request(("close_session", self._session_id, reuse_last))


class ProcessAsrEngine:
    """ASR engine wrapper that runs the backend in a separate process."""

    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        context: str = "spawn",
        timeout: float = 300.0,
    ):
        self._service = _ProcessService(
            "asr", factory, context=context, timeout=timeout
        )

    def open_stream(self, **kwargs) -> ProcessAsrSession:
        session_id = self._service.request(("open_stream", kwargs))
        return ProcessAsrSession(self._service, session_id)

    def close(self) -> None:
        self._service.close()

    def runtime_metrics(self) -> dict[str, Any]:
        return dict(self._service.startup_metrics)


class ProcessNanoLlmBackend:
    """Concurrent LLM process wrapper with request-id stream demultiplexing.

    The child process owns one resident LLM backend. Each parent `chat_stream`
    call gets an independent request id, and the response reader dispatches
    chunks to the matching caller. With `NanoVllmStepBatchingBackend` inside the
    child, concurrent callers can reach the backend's `add_request`/`step`
    batching loop without arbitrary parent-side `to_thread` access to CUDA state.
    """

    resident_runner = True
    supports_session_history = True

    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        context: str = "spawn",
        timeout: float = 300.0,
    ):
        self._timeout = float(timeout)
        self._ctx = mp.get_context(context)
        self._requests = self._ctx.Queue()
        self._responses = self._ctx.Queue()
        ready = self._ctx.Queue(maxsize=1)
        self._outs: dict[int, queue.Queue[Any]] = {}
        self._outs_lock = threading.Lock()
        self._closed = False
        self._ids = count(1)
        self._shutdown_seen = threading.Event()
        self._reader: threading.Thread | None = None
        self._process = self._ctx.Process(
            target=_process_concurrent_llm_service_main,
            args=(factory, self._requests, self._responses, ready),
            daemon=True,
        )
        self._process.start()
        kind, payload = ready.get(timeout=self._timeout)
        if kind == "error":
            self.close()
            raise RuntimeError(payload)
        self._startup_metrics = _normalize_ready_metrics(payload)
        self._reader = threading.Thread(
            target=self._read_responses,
            name="llm-process-response-reader",
            daemon=True,
        )
        self._reader.start()

    def chat_stream(self, text: str, *, session_id: Any = None):
        if self._closed:
            raise RuntimeError("process nano LLM backend is closed")
        request_id = next(self._ids)
        out: queue.Queue[Any] = queue.Queue()
        done = False
        with self._outs_lock:
            if self._closed:
                raise RuntimeError("process nano LLM backend is closed")
            self._outs[request_id] = out
        try:
            self._requests.put(("chat_stream", request_id, text, session_id))
            while True:
                try:
                    kind, payload = out.get(timeout=self._timeout)
                except queue.Empty as exc:
                    self.cancel(request_id)
                    raise TimeoutError("process nano LLM stream timed out") from exc
                if kind == "chunk":
                    yield payload
                    continue
                if kind == "done":
                    done = True
                    return
                if kind == "error":
                    done = True
                    raise RuntimeError(payload)
                raise RuntimeError(f"unexpected process LLM response: {kind}")
        finally:
            if not done:
                self.cancel(request_id)
            with self._outs_lock:
                self._outs.pop(request_id, None)

    def cancel(self, request_id: int | None = None) -> None:
        if self._closed:
            return
        self._requests.put(("cancel", request_id))

    def reset(self, *, session_id: Any = None) -> None:
        if self._closed:
            raise RuntimeError("process nano LLM backend is closed")
        request_id = next(self._ids)
        out: queue.Queue[Any] = queue.Queue()
        with self._outs_lock:
            if self._closed:
                raise RuntimeError("process nano LLM backend is closed")
            self._outs[request_id] = out
        try:
            self._requests.put(("reset", request_id, session_id))
            try:
                kind, payload = out.get(timeout=self._timeout)
            except queue.Empty as exc:
                raise TimeoutError("process nano LLM reset timed out") from exc
            if kind == "done":
                return
            if kind == "error":
                raise RuntimeError(payload)
            raise RuntimeError(f"unexpected process LLM reset response: {kind}")
        finally:
            with self._outs_lock:
                self._outs.pop(request_id, None)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.is_alive():
            try:
                self._requests.put(("shutdown",))
            except Exception:
                pass
            self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._wake_streams(RuntimeError("process nano LLM backend is closed"))
        self._shutdown_seen.set()
        if self._reader is not None:
            self._reader.join(timeout=5)

    def runtime_metrics(self) -> dict[str, Any]:
        return dict(self._startup_metrics)

    def _read_responses(self) -> None:
        while not self._shutdown_seen.is_set():
            try:
                item = self._responses.get(timeout=0.1)
            except queue.Empty:
                if self._closed and not self._process.is_alive():
                    return
                continue
            kind = item[0]
            if kind == "shutdown_done":
                self._shutdown_seen.set()
                return
            _, request_id, payload = item
            with self._outs_lock:
                out = self._outs.get(request_id)
            if out is not None:
                out.put((kind, payload))

    def _wake_streams(self, exc: BaseException) -> None:
        with self._outs_lock:
            outs = list(self._outs.values())
        for out in outs:
            out.put(("error", exc))


class ProcessConcurrentTtsBackend:
    """Concurrent TTS process wrapper with request-id stream demultiplexing.

    The child process owns one resident TTS backend. Parent callers may consume
    independent `synthesize_stream` generators concurrently; the child limits
    concurrent backend calls with `max_workers`.
    """

    resident_runner = True

    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        context: str = "spawn",
        timeout: float = 300.0,
        max_workers: int = 2,
        batch_window_ms: float = 0.0,
        max_batch_size: int = 8,
        shared_memory_threshold: int = 64 * 1024,
    ):
        self._timeout = float(timeout)
        self._ctx = mp.get_context(context)
        self._requests = self._ctx.Queue()
        self._responses = self._ctx.Queue()
        ready = self._ctx.Queue(maxsize=1)
        self._outs: dict[int, queue.Queue[Any]] = {}
        self._outs_lock = threading.Lock()
        self._closed = False
        self._ids = count(1)
        self._shutdown_seen = threading.Event()
        self._metrics_lock = threading.Lock()
        self._scheduler_metrics: dict[str, Any] = {}
        self._payload_transport = SharedBytesTransport(
            threshold=shared_memory_threshold
        )
        self._reader: threading.Thread | None = None
        self._process = self._ctx.Process(
            target=_process_concurrent_tts_service_main,
            args=(
                factory,
                self._requests,
                self._responses,
                ready,
                max(1, int(max_workers)),
                max(0.0, float(batch_window_ms)),
                max(1, int(max_batch_size)),
                max(1, int(shared_memory_threshold)),
            ),
            daemon=True,
        )
        self._process.start()
        kind, payload = ready.get(timeout=self._timeout)
        if kind == "error":
            self.close()
            raise RuntimeError(payload)
        self._startup_metrics = _normalize_ready_metrics(payload)
        self.supports_streaming_tts = bool(
            self._startup_metrics.get("supports_streaming", False)
        )
        self._reader = threading.Thread(
            target=self._read_responses,
            name="tts-process-response-reader",
            daemon=True,
        )
        self._reader.start()

    def synthesize(self, text: str) -> bytes | None:
        if self._closed:
            raise RuntimeError("process concurrent TTS backend is closed")
        request_id = next(self._ids)
        out = self._register_out(request_id)
        done = False
        try:
            self._requests.put(("synthesize", request_id, text))
            while True:
                kind, payload = self._wait_out(out, request_id)
                if kind == "result":
                    done = True
                    return payload
                if kind == "done":
                    done = True
                    return None
                if kind == "error":
                    done = True
                    raise RuntimeError(payload)
                raise RuntimeError(f"unexpected process TTS response: {kind}")
        finally:
            if not done:
                self.cancel(request_id)
            self._drop_out(request_id)

    def synthesize_stream(self, text: str):
        if self._closed:
            raise RuntimeError("process concurrent TTS backend is closed")
        request_id = next(self._ids)
        out = self._register_out(request_id)
        done = False
        try:
            self._requests.put(("synthesize_stream", request_id, text))
            while True:
                kind, payload = self._wait_out(out, request_id)
                if kind == "chunk":
                    yield payload
                    continue
                if kind == "done":
                    done = True
                    return
                if kind == "error":
                    done = True
                    raise RuntimeError(payload)
                raise RuntimeError(f"unexpected process TTS response: {kind}")
        finally:
            if not done:
                self.cancel(request_id)
            self._drop_out(request_id)

    def cancel(self, request_id: int | None = None) -> None:
        if self._closed:
            return
        self._requests.put(("cancel", request_id))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.is_alive():
            try:
                self._requests.put(("shutdown",))
            except Exception:
                pass
            self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._wake_streams(RuntimeError("process concurrent TTS backend is closed"))
        self._shutdown_seen.set()
        if self._reader is not None:
            self._reader.join(timeout=5)
        self._payload_transport.close()

    def runtime_metrics(self) -> dict[str, Any]:
        metrics = dict(self._startup_metrics)
        with self._metrics_lock:
            if self._scheduler_metrics:
                metrics["scheduler"] = dict(self._scheduler_metrics)
        metrics["shared_memory"] = self._payload_transport.metrics()
        return metrics

    def _register_out(self, request_id: int) -> queue.Queue[Any]:
        out: queue.Queue[Any] = queue.Queue()
        with self._outs_lock:
            if self._closed:
                raise RuntimeError("process concurrent TTS backend is closed")
            self._outs[request_id] = out
        return out

    def _drop_out(self, request_id: int) -> None:
        with self._outs_lock:
            self._outs.pop(request_id, None)

    def _wait_out(
        self, out: queue.Queue[Any], request_id: int
    ) -> tuple[str, Any]:
        try:
            return out.get(timeout=self._timeout)
        except queue.Empty as exc:
            self.cancel(request_id)
            raise TimeoutError("process concurrent TTS stream timed out") from exc

    def _read_responses(self) -> None:
        while not self._shutdown_seen.is_set():
            try:
                item = self._responses.get(timeout=0.1)
            except queue.Empty:
                if self._closed and not self._process.is_alive():
                    return
                continue
            kind = item[0]
            if kind == "shutdown_done":
                self._shutdown_seen.set()
                return
            _, request_id, payload = item
            if kind == "metrics":
                with self._metrics_lock:
                    self._scheduler_metrics = dict(payload)
                continue
            if kind in {"chunk", "result"}:
                try:
                    payload = self._payload_transport.decode(payload)
                except BaseException as exc:  # noqa: BLE001 - wake request owner
                    kind, payload = "error", exc
            with self._outs_lock:
                out = self._outs.get(request_id)
            if out is not None:
                out.put((kind, payload))

    def _wake_streams(self, exc: BaseException) -> None:
        with self._outs_lock:
            outs = list(self._outs.values())
        for out in outs:
            out.put(("error", exc))


class ProcessTtsBackend:
    """TTS backend wrapper that synthesizes in a separate process."""

    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        context: str = "spawn",
        timeout: float = 300.0,
        shared_memory_threshold: int = 64 * 1024,
    ):
        self._service = _ProcessService(
            "tts",
            factory,
            context=context,
            timeout=timeout,
            shared_memory_threshold=shared_memory_threshold,
        )
        self.supports_streaming_tts = bool(
            self._service.request(("supports_streaming",))
        )

    def synthesize(self, text: str) -> bytes | None:
        return self._service.request(("synthesize", text))

    def synthesize_stream(self, text: str):
        yield from self._service.stream(("synthesize_stream", text))

    def close(self) -> None:
        self._service.close()

    def runtime_metrics(self) -> dict[str, Any]:
        metrics = dict(self._service.startup_metrics)
        metrics["shared_memory"] = self._service.shared_memory_metrics()
        return metrics
