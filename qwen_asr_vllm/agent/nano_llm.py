from __future__ import annotations

import queue
import sys
import threading
from dataclasses import dataclass
from itertools import count
from typing import Any, Callable, Sequence


def ensure_nanovllm_on_path(root: str = "/home/nano-vllm") -> None:
    if root and root not in sys.path:
        sys.path.insert(0, root)


def cuda_device_index(spec: str | None) -> int:
    if not spec:
        return 0
    spec = str(spec)
    if ":" in spec:
        return int(spec.rsplit(":", 1)[-1])
    return 0


def normalize_warmup_batch_sizes(
    sizes: Sequence[int],
    *,
    max_num_seqs: int,
) -> tuple[int, ...]:
    limit = max(1, int(max_num_seqs))
    normalized = []
    for size in sizes:
        size = int(size)
        if size <= 0:
            raise ValueError("warmup batch sizes must be positive")
        size = min(size, limit)
        if size not in normalized:
            normalized.append(size)
    return tuple(normalized)


@dataclass
class _ActiveNanoRequest:
    out: queue.Queue
    user_text: str
    session_id: Any = None
    emitted: str = ""
    full_reply: str = ""


class NanoVllmStepBatchingBackend:
    """nano-vLLM chat backend that batches concurrent streams via add_request/step."""

    resident_runner = True
    supports_session_history = True

    def __init__(
        self,
        *,
        engine: Any | None = None,
        model_path: str | None = None,
        sampling_cls: Callable[..., Any] | None = None,
        nanovllm_root: str = "/home/nano-vllm",
        device: str | None = "cuda:0",
        max_model_len: int = 4096,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int = 4,
        gpu_memory_utilization: float = 0.85,
        num_kvcache_blocks: int = -1,
        enforce_eager: bool = False,
        system_prompt: str = (
            "You are a helpful voice assistant. Keep answers concise and "
            "conversational."
        ),
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
        warmup: bool = True,
        warmup_batch_sizes: Sequence[int] = (1,),
        admission_window_ms: float = 5.0,
        name: str = "nano-vllm-step-batching-runner",
    ):
        num_kvcache_blocks = int(num_kvcache_blocks)
        if num_kvcache_blocks == 0 or num_kvcache_blocks < -1:
            raise ValueError("num_kvcache_blocks must be -1 or positive")
        if engine is None:
            if model_path is None:
                raise ValueError("model_path is required when engine is not provided")
            ensure_nanovllm_on_path(nanovllm_root)
            from nanovllm.llm import LLM
            from nanovllm.sampling_params import SamplingParams

            sampling_cls = sampling_cls or SamplingParams
            engine = LLM(
                model_path,
                device_index=cuda_device_index(device),
                max_model_len=int(max_model_len),
                max_num_batched_tokens=int(
                    max_num_batched_tokens or max(max_model_len, 2048)
                ),
                max_num_seqs=int(max_num_seqs),
                gpu_memory_utilization=float(gpu_memory_utilization),
                num_kvcache_blocks=num_kvcache_blocks,
                enforce_eager=bool(enforce_eager),
            )
        elif sampling_cls is None:
            ensure_nanovllm_on_path(nanovllm_root)
            from nanovllm.sampling_params import SamplingParams

            sampling_cls = SamplingParams

        self._engine = engine
        self._tokenizer = engine.tokenizer
        self._sampling_cls = sampling_cls
        self._system_prompt = system_prompt
        self._temperature = float(temperature)
        self._max_new_tokens = int(max_new_tokens)
        self._admission_window = max(0.0, admission_window_ms / 1000.0)
        self._histories: dict[Any, list[dict[str, str]]] = {}
        self._history_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = {"step_calls": 0, "max_step_batch_size": 0}
        self._warmup_stats = {
            "batch_sizes": [],
            "step_calls": 0,
            "max_step_batch_size": 0,
        }
        self._requests: queue.Queue[Any] = queue.Queue()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._closed = False
        self._ids = count(1)
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise self._startup_error
        if warmup:
            batches = normalize_warmup_batch_sizes(
                warmup_batch_sizes,
                max_num_seqs=max_num_seqs,
            )
            try:
                self._warmup_batches(batches)
            except BaseException:
                self.close()
                raise

    def chat_stream(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
        session_id: Any = None,
    ):
        if self._closed:
            raise RuntimeError("nano-vLLM step batching backend is closed")
        out: queue.Queue[Any] = queue.Queue()
        prompt = self._format_prompt(text, session_id=session_id)
        sampling = self._sampling_params(max_tokens=max_tokens)
        self._requests.put(
            ("stream", next(self._ids), text, prompt, sampling, out, session_id)
        )
        while True:
            kind, value = out.get()
            if kind == "chunk":
                yield value
                continue
            if kind == "done":
                return
            raise value

    def reset(self, *, session_id: Any = None) -> None:
        with self._history_lock:
            if session_id is None:
                self._histories.clear()
            else:
                self._histories.pop(session_id, None)

    @property
    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    @property
    def warmup_stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return dict(self._warmup_stats)

    def runtime_metrics(self) -> dict[str, Any]:
        return {
            "online": self.stats,
            "warmup": self.warmup_stats,
        }

    def _warmup_batches(self, batch_sizes: Sequence[int]) -> None:
        for batch_size in batch_sizes:
            commands = []
            outputs = []
            for _ in range(batch_size):
                out: queue.Queue[Any] = queue.Queue()
                outputs.append(out)
                commands.append(
                    (
                        "stream",
                        next(self._ids),
                        "hi",
                        self._format_prompt("hi"),
                        self._sampling_params(max_tokens=1),
                        out,
                        None,
                    )
                )
            self._requests.put(("warmup_batch", commands))
            for out in outputs:
                while True:
                    kind, value = out.get()
                    if kind == "chunk":
                        continue
                    if kind == "done":
                        break
                    raise value
            self.reset()
        with self._stats_lock:
            self._warmup_stats = {
                "batch_sizes": list(batch_sizes),
                **self._stats,
            }
            self._stats = {"step_calls": 0, "max_step_batch_size": 0}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._requests.put(("close",))
        self._thread.join(timeout=5)

    def _sampling_params(self, *, max_tokens: int | None = None):
        return self._sampling_cls(
            temperature=self._temperature,
            max_tokens=int(max_tokens or self._max_new_tokens),
        )

    def _format_prompt(self, user_text: str, *, session_id: Any = None) -> str:
        with self._history_lock:
            messages = list(self._histories.get(session_id, ()))
        if self._system_prompt:
            messages = [{"role": "system", "content": self._system_prompt}] + messages
        messages.append({"role": "user", "content": user_text})
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return self._tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    def _run(self) -> None:
        active: dict[int, _ActiveNanoRequest] = {}
        seq_to_request: dict[int, int] = {}
        closed = False
        try:
            self._activate_engine_device()
            self._ready.set()
            while not closed or active:
                if not active:
                    command = self._requests.get()
                    closed = self._handle_command(command, active, seq_to_request)
                    closed = (
                        self._drain_commands(
                            active, seq_to_request, wait=self._admission_window
                        )
                        or closed
                    )
                    if not active:
                        continue
                else:
                    closed = (
                        self._drain_commands(active, seq_to_request, wait=0.0)
                        or closed
                    )

                if active and not self._engine.is_finished():
                    finished = self._step_engine()
                    self._emit_running(active, seq_to_request)
                    self._emit_finished(finished, active, seq_to_request)
        except BaseException as exc:  # noqa: BLE001 - propagate to request streams
            for request in active.values():
                request.out.put(("error", exc))
        finally:
            self._safe_close_engine()

    def _activate_engine_device(self) -> None:
        model_runner = getattr(self._engine, "model_runner", None)
        config = getattr(model_runner, "config", None)
        device_index = getattr(config, "device_index", None)
        if device_index is None:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.set_device(int(device_index))
        except Exception:
            return

    def _step_engine(self):
        scheduler = getattr(self._engine, "scheduler", None)
        waiting_before = {
            getattr(seq, "seq_id", None)
            for seq in list(getattr(scheduler, "waiting", []) or [])
        }
        finished, num_tokens = self._engine.step()
        self._record_step_batch_size(waiting_before, finished, num_tokens)
        return finished

    def _record_step_batch_size(self, waiting_before, finished, num_tokens) -> None:
        scheduler = getattr(self._engine, "scheduler", None)
        if num_tokens < 0:
            batch_size = -int(num_tokens)
        else:
            running_after = {
                getattr(seq, "seq_id", None)
                for seq in list(getattr(scheduler, "running", []) or [])
            }
            finished_after = {item[0] for item in finished or []}
            batch_size = len(waiting_before & (running_after | finished_after))
        with self._stats_lock:
            self._stats["step_calls"] += 1
            self._stats["max_step_batch_size"] = max(
                self._stats["max_step_batch_size"],
                int(batch_size),
            )

    def _drain_commands(
        self,
        active: dict[int, _ActiveNanoRequest],
        seq_to_request: dict[int, int],
        *,
        wait: float,
    ) -> bool:
        closed = False
        while True:
            try:
                command = self._requests.get(timeout=wait)
            except queue.Empty:
                return closed
            if self._handle_command(command, active, seq_to_request):
                closed = True
            wait = 0.0

    def _handle_command(
        self,
        command,
        active: dict[int, _ActiveNanoRequest],
        seq_to_request: dict[int, int],
    ) -> bool:
        kind = command[0]
        if kind == "close":
            return True
        if kind == "warmup_batch":
            for stream_command in command[1]:
                self._handle_command(stream_command, active, seq_to_request)
            return False
        if kind != "stream":
            return False
        _, request_id, user_text, prompt, sampling, out, session_id = command
        try:
            seq_id = self._add_engine_request(prompt, sampling)
        except BaseException as exc:  # noqa: BLE001 - deliver through chat stream
            out.put(("error", exc))
            return False
        active[request_id] = _ActiveNanoRequest(
            out=out,
            user_text=user_text,
            session_id=session_id,
        )
        seq_to_request[seq_id] = request_id
        return False

    def _add_engine_request(self, prompt, sampling) -> int:
        returned = self._engine.add_request(prompt, sampling)
        if isinstance(returned, int):
            return returned
        seq_id = getattr(returned, "seq_id", None)
        if seq_id is not None:
            return int(seq_id)
        waiting = getattr(getattr(self._engine, "scheduler", None), "waiting", None)
        if waiting:
            return int(waiting[-1].seq_id)
        raise RuntimeError("nano-vLLM add_request did not expose a sequence id")

    def _emit_running(
        self,
        active: dict[int, _ActiveNanoRequest],
        seq_to_request: dict[int, int],
    ) -> None:
        running = getattr(getattr(self._engine, "scheduler", None), "running", [])
        for seq in list(running):
            request_id = seq_to_request.get(seq.seq_id)
            if request_id is None:
                continue
            self._emit_delta(active[request_id], seq.completion_token_ids)

    def _emit_finished(
        self,
        finished,
        active: dict[int, _ActiveNanoRequest],
        seq_to_request: dict[int, int],
    ) -> None:
        for item in finished or []:
            seq_id, token_ids = item[0], item[1]
            request_id = seq_to_request.pop(seq_id, None)
            if request_id is None:
                continue
            request = active.pop(request_id)
            self._emit_delta(request, token_ids)
            with self._history_lock:
                history = self._histories.setdefault(request.session_id, [])
                history.extend(
                    [
                        {"role": "user", "content": request.user_text},
                        {
                            "role": "assistant",
                            "content": request.full_reply.strip(),
                        },
                    ]
                )
            request.out.put(("done", None))

    def _emit_delta(self, request: _ActiveNanoRequest, token_ids) -> None:
        if not token_ids:
            return
        decoded = self._tokenizer.decode(token_ids, skip_special_tokens=True)
        if decoded.startswith(request.emitted):
            delta = decoded[len(request.emitted) :]
        elif len(decoded) > len(request.emitted):
            delta = decoded[len(request.emitted) :]
        else:
            delta = ""
        request.emitted = decoded
        if delta:
            request.full_reply += delta
            request.out.put(("chunk", delta))

    def _safe_close_engine(self) -> None:
        close = getattr(self._engine, "close", None)
        if callable(close):
            close()
            return
        exit_engine = getattr(self._engine, "exit", None)
        if callable(exit_engine):
            exit_engine()
