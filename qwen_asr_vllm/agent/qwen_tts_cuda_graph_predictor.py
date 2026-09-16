from __future__ import annotations

import queue
import threading
import time
import hashlib
import os
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import count
from types import SimpleNamespace
from typing import Any

from qwen_asr_vllm.agent.qwen_tts_fast_predictor import (
    FastCodePredictorError,
    _sample_next_token,
)


class CUDAGraphCodePredictorEngine:
    def __init__(
        self,
        code_predictor: Any,
        *,
        max_graph_batch_size: int = 1,
        bundle_factory=None,
        device_context_factory=None,
        allow_non_cuda_for_testing: bool = False,
        prewarm_batch_sizes: tuple[int, ...] = (),
    ):
        self._code_predictor = code_predictor
        self._max_graph_batch_size = max(1, int(max_graph_batch_size))
        self._bundle_factory = bundle_factory or _create_cuda_graph_bundle
        self._device_context_factory = (
            device_context_factory or _input_device_context
        )
        self._allow_non_cuda_for_testing = bool(allow_non_cuda_for_testing)
        self._prewarm_batch_sizes = tuple(
            sorted(
                {
                    int(batch_size)
                    for batch_size in prewarm_batch_sizes
                    if 1 < int(batch_size) <= self._max_graph_batch_size
                }
            )
        )
        self._bundles: dict[tuple[Any, ...], Any] = {}
        self._bundle_lock = threading.Lock()
        self._prewarmed_shapes: set[tuple[Any, ...]] = set()
        self._closed = False
        self._steady_state = False
        self.graph_captures = 0
        self.graph_replays = 0
        self.lazy_captures = 0
        self._trace_tokens: list[str] = []
        self._trace_token_values: list[list[list[int]]] = []
        self._trace_logits: list[list[dict[str, float | int]]] = []
        self._trace_cache: list[list[str]] = []
        self._trace_hidden: list[list[dict[str, float] | None]] = []

    def generate(
        self,
        *,
        inputs_embeds,
        max_new_tokens: int,
        do_sample: bool | None = True,
        top_p: float | None = 1.0,
        top_k: int | None = 50,
        temperature: float | None = 0.9,
        output_hidden_states: bool | None = None,
        return_dict_in_generate: bool | None = True,
        **kwargs,
    ):
        if self._closed:
            raise FastCodePredictorError("CUDA Graph code predictor is closed")
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise FastCodePredictorError(
                f"unsupported CUDA Graph code predictor kwargs: {unsupported}"
            )
        if inputs_embeds is None:
            raise FastCodePredictorError("inputs_embeds is required")
        if max_new_tokens <= 0:
            raise FastCodePredictorError("max_new_tokens must be positive")
        self._require_graph_supported(inputs_embeds)
        call_kwargs = {
            "inputs_embeds": inputs_embeds,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "output_hidden_states": output_hidden_states,
            "return_dict_in_generate": return_dict_in_generate,
        }

        output_hidden_states = bool(output_hidden_states)
        key = self._bundle_key(
            inputs_embeds,
            int(max_new_tokens),
            output_hidden_states=output_hidden_states,
        )
        with self._bundle_lock:
            was_cached = key in self._bundles
        bundle = self._get_bundle(
            key,
            inputs_embeds=inputs_embeds,
            max_new_tokens=int(max_new_tokens),
            output_hidden_states=output_hidden_states,
        )
        with self._device_context_factory(inputs_embeds):
            result = bundle.generate(**call_kwargs)
        if getattr(result, "sequences", None) is not None:
            self._trace_token_values.append(result.sequences.detach().to("cpu").tolist())
            self._trace_tokens.append(
                hashlib.sha256(
                    result.sequences.detach().to("cpu").contiguous().numpy().tobytes()
                ).hexdigest()
            )
            if os.environ.get("VOICE_TTS_TRACE_INNER_LOGITS", "0") == "1":
                trace = getattr(bundle, "last_logits_trace", None)
                if trace is not None:
                    self._trace_logits.append(trace)
                cache_trace = getattr(bundle, "last_cache_trace", None)
                if cache_trace is not None:
                    self._trace_cache.append(cache_trace)
                hidden_trace = getattr(bundle, "last_hidden_trace", None)
                if hidden_trace is not None:
                    self._trace_hidden.append(hidden_trace)
        if not was_cached and self._steady_state:
            self.lazy_captures += 1
        self.graph_replays += 1
        return result

    def prewarm(
        self,
        *,
        inputs_embeds,
        max_new_tokens: int,
        output_hidden_states: bool = False,
    ) -> None:
        """Capture the request shape before serving traffic."""
        if self._closed:
            raise FastCodePredictorError("CUDA Graph code predictor is closed")
        self._require_graph_supported(inputs_embeds)
        key = self._bundle_key(
            inputs_embeds,
            int(max_new_tokens),
            output_hidden_states=bool(output_hidden_states),
        )
        self._get_bundle(
            key,
            inputs_embeds=inputs_embeds,
            max_new_tokens=int(max_new_tokens),
            output_hidden_states=bool(output_hidden_states),
        )

    def mark_steady_state(self) -> None:
        """Declare that resident warmup is complete."""
        if self._closed:
            raise FastCodePredictorError("CUDA Graph code predictor is closed")
        self._steady_state = True

    def metrics_snapshot(self) -> dict[str, Any]:
        with self._bundle_lock:
            return {
                "graph_captures": int(self.graph_captures),
                "graph_replays": int(self.graph_replays),
                "lazy_captures": int(self.lazy_captures),
                "bundles": len(self._bundles),
                "bundle_keys": [repr(key) for key in self._bundles],
                "steady_state": bool(self._steady_state),
                "closed": bool(self._closed),
                "trace_tokens": list(self._trace_tokens),
                "trace_token_values": list(self._trace_token_values),
                "trace_logits": list(self._trace_logits),
                "trace_cache": list(self._trace_cache),
                "trace_hidden": list(self._trace_hidden),
            }

    def close(self) -> None:
        with self._bundle_lock:
            if self._closed:
                return
            self._closed = True
            bundles = list(self._bundles.values())
            self._bundles.clear()
        for bundle in bundles:
            close = getattr(bundle, "close", None)
            if callable(close):
                close()

    def _require_graph_supported(self, inputs_embeds) -> None:
        if inputs_embeds is None:
            raise FastCodePredictorError("inputs_embeds is required")
        if int(inputs_embeds.shape[0]) > self._max_graph_batch_size:
            raise FastCodePredictorError(
                "CUDA Graph batch exceeds max_graph_batch_size: "
                f"batch={int(inputs_embeds.shape[0])}, "
                f"max={self._max_graph_batch_size}"
            )
        if (
            getattr(inputs_embeds.device, "type", None) != "cuda"
            and not self._allow_non_cuda_for_testing
        ):
            raise FastCodePredictorError(
                "CUDA Graph code predictor requires CUDA inputs"
            )

    def _get_bundle(
        self,
        key,
        *,
        inputs_embeds,
        max_new_tokens: int,
        output_hidden_states: bool,
    ):
        with self._bundle_lock:
            if self._closed:
                raise FastCodePredictorError("CUDA Graph code predictor is closed")
            bundle = self._bundles.get(key)
            if bundle is None:
                bundle = self._capture_bundle(
                    key,
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=max_new_tokens,
                    output_hidden_states=output_hidden_states,
                )
            self._prewarm_fixed_batches(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                output_hidden_states=output_hidden_states,
            )
            return bundle

    @staticmethod
    def _bundle_key(
        inputs_embeds,
        max_new_tokens: int,
        *,
        output_hidden_states: bool = False,
    ):
        return (
            tuple(int(item) for item in inputs_embeds.shape),
            str(inputs_embeds.device),
            str(inputs_embeds.dtype),
            bool(output_hidden_states),
            int(max_new_tokens),
        )

    def _capture_bundle(
        self,
        key,
        *,
        inputs_embeds,
        max_new_tokens: int,
        output_hidden_states: bool,
    ):
        with self._device_context_factory(inputs_embeds):
            bundle = self._bundle_factory(
                code_predictor=self._code_predictor,
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                output_hidden_states=output_hidden_states,
            )
        self._bundles[key] = bundle
        self.graph_captures += 1
        return bundle

    def _prewarm_fixed_batches(
        self,
        *,
        inputs_embeds,
        max_new_tokens: int,
        output_hidden_states: bool,
    ) -> None:
        if int(inputs_embeds.shape[0]) != 1 or not self._prewarm_batch_sizes:
            return
        shape_key = (
            tuple(int(item) for item in inputs_embeds.shape[1:]),
            str(inputs_embeds.device),
            str(inputs_embeds.dtype),
            bool(output_hidden_states),
            int(max_new_tokens),
        )
        if shape_key in self._prewarmed_shapes:
            return
        for batch_size in self._prewarm_batch_sizes:
            repeats = (batch_size,) + (1,) * (inputs_embeds.ndim - 1)
            batched_inputs = inputs_embeds.repeat(repeats)
            key = self._bundle_key(
                batched_inputs,
                max_new_tokens,
                output_hidden_states=output_hidden_states,
            )
            if key not in self._bundles:
                self._capture_bundle(
                    key,
                    inputs_embeds=batched_inputs,
                    max_new_tokens=max_new_tokens,
                    output_hidden_states=output_hidden_states,
                )
        self._prewarmed_shapes.add(shape_key)


@dataclass
class _GraphRequest:
    request_id: int
    call_kwargs: dict[str, Any]
    signature: tuple[Any, ...]
    batch_size: int
    event: threading.Event
    result: Any = None
    error: BaseException | None = None


class FixedSlotCUDAGraphBatcher:
    """Batch complete predictor requests into pre-captured fixed CUDA slots."""

    def __init__(
        self,
        engine: CUDAGraphCodePredictorEngine,
        *,
        slot_count: int = 2,
        batch_window_ms: float = 2.0,
    ):
        self._engine = engine
        self._slot_count = max(1, int(slot_count))
        self._batch_window = max(0.0, float(batch_window_ms) / 1000.0)
        self._queue: queue.Queue[_GraphRequest | None] = queue.Queue()
        self._ids = count(1)
        self._closed = False
        self.request_batches: list[int] = []
        self.slot_batches: list[int] = []
        self.padded_slots = 0
        self._thread = threading.Thread(
            target=self._run,
            name="qwen-tts-cuda-graph-slot-batcher",
            daemon=True,
        )
        self._thread.start()

    def generate(
        self,
        *,
        inputs_embeds,
        max_new_tokens: int,
        do_sample: bool | None = True,
        top_p: float | None = 1.0,
        top_k: int | None = 50,
        temperature: float | None = 0.9,
        output_hidden_states: bool | None = None,
        return_dict_in_generate: bool | None = True,
        **kwargs,
    ):
        if self._closed:
            raise FastCodePredictorError("CUDA Graph slot batcher is closed")
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise FastCodePredictorError(
                f"unsupported CUDA Graph code predictor kwargs: {unsupported}"
            )
        if inputs_embeds is None:
            raise FastCodePredictorError("inputs_embeds is required")
        if max_new_tokens <= 0:
            raise FastCodePredictorError("max_new_tokens must be positive")

        call_kwargs = {
            "inputs_embeds": inputs_embeds,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "output_hidden_states": output_hidden_states,
            "return_dict_in_generate": return_dict_in_generate,
        }
        batch_size = int(inputs_embeds.shape[0])
        if batch_size > self._slot_count:
            raise FastCodePredictorError(
                "CUDA Graph slot batch exceeds slot_count: "
                f"batch={batch_size}, slots={self._slot_count}"
            )
        request = _GraphRequest(
            request_id=next(self._ids),
            call_kwargs=call_kwargs,
            signature=_graph_request_signature(call_kwargs),
            batch_size=batch_size,
            event=threading.Event(),
        )
        self._queue.put(request)
        request.event.wait()
        if request.error is not None:
            raise request.error
        return request.result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=5)
        if not self._thread.is_alive():
            close = getattr(self._engine, "close", None)
            if callable(close):
                close()

    def metrics_snapshot(self) -> dict[str, Any]:
        result = self._engine.metrics_snapshot()
        result.update(
            {
                "request_batches": list(self.request_batches),
                "slot_batches": list(self.slot_batches),
                "padded_slots": int(self.padded_slots),
                "slot_count": int(self._slot_count),
                "batch_window_ms": self._batch_window * 1000.0,
            }
        )
        return result

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                return
            batch = [first]
            occupied = first.batch_size
            if self._batch_window > 0.0 and occupied < self._slot_count:
                deadline = time.monotonic() + self._batch_window
                while occupied < self._slot_count:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        break
                    try:
                        candidate = self._queue.get(timeout=timeout)
                    except queue.Empty:
                        break
                    if candidate is None:
                        self._queue.put(None)
                        break
                    if (
                        candidate.signature != first.signature
                        or occupied + candidate.batch_size > self._slot_count
                    ):
                        self._queue.put(candidate)
                        break
                    batch.append(candidate)
                    occupied += candidate.batch_size
            self._execute_batch(batch)

    def _execute_batch(self, batch: list[_GraphRequest]) -> None:
        try:
            inputs = _cat_graph_inputs(
                [request.call_kwargs["inputs_embeds"] for request in batch]
            )
            real_rows = int(inputs.shape[0])
            if real_rows > 1 and real_rows < self._slot_count:
                inputs = _pad_graph_inputs(inputs, self._slot_count)
                self.padded_slots += self._slot_count - real_rows
            call_kwargs = dict(batch[0].call_kwargs)
            call_kwargs["inputs_embeds"] = inputs
            result = self._engine.generate(**call_kwargs)
            sequences = getattr(result, "sequences", result)[:real_rows]
            offset = 0
            for request in batch:
                request_sequences = sequences[offset : offset + request.batch_size]
                if request.call_kwargs["return_dict_in_generate"]:
                    request.result = SimpleNamespace(sequences=request_sequences)
                else:
                    request.result = request_sequences
                offset += request.batch_size
            self.request_batches.append(len(batch))
            self.slot_batches.append(int(inputs.shape[0]))
        except BaseException as exc:  # noqa: BLE001 - propagate to callers
            for request in batch:
                request.error = exc
        finally:
            for request in batch:
                request.event.set()


def _graph_request_signature(call_kwargs: dict[str, Any]) -> tuple[Any, ...]:
    inputs = call_kwargs["inputs_embeds"]
    return (
        tuple(int(item) for item in inputs.shape[1:]),
        str(inputs.device),
        str(inputs.dtype),
        int(call_kwargs["max_new_tokens"]),
        bool(call_kwargs["do_sample"]),
        call_kwargs["top_p"],
        call_kwargs["top_k"],
        call_kwargs["temperature"],
        bool(call_kwargs["output_hidden_states"]),
        bool(call_kwargs["return_dict_in_generate"]),
    )


def _cat_graph_inputs(values: list[Any]):
    import torch

    return torch.cat(values, dim=0)


def _pad_graph_inputs(inputs, slot_count: int):
    padding = int(slot_count) - int(inputs.shape[0])
    repeats = (padding,) + (1,) * (inputs.ndim - 1)
    return _cat_graph_inputs([inputs, inputs[:1].repeat(repeats)])


def install_cuda_graph_code_predictor(
    talker: Any,
    *,
    max_graph_batch_size: int = 1,
    fixed_slot_count: int = 1,
    batch_window_ms: float = 0.0,
    bundle_factory=None,
    allow_non_cuda_for_testing: bool = False,
) -> bool:
    code_predictor = getattr(talker, "code_predictor", None)
    if code_predictor is None:
        raise FastCodePredictorError("talker has no code_predictor")
    if getattr(code_predictor.generate, "_qav_cuda_graph_code_predictor", False):
        return False

    original_generate = code_predictor.generate
    slot_count = max(1, int(fixed_slot_count))
    max_graph_batch_size = max(int(max_graph_batch_size), slot_count)
    engine = CUDAGraphCodePredictorEngine(
        code_predictor,
        max_graph_batch_size=max_graph_batch_size,
        bundle_factory=bundle_factory,
        allow_non_cuda_for_testing=allow_non_cuda_for_testing,
        prewarm_batch_sizes=(slot_count,) if slot_count > 1 else (),
    )
    scheduler = (
        FixedSlotCUDAGraphBatcher(
            engine,
            slot_count=slot_count,
            batch_window_ms=batch_window_ms,
        )
        if slot_count > 1
        else None
    )

    def graph_generate(**kwargs):
        target = scheduler or engine
        return target.generate(**kwargs)

    graph_generate._qav_fast_code_predictor = True  # type: ignore[attr-defined]
    graph_generate._qav_static_code_predictor = True  # type: ignore[attr-defined]
    graph_generate._qav_cuda_graph_code_predictor = True  # type: ignore[attr-defined]
    graph_generate._qav_original_generate = original_generate  # type: ignore[attr-defined]
    graph_generate._qav_engine = engine  # type: ignore[attr-defined]
    graph_generate._qav_scheduler = scheduler  # type: ignore[attr-defined]
    code_predictor.generate = graph_generate
    return True


def _create_cuda_graph_bundle(
    *,
    code_predictor,
    inputs_embeds,
    max_new_tokens: int,
    output_hidden_states: bool = False,
):
    return _CUDAGraphBundle(
        code_predictor,
        inputs_embeds=inputs_embeds,
        max_new_tokens=max_new_tokens,
        output_hidden_states=output_hidden_states,
    )


def _input_device_context(inputs_embeds):
    if getattr(inputs_embeds.device, "type", None) != "cuda":
        return nullcontext()
    import torch

    return torch.cuda.device(inputs_embeds.device)


class _CUDAGraphBundle:
    def __init__(
        self,
        code_predictor,
        *,
        inputs_embeds,
        max_new_tokens: int,
        output_hidden_states: bool = False,
    ):
        import torch
        from qwen_asr_vllm.agent.qwen_tts_static_predictor import PrefixStaticCache

        if inputs_embeds.device.type != "cuda":
            raise FastCodePredictorError("CUDA Graph predictor requires CUDA inputs")

        self._torch = torch
        self._code_predictor = code_predictor
        self._shape = tuple(int(item) for item in inputs_embeds.shape)
        self._max_new_tokens = int(max_new_tokens)
        self._output_hidden_states = bool(output_hidden_states)
        prompt_length = int(inputs_embeds.shape[1])
        # Qwen-TTS returns prompt_length - 1 as the first decode
        # generation_step after prefill. This is an absolute MTP head index,
        # not the local zero-based decode step.
        self._first_decode_generation_step = max(0, prompt_length - 1)
        self._lock = threading.Lock()
        self._inputs_embeds = inputs_embeds.detach().clone()
        self._input_ids = [
            torch.zeros(
                (inputs_embeds.shape[0], 1),
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            for _ in range(1, self._max_new_tokens)
        ]
        self._cache_positions = [
            torch.arange(prompt_length, device=inputs_embeds.device)
        ] + [
            torch.tensor(
                [prompt_length + step_index - 1],
                device=inputs_embeds.device,
            )
            for step_index in range(1, self._max_new_tokens)
        ]
        self._cache = PrefixStaticCache(
            config=code_predictor.config,
            max_cache_len=prompt_length + self._max_new_tokens - 1,
        )
        self.graphs = []
        self.logits = []
        self.hidden_states = []
        self.last_logits_trace = None
        self.last_cache_trace = None
        self.last_hidden_trace = None
        started = time.perf_counter()
        self._capture()
        self.capture_ms = (time.perf_counter() - started) * 1000.0
        self._closed = False

    def _capture(self) -> None:
        torch = self._torch
        with torch.inference_mode():
            self._warmup()
            self._cache.reset()
            self._cache.prepare_step(
                past_length=0,
                query_length=int(self._inputs_embeds.shape[1]),
            )
        torch.cuda.synchronize(self._inputs_embeds.device)

        pool = torch.cuda.graph_pool_handle()
        with torch.inference_mode():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                outputs = self._code_predictor(
                    inputs_embeds=self._inputs_embeds,
                    past_key_values=self._cache,
                    cache_position=self._cache_positions[0],
                    use_cache=True,
                    output_hidden_states=self._output_hidden_states,
                )
                logits = outputs.logits[:, -1, :]
            self.graphs.append(graph)
            self.logits.append(logits)
            self.hidden_states.append(getattr(outputs, "hidden_states", None))

            for step_index in range(1, self._max_new_tokens):
                self._cache.prepare_step(
                    past_length=int(self._inputs_embeds.shape[1]) + step_index - 1,
                    query_length=1,
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    outputs = self._code_predictor(
                        input_ids=self._input_ids[step_index - 1],
                        past_key_values=self._cache,
                        cache_position=self._cache_positions[step_index],
                        use_cache=True,
                        output_hidden_states=self._output_hidden_states,
                        generation_steps=_decode_generation_step(
                            self._first_decode_generation_step, step_index
                        ),
                    )
                    logits = outputs.logits[:, -1, :]
                self.graphs.append(graph)
                self.logits.append(logits)
                self.hidden_states.append(getattr(outputs, "hidden_states", None))

            self._cache.reset()
        torch.cuda.synchronize(self._inputs_embeds.device)

    def _warmup(self) -> None:
        torch = self._torch
        self._cache.reset()
        self._cache.prepare_step(
            past_length=0,
            query_length=int(self._inputs_embeds.shape[1]),
        )
        outputs = self._code_predictor(
            inputs_embeds=self._inputs_embeds,
            past_key_values=self._cache,
            cache_position=self._cache_positions[0],
            use_cache=True,
            output_hidden_states=self._output_hidden_states,
        )
        token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        for step_index in range(1, self._max_new_tokens):
            self._cache.prepare_step(
                past_length=int(self._inputs_embeds.shape[1]) + step_index - 1,
                query_length=1,
            )
            self._input_ids[step_index - 1].copy_(token)
            outputs = self._code_predictor(
                input_ids=self._input_ids[step_index - 1],
                past_key_values=self._cache,
                cache_position=self._cache_positions[step_index],
                use_cache=True,
                output_hidden_states=self._output_hidden_states,
                generation_steps=_decode_generation_step(
                    self._first_decode_generation_step, step_index
                ),
            )
            token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

    def generate(
        self,
        *,
        inputs_embeds,
        max_new_tokens: int,
        do_sample: bool | None = True,
        top_p: float | None = 1.0,
        top_k: int | None = 50,
        temperature: float | None = 0.9,
        output_hidden_states: bool | None = None,
        return_dict_in_generate: bool | None = True,
        **kwargs,
    ):
        if self._closed:
            raise FastCodePredictorError("CUDA Graph bundle is closed")
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise FastCodePredictorError(
                f"unsupported CUDA Graph bundle kwargs: {unsupported}"
            )
        if tuple(int(item) for item in inputs_embeds.shape) != self._shape:
            raise FastCodePredictorError("CUDA Graph predictor input shape changed")
        if int(max_new_tokens) != self._max_new_tokens:
            raise FastCodePredictorError("CUDA Graph predictor step count changed")
        if bool(output_hidden_states) != self._output_hidden_states:
            raise FastCodePredictorError(
                "CUDA Graph predictor output_hidden_states mode changed"
            )

        torch = self._torch
        sequences = []
        with self._lock, torch.inference_mode():
            self._cache.reset()
            self._inputs_embeds.copy_(inputs_embeds)
            self.graphs[0].replay()
            token = _sample_next_token(
                self.logits[0],
                do_sample=bool(do_sample),
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
            )
            sequences.append(token)
            for step_index in range(1, self._max_new_tokens):
                self._input_ids[step_index - 1].copy_(token)
                self.graphs[step_index].replay()
                token = _sample_next_token(
                    self.logits[step_index],
                    do_sample=bool(do_sample),
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                )
                sequences.append(token)

            generated = torch.cat(sequences, dim=-1)
            if self._trace_enabled():
                self.last_logits_trace = [
                    {
                        "argmax": int(logit.argmax(dim=-1)[0].item()),
                        "max": float(logit.max().item()),
                        "min": float(logit.min().item()),
                        "mean": float(logit.float().mean().item()),
                        "l2": float(torch.linalg.vector_norm(logit.float()).item()),
                    }
                    for logit in self.logits
                ]
                self.last_cache_trace = []
                for layer in self._cache.layers:
                    digest = hashlib.sha256()
                    digest.update(layer.keys[..., : self._cache.get_seq_length()].detach().cpu().numpy().tobytes())
                    digest.update(layer.values[..., : self._cache.get_seq_length()].detach().cpu().numpy().tobytes())
                    self.last_cache_trace.append(digest.hexdigest())
                self.last_hidden_trace = []
                for hidden in self.hidden_states:
                    if not hidden:
                        self.last_hidden_trace.append(None)
                        continue
                    value = hidden[-1][:, -1, :]
                    self.last_hidden_trace.append({
                        "max": float(value.max().item()),
                        "min": float(value.min().item()),
                        "mean": float(value.float().mean().item()),
                        "l2": float(torch.linalg.vector_norm(value.float()).item()),
                    })
        if return_dict_in_generate:
            return SimpleNamespace(sequences=generated)
            return generated

    @staticmethod
    def _trace_enabled():
        import os
        return os.environ.get("VOICE_TTS_TRACE_INNER_LOGITS", "0") == "1"


    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.graphs.clear()
            self.logits.clear()
            self._input_ids.clear()
            self._cache = None
            self._inputs_embeds = None


def _decode_generation_step(first_decode_generation_step: int, step_index: int) -> int:
    """Map a local decode step to Qwen-TTS's absolute MTP head index."""

    if int(step_index) < 1:
        raise ValueError("step_index must be positive")
    return int(first_decode_generation_step) + int(step_index) - 1
