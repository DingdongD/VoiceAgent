from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


class FastCodePredictorError(RuntimeError):
    """Raised when the Qwen-TTS fast code predictor path is misconfigured."""


def install_batched_fast_code_predictor(
    talker: Any,
    *,
    batch_window_ms: float = 2.0,
    max_batch_size: int = 8,
    compile_step: bool = False,
    compile_mode: str = "reduce-overhead",
    compiler=None,
) -> bool:
    """Install a request-id level batching scheduler for codebook prediction."""

    code_predictor = getattr(talker, "code_predictor", None)
    if code_predictor is None:
        raise FastCodePredictorError("talker has no code_predictor")
    if getattr(code_predictor.generate, "_qav_batched_fast_code_predictor", False):
        return False

    original_generate = code_predictor.generate
    scheduler = _BatchedCodePredictorScheduler(
        code_predictor,
        batch_window_ms=batch_window_ms,
        max_batch_size=max_batch_size,
        compile_step=compile_step,
        compile_mode=compile_mode,
        compiler=compiler,
    )

    def fast_generate(**kwargs):
        return scheduler.generate(**kwargs)

    fast_generate._qav_fast_code_predictor = True  # type: ignore[attr-defined]
    fast_generate._qav_batched_fast_code_predictor = True  # type: ignore[attr-defined]
    fast_generate._qav_original_generate = original_generate  # type: ignore[attr-defined]
    fast_generate._qav_scheduler = scheduler  # type: ignore[attr-defined]
    code_predictor.generate = fast_generate
    return True


def fast_code_predictor_generate(
    code_predictor: Any,
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
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise FastCodePredictorError(
            f"unsupported fast code predictor kwargs: {unsupported}"
        )
    if inputs_embeds is None:
        raise FastCodePredictorError("inputs_embeds is required")
    if max_new_tokens <= 0:
        raise FastCodePredictorError("max_new_tokens must be positive")

    import torch

    sequences = []
    past_key_values = None
    generation_steps = None
    next_token = None

    with torch.inference_mode():
        for step_index in range(int(max_new_tokens)):
            if step_index == 0:
                outputs = code_predictor(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    output_hidden_states=bool(output_hidden_states),
                )
            else:
                outputs = code_predictor(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=bool(output_hidden_states),
                    generation_steps=generation_steps,
                )

            logits = outputs.logits[:, -1, :]
            next_token = _sample_next_token(
                logits,
                do_sample=bool(do_sample),
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
            )
            sequences.append(next_token)
            past_key_values = outputs.past_key_values
            generation_steps = outputs.generation_steps

    generated = torch.cat(sequences, dim=-1)
    if return_dict_in_generate:
        return SimpleNamespace(sequences=generated)
    return generated


def _sample_next_token(
    logits,
    *,
    do_sample: bool,
    top_p: float | None,
    top_k: int | None,
    temperature: float | None,
):
    import torch

    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    filtered = logits.float()
    if temperature is not None and float(temperature) > 0:
        filtered = filtered / float(temperature)
    if top_k is not None and int(top_k) > 0 and int(top_k) < filtered.shape[-1]:
        kth_values = torch.topk(filtered, int(top_k), dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < kth_values, float("-inf"))
    if top_p is not None and 0 < float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > float(top_p)
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        restored = torch.full_like(filtered, float("-inf"))
        filtered = restored.scatter(-1, sorted_indices, sorted_logits)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@dataclass
class _StepRequest:
    inputs_embeds: Any
    input_ids: Any
    past_key_values: Any
    output_hidden_states: bool
    generation_steps: Any
    batch_size: int
    event: threading.Event
    result: Any = None
    error: BaseException | None = None


class _BatchedCodePredictorScheduler:
    def __init__(
        self,
        code_predictor: Any,
        *,
        batch_window_ms: float,
        max_batch_size: int,
        compile_step: bool = False,
        compile_mode: str = "reduce-overhead",
        compiler=None,
    ):
        self._code_predictor = code_predictor
        self._forward = (
            _compile_callable(code_predictor, mode=compile_mode, compiler=compiler)
            if compile_step
            else code_predictor
        )
        self._batch_window = max(0.0, float(batch_window_ms) / 1000.0)
        self._max_batch_size = max(1, int(max_batch_size))
        self._queue: queue.Queue[_StepRequest] = queue.Queue()
        self.step_batches: list[int] = []
        self._thread = threading.Thread(
            target=self._run,
            name="qwen-tts-code-predictor-batcher",
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
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise FastCodePredictorError(
                f"unsupported fast code predictor kwargs: {unsupported}"
            )
        if inputs_embeds is None:
            raise FastCodePredictorError("inputs_embeds is required")
        if max_new_tokens <= 0:
            raise FastCodePredictorError("max_new_tokens must be positive")

        import torch

        sequences = []
        past_key_values = None
        generation_steps = None
        next_token = None

        with torch.inference_mode():
            for step_index in range(int(max_new_tokens)):
                if step_index == 0:
                    outputs = self._run_step(
                        inputs_embeds=inputs_embeds,
                        input_ids=None,
                        past_key_values=None,
                        output_hidden_states=bool(output_hidden_states),
                        generation_steps=None,
                    )
                else:
                    outputs = self._run_step(
                        inputs_embeds=None,
                        input_ids=next_token,
                        past_key_values=past_key_values,
                        output_hidden_states=bool(output_hidden_states),
                        generation_steps=generation_steps,
                    )

                logits = outputs.logits[:, -1, :]
                next_token = _sample_next_token(
                    logits,
                    do_sample=bool(do_sample),
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                )
                sequences.append(next_token)
                past_key_values = outputs.past_key_values
                generation_steps = outputs.generation_steps

        generated = torch.cat(sequences, dim=-1)
        if return_dict_in_generate:
            return SimpleNamespace(sequences=generated)
        return generated

    def _run_step(
        self,
        *,
        inputs_embeds,
        input_ids,
        past_key_values,
        output_hidden_states: bool,
        generation_steps,
    ):
        batch_source = inputs_embeds if inputs_embeds is not None else input_ids
        request = _StepRequest(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            generation_steps=generation_steps,
            batch_size=_batch_size(batch_source),
            event=threading.Event(),
        )
        self._queue.put(request)
        request.event.wait()
        if request.error is not None:
            raise request.error
        return request.result

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            batch = [first]
            if self._batch_window > 0.0 and self._max_batch_size > 1:
                deadline = time.monotonic() + self._batch_window
                while len(batch) < self._max_batch_size:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        break
                    try:
                        candidate = self._queue.get(timeout=timeout)
                    except queue.Empty:
                        break
                    if _compatible_step(first, candidate):
                        batch.append(candidate)
                    else:
                        self._queue.put(candidate)
                        break
            self._execute_batch(batch)

    def _execute_batch(self, batch: list[_StepRequest]) -> None:
        try:
            outputs = self._forward_batch(batch)
            self.step_batches.append(len(batch))
            self._assign_batch_outputs(batch, outputs)
        except BaseException as exc:  # noqa: BLE001 - propagate to callers
            for request in batch:
                request.error = exc
                request.event.set()

    def _forward_batch(self, batch: list[_StepRequest]):
        first = batch[0]
        kwargs = {
            "use_cache": True,
            "output_hidden_states": first.output_hidden_states,
        }
        if first.inputs_embeds is not None:
            kwargs["inputs_embeds"] = _cat_tensors(
                [request.inputs_embeds for request in batch]
            )
        else:
            kwargs["input_ids"] = _cat_tensors([request.input_ids for request in batch])
            kwargs["past_key_values"] = _cat_past_key_values(
                [request.past_key_values for request in batch]
            )
            kwargs["generation_steps"] = first.generation_steps
        return self._forward(**kwargs)

    def _assign_batch_outputs(self, batch: list[_StepRequest], outputs: Any) -> None:
        sizes = [request.batch_size for request in batch]
        logits = getattr(outputs, "logits", None)
        logits_parts = _split_tensor(logits, sizes)
        past_parts = _split_past_key_values(getattr(outputs, "past_key_values", None), sizes)
        for index, request in enumerate(batch):
            request.result = SimpleNamespace(
                logits=logits_parts[index],
                past_key_values=past_parts[index],
                generation_steps=getattr(outputs, "generation_steps", None),
            )
            request.event.set()


def _compatible_step(first: _StepRequest, other: _StepRequest) -> bool:
    return (
        (first.inputs_embeds is not None) == (other.inputs_embeds is not None)
        and first.output_hidden_states == other.output_hidden_states
        and first.generation_steps == other.generation_steps
    )


def _batch_size(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        raise FastCodePredictorError("cannot infer request batch size")
    return int(shape[0])


def _cat_tensors(values: list[Any]):
    import torch

    return torch.cat(values, dim=0)


def _split_tensor(value: Any, sizes: list[int]) -> list[Any]:
    import torch

    return list(torch.split(value, sizes, dim=0))


def _cat_past_key_values(values: list[Any]):
    import torch
    from transformers.cache_utils import DynamicCache

    if any(value is None for value in values):
        return None
    legacy_values = [_to_legacy_cache(value) for value in values]
    layer_count = len(legacy_values[0])
    merged = []
    for layer_index in range(layer_count):
        keys = torch.cat(
            [cache[layer_index][0] for cache in legacy_values],
            dim=0,
        )
        vals = torch.cat(
            [cache[layer_index][1] for cache in legacy_values],
            dim=0,
        )
        merged.append((keys, vals))
    return DynamicCache.from_legacy_cache(tuple(merged))


def _split_past_key_values(value: Any, sizes: list[int]) -> list[Any]:
    from transformers.cache_utils import DynamicCache

    if value is None:
        return [None for _ in sizes]
    legacy = _to_legacy_cache(value)
    starts = []
    offset = 0
    for size in sizes:
        starts.append(offset)
        offset += size
    outputs = []
    for start, size in zip(starts, sizes):
        layers = []
        for key, val in legacy:
            layers.append((key[start : start + size], val[start : start + size]))
        outputs.append(DynamicCache.from_legacy_cache(tuple(layers)))
    return outputs


def _to_legacy_cache(value: Any):
    to_legacy = getattr(value, "to_legacy_cache", None)
    if callable(to_legacy):
        return to_legacy()
    return value


def _compile_callable(fn, *, mode: str, compiler=None):
    if compiler is None:
        import torch

        compiler = torch.compile
    return compiler(fn, mode=mode)
