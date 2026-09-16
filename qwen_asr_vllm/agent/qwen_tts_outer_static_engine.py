from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock, local
from types import SimpleNamespace
from typing import Any

from qwen_asr_vllm.agent.qwen_tts_fast_predictor import _sample_next_token


class OuterTalkerEngineError(RuntimeError):
    """Raised when the outer talker static engine cannot handle a request."""


@dataclass
class OuterRequestState:
    request_id: int
    rope_delta: Any
    past_hidden: Any
    generation_step: int
    trailing_text_hidden: Any
    tts_pad_embed: Any
    active: bool = True
    done_steps: int = 0
    first_codebook_history: list[Any] = field(default_factory=list)
    hidden_history: list[Any] = field(default_factory=list)


@dataclass
class OuterCohortState:
    cache: Any
    cache_position: int
    requests: list[OuterRequestState]
    last_logits: Any
    lease: Any = None
    prompt_attention_mask: Any = None

    @property
    def active_indices(self) -> list[int]:
        return [index for index, state in enumerate(self.requests) if state.active]


def build_decode_position_ids(cache_position: int, requests: list[OuterRequestState]):
    """Build Qwen's three-axis decode positions for a fixed cache position."""

    import torch

    rope_deltas = torch.cat([state.rope_delta for state in requests], dim=0)
    return rope_deltas.reshape(1, len(requests), 1).expand(3, -1, -1) + cache_position


def select_text_condition(requests: list[OuterRequestState]):
    """Return each request's current text hidden state or its TTS padding state."""

    import torch

    conditions = [
        (
            state.trailing_text_hidden[:, state.generation_step : state.generation_step + 1]
            if state.generation_step < state.trailing_text_hidden.shape[1]
            else state.tts_pad_embed
        )
        for state in requests
    ]
    return torch.cat(conditions, dim=0)


@dataclass
class PreparedOuterDecode:
    codec_ids: Any
    condition: Any
    attention_mask: Any
    position_ids: Any
    cache_position: Any
    output_hidden_states: bool


@dataclass
class OuterStepOutput:
    logits: Any
    past_hidden: Any
    hidden_states: Any


class OuterRuntimeLease:
    """Cache-bound execution interface retained for one outer cohort."""

    def __init__(self, cache: Any):
        self.cache = cache

    def reset(self) -> None:
        self.cache.reset()

    def run(self, prepared_decode: PreparedOuterDecode) -> OuterStepOutput:
        raise NotImplementedError

    def release(self) -> None:
        """Release any runtime ownership held by this lease."""


class _EagerOuterRuntimeLease(OuterRuntimeLease):
    def __init__(self, talker: Any, cache: Any):
        super().__init__(cache)
        self._talker = talker

    def run(self, prepared_decode: PreparedOuterDecode) -> OuterStepOutput:
        return _run_outer_decode_body(self._talker, self.cache, prepared_decode)


class EagerOuterStepRuntime:
    """Acquire an eager outer-step lease with its own static KV cache."""

    def __init__(self, talker: Any, *, cache_factory=None):
        self._talker = talker
        self._cache_factory = cache_factory

    def acquire(self, batch_size, device, dtype, max_cache_len) -> OuterRuntimeLease:
        if self._cache_factory is None:
            from transformers import StaticCache

            cache_factory = StaticCache
        else:
            cache_factory = self._cache_factory
        cache = cache_factory(
            config=self._talker.config,
            max_cache_len=max_cache_len,
            max_batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        return _EagerOuterRuntimeLease(self._talker, cache)

    def build_attention_mask(
        self, prompt_attention_mask, cache_position, max_cache_len, dtype
    ):
        return _build_static_decode_mask(
            prompt_attention_mask,
            cache_position,
            max_cache_len,
            dtype=dtype,
        )


class OuterTalkerStaticEngine:
    """Run Qwen-TTS outer prefill and decode against a lease-owned static cache."""

    def __init__(self, talker: Any, max_cache_len: int = 1024, step_runtime=None):
        if int(max_cache_len) <= 0:
            raise OuterTalkerEngineError("max_cache_len must be positive")
        self.talker = talker
        self.max_cache_len = int(max_cache_len)
        self.step_runtime = step_runtime or EagerOuterStepRuntime(talker)
        self._metrics_lock = Lock()
        self._codec_frame_callbacks = local()
        self.prefill_calls = 0
        self.static_steps_by_batch: dict[int, int] = {}
        self.active_slots_per_step: list[int] = []
        self.errors = 0

    @contextmanager
    def codec_frame_callback(self, callback):
        """Expose generated codec frames to the streaming decoder per thread."""

        previous = getattr(self._codec_frame_callbacks, "callback", None)
        self._codec_frame_callbacks.callback = callback
        try:
            yield
        finally:
            if previous is None:
                try:
                    del self._codec_frame_callbacks.callback
                except AttributeError:
                    pass
            else:
                self._codec_frame_callbacks.callback = previous

    def _emit_codec_frame(self, codec_ids) -> None:
        callback = getattr(self._codec_frame_callbacks, "callback", None)
        if callback is not None:
            callback(codec_ids)

    def generate(self, **kwargs):
        steps = self.iterate(**kwargs)
        try:
            pending = next(steps)
            while True:
                try:
                    output = self._decode_step(**pending)
                except BaseException:
                    self._record_error()
                    raise
                pending = steps.send(output)
        except StopIteration as done:
            return done.value
        finally:
            steps.close()

    def iterate(
        self,
        *,
        inputs_embeds,
        attention_mask,
        trailing_text_hidden,
        tts_pad_embed,
        max_new_tokens: int,
        min_new_tokens: int = 0,
        do_sample: bool | None = True,
        top_k: int | None = 50,
        top_p: float | None = 1.0,
        temperature: float | None = 0.9,
        subtalker_dosample: bool | None = True,
        subtalker_top_k: int | None = 50,
        subtalker_top_p: float | None = 1.0,
        subtalker_temperature: float | None = 0.9,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        repetition_penalty: float | None = None,
        suppress_tokens: list[int] | None = None,
        output_hidden_states: bool | None = True,
        return_dict_in_generate: bool | None = True,
        **kwargs,
    ):
        lease = None
        try:
            try:
                if kwargs:
                    unsupported = ", ".join(sorted(kwargs))
                    raise OuterTalkerEngineError(
                        f"unsupported static outer talker kwargs: {unsupported}"
                    )
                resolved_eos_token_id = _resolve_eos(self.talker, eos_token_id)
                if (
                    pad_token_id is not None
                    and int(pad_token_id) != resolved_eos_token_id
                ):
                    raise OuterTalkerEngineError(
                        "pad_token_id must equal eos_token_id for static outer generation"
                    )
                self._validate_request(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    trailing_text_hidden=trailing_text_hidden,
                    tts_pad_embed=tts_pad_embed,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                )

                import torch

                with torch.inference_mode():
                    lease = self.step_runtime.acquire(
                        inputs_embeds.shape[0],
                        inputs_embeds.device,
                        inputs_embeds.dtype,
                        self.max_cache_len,
                    )
                    lease.reset()
                    return (yield from self.iterate_with_lease(
                        lease=lease,
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        trailing_text_hidden=trailing_text_hidden,
                        tts_pad_embed=tts_pad_embed,
                        max_new_tokens=int(max_new_tokens),
                        min_new_tokens=int(min_new_tokens),
                        do_sample=bool(do_sample),
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        subtalker_dosample=subtalker_dosample,
                        subtalker_top_k=subtalker_top_k,
                        subtalker_top_p=subtalker_top_p,
                        subtalker_temperature=subtalker_temperature,
                        eos_token_id=resolved_eos_token_id,
                        repetition_penalty=repetition_penalty,
                        suppress_tokens=suppress_tokens,
                        output_hidden_states=bool(output_hidden_states),
                        return_dict_in_generate=bool(return_dict_in_generate),
                    ))
            finally:
                if lease is not None:
                    with torch.inference_mode():
                        try:
                            lease.reset()
                        finally:
                            lease.release()
        except GeneratorExit:
            raise
        except BaseException:
            self._record_error()
            raise

    def _validate_request(
        self,
        *,
        inputs_embeds,
        attention_mask,
        trailing_text_hidden,
        tts_pad_embed,
        max_new_tokens,
        min_new_tokens,
    ) -> None:
        import torch

        tensors = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "trailing_text_hidden": trailing_text_hidden,
            "tts_pad_embed": tts_pad_embed,
        }
        for name, value in tensors.items():
            if not torch.is_tensor(value):
                raise OuterTalkerEngineError(f"{name} must be a tensor")
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[1] <= 0:
            raise OuterTalkerEngineError(
                "inputs_embeds must have shape [batch, prompt, hidden]"
            )
        batch_size, prompt_length, hidden_size = inputs_embeds.shape
        if attention_mask.shape != (batch_size, prompt_length):
            raise OuterTalkerEngineError(
                "attention_mask must match inputs_embeds batch and prompt dimensions"
            )
        if trailing_text_hidden.ndim != 3 or (
            trailing_text_hidden.shape[0] != batch_size
            or trailing_text_hidden.shape[2] != hidden_size
        ):
            raise OuterTalkerEngineError(
                "trailing_text_hidden must match inputs_embeds batch and hidden dimensions"
            )
        if tts_pad_embed.ndim != 3 or (
            tts_pad_embed.shape[0] not in (1, batch_size)
            or tts_pad_embed.shape[1] != 1
            or tts_pad_embed.shape[2] != hidden_size
        ):
            raise OuterTalkerEngineError("tts_pad_embed must have shape [1|batch, 1, hidden]")
        if int(max_new_tokens) <= 0:
            raise OuterTalkerEngineError("max_new_tokens must be positive")
        if int(min_new_tokens) < 0:
            raise OuterTalkerEngineError("min_new_tokens must be non-negative")
        if prompt_length + int(max_new_tokens) - 1 > self.max_cache_len:
            raise OuterTalkerEngineError(
                "requested prompt and decode tokens exceed static cache capacity"
            )

    def iterate_with_lease(
        self,
        *,
        lease,
        inputs_embeds,
        attention_mask,
        trailing_text_hidden,
        tts_pad_embed,
        max_new_tokens,
        min_new_tokens,
        do_sample,
        top_k,
        top_p,
        temperature,
        subtalker_dosample,
        subtalker_top_k,
        subtalker_top_p,
        subtalker_temperature,
        eos_token_id,
        repetition_penalty,
        suppress_tokens,
        output_hidden_states,
        return_dict_in_generate,
    ):
        import torch

        batch_size, prompt_length, _ = inputs_embeds.shape
        position_ids, rope_deltas = self.talker.get_rope_index(attention_mask)
        rope_deltas = rope_deltas - (1 - attention_mask).sum(-1, keepdim=True)
        cache_position = torch.arange(prompt_length, device=inputs_embeds.device)
        _validate_contiguous_cache_position(lease.cache, cache_position)
        self._record_prefill_call()
        outputs = self.talker.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=lease.cache,
            cache_position=cache_position,
            use_cache=True,
            output_hidden_states=output_hidden_states,
        )
        last_hidden = outputs.last_hidden_state
        pad_embeds = (
            tts_pad_embed.expand(batch_size, -1, -1)
            if tts_pad_embed.shape[0] == 1
            else tts_pad_embed
        )
        requests = [
            OuterRequestState(
                request_id=row,
                rope_delta=rope_deltas[row : row + 1],
                past_hidden=last_hidden[row : row + 1, -1:, :],
                generation_step=0,
                trailing_text_hidden=trailing_text_hidden[row : row + 1],
                tts_pad_embed=pad_embeds[row : row + 1],
            )
            for row in range(batch_size)
        ]
        cohort = OuterCohortState(
            cache=lease.cache,
            cache_position=prompt_length,
            requests=requests,
            last_logits=self.talker.codec_head(last_hidden)[:, -1, :],
            lease=lease,
            prompt_attention_mask=attention_mask,
        )
        hidden_history = []
        if output_hidden_states:
            hidden_history.append((getattr(outputs, "hidden_states", None), None))
        generated_first_codebook = []
        terminal_sampled = False

        # HF generation performs at most max_new_tokens model calls total. The
        # prefill above is the first call, so only N-1 sampled tokens are ever
        # forwarded into decode history.
        for step_index in range(max_new_tokens - 1):
            if not cohort.active_indices and step_index >= min_new_tokens:
                terminal_sampled = True
                break
            first_ids = self._sample_first_ids(
                cohort,
                generated_first_codebook=generated_first_codebook,
                step_index=step_index,
                min_new_tokens=min_new_tokens,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                suppress_tokens=suppress_tokens,
            )
            stop_allowed = step_index + 1 > min_new_tokens
            if stop_allowed and all(
                (not request.active) or int(first_ids[row, 0].item()) == eos_token_id
                for row, request in enumerate(cohort.requests)
            ):
                for request in cohort.requests:
                    request.active = False
                terminal_sampled = True
                break
            active_count = len(cohort.active_indices)
            self._record_logical_step(batch_size, active_count)
            codec_ids, step_output = yield dict(
                cohort=cohort,
                first_ids=first_ids,
                subtalker_dosample=subtalker_dosample,
                subtalker_top_k=subtalker_top_k,
                subtalker_top_p=subtalker_top_p,
                subtalker_temperature=subtalker_temperature,
                output_hidden_states=output_hidden_states,
            )
            self._emit_codec_frame(codec_ids)
            if output_hidden_states:
                hidden_history.append((step_output.hidden_states, codec_ids))
            generated_first_codebook.append(codec_ids[:, :1])
            for row, request in enumerate(cohort.requests):
                request.first_codebook_history.append(codec_ids[row : row + 1, :1])
                request.past_hidden = step_output.past_hidden[row : row + 1]
                request.generation_step += 1
                request.done_steps += 1
                if output_hidden_states:
                    request.hidden_history.append(
                        (
                            _slice_hidden_states(step_output.hidden_states, row),
                            codec_ids[row : row + 1],
                        )
                    )
                if (
                    request.active
                    and step_index + 1 > min_new_tokens
                    and int(codec_ids[row, 0].item()) == eos_token_id
                ):
                    request.active = False
            cohort.last_logits = step_output.logits[:, -1, :]
            cohort.cache_position += 1

        if not terminal_sampled:
            # GenerationMixin still applies processors and samples once after
            # the final allowed model call. That token is returned in sequences
            # but is never forwarded, so Qwen's codec hidden history excludes it.
            self._sample_first_ids(
                cohort,
                generated_first_codebook=generated_first_codebook,
                step_index=max_new_tokens - 1,
                min_new_tokens=min_new_tokens,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                suppress_tokens=suppress_tokens,
            )

        if return_dict_in_generate:
            return SimpleNamespace(
                hidden_states=tuple(hidden_history) if output_hidden_states else None
            )
        return tuple(hidden_history)

    def _sample_first_ids(
        self,
        cohort: OuterCohortState,
        *,
        generated_first_codebook,
        step_index,
        min_new_tokens,
        do_sample,
        top_k,
        top_p,
        temperature,
        eos_token_id,
        repetition_penalty,
        suppress_tokens,
    ):
        import torch

        sample_suppress_tokens = list(suppress_tokens or ())
        if step_index + 1 <= min_new_tokens:
            sample_suppress_tokens.append(eos_token_id)
        first_ids = _sample_outer_token(
            cohort.last_logits,
            generated_first_codebook,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            suppress_tokens=sample_suppress_tokens,
        )
        inactive = torch.tensor(
            [not request.active for request in cohort.requests],
            dtype=torch.bool,
            device=first_ids.device,
        ).unsqueeze(1)
        return first_ids.masked_fill(inactive, eos_token_id)

    def _decode_step(
        self,
        cohort: OuterCohortState,
        *,
        first_ids,
        subtalker_dosample,
        subtalker_top_k,
        subtalker_top_p,
        subtalker_temperature,
        output_hidden_states,
    ):
        import torch
        first_hidden = self.talker.get_input_embeddings()(first_ids)
        past_hidden = torch.cat(
            [request.past_hidden for request in cohort.requests], dim=0
        )

        # Predictor generation must remain outside lease.run so Task 4 never nests graphs.
        predictor_result = self.talker.code_predictor.generate(
            inputs_embeds=torch.cat((past_hidden, first_hidden), dim=1),
            max_new_tokens=self.talker.config.num_code_groups - 1,
            do_sample=subtalker_dosample,
            top_p=subtalker_top_p,
            top_k=subtalker_top_k,
            temperature=subtalker_temperature,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        remaining_ids = predictor_result.sequences
        codec_ids = torch.cat((first_ids, remaining_ids), dim=-1)
        condition = select_text_condition(cohort.requests)
        cache_position = torch.tensor(
            [cohort.cache_position], dtype=torch.long, device=codec_ids.device
        )
        prepared = PreparedOuterDecode(
            codec_ids=codec_ids,
            condition=condition,
            attention_mask=self.step_runtime.build_attention_mask(
                cohort.prompt_attention_mask,
                cohort.cache_position,
                self.max_cache_len,
                condition.dtype,
            ),
            position_ids=build_decode_position_ids(
                cohort.cache_position, cohort.requests
            ),
            cache_position=cache_position,
            output_hidden_states=output_hidden_states,
        )
        _validate_contiguous_cache_position(cohort.lease.cache, cache_position)
        return codec_ids, cohort.lease.run(prepared)

    def metrics_snapshot(self) -> dict[str, Any]:
        with self._metrics_lock:
            prefill_calls = self.prefill_calls
            static_steps_by_batch = dict(self.static_steps_by_batch)
            active_slots_per_step = list(self.active_slots_per_step)
            errors = self.errors
        total_slots = sum(
            batch_size * steps for batch_size, steps in static_steps_by_batch.items()
        )
        active_slots = sum(active_slots_per_step)
        snapshot = {
            "prefill_calls": prefill_calls,
            "static_steps_by_batch": static_steps_by_batch,
            "active_slots_per_step": active_slots_per_step,
            "slot_occupancy": active_slots / total_slots if total_slots else 0.0,
            "errors": errors,
        }
        runtime_metrics = getattr(self.step_runtime, "metrics_snapshot", None)
        if callable(runtime_metrics):
            snapshot.update(runtime_metrics())
        return snapshot

    def _record_prefill_call(self) -> None:
        with self._metrics_lock:
            self.prefill_calls += 1

    def _record_logical_step(self, batch_size: int, active_count: int) -> None:
        with self._metrics_lock:
            self.active_slots_per_step.append(active_count)
            self.static_steps_by_batch[batch_size] = (
                self.static_steps_by_batch.get(batch_size, 0) + 1
            )

    def _record_error(self) -> None:
        with self._metrics_lock:
            self.errors += 1


def _run_outer_decode_body(
    talker: Any, cache: Any, prepared_decode: PreparedOuterDecode
) -> OuterStepOutput:
    codec_ids = prepared_decode.codec_ids
    codec_hiddens = [talker.get_input_embeddings()(codec_ids[:, :1])]
    predictor_embeddings = talker.code_predictor.get_input_embeddings()
    codec_hiddens.extend(
        predictor_embeddings[index](codec_ids[:, index + 1 : index + 2])
        for index in range(talker.config.num_code_groups - 1)
    )

    import torch

    inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)
    inputs_embeds = inputs_embeds + prepared_decode.condition
    outputs = talker.model(
        inputs_embeds=inputs_embeds,
        attention_mask=prepared_decode.attention_mask,
        position_ids=prepared_decode.position_ids,
        past_key_values=cache,
        cache_position=prepared_decode.cache_position,
        use_cache=True,
        output_hidden_states=prepared_decode.output_hidden_states,
    )
    hidden_states = outputs.last_hidden_state
    return OuterStepOutput(
        logits=talker.codec_head(hidden_states),
        past_hidden=hidden_states[:, -1:, :],
        hidden_states=getattr(outputs, "hidden_states", None),
    )


def _validate_contiguous_cache_position(cache, cache_position) -> None:
    import torch

    if (
        not torch.is_tensor(cache_position)
        or cache_position.ndim != 1
        or cache_position.dtype != torch.long
        or cache_position.numel() == 0
    ):
        raise OuterTalkerEngineError("cache position is not contiguous")
    start = cache.get_seq_length()
    expected = torch.arange(
        start,
        start + cache_position.shape[0],
        dtype=torch.long,
        device=cache_position.device,
    )
    if not bool(torch.equal(cache_position, expected)):
        raise OuterTalkerEngineError("cache position is not contiguous")


def _build_static_decode_mask(prompt_attention_mask, cache_position, max_cache_len, *, dtype):
    import torch

    batch_size, prompt_length = prompt_attention_mask.shape
    allowed = torch.zeros(
        (batch_size, max_cache_len),
        dtype=torch.bool,
        device=prompt_attention_mask.device,
    )
    allowed[:, :prompt_length] = prompt_attention_mask.to(dtype=torch.bool)
    allowed[:, prompt_length : cache_position + 1] = True
    mask = torch.full(
        (batch_size, 1, 1, max_cache_len),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=prompt_attention_mask.device,
    )
    return mask.masked_fill(allowed[:, None, None, :], 0)


def _slice_hidden_states(hidden_states, row):
    if hidden_states is None:
        return None
    if isinstance(hidden_states, tuple):
        return tuple(hidden[row : row + 1] for hidden in hidden_states)
    return hidden_states[row : row + 1]


def _resolve_eos(talker: Any, eos_token_id: int | None) -> int:
    if eos_token_id is not None:
        return int(eos_token_id)
    eos = getattr(getattr(talker, "config", None), "codec_eos_token_id", None)
    if eos is None:
        raise OuterTalkerEngineError("eos_token_id is required")
    return int(eos)


def _sample_outer_token(
    logits,
    generated_first_codebook,
    *,
    do_sample,
    top_p,
    top_k,
    temperature,
    repetition_penalty,
    suppress_tokens,
):
    filtered = logits.float()
    if suppress_tokens:
        filtered[:, suppress_tokens] = float("-inf")
    if repetition_penalty is not None and float(repetition_penalty) != 1.0:
        filtered = _apply_repetition_penalty(
            filtered, generated_first_codebook, float(repetition_penalty)
        )
    return _sample_next_token(
        filtered,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )


def _apply_repetition_penalty(logits, generated_first_codebook, penalty):
    if not generated_first_codebook:
        return logits
    import torch

    tokens = torch.cat(generated_first_codebook, dim=-1)
    adjusted = logits.clone()
    for batch_index in range(tokens.shape[0]):
        for token in torch.unique(tokens[batch_index]):
            token_id = int(token.item())
            score = adjusted[batch_index, token_id]
            adjusted[batch_index, token_id] = (
                score * penalty if score < 0 else score / penalty
            )
    return adjusted


def install_static_outer_talker(talker: Any, *, max_cache_len: int = 1024) -> bool:
    """Install the parity-preserving eager outer engine under the legacy name.

    The former implementation exposed the full StaticCache extent to
    attention and could change FP16 SDPA/GQA dispatch. Keep the public
    installer for compatibility, but use active-prefix cache semantics.
    """

    if getattr(talker.generate, "_qav_static_outer_talker", False):
        return False
    from qwen_asr_vllm.agent.qwen_tts_outer_active_prefix_cache import (
        ActivePrefixOuterStepRuntime,
    )

    original_generate = talker.generate
    runtime = ActivePrefixOuterStepRuntime(talker)
    engine = OuterTalkerStaticEngine(
        talker,
        max_cache_len=max_cache_len,
        step_runtime=runtime,
    )

    def static_generate(**kwargs):
        return engine.generate(**kwargs)

    static_generate._qav_static_outer_talker = True  # type: ignore[attr-defined]
    static_generate._qav_active_prefix_cache = True  # type: ignore[attr-defined]
    static_generate._qav_original_generate = original_generate  # type: ignore[attr-defined]
    static_generate._qav_outer_engine = engine  # type: ignore[attr-defined]
    talker.generate = static_generate
    return True
