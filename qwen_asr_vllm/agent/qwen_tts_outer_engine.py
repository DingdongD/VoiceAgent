from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from qwen_asr_vllm.agent.qwen_tts_fast_predictor import _sample_next_token


class ExplicitTalkerEngineError(RuntimeError):
    """Raised when the explicit Qwen-TTS talker loop cannot handle a request."""


def install_explicit_talker_step_engine(
    talker: Any,
    *,
    compile_step: bool = False,
    compile_mode: str = "reduce-overhead",
    compiler=None,
) -> bool:
    """Replace HF `talker.generate()` with an explicit batched codec AR loop."""

    if getattr(talker.generate, "_qav_explicit_talker_step_engine", False):
        return False
    original_generate = talker.generate
    compiled_forward = (
        _compile_callable(talker.forward, mode=compile_mode, compiler=compiler)
        if compile_step
        else None
    )

    def explicit_generate(**kwargs):
        return explicit_talker_generate(talker, forward_fn=compiled_forward, **kwargs)

    explicit_generate._qav_explicit_talker_step_engine = True  # type: ignore[attr-defined]
    explicit_generate._qav_original_generate = original_generate  # type: ignore[attr-defined]
    talker.generate = explicit_generate
    return True


def explicit_talker_generate(
    talker: Any,
    *,
    forward_fn=None,
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
    repetition_penalty: float | None = None,
    suppress_tokens: list[int] | None = None,
    output_hidden_states: bool | None = True,
    return_dict_in_generate: bool | None = True,
    **kwargs,
):
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise ExplicitTalkerEngineError(
            f"unsupported explicit talker kwargs: {unsupported}"
        )
    if inputs_embeds is None or attention_mask is None:
        raise ExplicitTalkerEngineError("inputs_embeds and attention_mask are required")
    if max_new_tokens <= 0:
        raise ExplicitTalkerEngineError("max_new_tokens must be positive")

    import torch

    eos_token_id = _resolve_eos(talker, eos_token_id)
    hidden_history = []
    generated_first_codebook = []
    past_key_values = None
    past_hidden = None
    generation_step = None
    next_token = None
    done = torch.zeros(inputs_embeds.shape[0], dtype=torch.bool, device=inputs_embeds.device)

    with torch.inference_mode():
        outputs = _talker_forward(
            talker,
            forward_fn,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=bool(output_hidden_states),
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
        )
        hidden_history.append(outputs.hidden_states)
        past_key_values = outputs.past_key_values
        past_hidden = outputs.past_hidden
        generation_step = outputs.generation_step
        next_token = _sample_outer_token(
            outputs.logits[:, -1, :],
            generated_first_codebook,
            do_sample=bool(do_sample),
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            suppress_tokens=suppress_tokens,
        )

        for step_index in range(int(max_new_tokens)):
            if done.all() and step_index >= int(min_new_tokens):
                break
            input_ids = next_token.masked_fill(done.unsqueeze(1), eos_token_id)
            outputs = _talker_forward(
                talker,
                forward_fn,
                input_ids=input_ids,
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=bool(output_hidden_states),
                past_hidden=past_hidden,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                generation_step=generation_step,
                subtalker_dosample=subtalker_dosample,
                subtalker_top_p=subtalker_top_p,
                subtalker_top_k=subtalker_top_k,
                subtalker_temperature=subtalker_temperature,
            )
            hidden_history.append(outputs.hidden_states)
            codec_ids = outputs.hidden_states[-1]
            if codec_ids is not None:
                generated_first_codebook.append(codec_ids[:, :1])
                if step_index + 1 >= int(min_new_tokens):
                    done |= codec_ids[:, 0] == int(eos_token_id)
            past_key_values = outputs.past_key_values
            past_hidden = outputs.past_hidden
            generation_step = outputs.generation_step
            next_token = _sample_outer_token(
                outputs.logits[:, -1, :],
                generated_first_codebook,
                do_sample=bool(do_sample),
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                suppress_tokens=suppress_tokens,
            )

    if return_dict_in_generate:
        return SimpleNamespace(hidden_states=tuple(hidden_history))
    return tuple(hidden_history)


def _resolve_eos(talker: Any, eos_token_id: int | None) -> int:
    if eos_token_id is not None:
        return int(eos_token_id)
    config = getattr(talker, "config", None)
    eos = getattr(config, "codec_eos_token_id", None)
    if eos is None:
        raise ExplicitTalkerEngineError("eos_token_id is required")
    return int(eos)


def _sample_outer_token(
    logits,
    generated_first_codebook: list[Any],
    *,
    do_sample: bool,
    top_p: float | None,
    top_k: int | None,
    temperature: float | None,
    repetition_penalty: float | None,
    suppress_tokens: list[int] | None,
):
    filtered = logits.float()
    if suppress_tokens:
        filtered[:, suppress_tokens] = float("-inf")
    if repetition_penalty is not None and float(repetition_penalty) != 1.0:
        filtered = _apply_repetition_penalty(
            filtered,
            generated_first_codebook,
            float(repetition_penalty),
        )
    return _sample_next_token(
        filtered,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )


def _apply_repetition_penalty(logits, generated_first_codebook, penalty: float):
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


def _talker_forward(talker: Any, forward_fn, **kwargs):
    if forward_fn is None:
        forward_fn = talker.forward
    return forward_fn(**kwargs)


def _compile_callable(fn, *, mode: str, compiler=None):
    if compiler is None:
        import torch

        compiler = torch.compile
    return compiler(fn, mode=mode)
