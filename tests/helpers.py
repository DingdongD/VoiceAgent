"""Synthetic requests for CPU-only scheduler and cache tests.

Real requests need a checkpoint for the tokenizer and a GPU for the encoder. The
scheduler and block manager only ever look at token ids, lengths and the audio
span, so they can be exercised against hand-built layouts instead.
"""
from __future__ import annotations

import torch

from qwen_asr_vllm.audio.frontend import AudioFeatures
from qwen_asr_vllm.config import EngineConfig
from qwen_asr_vllm.engine.request import AsrRequest, RequestStage, SamplingParams
from qwen_asr_vllm.prompt import PromptLayout

from .conftest import model_path

BLOCK_SIZE = 256
AUDIO_TOKEN_ID = 151676
STOP_TOKEN_IDS = {151643, 151645}

# Tokens per second of audio, fixed by the encoder's 100-frame chunking.
TOKENS_PER_SECOND = 13


def make_request(
    prompt_len: int,
    audio_length: int = TOKENS_PER_SECOND,
    audio_offset: int = 8,
    max_new_tokens: int = 440,
    block_size: int = BLOCK_SIZE,
) -> AsrRequest:
    """A request whose ids are textual prefix, audio span, then textual suffix."""
    suffix_len = prompt_len - audio_offset - audio_length
    if suffix_len < 0:
        raise ValueError(f"prompt_len {prompt_len} is too small for the audio span")
    token_ids = (
        list(range(1000, 1000 + audio_offset))
        + [AUDIO_TOKEN_ID] * audio_length
        + list(range(2000, 2000 + suffix_len))
    )
    mel_frames = audio_length * 100 // TOKENS_PER_SECOND
    return AsrRequest(
        features=AudioFeatures(
            mel=torch.zeros(128, mel_frames),
            mel_frames=mel_frames,
            num_audio_tokens=audio_length,
            audio_seconds=audio_length / TOKENS_PER_SECOND,
        ),
        layout=PromptLayout(
            token_ids=token_ids, audio_offset=audio_offset, audio_length=audio_length
        ),
        sampling=SamplingParams(max_new_tokens=max_new_tokens),
        block_size=block_size,
        stop_token_ids=STOP_TOKEN_IDS,
    )


def make_config(**overrides) -> EngineConfig:
    settings = {
        "model": model_path(),
        "max_num_seqs": 8,
        "max_model_len": 4096,
        "max_num_batched_tokens": 4096,
    }
    settings.update(overrides)
    return EngineConfig(**settings)


def admit_for_prefill(scheduler, request: AsrRequest, audio_embeds=None) -> AsrRequest:
    """Move a request past the encode stage without running an encoder."""
    request.audio_embeds = (
        audio_embeds
        if audio_embeds is not None
        else torch.zeros(request.layout.audio_length, 4)
    )
    request.stage = RequestStage.WAITING_PREFILL
    scheduler.admit_prefill([request])
    return request


def start_decoding(scheduler, manager, request: AsrRequest) -> AsrRequest:
    """Put a request into the running queue with blocks held, as if prefilled."""
    admit_for_prefill(scheduler, request)
    manager.allocate(request)
    request.num_computed_tokens = len(request)
    request.stage = RequestStage.RUNNING_DECODE
    scheduler.waiting_prefill.remove(request)
    scheduler.running.append(request)
    return request
