"""Audio encoding stage.

One encoder call covers every request in the batch. Audio embeddings are kept
until the request finishes rather than freed after prefill: at roughly 13 tokens
per second of audio they cost about 27KB per second per request, so holding them
lets a preempted request re-prefill without re-encoding.
"""

from __future__ import annotations

import torch

from qwen_asr_vllm.audio.batcher import pack_audio_batch
from qwen_asr_vllm.config import EngineConfig
from qwen_asr_vllm.engine.request import AsrRequest
from qwen_asr_vllm.models.qwen3_asr import Qwen3ASRForConditionalGeneration


class AudioRunner:
    def __init__(self, model: Qwen3ASRForConditionalGeneration, config: EngineConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        self.n_window_infer = config.model_config.audio.n_window_infer

    @torch.inference_mode()
    def encode(self, requests: list[AsrRequest]) -> None:
        if not requests:
            return
        batch = pack_audio_batch(
            [request.features.mel.to(self.device, self.dtype) for request in requests],
            n_window_infer=self.n_window_infer,
            device=self.device,
        )
        embeddings = self.model.encode_audio(batch)
        for request, audio_embeds in zip(requests, embeddings):
            if audio_embeds.size(0) != request.layout.audio_length:
                raise RuntimeError(
                    f"request {request.request_id}: encoder produced {audio_embeds.size(0)} "
                    f"audio tokens but the prompt reserved {request.layout.audio_length}"
                )
            request.audio_embeds = audio_embeds
            request.mark_encoding()
