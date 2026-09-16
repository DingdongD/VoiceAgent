"""Top-level Qwen3-ASR assembly.

Audio encoding is *not* performed here. By the time a request reaches the text
decoder its audio embeddings already exist, produced by an earlier pipeline
stage that batched them together with other requests. All this model does is drop
them into the placeholder span, whose offset and length were fixed when the
prompt was built.
"""

from __future__ import annotations

import torch
from torch import nn

from qwen_asr_vllm.audio.batcher import PackedAudioBatch
from qwen_asr_vllm.config import Qwen3ASRConfig
from qwen_asr_vllm.models.audio_encoder import AudioEncoder
from qwen_asr_vllm.models.qwen3 import Qwen3Model


class Qwen3ASRForConditionalGeneration(nn.Module):
    def __init__(self, config: Qwen3ASRConfig, max_position: int):
        super().__init__()
        self.config = config
        self.audio_tower = AudioEncoder(config.audio)
        self.model = Qwen3Model(config.text, max_position)
        self.lm_head = nn.Linear(config.text.hidden_size, config.text.vocab_size, bias=False)
        if config.text.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def encode_audio(self, batch: PackedAudioBatch) -> list[torch.Tensor]:
        """Encode a cross-request batch and split it back per request."""
        return batch.split_outputs(self.audio_tower(batch))

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def scatter_audio_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        spans: list[tuple[int, torch.Tensor]],
    ) -> torch.Tensor:
        """Write audio embeddings into their placeholder spans.

        ``spans`` holds ``(offset_in_batch, embeddings)`` pairs. A slice write is
        possible because the placeholder run is contiguous and its position is
        known from prompt construction, so there is no need to build a boolean
        mask over the whole batch.
        """
        for offset, audio_embeds in spans:
            length = audio_embeds.size(0)
            inputs_embeds[offset : offset + length] = audio_embeds.to(inputs_embeds.dtype)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
