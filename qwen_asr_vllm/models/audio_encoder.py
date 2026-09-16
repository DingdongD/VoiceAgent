"""Qwen3-ASR audio tower, batched across requests.

The upstream encoder already consumes a time-concatenated mel tensor plus
per-request ``feature_lens``, but every caller drives it one request at a time.
This version takes a :class:`~qwen_asr_vllm.audio.batcher.PackedAudioBatch` so
many requests share a single pass with no padding waste.

Module and parameter names match the checkpoint so weights load without a
mapping table.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from qwen_asr_vllm.audio.batcher import PackedAudioBatch
from qwen_asr_vllm.audio.tokens import CHUNK_FRAMES, TOKENS_PER_FULL_CHUNK
from qwen_asr_vllm.config import AudioEncoderConfig
from qwen_asr_vllm.layers.attention import VarlenSelfAttention
from qwen_asr_vllm.profiling import NULL_TIMER


class SinusoidsPositionEmbedding(nn.Module):
    """Whisper-style fixed sinusoidal table, recomputed rather than checkpointed."""

    def __init__(self, length: int, channels: int, max_timescale: int = 10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding requires an even channel count")
        log_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        return self.positional_embedding[:seqlen, :]


class AudioAttention(nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.attn = VarlenSelfAttention(self.num_heads, self.head_dim, self.head_dim**-0.5)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        shape = (num_tokens, self.num_heads, self.head_dim)
        q = self.q_proj(hidden_states).view(shape)
        k = self.k_proj(hidden_states).view(shape)
        v = self.v_proj(hidden_states).view(shape)
        out = self.attn(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.out_proj(out.reshape(num_tokens, self.embed_dim))


class AudioEncoderLayer(nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.self_attn = AudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        if config.activation_function != "gelu":
            raise ValueError(f"unsupported audio activation: {config.activation_function}")

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, cu_seqlens, max_seqlen)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(F.gelu(self.fc1(hidden_states)))
        return residual + hidden_states


class AudioEncoder(nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        self.chunk_frames = config.chunk_frames
        if self.chunk_frames != CHUNK_FRAMES:
            raise ValueError(
                f"audio token accounting assumes {CHUNK_FRAMES}-frame chunks, "
                f"config asks for {self.chunk_frames}"
            )
        self.tokens_per_chunk = TOKENS_PER_FULL_CHUNK
        self.conv_chunksize = config.conv_chunksize

        self.positional_embedding = SinusoidsPositionEmbedding(
            config.max_source_positions, config.d_model
        )
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1
        )
        mel_after_conv = (((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * mel_after_conv, config.d_model, bias=False
        )
        self.layers = nn.ModuleList(
            [AudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.ln_post = nn.LayerNorm(config.d_model)
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.proj2 = nn.Linear(config.d_model, config.output_dim)
        # Replaced by a profiler to split this forward into its three phases.
        self.timer = NULL_TIMER

    def _chunk_and_pad(self, batch: PackedAudioBatch) -> torch.Tensor:
        """Reshape packed mel into ``(num_chunks, 1, num_mel_bins, chunk_frames)``.

        Chunks are padded to the full ``chunk_frames`` regardless of batch
        contents, which keeps the post-convolution width -- and therefore the
        position-embedding slice -- independent of how requests were grouped.
        """
        chunks = torch.split(batch.mel.transpose(0, 1), batch.chunk_lengths.tolist(), dim=0)
        padded = batch.mel.new_zeros(
            (len(chunks), self.chunk_frames, self.config.num_mel_bins)
        )
        for index, chunk in enumerate(chunks):
            padded[index, : chunk.size(0)] = chunk
        return padded.transpose(1, 2).unsqueeze(1)

    def _downsample(self, padded_mel: torch.Tensor) -> torch.Tensor:
        """Three stride-2 convolutions, sliced to bound peak activation memory."""
        embeds = []
        for chunk in padded_mel.split(self.conv_chunksize, dim=0):
            hidden = F.gelu(self.conv2d1(chunk))
            hidden = F.gelu(self.conv2d2(hidden))
            hidden = F.gelu(self.conv2d3(hidden))
            embeds.append(hidden)
        embed = torch.cat(embeds, dim=0)
        num_chunks, channels, mel_bins, time = embed.size()
        embed = embed.permute(0, 3, 1, 2).contiguous().view(num_chunks, time, channels * mel_bins)
        return self.conv_out(embed)

    def _valid_token_mask(self, batch: PackedAudioBatch) -> torch.Tensor:
        positions = torch.arange(self.tokens_per_chunk, device=batch.chunk_valid_lens.device)
        return positions.unsqueeze(0) < batch.chunk_valid_lens.unsqueeze(1)

    @torch.inference_mode()
    def forward(self, batch: PackedAudioBatch) -> torch.Tensor:
        """Encode a packed batch into ``(total_audio_tokens, output_dim)``."""
        with self.timer.phase("conv"):
            with self.timer.phase("conv.chunk_pad"):
                padded_mel = self._chunk_and_pad(batch)
            with self.timer.phase("conv.downsample"):
                embed = self._downsample(padded_mel)
            with self.timer.phase("conv.gather"):
                embed = embed + self.positional_embedding(embed.size(1)).unsqueeze(0).to(
                    embed.dtype
                )
                hidden_states = embed[self._valid_token_mask(batch)]

        if hidden_states.size(0) != batch.total_tokens:
            raise RuntimeError(
                f"encoder produced {hidden_states.size(0)} tokens but the closed-form "
                f"accounting expected {batch.total_tokens}"
            )

        with self.timer.phase("audio_transformer"):
            for layer in self.layers:
                hidden_states = layer(hidden_states, batch.cu_seqlens, batch.max_seqlen)

        with self.timer.phase("projector"):
            hidden_states = self.ln_post(hidden_states)
            hidden_states = self.proj2(F.gelu(self.proj1(hidden_states)))
        return hidden_states
