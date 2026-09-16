"""Qwen3 text decoder.

Single-GPU only: Qwen3-ASR ships at 0.6B and 1.7B, which leaves plenty of room
on one 40GB card, so there is no tensor-parallel machinery to reason about. The
QKV and gate/up projections are stored fused; the weight loader writes the
per-projection checkpoint tensors into the right slices.
"""

from __future__ import annotations

import torch
from torch import nn

from qwen_asr_vllm.config import TextConfig
from qwen_asr_vllm.layers.activation import SiluAndMul
from qwen_asr_vllm.layers.attention import PagedAttention
from qwen_asr_vllm.layers.layernorm import RMSNorm
from qwen_asr_vllm.layers.rotary_embedding import get_rope


class Qwen3Attention(nn.Module):
    def __init__(self, config: TextConfig, max_position: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = nn.Linear(
            config.hidden_size, self.q_size + 2 * self.kv_size, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)
        # Qwen3 normalises each head of Q and K before applying RoPE.
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(self.head_dim, max_position, config.rope_theta)
        self.attn = PagedAttention(
            self.num_heads, self.head_dim, self.head_dim**-0.5, self.num_kv_heads
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn(q, k, v)
        return self.o_proj(attn_out.flatten(1, -1))


class Qwen3MLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"unsupported activation: {config.hidden_act}")
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, max_position: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, max_position)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class Qwen3Model(nn.Module):
    def __init__(self, config: TextConfig, max_position: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, max_position) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids or inputs_embeds")
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
