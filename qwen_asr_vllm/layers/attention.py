"""Paged attention over a single varlen kernel call.

Both prefill and decode go through the same ``flash_attn_varlen_func`` on the
paged KV cache: a decode step is just a request whose query length is one. That
is what lets the scheduler put prefill and decode requests in the same batch
instead of alternating between all-prefill and all-decode steps.

FlashAttention aligns the causal mask to the bottom-right when the query is
shorter than the key, so a query length of one attends to the whole cached
context and a partial prefill attends to its prefix plus itself.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import nn

try:
    from flash_attn import flash_attn_varlen_func

    HAS_FLASH_ATTN = True
except ImportError:  # pragma: no cover - the engine refuses to start without it
    flash_attn_varlen_func = None
    HAS_FLASH_ATTN = False

from qwen_asr_vllm.layers.context import get_attention_context


@triton.jit
def _store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    index = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + index)
    if slot == -1:
        return
    key = tl.load(key_ptr + index * key_stride + tl.arange(0, D))
    value = tl.load(value_ptr + index * value_stride + tl.arange(0, D))
    tl.store(k_cache_ptr + slot * D + tl.arange(0, D), key)
    tl.store(v_cache_ptr + slot * D + tl.arange(0, D), value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    num_tokens, num_heads, head_dim = key.shape
    hidden = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == hidden and v_cache.stride(1) == hidden
    assert slot_mapping.numel() == num_tokens
    _store_kvcache_kernel[(num_tokens,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, hidden
    )


class PagedAttention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        if not HAS_FLASH_ATTN:
            raise RuntimeError(
                "flash-attn is required. Without it attention falls back to a Python "
                "loop, which is what made earlier ASR benchmarks look slower than "
                "HuggingFace. Install a flash-attn build matching this torch/CUDA."
            )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # Bound to slices of the engine-owned cache by ModelRunner.
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_attention_context()
        if self.k_cache.numel():
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
            k, v = self.k_cache, self.v_cache
            block_table = context.block_tables
        else:
            # Memory-profiling pass before the cache exists.
            block_table = None
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            block_table=block_table,
        )


class VarlenSelfAttention(nn.Module):
    """Non-causal varlen attention for the audio encoder (no KV cache)."""

    def __init__(self, num_heads: int, head_dim: int, scale: float):
        super().__init__()
        if not HAS_FLASH_ATTN:
            raise RuntimeError("flash-attn is required for the audio encoder")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.scale,
            causal=False,
        )
