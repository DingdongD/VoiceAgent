"""Per-forward attention metadata.

Batch assembly happens in the model runner while attention needs the resulting
offsets deep inside the layer stack, so the two are bridged by a module-level
value rather than threaded through every signature.

There is deliberately no ``is_prefill`` flag: prefill and decode share one
kernel path, and a batch may hold both at once.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionContext:
    cu_seqlens_q: torch.Tensor | None = None
    """Cumulative query lengths; a decode request contributes 1."""
    cu_seqlens_k: torch.Tensor | None = None
    """Cumulative context lengths, including tokens already in the cache."""
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    """Physical KV slot for each query token; ``-1`` skips the write."""
    block_tables: torch.Tensor | None = None


_CONTEXT = AttentionContext()


def get_attention_context() -> AttentionContext:
    return _CONTEXT


def set_attention_context(context: AttentionContext) -> None:
    global _CONTEXT
    _CONTEXT = context


@contextmanager
def attention_context(context: AttentionContext):
    previous = get_attention_context()
    set_attention_context(context)
    try:
        yield
    finally:
        set_attention_context(previous)
