from __future__ import annotations

from functools import lru_cache

import torch
from torch import nn


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate using the half-split convention (first half paired with second)."""
    first, second = torch.chunk(x.float(), 2, dim=-1)
    return torch.cat(
        (first * cos - second * sin, second * cos + first * sin), dim=-1
    ).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position: int, base: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        positions = torch.arange(max_position, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.cos_sin_cache[positions].chunk(2, dim=-1)
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)


@lru_cache(maxsize=4)
def get_rope(head_dim: int, max_position: int, base: float) -> RotaryEmbedding:
    return RotaryEmbedding(head_dim, max_position, base)
