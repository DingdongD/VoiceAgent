from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMS norm with an optional fused residual add.

    The fused form returns the pre-norm sum as the next residual so a decoder
    layer never has to materialise it separately.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(variance + self.eps))
        return x.to(orig_dtype).mul_(self.weight)

    @torch.compile
    def _add_norm(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(variance + self.eps))
        return x.to(orig_dtype).mul_(self.weight), residual

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._norm(x)
        return self._add_norm(x, residual)
