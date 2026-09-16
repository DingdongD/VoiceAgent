from __future__ import annotations

import torch
from torch import nn

GREEDY_TEMPERATURE_EPS = 1e-4


class Sampler(nn.Module):
    """Greedy or temperature sampling, chosen per request.

    ASR decoding is greedy in practice, so the greedy result is always computed
    and the sampled branch only replaces it where a request asked for it.
    """

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        greedy = logits.argmax(dim=-1)
        if bool((temperatures < GREEDY_TEMPERATURE_EPS).all()):
            return greedy

        scaled = logits / temperatures.clamp_min(GREEDY_TEMPERATURE_EPS).unsqueeze(-1)
        probs = torch.softmax(scaled, dim=-1)
        # Gumbel-max: one exponential draw per row instead of a multinomial call.
        noise = torch.empty_like(probs).exponential_(1.0)
        sampled = probs.div_(noise).argmax(dim=-1)
        return torch.where(temperatures < GREEDY_TEMPERATURE_EPS, greedy, sampled)
