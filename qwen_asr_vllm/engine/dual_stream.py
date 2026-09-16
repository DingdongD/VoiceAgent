"""Same-thread dual CUDA streams for encode ∥ decode overlap.

Schedule both batches before calling ``run_*``; admit/postprocess only after
return. Decode must not depend on this step's encode outputs.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch


@dataclass
class DualStreamReport:
    mode: str
    model: str
    speedup_wall: float
    speedup_mixed_steps: float
    correctness_ok: bool
    gate_1_15: bool
    n_mixed_steps: int
    notes: str = ""
    serial_mixed_seconds: float = 0.0
    overlap_mixed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_gate(*, correctness_ok: bool, speedup_mixed_steps: float) -> bool:
    return bool(correctness_ok) and speedup_mixed_steps >= 1.15


class DualStreamExecutor:
    """Launch encode and decode on separate CUDA streams (or serially on CPU)."""

    def __init__(self, device: torch.device | str = "cuda"):
        self.device = torch.device(device)
        self.encode_stream: torch.cuda.Stream | None = None
        self.decode_stream: torch.cuda.Stream | None = None
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but unavailable")
            idx = self.device.index if self.device.index is not None else 0
            self.encode_stream = torch.cuda.Stream(device=idx)
            self.decode_stream = torch.cuda.Stream(device=idx)

    def run_serial(
        self, encode_fn: Callable[[], None], decode_fn: Callable[[], None]
    ) -> float:
        start = time.perf_counter()
        encode_fn()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        decode_fn()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter() - start

    def run_overlap(
        self, encode_fn: Callable[[], None], decode_fn: Callable[[], None]
    ) -> float:
        if self.encode_stream is None or self.decode_stream is None:
            return self.run_serial(encode_fn, decode_fn)
        start = time.perf_counter()
        with torch.cuda.stream(self.encode_stream):
            encode_fn()
        with torch.cuda.stream(self.decode_stream):
            decode_fn()
        self.encode_stream.synchronize()
        self.decode_stream.synchronize()
        return time.perf_counter() - start
