"""Stage timing with CUDA events, off unless a profiler turns it on.

Wall-clock timers around GPU phases measure when work was *issued*, not when it
ran, and the decode path showed how badly that misleads: the CPU was five steps
ahead of the GPU. CUDA events are recorded in the stream, so they time the device.

Disabled by default and a no-op in that state, because the phases being timed are
inside the per-layer hot path -- the audio encoder's projector runs once per encode
call, but ``phase`` is entered on every request in a batch.

Events are collected during the run and only read afterwards: ``elapsed_time``
synchronises, and calling it mid-run would serialise exactly the CPU/GPU overlap the
engine works to create.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager

import torch


class StageTimer:
    """Accumulates device time per named phase."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def phase(self, name: str):
        if not self.enabled:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pending.append((name, start, end))

    def collect(self) -> None:
        """Drain recorded events into totals. Synchronises; call between runs."""
        if not self._pending:
            return
        torch.cuda.synchronize()
        for name, start, end in self._pending:
            self._totals[name] += start.elapsed_time(end) / 1000
            self._counts[name] += 1
        self._pending.clear()

    @property
    def totals(self) -> dict[str, float]:
        self.collect()
        return dict(self._totals)

    @property
    def counts(self) -> dict[str, int]:
        self.collect()
        return dict(self._counts)

    def reset(self) -> None:
        self._pending.clear()
        self._totals.clear()
        self._counts.clear()


# A single disabled timer shared by every component that has nothing to report, so
# the normal path allocates nothing and the enabled check is one attribute read.
NULL_TIMER = StageTimer(enabled=False)
