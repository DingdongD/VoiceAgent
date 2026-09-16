# ASR Dual-Stream Encode∥Decode Overlap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a same-thread dual-CUDA-stream primitive and a comparison bench that measures serial vs encode∥decode overlap; only merge into `AsrEngine.step` later if mixed-step speedup ≥ 1.15× with correctness.

**Architecture:** A thin `DualStreamExecutor` owns two CUDA streams and runs `encode_fn` ∥ `decode_fn` without touching the scheduler. A CLI bench builds synthetic mixed steps and real LibriSpeech replay, writes JSON reports with a `gate_1_15` flag. Default `AsrEngine.step` stays unchanged in Phase 1.

**Tech Stack:** PyTorch CUDA streams/events, existing `AudioRunner` / `ModelRunner` / `Scheduler`, pytest (`gpu` markers), LibriSpeech via `bench.data`.

**Spec:** `docs/superpowers/specs/2026-08-20-asr-dual-stream-overlap-design.md`

## Global Constraints

- Phase 1 must **not** change default `AsrEngine.step` behavior.
- Overlap path uses **eager** decode only (`enforce_eager=True` for overlap measurements).
- Overlap only when model batch does not consume same-step encode outputs: schedule both batches first, launch in parallel, then `admit_prefill` / `postprocess`.
- Gate: `correctness_ok` and `speedup_mixed_steps >= 1.15` before any Phase 2.
- Do not import `vla_lib`.
- Commits: only when the user explicitly asks (do not auto-commit).

## File map

| path | responsibility |
|---|---|
| `qwen_asr_vllm/engine/dual_stream.py` | Stream pair + `run_serial` / `run_overlap` |
| `tests/test_dual_stream.py` | Unit tests for ordering / sync (CPU-mockable + optional GPU) |
| `bench/dual_stream_overlap.py` | Synthetic + real replay CLI + JSON report |
| `results/dual_stream_*.json` | Written by bench (gitignored if already) |
| `qwen_asr_vllm/engine/engine.py` | Phase 1: optional one-line comment only pointing at dual_stream |
| `docs/superpowers/plans/2026-08-20-asr-dual-stream-overlap.md` | This plan |

---

### Task 1: DualStreamExecutor primitive

**Files:**
- Create: `qwen_asr_vllm/engine/dual_stream.py`
- Test: `tests/test_dual_stream.py`
- Modify: `qwen_asr_vllm/engine/__init__.py` (export if other engine symbols are exported there)

**Interfaces:**
- Produces:
  - `class DualStreamExecutor`
  - `DualStreamExecutor.__init__(self, device: torch.device | str = "cuda")`
  - `DualStreamExecutor.run_serial(self, encode_fn: Callable[[], None], decode_fn: Callable[[], None]) -> float` — wall seconds
  - `DualStreamExecutor.run_overlap(self, encode_fn: Callable[[], None], decode_fn: Callable[[], None]) -> float` — wall seconds
  - Both callables must be side-effecting runners (mutate request state); return wall time via `time.perf_counter`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dual_stream.py`:

```python
"""DualStreamExecutor orders encode/decode without scheduler involvement."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_asr_vllm.engine.dual_stream import DualStreamExecutor


def test_serial_runs_encode_before_decode():
    order: list[str] = []
    ex = DualStreamExecutor(device="cpu")  # or skip if ctor requires CUDA

    def encode():
        order.append("encode")

    def decode():
        order.append("decode")

    elapsed = ex.run_serial(encode, decode)
    assert order == ["encode", "decode"]
    assert elapsed >= 0.0


def test_overlap_invokes_both(monkeypatch):
    """On CPU fallback, overlap may degrade to serial; both must still run."""
    order: list[str] = []
    ex = DualStreamExecutor(device="cpu")

    def encode():
        order.append("encode")

    def decode():
        order.append("decode")

    elapsed = ex.run_overlap(encode, decode)
    assert set(order) == {"encode", "decode"}
    assert elapsed >= 0.0


@pytest.mark.gpu
def test_overlap_uses_two_cuda_streams():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    ex = DualStreamExecutor(device="cuda")
    assert ex.encode_stream is not None
    assert ex.decode_stream is not None
    assert ex.encode_stream.cuda_stream != ex.decode_stream.cuda_stream
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/qwen-asr-vllm && /opt/conda/envs/nano-vllm/bin/python -m pytest tests/test_dual_stream.py -v --ignore-glob='*gpu*' -k 'serial or overlap_invokes' 2>&1 | tail -40`

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `dual_stream`.

- [ ] **Step 3: Implement DualStreamExecutor**

Create `qwen_asr_vllm/engine/dual_stream.py`:

```python
"""Same-thread dual CUDA streams for encode ∥ decode overlap.

Schedule both batches before calling run_*; admit/postprocess only after return.
Decode must not depend on this step's encode outputs.
"""

from __future__ import annotations

import time
from typing import Callable

import torch


class DualStreamExecutor:
    def __init__(self, device: torch.device | str = "cuda"):
        self.device = torch.device(device)
        self.encode_stream: torch.cuda.Stream | None = None
        self.decode_stream: torch.cuda.Stream | None = None
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but unavailable")
            idx = self.device.index or 0
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
```

If `engine/__init__.py` exports public symbols, add `DualStreamExecutor` there; otherwise leave it importable via package path only.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/qwen-asr-vllm && /opt/conda/envs/nano-vllm/bin/python -m pytest tests/test_dual_stream.py -v -k 'not gpu' 2>&1 | tail -40`

Expected: PASS for CPU tests. Optionally run GPU stream identity test if a free GPU is available:

`pytest tests/test_dual_stream.py::test_overlap_uses_two_cuda_streams -v`

- [ ] **Step 5: Commit only if user asked**

Skip unless the user requested a commit.

---

### Task 2: Synthetic overlap harness inside the bench module

**Files:**
- Create: `bench/dual_stream_overlap.py` (scaffold + synthetic path)
- Test: extend `tests/test_dual_stream.py` with a lightweight harness unit that mocks runners **or** a pure helper test for report JSON

**Interfaces:**
- Consumes: `DualStreamExecutor.run_serial` / `run_overlap`
- Produces:
  - `dataclass DualStreamReport` with fields matching the spec JSON
  - `def run_synthetic(engine: AsrEngine, *, rounds: int, warmup: int) -> DualStreamReport`
  - Helper that, given encode_batch and decode_batch lists already scheduled, times serial vs overlap of `audio_runner.encode` and `model_runner.run`

- [ ] **Step 1: Write failing test for report gate logic**

Add to `tests/test_dual_stream.py`:

```python
from qwen_asr_vllm.engine.dual_stream import DualStreamReport, compute_gate


def test_compute_gate_requires_correctness_and_1_15():
    assert compute_gate(correctness_ok=True, speedup_mixed_steps=1.15) is True
    assert compute_gate(correctness_ok=True, speedup_mixed_steps=1.149) is False
    assert compute_gate(correctness_ok=False, speedup_mixed_steps=2.0) is False
```

Put `DualStreamReport` / `compute_gate` in `dual_stream.py` (shared) or at top of bench with re-export — prefer `dual_stream.py` so tests do not import `bench`.

- [ ] **Step 2: Run test — expect fail**

Run: `pytest tests/test_dual_stream.py::test_compute_gate_requires_correctness_and_1_15 -v`

Expected: FAIL import of `DualStreamReport` / `compute_gate`.

- [ ] **Step 3: Add report types + synthetic runner**

Append to `dual_stream.py`:

```python
from dataclasses import asdict, dataclass, field


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
```

In `bench/dual_stream_overlap.py`, implement synthetic path (outline to flesh fully during coding):

1. Build `AsrEngine(model, enforce_eager=True, max_num_seqs=..., gpu_memory_utilization=0.25)`.
2. Load N LibriSpeech clips (or sine tones via numpy if offline).
3. Warmup: `transcribe` a few clips so KV/graphs paths are exercised (graphs off).
4. Manually: add requests, step until some are decoding and some wait encode; capture `encode_batch = schedule_audio()` and `model_batch = schedule_model()` **without** running yet — may need a small test hook or duplicate schedule logic carefully.

**Practical approach for schedule-then-execute without changing engine:**

Add optional methods on the bench only by subclassing:

```python
class InstrumentedAsrEngine(AsrEngine):
    def step_timed(self, executor: DualStreamExecutor, mode: str) -> tuple[list, float, bool]:
        """Returns (outputs, elapsed, was_mixed). mode in {'serial','overlap'}."""
        encode_batch = self.scheduler.schedule_audio()
        model_batch = self.scheduler.schedule_model() if False else None
        # WRONG: schedule_model after encode admit — follow spec:
        # schedule_audio, schedule_model BEFORE execute; admit AFTER.
```

Correct instrumented step:

```python
def step_instrumented(self, executor, mode: str):
    encode_batch = self.scheduler.schedule_audio()
    # Peek model schedule without consuming encode→prefill: schedule_model only
    # sees waiting_prefill + running. So we MUST schedule model BEFORE admit.
    batch = self.scheduler.schedule_model()
    was_mixed = bool(encode_batch) and bool(batch)

    def encode_fn():
        if encode_batch:
            self.audio_runner.encode(encode_batch)

    def decode_fn():
        nonlocal token_ids
        token_ids = None
        if batch:
            token_ids = self.model_runner.run(batch)

    token_ids = None
    if mode == "serial":
        elapsed = executor.run_serial(encode_fn, decode_fn)
    else:
        elapsed = executor.run_overlap(encode_fn, decode_fn)

    finished = []
    if encode_batch:
        self.scheduler.admit_prefill(encode_batch)
    if batch and token_ids is not None:
        finished = self.scheduler.postprocess(batch, token_ids)
        self.scheduler.relax_limits()
    outputs = [self._finalize(r) for r in self.scheduler.drain_aborted() + finished]
    return outputs, elapsed, was_mixed
```

**Important:** Production `step` calls `schedule_model` *after* admit, so a
just-encoded request can prefill in the **same** step. The prototype uses
**overlap-eligible order** for *both* serial and overlap timings: schedule
audio + model first, execute, then admit. That defers new prefills by one step
but keeps serial vs overlap an apples-to-apples compare of launch overlap only.
Document this in the bench module docstring. Do not claim the prototype matches
production admission timing until Phase 2 redesigns `step`.

Synthetic correctness: run the same prepared batches twice (clone request state is hard). Prefer: for each mixed opportunity, run serial once and overlap once on **fresh identical engines** with same seed/order — expensive. Lighter approach for synthetic:

- After warmup, for K rounds: deep-copy is unavailable for requests; instead compare `model_runner.run` outputs by:
  1. Run overlap encode+decode
  2. Re-encode the same mel on serial path into temporary tensors and `allclose` embeds
  3. For decode: save KV snapshot is hard

**Simpler correctness for Task 2 (synthetic):**

- Run `run_overlap` and `run_serial` back-to-back on the **same** encode_fn/decode_fn that only compute into local tensors (no KV write) — too fake.

**Spec-aligned pragmatic correctness for synthetic:**

1. Full engine path serial-instrumented over a fixed clip set → list of texts T_s  
2. Full engine path overlap-instrumented over same clips → texts T_o  
3. `correctness_ok = (T_s == T_o)`  

Timing: accumulate `elapsed` only when `was_mixed`.

Implement `run_synthetic` and `run_real` both using `step_instrumented`.

- [ ] **Step 4: Run gate unit test — expect pass**

`pytest tests/test_dual_stream.py::test_compute_gate_requires_correctness_and_1_15 -v`

- [ ] **Step 5: Commit only if user asked**

---

### Task 3: CLI bench — synthetic + real + JSON

**Files:**
- Modify: `bench/dual_stream_overlap.py` (complete CLI)
- Create results under `results/`

**Interfaces:**
- Consumes: `InstrumentedAsrEngine.step_instrumented`, `DualStreamReport`, `compute_gate`, `bench.data.load_librispeech`
- Produces: CLI entrypoint writing `results/dual_stream_{mode}_{timestamp}.json`

- [ ] **Step 1: Implement CLI**

```python
# bench/dual_stream_overlap.py — main
"""Compare serial vs dual-stream encode∥decode (Phase 1 prototype).

Usage:
  /opt/conda/envs/nano-vllm/bin/python bench/dual_stream_overlap.py \\
      --model /mnt/llm_data/voice_ckpt/Qwen3-ASR-0.6B \\
      --mode both --num-samples 32 --concurrency 8
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/llm_data/voice_ckpt/Qwen3-ASR-0.6B")
    parser.add_argument("--mode", choices=("synthetic", "real", "both"), default="both")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    # run modes, write JSON, print gate_1_15
```

For each mode, construct report:

```python
speedup = serial_s / overlap_s if overlap_s > 0 else 0.0
report = DualStreamReport(
    mode=mode,
    model=args.model,
    speedup_wall=wall_serial / wall_overlap,
    speedup_mixed_steps=speedup,
    correctness_ok=texts_match,
    gate_1_15=compute_gate(correctness_ok=texts_match, speedup_mixed_steps=speedup),
    n_mixed_steps=n_mixed,
    serial_mixed_seconds=serial_s,
    overlap_mixed_seconds=overlap_s,
)
```

Print a one-line verdict: `GATE PASS` or `GATE FAIL`.

- [ ] **Step 2: Smoke-run synthetic on GPU**

Run:

```bash
cd /home/qwen-asr-vllm
/opt/conda/envs/nano-vllm/bin/python bench/dual_stream_overlap.py \
  --mode synthetic --num-samples 16 --max-num-seqs 8
```

Expected: JSON written; `correctness_ok` true or false printed; no crash.

- [ ] **Step 3: Smoke-run real mode**

```bash
/opt/conda/envs/nano-vllm/bin/python bench/dual_stream_overlap.py \
  --mode real --num-samples 32 --max-num-seqs 8
```

Expected: `n_mixed_steps > 0` (if not, increase samples / lower max_num_seqs pressure differently — try max_num_seqs=4 with 32 samples).

- [ ] **Step 4: Commit only if user asked**

---

### Task 4: Engine comment + README pointer (no behavior change)

**Files:**
- Modify: `qwen_asr_vllm/engine/engine.py` (comment above `step` only)
- Modify: `README.md` — short "Dual-stream prototype" blurb under Architecture or a Experiments subsection pointing to the bench and spec

- [ ] **Step 1: Add comment in `engine.py` above `def step`**

```python
    def step(self) -> list[AsrOutput]:
        # Phase 1 dual-stream prototype lives in engine/dual_stream.py and
        # bench/dual_stream_overlap.py. Default path stays serial encode→model.
```

- [ ] **Step 2: Add README subsection (~8 lines)** describing how to run the bench and the 1.15× gate.

- [ ] **Step 3: Commit only if user asked**

---

### Task 5: Record results and Phase-2 go/no-go

**Files:**
- Write: `results/dual_stream_summary.md` (short human summary)

- [ ] **Step 1: Run `--mode both` and save JSON**

- [ ] **Step 2: Write `results/dual_stream_summary.md`** with:
  - measured `speedup_mixed_steps`
  - `correctness_ok`
  - explicit **GO Phase 2** or **NO-GO** per gate

- [ ] **Step 3: If NO-GO — stop.** Do not modify `AsrEngine.step` execution.
- [ ] **Step 4: If GO — open a follow-up plan** for flagging `enable_dual_stream` into `step` (out of this plan's coding tasks).

---

## Spec coverage checklist

| Spec requirement | Task |
|---|---|
| `dual_stream.py` primitive | Task 1 |
| serial vs overlap runners | Task 1 |
| Report schema + `gate_1_15` | Task 2 |
| Synthetic workload | Task 2–3 |
| Real LibriSpeech replay | Task 3 |
| Eager-only overlap | Task 3 (`enforce_eager=True`) |
| Schedule before execute; admit after | Task 2 instrumented step |
| No default `step` change | Task 4 comment only |
| Phase 2 conditional | Task 5 go/no-go |

## Placeholder / consistency self-review

- No TBD left for required APIs; instrumented step order is specified explicitly.
- `DualStreamReport` / `compute_gate` live in `dual_stream.py` for unit testing without GPU.
- Gate threshold `1.15` appears only in `compute_gate` (single source of truth).
