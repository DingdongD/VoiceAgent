"""DualStreamExecutor orders encode/decode without scheduler involvement."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_asr_vllm.engine.dual_stream import DualStreamExecutor, DualStreamReport, compute_gate


def test_serial_runs_encode_before_decode():
    order: list[str] = []
    ex = DualStreamExecutor(device="cpu")

    def encode():
        order.append("encode")

    def decode():
        order.append("decode")

    elapsed = ex.run_serial(encode, decode)
    assert order == ["encode", "decode"]
    assert elapsed >= 0.0


def test_overlap_invokes_both():
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


def test_compute_gate_requires_correctness_and_1_15():
    assert compute_gate(correctness_ok=True, speedup_mixed_steps=1.15) is True
    assert compute_gate(correctness_ok=True, speedup_mixed_steps=1.149) is False
    assert compute_gate(correctness_ok=False, speedup_mixed_steps=2.0) is False
    report = DualStreamReport(
        mode="synthetic",
        model="dummy",
        speedup_wall=1.0,
        speedup_mixed_steps=1.2,
        correctness_ok=True,
        gate_1_15=True,
        n_mixed_steps=3,
    )
    assert report.to_dict()["gate_1_15"] is True


@pytest.mark.gpu
def test_overlap_uses_two_cuda_streams():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    ex = DualStreamExecutor(device="cuda")
    assert ex.encode_stream is not None
    assert ex.decode_stream is not None
    assert ex.encode_stream is not ex.decode_stream
