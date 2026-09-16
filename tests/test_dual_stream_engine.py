"""Phase 2: enable_dual_stream must match serial transcripts on real audio."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .conftest import model_path

pytestmark = [pytest.mark.gpu, pytest.mark.checkpoint, pytest.mark.slow]

MODEL_PATH = model_path()
GPU_FRACTION = 0.25
NUM_SAMPLES = 16


@pytest.fixture(scope="module")
def samples():
    from bench.data import load_librispeech

    return load_librispeech("test-clean", num_samples=NUM_SAMPLES, seed=1)


def _make_engine(*, dual: bool):
    from qwen_asr_vllm import AsrEngine

    return AsrEngine(
        MODEL_PATH,
        max_num_seqs=8,
        max_model_len=2048,
        gpu_memory_utilization=GPU_FRACTION,
        enable_dual_stream=dual,
        enforce_eager=True,
    )


def test_dual_stream_matches_serial_transcripts(samples):
    waves = [s.audio for s in samples]
    serial = _make_engine(dual=False)
    try:
        texts_serial = [o.text for o in serial.transcribe(waves, language="en")]
    finally:
        del serial
        gc.collect()
        torch.cuda.empty_cache()

    dual = _make_engine(dual=True)
    try:
        texts_dual = [o.text for o in dual.transcribe(waves, language="en")]
    finally:
        del dual
        gc.collect()
        torch.cuda.empty_cache()

    assert texts_dual == texts_serial


def test_enable_dual_stream_forces_eager():
    from qwen_asr_vllm.config import EngineConfig

    cfg = EngineConfig(
        model=MODEL_PATH,
        enable_dual_stream=True,
        enforce_eager=False,
        max_model_len=2048,
        max_num_batched_tokens=4096,
    )
    assert cfg.enforce_eager is True
