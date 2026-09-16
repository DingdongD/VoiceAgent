"""Shared fixtures and environment gating.

Most of this suite needs neither a GPU nor the weights, but a handful of tests do,
and they used to fail rather than skip on a machine without them -- which made the
whole suite unrunnable in CI. Tests marked ``gpu`` or ``checkpoint`` are skipped
automatically when the resource is absent, so ``pytest`` alone always passes on a
plain CPU box and the same command gains coverage on a GPU one.

Point ``QWEN_ASR_MODEL`` at a checkpoint to override the default path.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

DEFAULT_MODEL_PATH = "/mnt/llm_data/voice_ckpt/Qwen3-ASR-0.6B"


def model_path() -> str:
    return os.environ.get("QWEN_ASR_MODEL", DEFAULT_MODEL_PATH)


def has_checkpoint() -> bool:
    path = Path(model_path())
    return path.is_dir() and any(path.glob("*.safetensors"))


def has_longform_dataset() -> bool:
    from bench.data import TEDLIUM_LONG_FORM

    return TEDLIUM_LONG_FORM.is_dir() and any(TEDLIUM_LONG_FORM.glob("*.parquet"))


def has_gpu() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="needs a CUDA device with flash-attn")
    skip_checkpoint = pytest.mark.skip(reason=f"no checkpoint at {model_path()}")
    skip_dataset = pytest.mark.skip(reason="no long-form dataset shards on disk")
    gpu_available = has_gpu()
    checkpoint_available = has_checkpoint()
    dataset_available = has_longform_dataset()

    for item in items:
        if "gpu" in item.keywords and not gpu_available:
            item.add_marker(skip_gpu)
        if "checkpoint" in item.keywords and not checkpoint_available:
            item.add_marker(skip_checkpoint)
        if "dataset" in item.keywords and not dataset_available:
            item.add_marker(skip_dataset)


@pytest.fixture(scope="session")
def model_dir() -> str:
    return model_path()
