from types import SimpleNamespace

import pytest
import torch

from qwen_asr_vllm.config import EngineConfig


def _model_config():
    return SimpleNamespace(
        dtype=torch.float16,
        text=SimpleNamespace(max_position_embeddings=8192),
    )


@pytest.mark.parametrize("blocks", [-1, 1, 128])
def test_engine_config_accepts_automatic_or_positive_kv_blocks(tmp_path, blocks):
    config = EngineConfig(
        model=str(tmp_path),
        model_config=_model_config(),
        num_kvcache_blocks=blocks,
    )

    assert config.num_kvcache_blocks == blocks


@pytest.mark.parametrize("blocks", [0, -2])
def test_engine_config_rejects_invalid_fixed_kv_blocks(tmp_path, blocks):
    with pytest.raises(
        ValueError,
        match="num_kvcache_blocks must be -1 or positive",
    ):
        EngineConfig(
            model=str(tmp_path),
            model_config=_model_config(),
            num_kvcache_blocks=blocks,
        )
