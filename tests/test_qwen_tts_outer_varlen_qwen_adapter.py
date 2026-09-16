import types

import pytest

from qwen_asr_vllm.agent.qwen_tts_outer_varlen import (
    QwenOuterVarlenAdapter,
    VarlenCapabilityError,
)


def _talker():
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=4,
            num_attention_heads=2,
            head_dim=2,
        ),
        model=types.SimpleNamespace(layers=[object(), object()]),
    )


def test_varlen_adapter_exposes_page_contract():
    talker = _talker()

    adapter = QwenOuterVarlenAdapter(
        talker, max_cache_len=64, page_size=4, max_pages=32
    )
    assert adapter.page_size == 4
    assert adapter.max_cache_len == 64


def test_varlen_adapter_rejects_unsupported_model_geometry():
    talker = _talker()
    talker.config.num_attention_heads = 3
    with pytest.raises(VarlenCapabilityError, match="head geometry"):
        QwenOuterVarlenAdapter(
            talker, max_cache_len=64, page_size=4, max_pages=32
        )
