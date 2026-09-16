"""Prompt construction tests.

These need the checkpoint's tokenizer but no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_asr_vllm.config import Qwen3ASRConfig
from qwen_asr_vllm.prompt import ASR_TEXT_TAG, PromptBuilder

from .conftest import model_path

pytestmark = pytest.mark.checkpoint

MODEL_PATH = model_path()

IM_START = 151644
IM_END = 151645
AUDIO_START = 151669
AUDIO_END = 151670
AUDIO_PAD = 151676
ASR_TEXT = 151704


@pytest.fixture(scope="module")
def builder() -> PromptBuilder:
    return PromptBuilder(MODEL_PATH, Qwen3ASRConfig.from_pretrained(MODEL_PATH))


def test_bare_template_is_sixteen_tokens(builder):
    layout = builder.build(num_audio_tokens=1)
    assert layout.token_ids == [
        IM_START, 8948, 198,
        IM_END, 198,
        IM_START, 872, 198,
        AUDIO_START, AUDIO_PAD, AUDIO_END, IM_END, 198,
        IM_START, 77091, 198,
    ]  # fmt: skip
    assert layout.audio_offset == 9
    assert layout.audio_length == 1


def test_placeholder_expansion_equals_literal_repetition(builder):
    """Expanding the id list must match tokenizing a string of N placeholders."""
    for num_audio_tokens in (1, 2, 13, 46, 303):
        layout = builder.build(num_audio_tokens)
        literal = builder.tokenizer.encode(
            f"<|im_start|>system\n<|im_end|>\n"
            f"<|im_start|>user\n<|audio_start|>{'<|audio_pad|>' * num_audio_tokens}"
            f"<|audio_end|><|im_end|>\n<|im_start|>assistant\n"
        )
        assert layout.token_ids == literal, f"diverged at {num_audio_tokens} audio tokens"


def test_audio_span_is_contiguous_and_correctly_located(builder):
    layout = builder.build(num_audio_tokens=46, context="some domain hints")
    start, end = layout.audio_offset, layout.audio_offset + layout.audio_length
    assert layout.token_ids[start:end] == [AUDIO_PAD] * 46
    assert layout.token_ids[start - 1] == AUDIO_START
    assert layout.token_ids[end] == AUDIO_END
    assert AUDIO_PAD not in layout.token_ids[:start]
    assert AUDIO_PAD not in layout.token_ids[end:]


def test_context_shifts_the_audio_span(builder):
    without = builder.build(num_audio_tokens=46)
    with_context = builder.build(num_audio_tokens=46, context="acme corp, widget, sprocket")
    assert with_context.audio_offset > without.audio_offset
    assert len(with_context) > len(without)


def test_forced_language_primes_the_assistant_turn(builder):
    layout = builder.build(num_audio_tokens=46, language="English")
    assert layout.token_ids[-1] == ASR_TEXT
    assert builder.tokenizer.decode(layout.token_ids[-3:]).endswith(f"English{ASR_TEXT_TAG}")


def test_stop_tokens_come_from_generation_config(builder):
    assert builder.stop_token_ids == {151643, 151645}


def test_rejects_empty_audio(builder):
    with pytest.raises(ValueError):
        builder.build(num_audio_tokens=0)
