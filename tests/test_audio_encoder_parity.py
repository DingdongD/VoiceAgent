"""Numerical parity for the audio tower.

Cross-request batching is the premise of the whole engine design, so it needs a
sharper test than "the outputs look close". Encoding N requests together does not
reproduce encoding them one at a time bit for bit: cuDNN picks different
convolution algorithms for different batch sizes, and a one-ULP difference at the
convolution output grows to a few percent after eighteen residual layers.

That drift belongs to the model and the hardware, not to this implementation --
the upstream encoder also takes a time-concatenated mel tensor with per-request
``feature_lens``, and exhibits exactly the same drift when driven that way. So
the tests pin the property that actually matters: our packed output equals the
reference's packed output, and batching costs us no more accuracy than batching
costs the reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_asr_vllm.audio.batcher import pack_audio_batch
from qwen_asr_vllm.audio.frontend import AudioFrontend
from qwen_asr_vllm.audio.tokens import num_audio_tokens
from qwen_asr_vllm.config import Qwen3ASRConfig
from qwen_asr_vllm.loader import _iter_checkpoint_tensors
from qwen_asr_vllm.models.audio_encoder import AudioEncoder

from .conftest import model_path

MODEL_PATH = model_path()
AUDIO_TOWER_PREFIX = "thinker.audio_tower."
BF16_TOLERANCE = 2e-2
PACKED_TOLERANCE = 1e-3
# 2.0s lands on an exact chunk boundary; the rest leave a short tail chunk.
DURATIONS = [3.5, 2.0, 7.3, 1.05]

pytestmark = [pytest.mark.gpu, pytest.mark.checkpoint]


def _load_audio_tower_weights(model: torch.nn.Module) -> None:
    params = dict(model.named_parameters())
    for name, tensor in _iter_checkpoint_tensors(MODEL_PATH):
        if not name.startswith(AUDIO_TOWER_PREFIX):
            continue
        with torch.no_grad():
            params[name[len(AUDIO_TOWER_PREFIX) :]].copy_(tensor)


@pytest.fixture(scope="module")
def config() -> Qwen3ASRConfig:
    return Qwen3ASRConfig.from_pretrained(MODEL_PATH)


@pytest.fixture(scope="module")
def encoder(config: Qwen3ASRConfig) -> AudioEncoder:
    torch.set_default_dtype(config.dtype)
    try:
        with torch.device("cuda"):
            model = AudioEncoder(config.audio)
    finally:
        torch.set_default_dtype(torch.float32)
    _load_audio_tower_weights(model)
    return model.eval()


@pytest.fixture(scope="module")
def reference_encoder(config: Qwen3ASRConfig):
    from reference.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from reference.modeling_qwen3_asr import Qwen3ASRAudioEncoder

    audio = config.audio
    ref_config = Qwen3ASRAudioEncoderConfig(
        num_mel_bins=audio.num_mel_bins,
        d_model=audio.d_model,
        encoder_layers=audio.encoder_layers,
        encoder_attention_heads=audio.encoder_attention_heads,
        encoder_ffn_dim=audio.encoder_ffn_dim,
        downsample_hidden_size=audio.downsample_hidden_size,
        output_dim=audio.output_dim,
        n_window=audio.n_window,
        n_window_infer=audio.n_window_infer,
        conv_chunksize=audio.conv_chunksize,
        max_source_positions=audio.max_source_positions,
        activation_function=audio.activation_function,
    )
    ref_config._attn_implementation = "flash_attention_2"
    model = Qwen3ASRAudioEncoder(ref_config).to(device="cuda", dtype=config.dtype).eval()
    _load_audio_tower_weights(model)
    return model


@pytest.fixture(scope="module")
def mels(config: Qwen3ASRConfig) -> list[torch.Tensor]:
    frontend = AudioFrontend(MODEL_PATH)
    rng = np.random.default_rng(1234)
    out = []
    for duration in DURATIONS:
        waveform = rng.standard_normal(int(16000 * duration)).astype(np.float32)
        features = frontend(waveform, 16000)
        assert features.num_audio_tokens == num_audio_tokens(features.mel_frames)
        out.append(features.mel.to(device="cuda", dtype=config.dtype))
    return out


def _reference_encode(reference_encoder, mels: list[torch.Tensor]) -> torch.Tensor:
    """Drive the reference the same way we drive ours: one packed call."""
    packed = torch.cat(mels, dim=-1)
    feature_lens = torch.tensor([m.size(-1) for m in mels], device=packed.device)
    with torch.inference_mode():
        return reference_encoder(packed, feature_lens=feature_lens).last_hidden_state


def _max_abs_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return (left.float() - right.float()).abs().max().item()


def test_token_count_matches_encoder_output(config, encoder, mels):
    """The closed-form accounting must predict the encoder's real output length."""
    for mel in mels:
        batch = pack_audio_batch([mel], config.audio.n_window_infer, device="cuda")
        assert encoder(batch).size(0) == num_audio_tokens(mel.size(-1))


def test_single_request_matches_reference(config, encoder, reference_encoder, mels):
    for index, mel in enumerate(mels):
        batch = pack_audio_batch([mel], config.audio.n_window_infer, device="cuda")
        ours = encoder(batch)
        expected = _reference_encode(reference_encoder, [mel])
        assert ours.shape == expected.shape
        delta = _max_abs_delta(ours, expected)
        assert delta < BF16_TOLERANCE, f"request {index} ({DURATIONS[index]}s) drifted by {delta}"


def test_packed_batch_matches_reference_packed_batch(config, encoder, reference_encoder, mels):
    """Cross-request packing reproduces the reference's own packed path."""
    batch = pack_audio_batch(mels, config.audio.n_window_infer, device="cuda")
    ours = encoder(batch)
    expected = _reference_encode(reference_encoder, mels)

    assert ours.shape == expected.shape
    delta = _max_abs_delta(ours, expected)
    assert delta < PACKED_TOLERANCE, f"packed output drifted from reference by {delta}"


def test_batching_costs_no_more_accuracy_than_reference(config, encoder, reference_encoder, mels):
    """Our batch-vs-single drift must not exceed the reference's own."""
    window = config.audio.n_window_infer
    batch = pack_audio_batch(mels, window, device="cuda")
    ours_batched = batch.split_outputs(encoder(batch))
    reference_batched = batch.split_outputs(_reference_encode(reference_encoder, mels))

    for index, mel in enumerate(mels):
        ours_alone = encoder(pack_audio_batch([mel], window, device="cuda"))
        reference_alone = _reference_encode(reference_encoder, [mel])

        our_drift = _max_abs_delta(ours_alone, ours_batched[index])
        reference_drift = _max_abs_delta(reference_alone, reference_batched[index])
        assert our_drift <= reference_drift + PACKED_TOLERANCE, (
            f"request {index} ({DURATIONS[index]}s): batching cost us {our_drift} "
            f"but only cost the reference {reference_drift}"
        )
