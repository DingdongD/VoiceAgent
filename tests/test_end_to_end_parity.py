"""End-to-end parity against the official implementation.

The engine is not expected to be bit-identical to a HuggingFace ``generate``
loop -- paged attention, fused projections and cross-request audio batching all
reorder floating-point work. What must hold is that the recognition quality is
indistinguishable: the same WER against the LibriSpeech references, and near
identical transcriptions.

Both paths share this project's mel frontend and prompt construction, so any
difference measured here comes from the inference engine itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.baselines import ReferenceTranscriber
from bench.data import load_librispeech
from bench.metrics import compute_wer, exact_match_rate

from .conftest import model_path

MODEL_PATH = model_path()
NUM_SAMPLES = 32
# The engine may reorder floating-point work, so a handful of tokens can differ.
MAX_WER_GAP = 0.005
MIN_EXACT_MATCH = 0.90

pytestmark = [pytest.mark.gpu, pytest.mark.checkpoint, pytest.mark.slow]


@pytest.fixture(scope="module")
def samples():
    return load_librispeech("test-clean", num_samples=NUM_SAMPLES, seed=0)


@pytest.fixture(scope="module")
def engine_outputs(samples):
    import gc

    import torch

    from qwen_asr_vllm import AsrEngine

    engine = AsrEngine(MODEL_PATH, max_num_seqs=16, max_model_len=2048)
    try:
        outputs = engine.transcribe([sample.audio for sample in samples])
    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    return outputs


@pytest.fixture(scope="module")
def reference_outputs(samples):
    import gc

    import torch

    transcriber = ReferenceTranscriber(MODEL_PATH)
    try:
        result = transcriber.transcribe([sample.audio for sample in samples])
    finally:
        del transcriber
        gc.collect()
        torch.cuda.empty_cache()
    return result


def test_engine_wer_matches_reference(samples, engine_outputs, reference_outputs):
    references = [sample.text for sample in samples]
    engine_wer = compute_wer(references, [output.text for output in engine_outputs])
    reference_wer = compute_wer(references, reference_outputs.texts)

    assert abs(engine_wer - reference_wer) <= MAX_WER_GAP, (
        f"engine WER {engine_wer:.4f} vs reference WER {reference_wer:.4f}"
    )


def test_engine_transcriptions_match_reference(engine_outputs, reference_outputs):
    match_rate = exact_match_rate(
        [output.text for output in engine_outputs], reference_outputs.texts
    )
    assert match_rate >= MIN_EXACT_MATCH, f"only {match_rate:.1%} of transcriptions matched"


def test_detected_language_matches_reference(engine_outputs, reference_outputs):
    engine_languages = [output.language for output in engine_outputs]
    assert engine_languages == reference_outputs.languages
