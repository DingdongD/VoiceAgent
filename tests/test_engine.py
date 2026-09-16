"""Engine integration tests on real audio."""

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
NUM_SAMPLES = 24
# Several engines coexist across these tests, so none may claim the whole card.
GPU_FRACTION = 0.25


@pytest.fixture(scope="module")
def samples():
    from bench.data import load_librispeech

    return load_librispeech("test-clean", num_samples=NUM_SAMPLES, seed=0)


@pytest.fixture(scope="module")
def engine():
    from qwen_asr_vllm import AsrEngine

    instance = AsrEngine(
        MODEL_PATH, max_num_seqs=8, max_model_len=2048, gpu_memory_utilization=GPU_FRACTION
    )
    yield instance
    del instance
    gc.collect()
    torch.cuda.empty_cache()


def test_transcribes_in_input_order(engine, samples):
    """Each output must align with its own reference, not a neighbour's."""
    from bench.metrics import compute_wer

    outputs = engine.transcribe([sample.audio for sample in samples])
    assert len(outputs) == len(samples)
    for sample, output in zip(samples, outputs):
        assert output.text, "empty transcription"
        aligned = compute_wer([sample.text], [output.text])
        shifted = compute_wer([sample.text], [outputs[0].text])
        assert aligned < 0.35, f"{output.text!r} does not match {sample.text!r}"
        if output is not outputs[0]:
            assert aligned < shifted


def test_audio_duration_and_token_accounting(engine, samples):
    outputs = engine.transcribe([sample.audio for sample in samples[:4]])
    for sample, output in zip(samples, outputs):
        assert output.audio_seconds == pytest.approx(sample.duration, abs=1e-6)
        # 13 audio tokens per second, floor-aligned to whole mel chunks.
        assert output.num_audio_tokens == pytest.approx(sample.duration * 13, abs=13)
        assert output.num_prompt_tokens > output.num_audio_tokens


def test_mixed_prefill_decode_batches_occur(engine, samples):
    """More requests than slots forces arrivals to join in-flight decodes."""
    before = engine.scheduler.stats.num_mixed_steps
    engine.transcribe([sample.audio for sample in samples])
    assert engine.scheduler.stats.num_mixed_steps > before


def test_cross_request_audio_batching_happens(engine, samples):
    stats = engine.scheduler.stats
    encoded_before, batches_before = stats.num_encoded_requests, stats.num_encode_batches
    engine.transcribe([sample.audio for sample in samples])
    encoded = stats.num_encoded_requests - encoded_before
    batches = stats.num_encode_batches - batches_before
    assert encoded == len(samples)
    assert batches < encoded, f"{encoded} clips took {batches} encoder calls; no batching happened"


def test_forced_language_suppresses_metadata(engine, samples):
    outputs = engine.transcribe([sample.audio for sample in samples[:4]], language="en")
    for output in outputs:
        assert output.language == "English"
        assert "language" not in output.raw_text.lower().split("<")[0][:20]
        assert output.text


def test_shared_context_reuses_cached_prefix_blocks(engine, samples):
    """A shared system prompt spanning whole blocks should hit the block cache."""
    # Roughly 700 tokens of hotword context, i.e. two full 256-token blocks.
    long_context = ", ".join(f"term{index}" for index in range(180))
    shortest = sorted(samples, key=lambda sample: sample.duration)[:4]

    hits_before = engine.block_manager.num_cache_hit_blocks
    outputs = engine.transcribe([sample.audio for sample in shortest], context=long_context)

    assert all(output.text for output in outputs)
    assert engine.block_manager.num_cache_hit_blocks > hits_before


def test_short_context_yields_no_cache_hits(engine, samples):
    """With the bare template no block precedes the audio, so nothing is cacheable."""
    hits_before = engine.block_manager.num_cache_hit_blocks
    engine.transcribe([sample.audio for sample in samples[:4]])
    assert engine.block_manager.num_cache_hit_blocks == hits_before


def test_rejects_audio_longer_than_context_window(engine):
    import numpy as np

    from qwen_asr_vllm import EngineConfig

    # max_model_len 2048 tokens at 13 tokens/second is about 157 seconds.
    too_long = np.zeros(16000 * 200, dtype=np.float32)
    with pytest.raises(ValueError, match="max_model_len"):
        engine.add_request(too_long)
    assert isinstance(engine.config, EngineConfig)


def test_concurrency_does_not_change_quality(samples):
    from bench.metrics import compute_wer
    from qwen_asr_vllm import AsrEngine

    references = [sample.text for sample in samples]
    audios = [sample.audio for sample in samples]

    wers = []
    for max_num_seqs in (1, 8):
        engine = AsrEngine(
            MODEL_PATH,
            max_num_seqs=max_num_seqs,
            max_model_len=2048,
            gpu_memory_utilization=GPU_FRACTION,
        )
        try:
            outputs = engine.transcribe(audios)
        finally:
            del engine
            gc.collect()
            torch.cuda.empty_cache()
        wers.append(compute_wer(references, [output.text for output in outputs]))

    assert abs(wers[0] - wers[1]) < 0.01, f"WER moved with concurrency: {wers}"
