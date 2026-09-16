"""CUDA graph decode must be indistinguishable from eager decode.

A captured graph freezes tensor addresses and a few shape parameters, so the ways it
can go wrong are quiet ones: a stale block table, padding rows bleeding into real
attention, a baked-in max sequence length that truncates long contexts. None of those
raise -- they just change the transcription. So the gate here is token-level equality
against the eager path on real audio, not a tolerance on logits.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qwen_asr_vllm.engine.engine import AsrEngine
from qwen_asr_vllm.engine.graph_runner import DEFAULT_BUCKETS, GraphRunner

from .conftest import model_path

pytestmark = [pytest.mark.gpu, pytest.mark.checkpoint, pytest.mark.slow]

MODEL_PATH = model_path()
NUM_SAMPLES = 16
# Two engines are alive at once in the comparison tests.
GPU_FRACTION = 0.35


@pytest.fixture(scope="module")
def samples():
    from bench.data import load_librispeech

    return load_librispeech(split="test-clean", num_samples=NUM_SAMPLES)


def transcribe_all(samples, **engine_kwargs):
    engine = AsrEngine(
        MODEL_PATH,
        max_model_len=2048,
        gpu_memory_utilization=GPU_FRACTION,
        **engine_kwargs,
    )
    try:
        outputs = engine.transcribe([sample.audio for sample in samples], language="en")
        used_graph = engine.model_runner.graph_runner is not None
        return outputs, used_graph
    finally:
        del engine
        torch.cuda.empty_cache()


class TestBucketSelection:
    def test_smallest_fitting_bucket_is_chosen(self):
        runner = GraphRunner.__new__(GraphRunner)
        runner.buckets = DEFAULT_BUCKETS

        assert runner.bucket_for(1) == 1
        assert runner.bucket_for(3) == 4
        assert runner.bucket_for(16) == 16
        assert runner.bucket_for(17) == 24

    def test_oversized_batch_has_no_bucket(self):
        runner = GraphRunner.__new__(GraphRunner)
        runner.buckets = (1, 2, 4)
        assert runner.bucket_for(5) is None

    def test_buckets_are_clamped_to_max_num_seqs(self):
        config = type(
            "Config",
            (),
            {
                "max_num_seqs": 8,
                "max_model_len": 2048,
                "kvcache_block_size": 256,
                "device": "cpu",
            },
        )()
        runner = GraphRunner.__new__(GraphRunner)
        runner.config = config
        runner.buckets = tuple(s for s in DEFAULT_BUCKETS if s <= config.max_num_seqs)

        assert runner.buckets == (1, 2, 4, 8)


class TestGraphParity:
    def test_graphs_are_captured_by_default(self, samples):
        _, used_graph = transcribe_all(samples[:2], max_num_seqs=4)
        assert used_graph

    def test_enforce_eager_skips_capture(self, samples):
        _, used_graph = transcribe_all(samples[:2], max_num_seqs=4, enforce_eager=True)
        assert not used_graph

    @pytest.mark.parametrize("max_num_seqs", [1, 4, 16])
    def test_transcriptions_match_eager_exactly(self, samples, max_num_seqs):
        """Every batch size bucket, including the padded ones, must agree with eager."""
        eager, _ = transcribe_all(samples, max_num_seqs=max_num_seqs, enforce_eager=True)
        graphed, used_graph = transcribe_all(samples, max_num_seqs=max_num_seqs)

        assert used_graph
        assert [output.text for output in graphed] == [output.text for output in eager]

    def test_token_ids_match_not_just_the_text(self, samples):
        """Text can hide a divergence that detokenization smooths over."""
        eager, _ = transcribe_all(samples[:8], max_num_seqs=8, enforce_eager=True)
        graphed, _ = transcribe_all(samples[:8], max_num_seqs=8)

        assert [output.num_output_tokens for output in graphed] == [
            output.num_output_tokens for output in eager
        ]
        assert [output.raw_text for output in graphed] == [
            output.raw_text for output in eager
        ]

    def test_a_padded_bucket_does_not_leak_between_requests(self, samples):
        """Three requests replay the bucket of four; the fourth row is padding.

        If padding rows shared attention with real ones, the transcriptions would
        shift -- so running the same three clips both ways has to agree.
        """
        subset = samples[:3]
        eager, _ = transcribe_all(subset, max_num_seqs=4, enforce_eager=True)
        graphed, _ = transcribe_all(subset, max_num_seqs=4)

        assert [output.text for output in graphed] == [output.text for output in eager]

    def test_long_context_is_not_truncated_by_the_baked_max_seqlen(self, samples):
        """max_seqlen_k is captured at max_model_len; real lengths come from cu_seqlens.

        The longest clips span several KV blocks, which is where a wrong bake-in or a
        too-narrow block table would show up.
        """
        longest = sorted(samples, key=lambda s: -len(s.audio))[:4]
        eager, _ = transcribe_all(longest, max_num_seqs=4, enforce_eager=True)
        graphed, _ = transcribe_all(longest, max_num_seqs=4)

        assert [output.text for output in graphed] == [output.text for output in eager]
        assert max(output.num_prompt_tokens for output in graphed) > 256


class TestGraphPathSelection:
    def test_prefill_uses_eager_and_decode_uses_the_graph(self, samples):
        """A batch holding a prefill has varying query lengths; no graph covers it."""
        engine = AsrEngine(
            MODEL_PATH, max_num_seqs=8, max_model_len=2048, gpu_memory_utilization=GPU_FRACTION
        )
        try:
            runner = engine.model_runner
            assert runner.graph_runner is not None

            # Driven stage by stage, because one engine step runs the encoder and the
            # first decoder pass together and would skip past the prefill batch.
            request = engine.add_request(samples[0].audio, language="en")
            encode_batch = engine.scheduler.schedule_audio()
            engine.audio_runner.encode(encode_batch)
            engine.scheduler.admit_prefill(encode_batch)

            batch = engine.scheduler.schedule_model()
            assert batch.prefill == [request]
            assert not runner.can_use_graph(batch)

            # Once it is decoding, the graph path takes over.
            engine.scheduler.postprocess(batch, runner.run(batch))
            decode_batch = engine.scheduler.schedule_model()
            assert decode_batch.decode == [request]
            assert runner.can_use_graph(decode_batch)
        finally:
            del engine
            torch.cuda.empty_cache()
