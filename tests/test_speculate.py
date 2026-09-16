"""Speculative draft acceptance: the pure rule and the GPU verify path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_asr_vllm.engine.speculate import accepted_draft_length, longest_common_prefix


class TestAcceptedDraftLength:
    def test_empty_draft(self):
        assert accepted_draft_length([], [1]) == 0

    def test_full_accept_uses_only_draft_width(self):
        # Predictions carry a bonus next-token that must not inflate the count.
        assert accepted_draft_length([1, 2, 3], [1, 2, 3, 99]) == 3

    def test_rejects_at_first_mismatch(self):
        assert accepted_draft_length([1, 2, 3, 4], [1, 2, 9, 4, 5]) == 2

    def test_rejects_everything(self):
        assert accepted_draft_length([1, 2], [9, 2, 3]) == 0

    def test_truncated_predictions(self):
        assert accepted_draft_length([1, 2, 3], [1, 2]) == 2


class TestLongestCommonPrefix:
    def test_matches_accepted_draft_length_on_shared_prefix(self):
        left = [1, 2, 3, 4]
        right = [1, 2, 9, 4]
        assert longest_common_prefix(left, right) == accepted_draft_length(
            left, right + [0]
        )


@pytest.mark.gpu
@pytest.mark.checkpoint
@pytest.mark.slow
class TestVerifyDraft:
    """On the same audio, the model's own greedy output must be fully accepted."""

    def test_self_draft_is_fully_accepted(self, model_dir):
        from bench.data import load_librispeech
        from qwen_asr_vllm.engine.engine import AsrEngine
        from qwen_asr_vllm.engine.request import SamplingParams

        sample = load_librispeech(split="test-clean", num_samples=1)[0]
        engine = AsrEngine(
            model=model_dir,
            max_model_len=2048,
            max_num_batched_tokens=2048,
            max_num_seqs=1,
            gpu_memory_utilization=0.35,
        )
        try:
            output = engine.transcribe(
                [sample.audio],
                language="en",
                sampling=SamplingParams.for_audio(sample.duration),
            )[0]
            assert output.output_token_ids, "expected a non-empty transcript"
            accepted, predictions = engine.verify_draft(
                sample.audio, output.output_token_ids, language="en"
            )
            assert accepted == len(output.output_token_ids), (
                f"self-draft accepted {accepted}/{len(output.output_token_ids)}; "
                f"predictions[:8]={predictions[:8]}"
            )
            # Bonus token after a full accept should be a stop id, otherwise the
            # original transcription would have continued.
            assert predictions[-1] in engine.prompt_builder.stop_token_ids
        finally:
            del engine

    def test_growing_audio_accept_matches_lcp(self, model_dir):
        from bench.data import load_librispeech
        from qwen_asr_vllm.engine.engine import AsrEngine
        from qwen_asr_vllm.engine.request import SamplingParams
        from qwen_asr_vllm.engine.speculate import longest_common_prefix

        sample = load_librispeech(split="test-clean", num_samples=4, seed=0)
        # Pick a clip long enough that a half-prefix still has content.
        sample = max(sample, key=lambda s: s.duration)
        half = sample.audio[: len(sample.audio) // 2]
        engine = AsrEngine(
            model=model_dir,
            max_model_len=2048,
            max_num_batched_tokens=2048,
            max_num_seqs=1,
            gpu_memory_utilization=0.35,
        )
        try:
            first = engine.transcribe(
                [half],
                language="en",
                sampling=SamplingParams.for_audio(len(half) / 16000),
            )[0]
            second = engine.transcribe(
                [sample.audio],
                language="en",
                sampling=SamplingParams.for_audio(sample.duration),
            )[0]
            accepted, _ = engine.verify_draft(
                sample.audio, first.output_token_ids, language="en"
            )
            lcp = longest_common_prefix(first.output_token_ids, second.output_token_ids)
            assert accepted == lcp, f"verify={accepted} lcp={lcp}"
        finally:
            del engine


@pytest.mark.gpu
@pytest.mark.checkpoint
@pytest.mark.slow
class TestTranscribeWithDraft:
    def test_self_draft_matches_greedy_transcribe(self, model_dir):
        from bench.data import load_librispeech
        from bench.metrics import normalize_text
        from qwen_asr_vllm.engine.engine import AsrEngine
        from qwen_asr_vllm.engine.request import SamplingParams

        sample = load_librispeech(split="test-clean", num_samples=4, seed=2)
        sample = max(sample, key=lambda s: s.duration)
        engine = AsrEngine(
            model=model_dir,
            max_model_len=2048,
            max_num_batched_tokens=2048,
            max_num_seqs=1,
            gpu_memory_utilization=0.35,
        )
        try:
            greedy = engine.transcribe(
                [sample.audio],
                language="en",
                sampling=SamplingParams.for_audio(sample.duration),
            )[0]
            drafted = engine.transcribe_with_draft(
                sample.audio,
                draft_token_ids=greedy.output_token_ids,
                language="en",
                sampling=SamplingParams.for_audio(sample.duration),
            )
            assert drafted.draft_accepted == len(greedy.output_token_ids)
            assert normalize_text(drafted.text) == normalize_text(greedy.text)
            assert drafted.output_token_ids == greedy.output_token_ids
        finally:
            del engine

    def test_partial_draft_matches_greedy_on_longer_audio(self, model_dir):
        from bench.data import load_librispeech
        from bench.metrics import normalize_text
        from qwen_asr_vllm.engine.engine import AsrEngine
        from qwen_asr_vllm.engine.request import SamplingParams

        sample = load_librispeech(split="test-clean", num_samples=4, seed=3)
        sample = max(sample, key=lambda s: s.duration)
        half = sample.audio[: len(sample.audio) // 2]
        engine = AsrEngine(
            model=model_dir,
            max_model_len=2048,
            max_num_batched_tokens=2048,
            max_num_seqs=1,
            gpu_memory_utilization=0.35,
        )
        try:
            first = engine.transcribe(
                [half],
                language="en",
                sampling=SamplingParams.for_audio(len(half) / 16000),
            )[0]
            greedy = engine.transcribe(
                [sample.audio],
                language="en",
                sampling=SamplingParams.for_audio(sample.duration),
            )[0]
            drafted = engine.transcribe_with_draft(
                sample.audio,
                draft_token_ids=first.output_token_ids,
                language="en",
                sampling=SamplingParams.for_audio(sample.duration),
            )
            assert normalize_text(drafted.text) == normalize_text(greedy.text)
            assert drafted.output_token_ids == greedy.output_token_ids
            assert 0 <= drafted.draft_accepted <= drafted.draft_tokens
        finally:
            del engine
