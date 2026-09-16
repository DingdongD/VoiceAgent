"""CPU-only unit tests for the pieces the engine's correctness rests on."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_asr_vllm.audio.batcher import (
    attention_window_tokens,
    build_window_boundaries,
    pack_audio_batch,
)
from qwen_asr_vllm.audio.frontend import AudioFeatures
from qwen_asr_vllm.audio.tokens import (
    CHUNK_FRAMES,
    TOKENS_PER_FULL_CHUNK,
    chunk_lengths_for,
    num_audio_tokens,
)
from qwen_asr_vllm.engine.block_manager import BlockManager
from qwen_asr_vllm.engine.request import AsrRequest, SamplingParams
from qwen_asr_vllm.postprocess import fix_repetitions, parse_asr_output
from qwen_asr_vllm.prompt import PromptLayout, normalize_language

BLOCK_SIZE = 256


def _official_output_length(input_lengths: int) -> int:
    """Verbatim copy of the upstream helper, used as the oracle."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


class TestAudioTokenAccounting:
    def test_matches_official_formula(self):
        mismatches = [
            frames for frames in range(1, 6000) if num_audio_tokens(frames) != _official_output_length(frames)
        ]
        assert mismatches == []

    def test_one_second_of_audio_is_thirteen_tokens(self):
        assert TOKENS_PER_FULL_CHUNK == 13
        assert num_audio_tokens(CHUNK_FRAMES) == 13
        assert num_audio_tokens(10 * CHUNK_FRAMES) == 130

    @pytest.mark.parametrize(
        "frames,expected",
        [
            (100, [100]),
            (200, [100, 100]),
            (350, [100, 100, 100, 50]),
            (105, [100, 5]),
            (7, [7]),
        ],
    )
    def test_chunk_split(self, frames, expected):
        assert chunk_lengths_for(frames) == expected

    def test_chunk_lengths_sum_to_input(self):
        for frames in range(1, 2000):
            assert sum(chunk_lengths_for(frames)) == frames

    def test_rejects_empty_audio(self):
        with pytest.raises(ValueError):
            chunk_lengths_for(0)


class TestAudioBatching:
    def test_window_size_from_config(self):
        assert attention_window_tokens(n_window_infer=800) == 104

    def test_boundaries_isolate_requests(self):
        boundaries = build_window_boundaries([46, 26, 95], window_tokens=104)
        assert boundaries == [0, 46, 72, 167]

    def test_long_request_is_split_into_windows(self):
        # 250 tokens at a 104-token window: two full windows plus a 42 remainder.
        assert build_window_boundaries([250], window_tokens=104) == [0, 104, 208, 250]

    def test_exact_multiple_of_window_leaves_no_remainder(self):
        assert build_window_boundaries([208], window_tokens=104) == [0, 104, 208]

    def test_pack_concatenates_and_accounts(self):
        mels = [torch.zeros(128, frames) for frames in (350, 200, 105)]
        batch = pack_audio_batch(mels, n_window_infer=800)

        assert batch.mel.shape == (128, 655)
        assert batch.audio_token_lens == [46, 26, 14]
        assert batch.total_tokens == 86
        assert batch.num_chunks == 4 + 2 + 2
        assert batch.chunk_lengths.tolist() == [100, 100, 100, 50, 100, 100, 100, 5]
        assert batch.cu_seqlens.dtype == torch.int32

    def test_split_outputs_restores_per_request_shapes(self):
        mels = [torch.zeros(128, frames) for frames in (350, 200)]
        batch = pack_audio_batch(mels, n_window_infer=800)
        parts = batch.split_outputs(torch.zeros(batch.total_tokens, 1024))
        assert [part.size(0) for part in parts] == [46, 26]

    def test_split_outputs_rejects_wrong_length(self):
        batch = pack_audio_batch([torch.zeros(128, 350)], n_window_infer=800)
        with pytest.raises(ValueError):
            batch.split_outputs(torch.zeros(45, 1024))


class TestPostprocess:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("language English<asr_text>hello there", ("English", "hello there")),
            ("language chinese<asr_text>ni hao", ("Chinese", "ni hao")),
            ("plain text, no tag", ("", "plain text, no tag")),
            ("language None<asr_text>", ("", "")),
            ("language None<asr_text>something", ("", "something")),
            ("language English\n\n<asr_text>  spaced  ", ("English", "spaced")),
            ("", ("", "")),
            (None, ("", "")),
        ],
    )
    def test_parse_cases(self, raw, expected):
        assert parse_asr_output(raw) == expected

    def test_forced_language_treats_output_as_plain_text(self):
        assert parse_asr_output("just the words", user_language="English") == (
            "English",
            "just the words",
        )

    def test_collapses_character_runs(self):
        assert fix_repetitions("ab" + "c" * 50 + "d") == "abcd"

    def test_collapses_pattern_runs(self):
        assert fix_repetitions("start" + "ha" * 40) == "startha"

    def test_leaves_short_runs_alone(self):
        text = "aaa bbb ccc"
        assert fix_repetitions(text) == text


class TestLanguageNormalization:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("en", "English"),
            ("EN", "English"),
            ("english", "English"),
            ("cHINese", "Chinese"),
            ("zh", "Chinese"),
            ("Japanese", "Japanese"),
            ("", None),
            (None, None),
        ],
    )
    def test_normalize(self, value, expected):
        assert normalize_language(value) == expected


def _make_request(prompt_len: int, audio_offset: int, audio_length: int) -> AsrRequest:
    """A request with synthetic ids: textual prefix, audio span, textual suffix."""
    audio_token_id = 151676
    token_ids = (
        list(range(1000, 1000 + audio_offset))
        + [audio_token_id] * audio_length
        + list(range(2000, 2000 + prompt_len - audio_offset - audio_length))
    )
    assert len(token_ids) == prompt_len
    return AsrRequest(
        features=AudioFeatures(
            mel=torch.zeros(128, audio_length * 100 // 13),
            mel_frames=audio_length * 100 // 13,
            num_audio_tokens=audio_length,
            audio_seconds=audio_length / 13,
        ),
        layout=PromptLayout(
            token_ids=token_ids, audio_offset=audio_offset, audio_length=audio_length
        ),
        sampling=SamplingParams(),
        block_size=BLOCK_SIZE,
        stop_token_ids={151643, 151645},
    )


class TestBlockCacheability:
    def test_short_prompt_has_no_cacheable_blocks(self):
        request = _make_request(prompt_len=62, audio_offset=9, audio_length=46)
        assert request.num_cacheable_blocks == 0

    def test_only_blocks_fully_before_audio_are_cacheable(self):
        # A 600-token system prompt puts the audio span at offset 609.
        request = _make_request(prompt_len=700, audio_offset=609, audio_length=46)
        assert request.num_cacheable_blocks == 609 // BLOCK_SIZE == 2

    def test_block_containing_audio_is_not_cacheable(self):
        request = _make_request(prompt_len=400, audio_offset=300, audio_length=46)
        # Block 1 spans tokens 256..511 and therefore holds audio.
        assert request.num_cacheable_blocks == 1


class TestBlockManager:
    def test_allocate_and_deallocate_are_balanced(self):
        manager = BlockManager(num_blocks=16, block_size=BLOCK_SIZE)
        request = _make_request(prompt_len=700, audio_offset=609, audio_length=46)

        assert manager.can_allocate(request)
        manager.allocate(request)
        assert len(request.block_table) == 3
        assert manager.num_free_blocks == 13

        manager.deallocate(request)
        assert manager.num_free_blocks == 16
        assert request.block_table == []

    def test_textual_prefix_is_reused_across_requests(self):
        manager = BlockManager(num_blocks=32, block_size=BLOCK_SIZE)
        first = _make_request(prompt_len=700, audio_offset=609, audio_length=46)
        manager.allocate(first)
        assert first.num_cached_tokens == 0

        # Same system prompt, different audio content of the same length.
        second = _make_request(prompt_len=700, audio_offset=609, audio_length=46)
        manager.allocate(second)

        assert second.num_cached_tokens == 2 * BLOCK_SIZE
        assert second.num_computed_tokens == 2 * BLOCK_SIZE
        assert second.block_table[:2] == first.block_table[:2]
        # The block holding audio must not be shared.
        assert second.block_table[2] != first.block_table[2]

    def test_audio_blocks_are_never_reused_even_when_ids_match(self):
        manager = BlockManager(num_blocks=32, block_size=BLOCK_SIZE)
        # audio_offset 0 means every block holds audio placeholders.
        first = _make_request(prompt_len=600, audio_offset=0, audio_length=590)
        second = _make_request(prompt_len=600, audio_offset=0, audio_length=590)
        manager.allocate(first)
        manager.allocate(second)

        assert second.num_cached_tokens == 0
        assert set(first.block_table).isdisjoint(second.block_table)

    def test_prefix_cache_can_be_disabled(self):
        manager = BlockManager(num_blocks=32, block_size=BLOCK_SIZE, enable_prefix_cache=False)
        first = _make_request(prompt_len=700, audio_offset=609, audio_length=46)
        second = _make_request(prompt_len=700, audio_offset=609, audio_length=46)
        manager.allocate(first)
        manager.allocate(second)
        assert second.num_cached_tokens == 0

    def test_append_allocates_only_on_block_boundary(self):
        manager = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
        request = _make_request(prompt_len=BLOCK_SIZE, audio_offset=9, audio_length=46)
        manager.allocate(request)
        assert len(request.block_table) == 1

        request.append_token(42)
        assert manager.can_append(request)
        manager.may_append(request)
        assert len(request.block_table) == 2

        request.append_token(43)
        manager.may_append(request)
        assert len(request.block_table) == 2


class TestRequestLifecycle:
    def test_stops_on_stop_token(self):
        request = _make_request(prompt_len=62, audio_offset=9, audio_length=46)
        request.append_token(151645)
        assert request.check_finished()
        assert request.finish_reason == "stop"

    def test_stops_at_max_new_tokens(self):
        request = _make_request(prompt_len=62, audio_offset=9, audio_length=46)
        request.sampling.max_new_tokens = 3
        for token_id in (10, 11, 12):
            request.append_token(token_id)
            finished = request.check_finished()
        assert finished
        assert request.finish_reason == "length"

    def test_query_length_is_one_while_decoding(self):
        request = _make_request(prompt_len=62, audio_offset=9, audio_length=46)
        assert request.num_tokens_to_compute == 62
        request.num_computed_tokens = 62
        request.append_token(7)
        assert request.num_tokens_to_compute == 1


class TestSamplingForAudio:
    """A token cap that does not silently truncate long audio."""

    def test_the_flat_default_is_unchanged(self):
        # Parity tests compare against upstream, which uses 440.
        assert SamplingParams().max_new_tokens == 440

    def test_short_audio_keeps_the_floor(self):
        assert SamplingParams.for_audio(5.0).max_new_tokens == 440

    def test_long_audio_scales_past_the_floor(self):
        # A 1299s talk really emitted 4849 tokens, so the cap has to clear that.
        assert SamplingParams.for_audio(1299.0).max_new_tokens > 4849

    def test_the_margin_holds_at_the_measured_token_rate(self):
        # Real speech emits 3.5-3.7 tokens per audio second, read or spontaneous.
        for seconds in (60, 300, 600, 1200, 1800):
            assert SamplingParams.for_audio(seconds).max_new_tokens > seconds * 3.7 * 2

    def test_an_explicit_cap_wins(self):
        assert SamplingParams.for_audio(1200.0, max_new_tokens=99).max_new_tokens == 99

    def test_other_fields_pass_through(self):
        params = SamplingParams.for_audio(600.0, temperature=0.5)
        assert params.temperature == 0.5
