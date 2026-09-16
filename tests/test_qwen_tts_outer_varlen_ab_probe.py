from pathlib import Path

import torch

from bench.qwen_tts_outer_varlen_ab_probe import (
    _codec_parity_details,
    _raw_codec_hash,
    build_parser,
)


def test_full_codec_probe_parser_defaults_are_repeatable():
    args = build_parser().parse_args(["--model-path", "/models/qwen-tts"])

    assert args.warmup == 1
    assert args.repeats == 2
    assert args.stagger_steps == 1
    assert args.out == Path("results/qwen_tts_outer_varlen_full_codec_ab_cuda2.json")


def test_full_codec_probe_parser_accepts_two_texts_and_timing_controls():
    args = build_parser().parse_args(
        [
            "--model-path",
            "/models/qwen-tts",
            "--text",
            "short",
            "--text",
            "a longer request",
            "--warmup",
            "0",
            "--repeats",
            "5",
            "--stagger-steps",
            "3",
            "--paged-attention-kernel",
        ]
    )

    assert args.texts == ["short", "a longer request"]
    assert args.warmup == 0
    assert args.repeats == 5
    assert args.stagger_steps == 3
    assert args.paged_attention_kernel is True


def test_raw_codec_hash_is_stable_across_device_representation():
    codec = torch.arange(8, dtype=torch.int64).reshape(1, 2, 4)

    assert _raw_codec_hash(codec) == _raw_codec_hash(codec.clone())


def test_codec_parity_details_reports_first_divergent_frame():
    active = torch.zeros((1, 4, 3), dtype=torch.long)
    candidate = active.clone()
    candidate[:, 2, 1] = 7

    assert _codec_parity_details(active, candidate) == {
        "shape_equal": True,
        "frame_exact": [True, True, False, True],
        "first_divergent_frame": 2,
    }
