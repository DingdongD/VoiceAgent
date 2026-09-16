import json
import inspect
import logging
import math
import warnings
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import bench.qwen_tts_active_prefix_parity_probe as parity_probe
from bench.qwen_tts_active_prefix_parity_probe import (
    AudioCapture,
    BoundaryCapture,
    HookRecorder,
    LayerKvCapture,
    ModeCapture,
    ProbeDependencies,
    TensorLayout,
    TensorSnapshot,
    _capture_talker_hooks,
    _execute_probe,
    _prepare_talker_inputs_once,
    _run_mode,
    build_audio_capture,
    build_codec_capture,
    build_parser,
    capture_tensor,
    clone_prepared_inputs,
    compare_tensors,
    first_codec_divergence,
    main,
    summarize_parity,
)


def _snapshot(name: str, value: torch.Tensor) -> TensorSnapshot:
    return capture_tensor(name, value)


def _mode_capture(
    mode: str,
    *,
    batch_size: int = 1,
    tensor_delta: float = 0.0,
    codec_delta: int = 0,
    sample_counts: tuple[int, ...] | None = None,
    warnings: tuple[str, ...] = (),
    errors: tuple[str, ...] = (),
) -> ModeCapture:
    boundaries = []
    raw_logits = []
    processed_logits = []
    for call_index in range(5):
        value = torch.tensor([float(call_index), 2.0], dtype=torch.float32)
        if call_index == 2:
            value = value + tensor_delta
        boundaries.append(
            BoundaryCapture(
                call_index=call_index,
                phase="prefill" if call_index == 0 else "decode",
                layers=(
                    LayerKvCapture(
                        layer_index=0,
                        key=_snapshot(f"kv.{call_index}.0.key", value.reshape(1, 1, 1, 2)),
                        value=_snapshot(
                            f"kv.{call_index}.0.value",
                            (value + 1).reshape(1, 1, 1, 2),
                        ),
                    ),
                ),
                last_hidden=_snapshot(f"hidden.{call_index}", value),
            )
        )
        raw_logits.append(_snapshot(f"raw_logits.{call_index}", value + 2))
        processed_logits.append(
            _snapshot(f"processed_logits.{call_index}", value + 3)
        )

    raw_codec = torch.tensor(
        [[[1, 10], [2, 20], [99, 30]]], dtype=torch.int64
    ).repeat(batch_size, 1, 1)
    if codec_delta:
        raw_codec[0, 1, 1] += codec_delta
    codec = build_codec_capture(
        raw_codec,
        predictor_sequences=raw_codec[:, :, 1:],
        decoded_codec_items=tuple(raw_codec[index, :2] for index in range(batch_size)),
        eos_token_id=99,
    )
    if sample_counts is None:
        sample_counts = tuple(8 for _ in range(batch_size))
    audio = AudioCapture(
        sample_rate=24000,
        sample_counts=sample_counts,
        finite=tuple(True for _ in sample_counts),
        nonempty=tuple(count > 0 for count in sample_counts),
    )
    return ModeCapture(
        mode=mode,
        batch_size=batch_size,
        boundaries=tuple(boundaries),
        raw_codec_logits=tuple(raw_logits),
        processed_outer_logits=tuple(processed_logits),
        sampled_first_codebook=_snapshot(
            "sampled_first_codebook", raw_codec[:, :, 0]
        ),
        codec=codec,
        audio=audio,
        attention_evidence={"attn_implementation": "sdpa"},
        runtime_metrics={"allocated_kv_bytes": 128},
        timings_ms={"generation": 12.5, "decode": 2.5},
        resolved_config={
            "speaker": "alice",
            "resolved_eos_token_id": 99,
            "repetition_penalty": 1.05,
            "suppress_tokens": [7, 8],
        },
        warnings=warnings,
        errors=errors,
    )


def test_compare_tensors_requires_bitwise_equality_and_reports_delta():
    reference = torch.tensor([1.0, 2.0], dtype=torch.float16)
    candidate = reference.clone()

    equal = compare_tensors("hidden", reference, candidate)
    candidate[1] = torch.nextafter(
        candidate[1], torch.tensor(float("inf"), dtype=torch.float16)
    )
    different = compare_tensors("hidden", reference, candidate)

    assert equal.bitwise_equal is True
    assert equal.max_abs == 0.0
    assert different.bitwise_equal is False
    assert different.max_abs == pytest.approx(0.001953125)
    assert different.first_difference == (1,)


def test_compare_tensors_distinguishes_signed_zero_bits():
    comparison = compare_tensors(
        "signed_zero",
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([-0.0], dtype=torch.float32),
    )

    assert comparison.bitwise_equal is False
    assert comparison.max_abs == 0.0
    assert comparison.first_difference == (0,)


@pytest.mark.parametrize(
    ("reference", "candidate", "shape_equal", "dtype_equal", "max_abs"),
    [
        (
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([[1, 2]], dtype=torch.int64),
            False,
            True,
            None,
        ),
        (
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([1, 2], dtype=torch.int32),
            True,
            False,
            0.0,
        ),
        (
            torch.tensor([float("inf")]),
            torch.tensor([float("-inf")]),
            True,
            True,
            None,
        ),
    ],
)
def test_compare_tensors_is_shape_dtype_strict_and_reports_max_abs_safely(
    reference,
    candidate,
    shape_equal,
    dtype_equal,
    max_abs,
):
    comparison = compare_tensors("codec", reference, candidate)

    assert comparison.shape_equal is shape_equal
    assert comparison.dtype_equal is dtype_equal
    assert comparison.bitwise_equal is False
    assert comparison.max_abs == max_abs


def test_capture_tensor_records_noncompact_layout_before_cpu_copy():
    backing = torch.arange(32, dtype=torch.float32).reshape(1, 2, 8, 2)
    active_prefix = backing[:, :, :3, :]

    snapshot = capture_tensor("layer.0.key", active_prefix)

    assert snapshot.layout == TensorLayout(
        shape=(1, 2, 3, 2),
        stride=(32, 16, 2, 1),
        contiguous=False,
        dtype="torch.float32",
        device="cpu",
    )
    assert torch.equal(snapshot.tensor, active_prefix)
    assert snapshot.tensor.data_ptr() != active_prefix.data_ptr()


def test_first_codec_divergence_reports_batch_frame_and_codebook():
    reference = torch.tensor(
        [[[1, 10], [2, 20]], [[3, 30], [4, 40]]], dtype=torch.int64
    )
    candidate = reference.clone()
    candidate[1, 0, 1] = 31

    assert first_codec_divergence(reference, candidate) == {
        "batch_index": 1,
        "frame_index": 0,
        "codebook_index": 1,
        "reference_id": 30,
        "candidate_id": 31,
    }
    assert first_codec_divergence(reference, reference.clone()) is None


def test_codec_capture_records_raw_ids_hashes_shapes_and_eos_positions():
    raw_codec = torch.tensor(
        [
            [[1, 10], [2, 20], [99, 30]],
            [[4, 40], [99, 50], [99, 60]],
        ],
        dtype=torch.int64,
    )

    capture = build_codec_capture(
        raw_codec,
        predictor_sequences=raw_codec[:, :, 1:],
        decoded_codec_items=(raw_codec[0, :2], raw_codec[1, :1]),
        eos_token_id=99,
    )

    assert torch.equal(capture.first_codebook_ids.tensor, raw_codec[:, :, 0])
    assert torch.equal(capture.predictor_sequences.tensor, raw_codec[:, :, 1:])
    assert capture.eos_positions == (2, 1)
    assert capture.item_shapes == ((2, 2), (1, 2))
    assert capture.item_dtypes == ("torch.int64", "torch.int64")
    assert capture.item_sha256 == (
        "9360c3018bdaf75ce7a4ae7d6034d04b03af22ed3349bfb2b99c630873d167b0",
        "e663ea27786217ac6fb83997871f4392c74bbad380ba8cb96afbc960d3d02af0",
    )
    assert capture.raw_sha256 == (
        "f5d2ebbfd5766df01faddb906017fd2cdce1b1692eb2723ba808a6b37b2aac1a"
    )
    assert capture.combined_sha256 == (
        "0150a918ff3a3b7b22fe8ff9682fbf629e236f33afaa4d99d2203103c55ce8ac"
    )


def test_audio_capture_records_finite_nonempty_and_sample_counts():
    capture = build_audio_capture(
        [np.array([0.0, 0.25], dtype=np.float32), np.array([np.nan])],
        sample_rate=24000,
    )

    assert capture == AudioCapture(
        sample_rate=24000,
        sample_counts=(2, 1),
        finite=(True, False),
        nonempty=(True, True),
    )


def test_summarize_parity_requires_every_boundary_codec_and_audio_gate():
    exact = summarize_parity(
        _mode_capture("upstream"),
        _mode_capture("active_prefix"),
    )
    mismatch = summarize_parity(
        _mode_capture("upstream"),
        _mode_capture(
            "active_prefix",
            tensor_delta=0.25,
            codec_delta=1,
            sample_counts=(7,),
            warnings=("attention warning",),
        ),
    )

    assert exact.parity_passed is True
    assert exact.failed_reasons == ()
    assert exact.first_codec_divergence is None
    assert len(exact.boundary_comparisons) == 5
    assert mismatch.parity_passed is False
    assert set(mismatch.failed_reasons) >= {
        "boundary_bitwise_mismatch",
        "raw_codec_mismatch",
        "predictor_sequence_mismatch",
        "codec_hash_mismatch",
        "audio_sample_count_mismatch",
        "warnings_present",
    }
    assert mismatch.first_codec_divergence == {
        "batch_index": 0,
        "frame_index": 1,
        "codebook_index": 1,
        "reference_id": 20,
        "candidate_id": 21,
    }


def test_summarize_parity_fails_missing_boundaries_nonfinite_audio_and_errors():
    candidate = _mode_capture("active_prefix", errors=("runtime error",))
    candidate = ModeCapture(
        **{
            **candidate.__dict__,
            "boundaries": candidate.boundaries[:4],
            "raw_codec_logits": candidate.raw_codec_logits[:4],
            "processed_outer_logits": candidate.processed_outer_logits[:4],
            "audio": AudioCapture(
                sample_rate=24000,
                sample_counts=(8,),
                finite=(False,),
                nonempty=(False,),
            ),
        }
    )

    summary = summarize_parity(_mode_capture("upstream"), candidate)

    assert summary.parity_passed is False
    assert set(summary.failed_reasons) >= {
        "required_boundary_count_mismatch",
        "raw_codec_logits_count_mismatch",
        "processed_outer_logits_count_mismatch",
        "audio_not_finite",
        "audio_empty",
        "errors_present",
    }


def test_clone_prepared_inputs_reuses_values_without_aliasing_storage():
    prepared = {
        "inputs_embeds": torch.arange(6).reshape(1, 3, 2),
        "nested": [torch.tensor([7]), None],
        "max_new_tokens": 64,
    }

    upstream = clone_prepared_inputs(prepared)
    active = clone_prepared_inputs(prepared)

    assert torch.equal(upstream["inputs_embeds"], active["inputs_embeds"])
    assert upstream["inputs_embeds"].data_ptr() != active["inputs_embeds"].data_ptr()
    assert upstream["nested"][0].data_ptr() != active["nested"][0].data_ptr()
    assert upstream["max_new_tokens"] == active["max_new_tokens"] == 64


class _FakeCacheLayer:
    def __init__(self):
        self.keys = torch.arange(24, dtype=torch.float32).reshape(1, 2, 6, 2)
        self.values = self.keys + 100


class _FakeCache:
    def __init__(self):
        self.layers = [_FakeCacheLayer()]

    def get_seq_length(self, layer_index=0):
        assert layer_index == 0
        return 3


def test_capture_hooks_record_boundaries_and_restore_every_mutation_on_error():
    cache = _FakeCache()

    def model_forward(**kwargs):
        return SimpleNamespace(
            last_hidden_state=torch.tensor([[[1.0, 2.0]]]),
            past_key_values=kwargs["past_key_values"],
        )

    def codec_head_forward(hidden):
        return hidden + 10

    def predictor_generate(**kwargs):
        return SimpleNamespace(sequences=torch.tensor([[20, 30]]))

    def original_get_logits_processor(*args, **kwargs):
        return [lambda input_ids, scores: scores + 1]

    def original_sampler(scores, **kwargs):
        return scores.argmax(dim=-1, keepdim=True)

    sampler_module = SimpleNamespace(_sample_next_token=original_sampler)

    def original_decode_step(*, scores):
        first_ids = sampler_module._sample_next_token(scores, do_sample=False)
        return torch.cat((first_ids, first_ids + 10), dim=-1), SimpleNamespace()

    talker = SimpleNamespace(
        model=SimpleNamespace(forward=model_forward),
        codec_head=SimpleNamespace(forward=codec_head_forward),
        code_predictor=SimpleNamespace(generate=predictor_generate),
        _get_logits_processor=original_get_logits_processor,
    )
    active_engine = SimpleNamespace(_decode_step=original_decode_step)
    recorder = HookRecorder()
    original_members = (
        talker.model.forward,
        talker.codec_head.forward,
        talker.code_predictor.generate,
        talker._get_logits_processor,
        sampler_module._sample_next_token,
        active_engine._decode_step,
    )

    with pytest.raises(RuntimeError, match="stop"):
        with _capture_talker_hooks(
            talker,
            mode="active_prefix",
                recorder=recorder,
                sampler_module=sampler_module,
                active_engine=active_engine,
            ):
            talker.model.forward(past_key_values=cache)
            talker.codec_head.forward(torch.tensor([[1.0, 3.0]]))
            talker.code_predictor.generate()
            codec_ids, _ = active_engine._decode_step(
                scores=torch.tensor([[1.0, 4.0]])
            )
            assert codec_ids[:, :1].tolist() == [[1]]
            raise RuntimeError("stop")

    assert recorder.boundaries[0].layers[0].key.layout.contiguous is False
    assert recorder.boundaries[0].layers[0].key.layout.shape == (1, 2, 3, 2)
    assert recorder.raw_codec_logits[0].tensor.tolist() == [[11.0, 13.0]]
    assert recorder.predictor_sequences[0].tensor.tolist() == [[20, 30]]
    assert recorder.processed_outer_logits[0].tensor.tolist() == [[1.0, 4.0]]
    assert recorder.sampled_first_codebook[0].tensor.tolist() == [[1]]
    assert (
        talker.model.forward,
        talker.codec_head.forward,
        talker.code_predictor.generate,
        talker._get_logits_processor,
        sampler_module._sample_next_token,
        active_engine._decode_step,
    ) == original_members


def test_hook_restoration_continues_after_first_failure_and_preserves_body_error(
    monkeypatch,
):
    body_error = RuntimeError("hook body failed")
    talker = SimpleNamespace(
        model=SimpleNamespace(forward=lambda **kwargs: None),
        codec_head=SimpleNamespace(forward=lambda hidden: hidden),
        code_predictor=SimpleNamespace(generate=lambda **kwargs: None),
        _get_logits_processor=lambda *args, **kwargs: [],
    )
    sampler_module = SimpleNamespace(
        _sample_next_token=lambda scores, **kwargs: scores.argmax(
            dim=-1, keepdim=True
        )
    )
    active_engine = SimpleNamespace(
        _decode_step=lambda **kwargs: (torch.ones(1, 2, dtype=torch.long), None)
    )
    originals = {
        "talker.model.forward": talker.model.forward,
        "talker.codec_head.forward": talker.codec_head.forward,
        "talker.code_predictor.generate": talker.code_predictor.generate,
        "talker._get_logits_processor": talker._get_logits_processor,
        "sampler_module._sample_next_token": sampler_module._sample_next_token,
        "active_engine._decode_step": active_engine._decode_step,
    }
    labels = {
        (id(talker.model), "forward"): "talker.model.forward",
        (id(talker.codec_head), "forward"): "talker.codec_head.forward",
        (id(talker.code_predictor), "generate"): "talker.code_predictor.generate",
        (id(talker), "_get_logits_processor"): "talker._get_logits_processor",
        (id(sampler_module), "_sample_next_token"): (
            "sampler_module._sample_next_token"
        ),
        (id(active_engine), "_decode_step"): "active_engine._decode_step",
    }
    attempts = []
    original_restore = parity_probe._restore_attribute_state

    def fail_first_restore(owner, name, state):
        attempts.append(labels[(id(owner), name)])
        if len(attempts) == 1:
            raise RuntimeError("first restore failed")
        original_restore(owner, name, state)

    monkeypatch.setattr(parity_probe, "_restore_attribute_state", fail_first_restore)

    with pytest.raises(RuntimeError) as raised:
        with _capture_talker_hooks(
            talker,
            mode="active_prefix",
            recorder=HookRecorder(),
            sampler_module=sampler_module,
            active_engine=active_engine,
        ):
            raise body_error

    assert raised.value is body_error
    assert attempts == [
        "active_engine._decode_step",
        "sampler_module._sample_next_token",
        "talker._get_logits_processor",
        "talker.code_predictor.generate",
        "talker.codec_head.forward",
        "talker.model.forward",
    ]
    assert body_error.__notes__ == [
        (
            "active_engine._decode_step restoration also failed: "
            "RuntimeError: first restore failed"
        )
    ]
    for label, owner, name in (
        ("sampler_module._sample_next_token", sampler_module, "_sample_next_token"),
        ("talker._get_logits_processor", talker, "_get_logits_processor"),
        ("talker.code_predictor.generate", talker.code_predictor, "generate"),
        ("talker.codec_head.forward", talker.codec_head, "forward"),
        ("talker.model.forward", talker.model, "forward"),
    ):
        assert getattr(owner, name) is originals[label]


def test_hook_install_failure_preserved_while_all_later_restores_run(monkeypatch):
    talker = SimpleNamespace(
        model=SimpleNamespace(forward=lambda **kwargs: None),
        codec_head=SimpleNamespace(forward=lambda hidden: hidden),
        code_predictor=SimpleNamespace(generate=lambda **kwargs: None),
        _get_logits_processor=lambda *args, **kwargs: [],
    )
    sampler_module = SimpleNamespace(
        _sample_next_token=lambda scores, **kwargs: scores
    )
    active_engine = SimpleNamespace(_decode_step=None)
    originals = (
        talker.model.forward,
        talker.codec_head.forward,
        talker.code_predictor.generate,
        talker._get_logits_processor,
    )
    attempts = []
    original_restore = parity_probe._restore_attribute_state

    def fail_first_restore(owner, name, state):
        attempts.append((owner, name))
        if len(attempts) == 1:
            raise RuntimeError("sampler restore failed")
        original_restore(owner, name, state)

    monkeypatch.setattr(parity_probe, "_restore_attribute_state", fail_first_restore)

    with pytest.raises(parity_probe.ParityProbeError) as raised:
        with _capture_talker_hooks(
            talker,
            mode="active_prefix",
            recorder=HookRecorder(),
            sampler_module=sampler_module,
            active_engine=active_engine,
        ):
            pytest.fail("hook body must not run after install validation fails")

    assert "installed API is missing callable _decode_step" in str(raised.value)
    assert raised.value.__notes__ == [
        (
            "sampler_module._sample_next_token restoration also failed: "
            "RuntimeError: sampler restore failed"
        )
    ]
    assert [(name) for _, name in attempts] == [
        "_sample_next_token",
        "_get_logits_processor",
        "generate",
        "forward",
        "forward",
    ]
    assert (
        talker.model.forward,
        talker.codec_head.forward,
        talker.code_predictor.generate,
        talker._get_logits_processor,
    ) == originals


def test_upstream_hook_wraps_public_processor_list_and_restores_it():
    transformers = pytest.importorskip("transformers")
    logits_module = transformers.generation.logits_process

    class AddTwo:
        def __call__(self, input_ids, scores):
            return scores + 2

    def original_get_logits_processor(*args, **kwargs):
        return logits_module.LogitsProcessorList([AddTwo()])

    talker = SimpleNamespace(
        model=SimpleNamespace(forward=lambda **kwargs: None),
        codec_head=SimpleNamespace(forward=lambda hidden: hidden),
        code_predictor=SimpleNamespace(generate=lambda **kwargs: None),
        _get_logits_processor=original_get_logits_processor,
        forward=lambda **kwargs: kwargs,
    )
    recorder = HookRecorder()

    with _capture_talker_hooks(talker, mode="upstream", recorder=recorder):
        processors = talker._get_logits_processor()
        assert isinstance(processors, logits_module.LogitsProcessorList)
        scores = torch.tensor([[1.0, 2.0]])
        scores = processors(torch.tensor([[0]]), scores)

    assert scores.tolist() == [[3.0, 4.0]]
    assert recorder.processed_outer_logits[0].tensor.tolist() == [[3.0, 4.0]]
    assert talker._get_logits_processor is original_get_logits_processor


def test_upstream_forward_hook_preserves_transformers_kwargs_validation_signature():
    transformers = pytest.importorskip("transformers")
    generation_mixin = transformers.generation.utils.GenerationMixin

    class SignatureTalker(generation_mixin):
        def __init__(self):
            self.config = SimpleNamespace(is_encoder_decoder=False)
            self.model = SimpleNamespace(forward=lambda **kwargs: None)
            self.codec_head = SimpleNamespace(forward=lambda hidden: hidden)
            self.code_predictor = SimpleNamespace(generate=lambda **kwargs: None)

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids, **kwargs}

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            trailing_text_hidden=None,
            qwen_extra=None,
        ):
            return input_ids

        def _get_logits_processor(self, *args, **kwargs):
            return transformers.generation.logits_process.LogitsProcessorList()

    talker = SignatureTalker()
    original_signature = inspect.signature(talker.forward)

    with _capture_talker_hooks(talker, mode="upstream", recorder=HookRecorder()):
        assert inspect.signature(talker.forward) == original_signature
        talker._validate_model_kwargs(
            {
                "attention_mask": torch.ones(1, 1, dtype=torch.long),
                "trailing_text_hidden": torch.ones(1, 1, 2),
                "qwen_extra": torch.ones(1),
            }
        )


def test_active_sampled_ids_observe_final_inactive_mask_boundary():
    def original_sampler(scores, **kwargs):
        return scores.argmax(dim=-1, keepdim=True)

    sampler_module = SimpleNamespace(_sample_next_token=original_sampler)

    def decode_step(*, scores, inactive, eos_token_id):
        proposals = sampler_module._sample_next_token(scores, do_sample=False)
        final_ids = proposals.masked_fill(inactive, eos_token_id)
        codec_ids = torch.cat((final_ids, final_ids + 100), dim=-1)
        return codec_ids, SimpleNamespace()

    active_engine = SimpleNamespace(_decode_step=decode_step)
    talker = SimpleNamespace(
        model=SimpleNamespace(forward=lambda **kwargs: None),
        codec_head=SimpleNamespace(forward=lambda hidden: hidden),
        code_predictor=SimpleNamespace(generate=lambda **kwargs: None),
        _get_logits_processor=lambda *args, **kwargs: [],
        forward=lambda **kwargs: kwargs,
    )
    active_recorder = HookRecorder()
    upstream_recorder = HookRecorder()

    first_scores = torch.zeros(2, 100)
    first_scores[0, 99] = 1
    first_scores[1, 5] = 1
    second_scores = torch.zeros(2, 100)
    second_scores[0, 7] = 1
    second_scores[1, 6] = 1

    with _capture_talker_hooks(
        talker,
        mode="active_prefix",
        recorder=active_recorder,
        sampler_module=sampler_module,
        active_engine=active_engine,
    ):
        first_codec, _ = active_engine._decode_step(
            scores=first_scores,
            inactive=torch.tensor([[False], [False]]),
            eos_token_id=99,
        )
        second_codec, _ = active_engine._decode_step(
            scores=second_scores,
            inactive=torch.tensor([[True], [False]]),
            eos_token_id=99,
        )

    with _capture_talker_hooks(
        talker,
        mode="upstream",
        recorder=upstream_recorder,
    ):
        talker.forward(input_ids=first_codec[:, :1])
        talker.forward(input_ids=second_codec[:, :1])

    active_ids = [item.tensor.tolist() for item in active_recorder.sampled_first_codebook]
    upstream_ids = [
        item.tensor.tolist() for item in upstream_recorder.sampled_first_codebook
    ]
    assert active_ids == upstream_ids == [[[99], [5]], [[99], [6]]]
    assert active_ids[1] != [[7], [6]]


def test_parser_defaults_match_controlled_parity_contract():
    args = build_parser().parse_args([])

    assert args.device == "cuda:0"
    assert args.texts == ["I'm here to help."]
    assert args.batch_sizes == "1,2"
    assert args.seed == 7
    assert args.max_new_tokens == 64
    assert args.max_cache_len == 1024
    assert args.requests == 0
    assert args.duration_minutes == 0
    assert str(args.out) == "results/qwen_tts_active_prefix_parity.json"


def test_parser_accepts_stability_request_and_duration_targets():
    args = build_parser().parse_args(
        ["--requests", "100", "--duration-minutes", "10.5"]
    )

    assert args.requests == 100
    assert args.duration_minutes == 10.5


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--requests", "-1"], "--requests must be non-negative"),
        (
            ["--duration-minutes", "-0.5"],
            "--duration-minutes must be finite and non-negative",
        ),
        (
            ["--requests", "99", "--duration-minutes", "10"],
            "stability mode with both targets requires at least 100 requests",
        ),
        (
            ["--requests", "100", "--duration-minutes", "9.9"],
            "stability mode with both targets requires at least 10 minutes",
        ),
    ],
)
def test_stability_arguments_reject_invalid_values(arguments, message, tmp_path):
    output = tmp_path / "invalid-stability.json"

    with pytest.raises(ValueError, match=message):
        main(
            ["--model-path", "/model", *arguments, "--out", str(output)],
            ProbeDependencies(
                execute=lambda args: pytest.fail("normal probe must not run"),
                environment=lambda args: {},
            ),
        )

    report = json.loads(output.read_text())
    assert report["status"] == "failure"
    assert any(message in error for error in report["errors"])


def test_stability_text_rotation_mixes_batch_one_and_left_padded_batch_two():
    texts = ["a", "longer", "cc"]
    first = parity_probe._stability_request_texts(texts, cursor=0, batch_size=1)
    second = parity_probe._stability_request_texts(
        texts, cursor=first[2], batch_size=2
    )
    third = parity_probe._stability_request_texts(
        texts, cursor=second[2], batch_size=1
    )

    assert first == (["a"], [0], 1)
    assert second == (["longer", "cc"], [1, 2], 3)
    assert third == (["a"], [0], 4)

    prepared = {
        "inputs_embeds": torch.tensor(
            [
                [[1.0], [2.0], [9.0], [9.0]],
                [[3.0], [4.0], [5.0], [6.0]],
            ]
        ),
        "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]),
    }
    left_padded = parity_probe._left_pad_prepared_inputs(prepared)

    assert left_padded["attention_mask"].tolist() == [
        [0, 0, 1, 1],
        [1, 1, 1, 1],
    ]
    assert left_padded["inputs_embeds"].squeeze(-1).tolist() == [
        [9.0, 9.0, 1.0, 2.0],
        [3.0, 4.0, 5.0, 6.0],
    ]
    assert prepared["attention_mask"].tolist()[0] == [1, 1, 0, 0]


def test_memory_snapshot_uses_injected_cuda_api_without_cuda_execution():
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def memory_allocated(device):
            return 11

        @staticmethod
        def memory_reserved(device):
            return 22

        @staticmethod
        def max_memory_allocated(device):
            return 33

        @staticmethod
        def max_memory_reserved(device):
            return 44

    assert parity_probe._memory_snapshot("cuda:0", cuda_api=FakeCuda()) == {
        "memory_allocated": 11,
        "memory_reserved": 22,
        "max_memory_allocated": 33,
        "max_memory_reserved": 44,
    }


def test_main_writes_schema_and_returns_two_for_completed_parity_mismatch(tmp_path):
    output = tmp_path / "parity.json"
    upstream = _mode_capture("upstream")
    active = _mode_capture("active_prefix", codec_delta=1)
    dependencies = ProbeDependencies(
        execute=lambda args: [(1, upstream, active)],
        environment=lambda args: {
            "gpu": {"name": "fake"},
            "metrics": {"nonfinite": float("nan")},
        },
    )

    exit_code = main(
        [
            "--model-path",
            "/model",
            "--batch-sizes",
            "1",
            "--out",
            str(output),
        ],
        dependencies,
    )
    report = json.loads(output.read_text())

    assert exit_code == 2
    assert report["schema"] == "qwen_tts_active_prefix_parity"
    assert report["schema_version"] == 1
    assert report["status"] == "completed"
    assert report["parity_passed"] is False
    assert report["environment"] == {
        "gpu": {"name": "fake"},
        "metrics": {"nonfinite": {"non_finite_float": "nan"}},
    }
    assert report["batches"][0]["summary"]["first_codec_divergence"] == {
        "batch_index": 0,
        "frame_index": 1,
        "codebook_index": 1,
        "reference_id": 20,
        "candidate_id": 21,
    }


def test_main_writes_failure_json_then_reraises_runtime_error(tmp_path):
    output = tmp_path / "failure.json"
    failure = RuntimeError("installed API changed")
    dependencies = ProbeDependencies(
        execute=lambda args: (_ for _ in ()).throw(failure),
        environment=lambda args: {"torch": "test"},
    )

    with pytest.raises(RuntimeError) as raised:
        main(["--model-path", "/model", "--out", str(output)], dependencies)

    report = json.loads(output.read_text())
    assert raised.value is failure
    assert report["schema"] == "qwen_tts_active_prefix_parity"
    assert report["status"] == "failure"
    assert report["parity_passed"] is False
    assert report["errors"] == ["RuntimeError: installed API changed"]
    assert report["batches"] == []
    assert report["exit_contract"]["runtime_failure"] == 1
    assert report["exit_contract"]["argument_parse_failure"] == 64


def test_main_writes_failure_json_for_invalid_batch_size_argument(tmp_path):
    output = tmp_path / "invalid.json"

    with pytest.raises(ValueError, match="positive integers"):
        main(
            [
                "--model-path",
                "/model",
                "--batch-sizes",
                "bad",
                "--out",
                str(output),
            ]
        )

    report = json.loads(output.read_text())
    assert report["status"] == "failure"
    assert report["config"]["batch_sizes"] == "bad"
    assert report["errors"] == [
        "ValueError: --batch-sizes must contain positive integers"
    ]


def test_main_rejects_max_new_tokens_below_five_as_runtime_failure(tmp_path):
    output = tmp_path / "short-boundary-request.json"

    def must_not_run(args):
        pytest.fail("probe execution must not run for an impossible boundary count")

    dependencies = ProbeDependencies(execute=must_not_run, environment=must_not_run)
    with pytest.raises(ValueError, match="--max-new-tokens must be at least 5"):
        main(
            [
                "--model-path",
                "/model",
                "--max-new-tokens",
                "4",
                "--out",
                str(output),
            ],
            dependencies,
        )

    report = json.loads(output.read_text())
    assert report["status"] == "failure"
    assert report["failure_kind"] == "runtime"
    assert report["parity_passed"] is False
    assert report["batches"] == []
    assert report["errors"] == [
        "ValueError: --max-new-tokens must be at least 5"
    ]


def test_main_writes_auditable_argparse_failure_with_distinct_exit(tmp_path):
    output = tmp_path / "argument-failure.json"

    def must_not_run(args):
        pytest.fail("probe dependencies must not run after argument parsing fails")

    dependencies = ProbeDependencies(execute=must_not_run, environment=must_not_run)
    exit_code = main(
        [
            "--model-path",
            "/model",
            "--out",
            str(output),
            "--seed",
            "not-an-integer",
        ],
        dependencies,
    )

    report = json.loads(output.read_text())
    assert exit_code == 64
    assert report["status"] == "failure"
    assert report["failure_kind"] == "argument_parse"
    assert report["config"]["out"] == str(output)
    assert report["exit_contract"]["argument_parse_failure"] == 64
    assert any("invalid int value" in error for error in report["errors"])


def test_main_accepts_injected_parser_failure_without_running_probe(tmp_path):
    output = tmp_path / "injected-argument-failure.json"

    def fail_parse(argv):
        raise parity_probe.ArgumentParseFailure("injected parser failure")

    def must_not_run(args):
        pytest.fail("probe dependencies must not run after argument parsing fails")

    dependencies = ProbeDependencies(
        execute=must_not_run,
        environment=must_not_run,
        parse_args=fail_parse,
    )
    exit_code = main(["--out", str(output)], dependencies)

    report = json.loads(output.read_text())
    assert exit_code == 64
    assert report["errors"] == [
        "ArgumentParseFailure: injected parser failure"
    ]
    assert report["config"]["out"] == str(output)


def test_active_hook_setup_rolls_back_if_sampler_validation_fails():
    def model_forward(**kwargs):
        return kwargs

    def head_forward(hidden):
        return hidden

    def predictor_generate(**kwargs):
        return kwargs

    def get_logits_processor(*args, **kwargs):
        return []

    talker = SimpleNamespace(
        model=SimpleNamespace(forward=model_forward),
        codec_head=SimpleNamespace(forward=head_forward),
        code_predictor=SimpleNamespace(generate=predictor_generate),
        _get_logits_processor=get_logits_processor,
    )
    sampler_module = SimpleNamespace(_sample_next_token=None)
    originals = (
        talker.model.forward,
        talker.codec_head.forward,
        talker.code_predictor.generate,
        talker._get_logits_processor,
    )

    with pytest.raises(parity_probe.ParityProbeError, match="_sample_next_token"):
        with _capture_talker_hooks(
            talker,
            mode="active_prefix",
            recorder=HookRecorder(),
            sampler_module=sampler_module,
        ):
            pytest.fail("hook context must not be entered")

    assert (
        talker.model.forward,
        talker.codec_head.forward,
        talker.code_predictor.generate,
        talker._get_logits_processor,
    ) == originals


def test_exact_gate_rejects_mode_batch_and_audio_cardinality_mismatch():
    reference = _mode_capture("upstream")
    candidate = _mode_capture("active_prefix")
    candidate = replace(
        candidate,
        batch_size=2,
        audio=AudioCapture(
            sample_rate=24000,
            sample_counts=(),
            finite=(),
            nonempty=(),
        ),
    )

    summary = summarize_parity(reference, candidate)

    assert summary.parity_passed is False
    assert set(summary.failed_reasons) >= {
        "mode_batch_size_mismatch",
        "audio_batch_size_mismatch",
    }


@pytest.mark.parametrize(
    ("configured", "result_sizes", "message"),
    [
        ("1,2", (1, 1), "duplicate result for configured batch size 1"),
        ("1,2", (1,), "missing result for configured batch size 2"),
        ("1,2", (1, 3), "unexpected result batch size 3"),
    ],
)
def test_main_rejects_duplicate_missing_and_unexpected_batch_results(
    configured,
    result_sizes,
    message,
    tmp_path,
):
    output = tmp_path / "cardinality.json"
    results = [
        (size, _mode_capture("upstream"), _mode_capture("active_prefix"))
        for size in result_sizes
    ]
    dependencies = ProbeDependencies(
        execute=lambda args: results,
        environment=lambda args: {},
    )

    with pytest.raises(parity_probe.ParityProbeError, match=message):
        main(
            [
                "--model-path",
                "/model",
                "--batch-sizes",
                configured,
                "--out",
                str(output),
            ],
            dependencies,
        )

    report = json.loads(output.read_text())
    assert report["status"] == "failure"
    assert message in report["errors"][0]


def test_main_rejects_mode_capture_batch_size_that_disagrees_with_result(tmp_path):
    reference = replace(_mode_capture("upstream"), batch_size=2)
    candidate = replace(_mode_capture("active_prefix"), batch_size=2)
    dependencies = ProbeDependencies(
        execute=lambda args: [(1, reference, candidate)],
        environment=lambda args: {},
    )

    with pytest.raises(
        parity_probe.ParityProbeError,
        match="upstream capture batch size 2 does not match configured batch size 1",
    ):
        main(
            [
                "--model-path",
                "/model",
                "--batch-sizes",
                "1",
                "--out",
                str(tmp_path / "capture-size.json"),
            ],
            dependencies,
        )


def test_warning_capture_collects_python_and_transformers_logs_without_leaks():
    root = logging.getLogger()
    transformers_logger = logging.getLogger("transformers.generation.test")
    root_handlers = tuple(root.handlers)
    transformer_handlers = tuple(transformers_logger.handlers)

    with pytest.raises(RuntimeError, match="stop"):
        with parity_probe._capture_probe_warnings() as captured:
            warnings.warn("python warning", UserWarning)
            transformers_logger.warning("transformers warning")
            raise RuntimeError("stop")

    assert any("python warning" in message for message in captured.messages())
    assert any("transformers warning" in message for message in captured.messages())
    assert tuple(root.handlers) == root_handlers
    assert tuple(transformers_logger.handlers) == transformer_handlers


def test_plain_report_copy_sanitizes_nested_nonfinite_floats():
    copied = parity_probe._plain_copy(
        {"values": [float("nan"), float("inf"), float("-inf"), 1.5]}
    )

    assert copied == {
        "values": [
            {"non_finite_float": "nan"},
            {"non_finite_float": "positive_infinity"},
            {"non_finite_float": "negative_infinity"},
            1.5,
        ]
    }
    assert not any(
        isinstance(value, float) and not math.isfinite(value)
        for value in copied["values"]
    )


def test_report_write_failure_rethrows_original_error_with_secondary_note(
    monkeypatch,
    tmp_path,
):
    primary = RuntimeError("primary failure")
    write_failure = OSError("disk full")
    dependencies = ProbeDependencies(
        execute=lambda args: (_ for _ in ()).throw(primary),
        environment=lambda args: {"metric": float("nan")},
    )
    monkeypatch.setattr(
        parity_probe,
        "_write_report",
        lambda path, report: (_ for _ in ()).throw(write_failure),
    )

    with pytest.raises(RuntimeError) as raised:
        main(
            ["--model-path", "/model", "--out", str(tmp_path / "failure.json")],
            dependencies,
        )

    assert raised.value is primary
    assert any("report write also failed: disk full" in note for note in primary.__notes__)


def test_sampled_first_codebook_is_reported_and_gated_against_raw_history():
    reference = _mode_capture("upstream")
    candidate = _mode_capture("active_prefix")

    assert "sampled_first_codebook" in ModeCapture.__dataclass_fields__
    candidate = replace(
        candidate,
        sampled_first_codebook=_snapshot(
            "sampled_first_codebook",
            torch.tensor([[1, 7, 99]], dtype=torch.int64),
        ),
    )
    summary = summarize_parity(reference, candidate)

    assert summary.parity_passed is False
    assert set(summary.failed_reasons) >= {
        "sampled_first_codebook_mismatch",
        "sampled_first_codebook_raw_inconsistent",
    }
    assert "sampled_first_codebook" in candidate.metadata()


def test_environment_records_cuda_visibility_and_public_physical_identity(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    properties = SimpleNamespace(
        uuid="GPU-123",
        pci_bus_id="00000000:81:00.0",
        pci_device_id=0x20B0,
    )

    identity = parity_probe._gpu_physical_identity(properties)
    environment = parity_probe._collect_environment(
        build_parser().parse_args(["--model-path", "/model"])
    )

    assert environment["cuda_visible_devices"] == "2"
    assert identity == {
        "uuid": "GPU-123",
        "pci_bus_id": "00000000:81:00.0",
        "pci_device_id": 0x20B0,
        "source": "torch.cuda.get_device_properties public attributes",
    }


def test_prepare_talker_inputs_tokenizes_once_and_restores_generate_and_rope():
    events = []

    def original_generate(**kwargs):
        return kwargs

    talker = SimpleNamespace(generate=original_generate, rope_deltas="original")

    class FakeModel:
        def __init__(self):
            self.talker = talker

        def generate(self, **kwargs):
            talker.rope_deltas = "mutated"
            return talker.generate(
                inputs_embeds=torch.ones(1, 2, 3),
                attention_mask=torch.ones(1, 2, dtype=torch.long),
                trailing_text_hidden=torch.ones(1, 1, 3),
                tts_pad_embed=torch.ones(1, 1, 3),
                min_new_tokens=2,
                eos_token_id=99,
                repetition_penalty=1.05,
                suppress_tokens=[7, 8],
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key
                    in {"do_sample", "max_new_tokens", "subtalker_dosample"}
                },
            )

    wrapper = SimpleNamespace(model=FakeModel())
    wrapper._build_assistant_text = lambda text: f"assistant:{text}"
    wrapper._validate_languages = lambda values: events.append(("languages", values))
    wrapper._validate_speakers = lambda values: events.append(("speakers", values))

    def tokenize(values):
        events.append(("tokenize", values))
        return [torch.tensor([[1, 2]])]

    wrapper._tokenize_texts = tokenize
    wrapper._merge_generate_kwargs = lambda **kwargs: kwargs
    backend = SimpleNamespace(_language="english", _speaker="alice")
    args = build_parser().parse_args(["--model-path", "/model"])

    prepared = _prepare_talker_inputs_once(wrapper, ["hello"], backend, args)

    assert [event[0] for event in events].count("tokenize") == 1
    assert prepared["eos_token_id"] == 99
    assert prepared["pad_token_id"] == 99
    assert prepared["temperature"] is None
    assert prepared["subtalker_temperature"] is None
    assert prepared["repetition_penalty"] == 1.05
    assert prepared["suppress_tokens"] == [7, 8]
    assert talker.generate is original_generate
    assert talker.rope_deltas == "original"


def test_prepare_restoration_continues_and_preserves_model_generate_error(
    monkeypatch,
):
    primary = RuntimeError("model preparation failed")

    def original_generate(**kwargs):
        return kwargs

    talker = SimpleNamespace(generate=original_generate, rope_deltas="original")

    class FakeModel:
        def __init__(self):
            self.talker = talker

        def generate(self, **kwargs):
            talker.rope_deltas = "mutated"
            raise primary

    wrapper = SimpleNamespace(model=FakeModel())
    wrapper._build_assistant_text = lambda text: text
    wrapper._tokenize_texts = lambda texts: [torch.tensor([[1]])]
    wrapper._validate_languages = lambda values: None
    wrapper._validate_speakers = lambda values: None
    wrapper._merge_generate_kwargs = lambda **kwargs: kwargs
    backend = SimpleNamespace(_language="english", _speaker="alice")
    args = build_parser().parse_args(["--model-path", "/model"])
    attempts = []
    original_restore = parity_probe._restore_attribute_state

    def fail_generate_restore(owner, name, state):
        attempts.append(name)
        if name == "generate":
            raise RuntimeError("generate restore failed")
        original_restore(owner, name, state)

    monkeypatch.setattr(
        parity_probe, "_restore_attribute_state", fail_generate_restore
    )

    with pytest.raises(RuntimeError) as raised:
        _prepare_talker_inputs_once(wrapper, ["hello"], backend, args)

    assert raised.value is primary
    assert attempts == ["generate", "rope_deltas"]
    assert talker.rope_deltas == "original"
    assert talker.generate is not original_generate
    assert primary.__notes__ == [
        (
            "talker.generate restoration also failed: "
            "RuntimeError: generate restore failed"
        )
    ]


class _RunModeTalker:
    def __init__(self):
        transformers = pytest.importorskip("transformers")
        self.cache = _FakeCache()
        self.model = SimpleNamespace(
            config=SimpleNamespace(_attn_implementation="sdpa"),
            forward=self._model_forward,
        )
        self.codec_head = SimpleNamespace(forward=lambda hidden: hidden + 5)
        self.code_predictor = SimpleNamespace(generate=self._predictor_generate)
        self._processor_list = transformers.generation.logits_process.LogitsProcessorList()

    def _get_logits_processor(self, *args, **kwargs):
        return self._processor_list

    def forward(self, **kwargs):
        return kwargs

    def _model_forward(self, **kwargs):
        return SimpleNamespace(
            last_hidden_state=torch.ones(1, 1, 2),
            past_key_values=self.cache,
        )

    def _predictor_generate(self, **kwargs):
        step = getattr(self, "step", 0)
        return SimpleNamespace(sequences=torch.tensor([[10 + step]]))

    def generate(self, **kwargs):
        histories = []
        for step in range(5):
            self.step = step
            first = torch.tensor([[step + 1]])
            self.forward(input_ids=first)
            predictor = self.code_predictor.generate()
            codec = torch.cat((first, predictor.sequences), dim=-1)
            output = self.model.forward(past_key_values=self.cache)
            logits = self.codec_head.forward(output.last_hidden_state)[:, -1, :]
            self._get_logits_processor()(first, logits)
            histories.append((None, codec))
        return SimpleNamespace(hidden_states=tuple(histories))


def test_run_mode_reconstructs_codec_sampled_ids_and_full_operation_warnings():
    talker = _RunModeTalker()

    class SpeechTokenizer:
        def decode(self, encoded):
            assert len(encoded) == 1
            assert encoded[0]["audio_codes"].shape == (5, 2)
            warnings.warn("decode warning", UserWarning)
            logging.getLogger("transformers.generation.test").warning(
                "decode logger warning"
            )
            return [np.ones(12, dtype=np.float32)], 24000

    wrapper = SimpleNamespace(
        model=SimpleNamespace(talker=talker, speech_tokenizer=SpeechTokenizer())
    )

    capture = _run_mode(
        wrapper,
        prepared={},
        mode="upstream",
        seed=7,
        eos_token_id=99,
        device="cpu",
        resolved_config={"speaker": "alice"},
    )

    assert capture.codec.raw_codec.tensor.shape == (1, 5, 2)
    assert capture.codec.predictor_sequences.tensor.tolist() == [
        [[10], [11], [12], [13], [14]]
    ]
    assert capture.sampled_first_codebook.tensor.tolist() == [[1, 2, 3, 4, 5]]
    assert capture.audio.sample_counts == (12,)
    assert capture.resolved_config == {"speaker": "alice"}
    assert any("decode warning" in message for message in capture.warnings)
    assert any("decode logger warning" in message for message in capture.warnings)


def test_real_run_mode_failure_warnings_reach_json_once_without_handler_leaks(
    tmp_path,
):
    output = tmp_path / "real-run-mode-warning-failure.json"
    primary = RuntimeError("real run mode generation failed")
    talker = _RunModeTalker()

    def failing_generate(**kwargs):
        warnings.warn("real run mode python warning", UserWarning)
        logging.getLogger("transformers.generation.test").warning(
            "real run mode logger warning"
        )
        raise primary

    talker.generate = failing_generate
    wrapper = SimpleNamespace(
        generate_defaults={},
        model=SimpleNamespace(
            talker=talker,
            speech_tokenizer=SimpleNamespace(),
            config=SimpleNamespace(
                talker_config=SimpleNamespace(codec_eos_token_id=99)
            ),
        ),
    )

    class Backend:
        _language = "english"
        _speaker = "alice"
        _model = wrapper

        def close(self):
            return None

    root = logging.getLogger()
    transformers_logger = logging.getLogger("transformers")
    root_state = (tuple(root.handlers), root.level, root.disabled)
    transformers_state = (
        tuple(transformers_logger.handlers),
        transformers_logger.level,
        transformers_logger.disabled,
    )

    def execute(args):
        return _execute_probe(
            args,
            backend_factory=lambda **kwargs: Backend(),
            active_installer=lambda *args, **kwargs: pytest.fail(
                "active install must not run after upstream failure"
            ),
            prepare_fn=lambda *args, **kwargs: {},
        )

    with pytest.raises(RuntimeError) as raised:
        main(
            [
                "--model-path",
                "/model",
                "--batch-sizes",
                "1",
                "--out",
                str(output),
            ],
            ProbeDependencies(execute=execute, environment=lambda args: {}),
        )

    report = json.loads(output.read_text())
    assert raised.value is primary
    for expected in (
        "real run mode python warning",
        "real run mode logger warning",
    ):
        matching_warnings = [
            warning for warning in report["warnings"] if expected in warning
        ]
        matching_errors = [error for error in report["errors"] if expected in error]
        assert len(matching_warnings) == 1
        assert len(matching_errors) == 1
    assert (tuple(root.handlers), root.level, root.disabled) == root_state
    assert (
        tuple(transformers_logger.handlers),
        transformers_logger.level,
        transformers_logger.disabled,
    ) == transformers_state


def test_execute_probe_prepares_once_per_batch_and_closes_active_runtime():
    events = []

    def original_generate(**kwargs):
        return kwargs

    talker = SimpleNamespace(generate=original_generate, rope_deltas="original")
    wrapper = SimpleNamespace(
        generate_defaults={"repetition_penalty": 1.05},
        model=SimpleNamespace(
            talker=talker,
            config=SimpleNamespace(
                talker_config=SimpleNamespace(codec_eos_token_id=99)
            ),
        ),
    )

    class Backend:
        _language = "english"
        _speaker = "alice"
        _model = wrapper

        def __init__(self):
            warnings.warn("backend construction warning", UserWarning)

        def close(self):
            events.append("backend.close")
            warnings.warn("backend close warning", UserWarning)

    class Runtime:
        def close(self):
            events.append("active.close")
            logging.getLogger("transformers.generation.test").warning(
                "active close warning"
            )

    runtime = Runtime()

    def install(talker_arg, *, max_cache_len):
        events.append("active.install")

        def active_generate(**kwargs):
            return kwargs

        active_generate._qav_outer_engine = SimpleNamespace(step_runtime=runtime)
        talker_arg.generate = active_generate
        return True

    prepared_sources = []

    def prepare(wrapper_arg, texts, backend_arg, args):
        events.append(f"prepare.{len(texts)}")
        warnings.warn("prepare warning", UserWarning)
        source = torch.tensor([len(texts)], dtype=torch.int64)
        prepared_sources.append(source)
        return {
            "inputs_embeds": source,
            "do_sample": False,
            "subtalker_dosample": False,
            "eos_token_id": 99,
            "repetition_penalty": 1.05,
            "suppress_tokens": [7, 8],
        }

    run_inputs = []
    resolved_configs = []

    def run_mode(wrapper_arg, *, prepared, mode, **kwargs):
        events.append(f"run.{mode}")
        run_inputs.append((mode, prepared["inputs_embeds"]))
        resolved_configs.append(kwargs["resolved_config"])
        return _mode_capture(mode)

    args = build_parser().parse_args(
        ["--model-path", "/model", "--batch-sizes", "1,2"]
    )
    results = _execute_probe(
        args,
        backend_factory=lambda **kwargs: Backend(),
        active_installer=install,
        prepare_fn=prepare,
        run_mode_fn=run_mode,
    )

    assert [result[0] for result in results] == [1, 2]
    assert events == [
        "prepare.1",
        "run.upstream",
        "active.install",
        "run.active_prefix",
        "active.close",
        "prepare.2",
        "run.upstream",
        "active.install",
        "run.active_prefix",
        "active.close",
        "backend.close",
    ]
    assert len(prepared_sources) == 2
    for index in range(0, len(run_inputs), 2):
        upstream = run_inputs[index][1]
        active = run_inputs[index + 1][1]
        assert torch.equal(upstream, active)
        assert upstream.data_ptr() != active.data_ptr()
    assert all(config["speaker"] == "alice" for config in resolved_configs)
    assert all(config["resolved_eos_token_id"] == 99 for config in resolved_configs)
    assert all(
        config["generation_defaults"]["repetition_penalty"] == 1.05
        for config in resolved_configs
    )
    assert all(
        config["generation"]["repetition_penalty"] == 1.05
        for config in resolved_configs
    )
    assert all(config["suppress_tokens"] == [7, 8] for config in resolved_configs)
    for _, reference, candidate in results:
        for capture in (reference, candidate):
            assert any("backend construction warning" in item for item in capture.warnings)
            assert any("prepare warning" in item for item in capture.warnings)
            assert any("active close warning" in item for item in capture.warnings)
            assert any("backend close warning" in item for item in capture.warnings)
    assert "warnings_present" in summarize_parity(results[0][1], results[0][2]).failed_reasons
    assert talker.generate is original_generate
    assert talker.rope_deltas == "original"


def test_execute_probe_error_warnings_reach_failure_report(tmp_path):
    output = tmp_path / "warning-failure.json"
    primary = RuntimeError("generation failed")

    def original_generate(**kwargs):
        return kwargs

    talker = SimpleNamespace(generate=original_generate, rope_deltas="original")
    wrapper = SimpleNamespace(
        generate_defaults={},
        model=SimpleNamespace(
            talker=talker,
            config=SimpleNamespace(
                talker_config=SimpleNamespace(codec_eos_token_id=99)
            ),
        ),
    )

    class Backend:
        _language = "english"
        _speaker = "alice"
        _model = wrapper

        def __init__(self):
            warnings.warn("constructor path warning", UserWarning)

        def close(self):
            warnings.warn("error close warning", UserWarning)

    def prepare(wrapper_arg, texts, backend_arg, args):
        return {
            "inputs_embeds": torch.ones(1),
            "eos_token_id": 99,
            "suppress_tokens": [],
        }

    def run_mode(wrapper_arg, *, mode, **kwargs):
        warnings.warn("generation path warning", UserWarning)
        raise primary

    def execute(args):
        return _execute_probe(
            args,
            backend_factory=lambda **kwargs: Backend(),
            active_installer=lambda *args, **kwargs: pytest.fail(
                "active install must not run after upstream failure"
            ),
            prepare_fn=prepare,
            run_mode_fn=run_mode,
        )

    with pytest.raises(RuntimeError) as raised:
        main(
            [
                "--model-path",
                "/model",
                "--batch-sizes",
                "1",
                "--out",
                str(output),
            ],
            ProbeDependencies(execute=execute, environment=lambda args: {}),
        )

    report = json.loads(output.read_text())
    assert raised.value is primary
    for expected in (
        "constructor path warning",
        "generation path warning",
        "error close warning",
    ):
        assert any(expected in warning for warning in report["warnings"])
        assert any(expected in error for error in report["errors"])


def test_execute_probe_preserves_generation_error_across_ordered_cleanup_failures(
    monkeypatch,
):
    generation_error = RuntimeError("candidate generation failed")
    active_installed = False

    def original_generate(**kwargs):
        return kwargs

    talker = SimpleNamespace(generate=original_generate, rope_deltas="original")
    wrapper = SimpleNamespace(
        generate_defaults={},
        model=SimpleNamespace(
            talker=talker,
            config=SimpleNamespace(
                talker_config=SimpleNamespace(codec_eos_token_id=99)
            ),
        ),
    )

    class Backend:
        _language = "english"
        _speaker = "alice"
        _model = wrapper

        def close(self):
            raise RuntimeError("backend close failed")

    class Runtime:
        def close(self):
            raise RuntimeError("runtime close failed")

    def install(talker_arg, *, max_cache_len):
        nonlocal active_installed
        active_installed = True

        def active_generate(**kwargs):
            return kwargs

        active_generate._qav_outer_engine = SimpleNamespace(step_runtime=Runtime())
        talker_arg.generate = active_generate
        talker_arg.rope_deltas = "active"
        return True

    def prepare(wrapper_arg, texts, backend_arg, args):
        return {
            "inputs_embeds": torch.ones(1),
            "eos_token_id": 99,
            "suppress_tokens": [],
        }

    def run_mode(wrapper_arg, *, mode, **kwargs):
        if mode == "upstream":
            return _mode_capture(mode)
        raise generation_error

    original_restore = parity_probe._restore_attribute_state

    def fail_active_restoration(owner, name, state):
        if active_installed and owner is talker:
            raise RuntimeError(f"{name} restore failed")
        return original_restore(owner, name, state)

    monkeypatch.setattr(parity_probe, "_restore_attribute_state", fail_active_restoration)
    args = build_parser().parse_args(
        ["--model-path", "/model", "--batch-sizes", "1"]
    )

    with pytest.raises(RuntimeError) as raised:
        _execute_probe(
            args,
            backend_factory=lambda **kwargs: Backend(),
            active_installer=install,
            prepare_fn=prepare,
            run_mode_fn=run_mode,
        )

    assert raised.value is generation_error
    assert generation_error.__notes__ == [
        "active runtime close also failed: RuntimeError: runtime close failed",
        "talker.generate restoration also failed: RuntimeError: generate restore failed",
        "talker.rope_deltas restoration also failed: RuntimeError: rope_deltas restore failed",
        (
            "talker.generate restoration validation also failed: ParityProbeError: "
            "talker.generate restoration failed"
        ),
        (
            "talker.rope_deltas restoration validation also failed: "
            "ParityProbeError: talker.rope_deltas restoration failed"
        ),
        "backend close also failed: RuntimeError: backend close failed",
    ]


def test_stability_target_waits_for_whichever_threshold_takes_longer():
    pending = parity_probe._stability_target_pending

    assert pending(2, 0.0, request_target=3, duration_seconds=0.0) is True
    assert pending(3, 0.0, request_target=3, duration_seconds=0.0) is False
    assert pending(10, 5.0, request_target=0, duration_seconds=6.0) is True
    assert pending(10, 6.0, request_target=0, duration_seconds=6.0) is False
    assert pending(100, 599.0, request_target=100, duration_seconds=600.0) is True
    assert pending(99, 600.0, request_target=100, duration_seconds=600.0) is True
    assert pending(100, 600.0, request_target=100, duration_seconds=600.0) is False


def test_stability_aggregates_request_failure_without_backend_reload(tmp_path):
    events = []
    prepare_index = -1

    def original_generate(**kwargs):
        return kwargs

    talker = SimpleNamespace(generate=original_generate, rope_deltas="original")
    wrapper = SimpleNamespace(
        generate_defaults={},
        model=SimpleNamespace(
            talker=talker,
            config=SimpleNamespace(
                talker_config=SimpleNamespace(codec_eos_token_id=99)
            ),
        ),
    )

    class Backend:
        _language = "english"
        _speaker = "alice"
        _model = wrapper

        def __init__(self):
            events.append("backend.load")

        def close(self):
            events.append("backend.close")

    cache = SimpleNamespace(
        layers=[
            SimpleNamespace(
                keys=torch.ones(1, 1, 2, 1),
                values=torch.ones(1, 1, 2, 1) * 2,
            )
        ]
    )

    class Runtime:
        def __init__(self):
            self._caches = {id(cache): cache}

        def close(self):
            events.append("runtime.close")

        def metrics_snapshot(self):
            return {"cache_allocations": 1, "cache_resets": 4}

    runtime = Runtime()

    def install(talker_arg, *, max_cache_len):
        events.append("active.install")

        def active_generate(**kwargs):
            return kwargs

        active_generate._qav_outer_engine = SimpleNamespace(
            step_runtime=runtime,
            metrics_snapshot=runtime.metrics_snapshot,
        )
        talker_arg.generate = active_generate
        return True

    def prepare(wrapper_arg, texts, backend_arg, args):
        nonlocal prepare_index
        prepare_index += 1
        lengths = [1 if text == "a" else 3 for text in texts]
        width = max(lengths)
        embeds = torch.full((len(texts), width, 1), 9.0)
        mask = torch.zeros(len(texts), width, dtype=torch.long)
        for row, length in enumerate(lengths):
            embeds[row, :length, 0] = torch.arange(1, length + 1)
            mask[row, :length] = 1
        return {"inputs_embeds": embeds, "attention_mask": mask}

    run_requests = []

    def run_mode(wrapper_arg, *, prepared, mode, **kwargs):
        batch_size = int(prepared["inputs_embeds"].shape[0])
        run_requests.append((prepare_index, mode, batch_size))
        if batch_size == 2:
            assert prepared["attention_mask"][1].tolist() == [0, 0, 1]
        if prepare_index == 1 and mode == "active_prefix":
            raise RuntimeError("request one active failure")
        return _mode_capture(mode, batch_size=batch_size)

    args = build_parser().parse_args(
        [
            "--model-path",
            "/model",
            "--texts",
            "a",
            "bbb",
            "a",
            "--requests",
            "4",
            "--out",
            str(tmp_path / "stability.json"),
        ]
    )
    memory_calls = []

    def memory_snapshot(device):
        memory_calls.append(device)
        value = len(memory_calls)
        return {
            "memory_allocated": value,
            "memory_reserved": value + 10,
            "max_memory_allocated": value + 20,
            "max_memory_reserved": value + 30,
        }

    execution = parity_probe._execute_stability(
        args,
        backend_factory=lambda **kwargs: Backend(),
        active_installer=install,
        prepare_fn=prepare,
        run_mode_fn=run_mode,
        memory_snapshot_fn=memory_snapshot,
    )

    assert events == ["backend.load", "active.install", "runtime.close", "backend.close"]
    assert [item["request_index"] for item in execution.request_results] == [0, 1, 2, 3]
    assert [item["batch_size"] for item in execution.request_results] == [1, 2, 1, 2]
    assert [item["text_indices"] for item in execution.request_results] == [
        [0],
        [1, 2],
        [0],
        [1, 2],
    ]
    assert execution.request_results[0]["parity_passed"] is True
    assert execution.request_results[1]["parity_passed"] is False
    assert any(
        "request one active failure" in error
        for error in execution.request_results[1]["errors"]
    )
    assert execution.request_results[2]["parity_passed"] is True
    assert execution.request_results[3]["parity_passed"] is True
    assert len(memory_calls) == 4
    assert execution.request_results[0]["memory"]["memory_allocated"] == 1
    assert execution.request_results[0]["active_cache"]["metrics"] == {
        "cache_allocations": 1,
        "cache_resets": 4,
    }
    assert execution.request_results[0]["active_cache"]["backing_pointers"]
    assert run_requests[-1][0] == 3

    output = tmp_path / "stability-report.json"
    exit_code = main(
        [
            "--model-path",
            "/model",
            "--texts",
            "a",
            "bbb",
            "a",
            "--requests",
            "4",
            "--out",
            str(output),
        ],
        ProbeDependencies(
            execute=lambda args: pytest.fail("normal probe must not run"),
            environment=lambda args: {"device": "cpu-test"},
            stability_execute=lambda args: execution,
        ),
    )

    report = json.loads(output.read_text())
    assert exit_code == 1
    assert report["mode"] == "stability"
    assert report["status"] == "failure"
    assert len(report["stability"]["requests"]) == 4
    assert report["stability"]["completed_requests"] == 4
    assert report["environment"] == {"device": "cpu-test"}
