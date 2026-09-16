import json
import time
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import bench.qwen_tts_internal_profile as internal_profile
from bench.qwen_tts_internal_profile import (
    _first_dim,
    _set_seed,
    build_parser,
    profile_qwen_tts_batch,
)
from qwen_asr_vllm.agent.qwen_tts_outer_static_engine import OuterTalkerEngineError


class FakeWrapper:
    def __init__(self):
        self._speaker = "alice"
        self._language = "english"
        self.model = FakeModel()

    def _ensure_list(self, value):
        return value if isinstance(value, list) else [value]

    def _validate_languages(self, languages):
        return None

    def _validate_speakers(self, speakers):
        return None

    def _build_assistant_text(self, text):
        return text

    def _tokenize_texts(self, texts):
        return [np.array([[index + 1]], dtype=np.int64) for index, _ in enumerate(texts)]

    def _merge_generate_kwargs(self, **kwargs):
        return {key: value for key, value in kwargs.items() if value is not None}


class FakeModel:
    tts_model_type = "custom_voice"

    def __init__(self):
        self.talker = FakeTalker()
        self.speech_tokenizer = FakeSpeechTokenizer()
        self.config = SimpleNamespace(
            talker_config=SimpleNamespace(codec_eos_token_id=999)
        )

    def generate(self, **kwargs):
        batch_size = len(kwargs["input_ids"])
        for _ in range(5):
            self.talker.forward(batch_size=batch_size)
            time.sleep(0.001)
        codes = [
            (np.arange(5, dtype=np.int64).reshape(5, 1) + request_index * 100)
            for request_index in range(batch_size)
        ]
        return codes, None


class FakeTalker:
    def __init__(self):
        self.calls = 0
        self.code_predictor = FakeCodePredictor()
        self.model = FakeTalkerModel()

    def forward(self, *, batch_size=1):
        self.calls += 1
        self.code_predictor.generate(batch_size=batch_size)
        self.model.forward(batch_size=batch_size)
        frames = np.full((batch_size, 1), self.calls, dtype=np.int64)
        return SimpleNamespace(hidden_states=(None, frames))


class FakeCodePredictor:
    def generate(self, *, batch_size=1):
        time.sleep(0.001)
        return SimpleNamespace(sequences=np.zeros((batch_size, 1), dtype=np.int64))


class FakeTalkerModel:
    def forward(self, *, batch_size=1):
        time.sleep(0.001)
        return SimpleNamespace(last_hidden_state=np.zeros((batch_size, 1, 1)))


class FakeSpeechTokenizer:
    def get_decode_upsample_rate(self):
        return 4

    def decode(self, encoded):
        audios = []
        for item in encoded:
            codes = np.asarray(item["audio_codes"])
            audios.append(np.repeat(codes[:, 0].astype(np.float32), 4))
        return audios, 16000


def test_profile_qwen_tts_batch_records_generation_decode_and_talker_steps():
    report = profile_qwen_tts_batch(
        FakeWrapper(),
        texts=["first fragment", "second fragment"],
        chunk_size=2,
        left_context_size=1,
        non_streaming_mode=True,
    )

    assert report["stage"] == "qwen_tts_internal"
    assert report["batch_size"] == 2
    assert report["codec_frames"] == [5, 5]
    assert report["outer_engine"] == "upstream"
    assert report["codec_ids_sha256"] == (
        "43a7ffae48bc7c35e8f01ef07d6cfa5e9cb9cc594b3b604b8e78d9b6cf2815d2"
    )
    assert report["codec_ids_item_sha256"] == [
        "281b02b10f5f4997e5bf8c93343e6f2aa8bc81ffad6d6813c593181ebceda12a",
        "1c24f5eb248b475d5b6596383bcd14c60e878e1f4f24f1cf9cb21c7113258c37",
    ]
    assert report["codec_ids_shapes"] == [[5, 1], [5, 1]]
    assert report["codec_ids_dtypes"] == ["torch.int64", "torch.int64"]
    assert report["outer_engine_metrics"] == {}
    assert report["audio_duration_s"] == [0.001, 0.001]
    assert report["audio_nonempty"] == [True, True]
    assert report["audio_finite"] == [True, True]
    assert report["generation_ms"] > 0
    assert report["full_decode_ms"] >= 0
    assert report["chunked_decode_total_ms"] >= 0
    assert report["chunked_chunks"] == [3, 3]
    assert report["talker_forward"]["steps"] == 5
    assert report["talker_forward"]["max_batch_size"] == 2
    assert report["talker_forward"]["mean_ms"] >= 0
    assert report["code_predictor_generate"]["steps"] == 5
    assert report["code_predictor_generate"]["max_batch_size"] == 2
    assert report["talker_model_forward"]["steps"] == 5
    assert report["talker_model_forward"]["max_batch_size"] == 2


def test_first_dim_uses_shape_without_materialising_cuda_tensor():
    class FakeCudaTensor:
        shape = (7, 1)

        def __array__(self):
            raise TypeError("can't convert cuda tensor to numpy")

    assert _first_dim(FakeCudaTensor()) == 7


def test_internal_profile_parser_accepts_static_predictor_flag():
    args = build_parser().parse_args(
        [
            "--static-code-predictor",
            "--cuda-graph-code-predictor",
            "--cuda-graph-fixed-slots",
            "2",
            "--cuda-graph-batch-window-ms",
            "3",
            "--outer-static-talker-engine",
            "--outer-graph-max-cache-len",
            "512",
            "--greedy",
            "--seed",
            "7",
        ]
    )

    assert args.static_code_predictor is True
    assert args.cuda_graph_code_predictor is True
    assert args.cuda_graph_fixed_slots == 2
    assert args.cuda_graph_batch_window_ms == 3
    assert args.outer_static_talker_engine is True
    assert args.outer_graph_max_cache_len == 512
    assert args.greedy is True
    assert args.seed == 7


def test_internal_profile_parser_accepts_active_prefix_outer_flag():
    args = build_parser().parse_args(
        [
            "--outer-active-prefix-talker-engine",
            "--outer-graph-max-cache-len",
            "512",
        ]
    )

    assert args.outer_active_prefix_talker_engine is True
    assert args.outer_graph_max_cache_len == 512


def test_internal_profile_parser_accepts_outer_cuda_graph_options():
    args = build_parser().parse_args(
        [
            "--outer-cuda-graph-talker-engine",
            "--outer-graph-fixed-slots",
            "3",
            "--outer-graph-fuse-qkv",
            "--outer-graph-fuse-attention",
            "--outer-graph-fused-projection-kernel",
            "--outer-graph-fusion-policy",
            "strict",
        ]
    )

    assert args.outer_cuda_graph_talker_engine is True
    assert args.outer_graph_fixed_slots == 3
    assert args.outer_graph_fuse_qkv is True
    assert args.outer_graph_fuse_attention is True
    assert args.outer_graph_fused_projection_kernel is True
    assert args.outer_graph_fusion_policy == "strict"


def test_internal_profile_cli_passes_engine_flags_before_warmup_and_reports_inputs(
    monkeypatch,
    tmp_path,
):
    backend_calls = []
    close_calls = []
    duplicate_installs = []
    profile_calls = []

    class FakeBackend:
        def __init__(self, **kwargs):
            backend_calls.append(kwargs)

            def generate(**generate_kwargs):
                return generate_kwargs

            if kwargs["outer_static_talker_engine"]:
                generate._qav_static_outer_talker = True
            talker = SimpleNamespace(generate=generate)
            self._model = SimpleNamespace(
                generate_defaults={},
                model=SimpleNamespace(talker=talker),
            )

        def close(self):
            close_calls.append(True)

    monkeypatch.setattr(internal_profile, "QwenTtsBackend", FakeBackend)
    monkeypatch.setattr(internal_profile, "_set_seed", lambda seed: None)
    monkeypatch.setattr(
        internal_profile,
        "profile_qwen_tts_batch",
        lambda wrapper, **kwargs: profile_calls.append(kwargs)
        or {"batch_size": len(kwargs["texts"])},
    )
    for installer_name in (
        "install_cuda_graph_code_predictor",
        "install_static_code_predictor",
        "install_batched_fast_code_predictor",
        "install_static_outer_talker",
        "install_explicit_talker_step_engine",
    ):
        monkeypatch.setattr(
            internal_profile,
            installer_name,
            lambda *args, _name=installer_name, **kwargs: duplicate_installs.append(
                _name
            ),
            raising=False,
        )

    output_path = tmp_path / "profile.json"
    assert (
        internal_profile.main(
            [
                "--model-path",
                "/models/tts",
                "--device",
                "cpu",
                "--texts",
                "same text",
                "--batch-sizes",
                "1,2",
                "--max-new-tokens",
                "64",
                "--no-non-streaming-mode",
                "--fast-code-predictor",
                "--static-code-predictor",
                "--cuda-graph-code-predictor",
                "--cuda-graph-fixed-slots",
                "2",
                "--cuda-graph-batch-window-ms",
                "3",
                "--fast-code-predictor-batch-window-ms",
                "5",
                "--fast-code-predictor-max-batch-size",
                "4",
                "--outer-static-talker-engine",
                "--outer-graph-max-cache-len",
                "512",
                "--explicit-talker-step-engine",
                "--compile-step-engine",
                "--compile-step-engine-mode",
                "max-autotune",
                "--out",
                str(output_path),
            ]
        )
        == 0
    )

    assert backend_calls == [
        {
            "model_path": "/models/tts",
            "device": "cpu",
            "language": "chinese",
            "speaker": "",
            "warmup": True,
            "fast_code_predictor": True,
            "static_code_predictor": True,
            "cuda_graph_code_predictor": True,
            "cuda_graph_fixed_slots": 2,
            "cuda_graph_batch_window_ms": 3.0,
            "fast_code_predictor_batch_window_ms": 5.0,
            "fast_code_predictor_max_batch_size": 4,
            "outer_static_talker_engine": True,
            "outer_active_prefix_talker_engine": False,
            "outer_graph_max_cache_len": 512,
            "outer_cuda_graph_talker_engine": False,
            "outer_graph_fixed_slots": 2,
            "outer_graph_fuse_qkv": False,
            "outer_graph_fuse_attention": False,
            "outer_graph_fused_projection_kernel": False,
            "outer_graph_fusion_policy": "allow",
            "outer_graph_fusion_max_abs_error": 0.002,
            "outer_graph_fusion_max_relative_l2": 0.0002,
            "explicit_talker_step_engine": True,
            "compile_step_engine": True,
            "compile_step_engine_mode": "max-autotune",
        }
    ]
    assert duplicate_installs == []
    assert close_calls == [True]
    assert [call["texts"] for call in profile_calls] == [
        ["same text"],
        ["same text", "same text"],
    ]

    report = json.loads(output_path.read_text())
    assert report["texts"] == ["same text"]
    assert report["max_new_tokens"] == 64
    assert report["non_streaming_mode"] is False
    assert report["status"] == "success"
    assert report["warnings"] == []
    assert report["errors"] == []
    assert report["cache_overflow"] is False


def test_internal_profile_cli_passes_active_flag_and_reports_actual_engine(
    monkeypatch, tmp_path
):
    backend_calls = []

    class FakeBackend:
        def __init__(self, **kwargs):
            backend_calls.append(kwargs)

            def generate(**generate_kwargs):
                return generate_kwargs

            generate._qav_active_prefix_outer_talker = True
            generate._qav_static_outer_talker = True
            generate._qav_outer_engine = SimpleNamespace(
                metrics_snapshot=lambda: {"cache_allocations": 1}
            )
            self._model = SimpleNamespace(
                generate_defaults={},
                model=SimpleNamespace(talker=SimpleNamespace(generate=generate)),
            )

        def close(self):
            return None

        def runtime_metrics(self):
            return {
                "code_predictor": {"graph_replays": 2},
                "outer_talker": {"cache_allocations": 1},
            }

    monkeypatch.setattr(internal_profile, "QwenTtsBackend", FakeBackend)
    monkeypatch.setattr(internal_profile, "_set_seed", lambda seed: None)
    monkeypatch.setattr(
        internal_profile,
        "profile_qwen_tts_batch",
        lambda wrapper, **kwargs: {"batch_size": len(kwargs["texts"])},
    )
    output_path = tmp_path / "active.json"

    assert internal_profile.main(
        [
            "--model-path",
            "/models/tts",
            "--device",
            "cpu",
            "--batch-sizes",
            "1",
            "--outer-active-prefix-talker-engine",
            "--outer-graph-max-cache-len",
            "512",
            "--no-warmup",
            "--out",
            str(output_path),
        ]
    ) == 0

    assert backend_calls[0]["outer_active_prefix_talker_engine"] is True
    report = json.loads(output_path.read_text())
    assert report["status"] == "success"
    assert report["outer_active_prefix_talker_engine"] is True
    assert report["outer_engine"] == "active_prefix"
    assert report["runtime_metrics"] == {
        "code_predictor": {"graph_replays": 2},
        "outer_talker": {"cache_allocations": 1},
    }


@pytest.mark.parametrize(
    "conflicting_flag",
    ["--outer-static-talker-engine", "--explicit-talker-step-engine"],
)
def test_internal_profile_rejects_active_outer_conflicts_before_backend(
    monkeypatch, tmp_path, conflicting_flag
):
    backend_calls = []

    class FakeBackend:
        def __init__(self, **kwargs):
            backend_calls.append(kwargs)

    monkeypatch.setattr(internal_profile, "QwenTtsBackend", FakeBackend)

    with pytest.raises(ValueError, match="mutually exclusive"):
        internal_profile.main(
            [
                "--model-path",
                "/models/tts",
                "--outer-active-prefix-talker-engine",
                conflicting_flag,
                "--out",
                str(tmp_path / "invalid.json"),
            ]
        )

    assert backend_calls == []


def test_internal_profile_constructor_failure_preserves_active_intent(
    monkeypatch, tmp_path
):
    class FailingBackend:
        def __init__(self, **kwargs):
            assert kwargs["outer_active_prefix_talker_engine"] is True
            raise RuntimeError("active constructor failed")

    monkeypatch.setattr(internal_profile, "QwenTtsBackend", FailingBackend)
    output_path = tmp_path / "active-failure.json"

    with pytest.raises(RuntimeError, match="active constructor failed"):
        internal_profile.main(
            [
                "--model-path",
                "/models/tts",
                "--outer-active-prefix-talker-engine",
                "--out",
                str(output_path),
            ]
        )

    report = json.loads(output_path.read_text())
    assert report["status"] == "failure"
    assert report["outer_active_prefix_talker_engine"] is True
    assert report["outer_engine"] == "active_prefix"
    assert report["runtime_metrics"] == {}
    assert report["errors"] == ["RuntimeError: active constructor failed"]


def test_outer_engine_mode_reads_only_installed_metadata():
    def fail_if_snapshotted():
        raise AssertionError("mode detection must not snapshot metrics")

    def generate(**kwargs):
        return kwargs

    generate._qav_active_prefix_outer_talker = True
    generate._qav_static_outer_talker = True
    generate._qav_outer_engine = SimpleNamespace(
        metrics_snapshot=fail_if_snapshotted
    )
    talker = SimpleNamespace(generate=generate)

    assert internal_profile._outer_engine_mode(talker) == "active_prefix"


def test_outer_engine_mode_prioritises_cuda_graph_metadata():
    def generate(**kwargs):
        return kwargs

    generate._qav_cuda_graph_outer_talker = True
    generate._qav_static_outer_talker = True

    assert internal_profile._outer_engine_mode(SimpleNamespace(generate=generate)) == "cuda_graph"


def test_internal_profile_freezes_mode_and_live_metrics_before_close(
    monkeypatch, tmp_path
):
    events = []
    live_metrics = {
        "code_predictor": {"graph_replays": 3},
        "outer_talker": {
            "allocated_kv_bytes": 4096,
            "backing_capacity_tokens": 512,
        },
    }

    class FakeBackend:
        def __init__(self, **kwargs):
            def generate(**generate_kwargs):
                return generate_kwargs

            generate._qav_active_prefix_outer_talker = True
            generate._qav_static_outer_talker = True
            self._talker = SimpleNamespace(generate=generate)
            self._model = SimpleNamespace(
                generate_defaults={},
                model=SimpleNamespace(talker=self._talker),
            )

        def runtime_metrics(self):
            events.append("snapshot")
            return live_metrics

        def close(self):
            events.append("close")
            live_metrics["outer_talker"]["allocated_kv_bytes"] = 0
            live_metrics["outer_talker"]["backing_capacity_tokens"] = 0
            self._talker.generate = lambda **kwargs: kwargs

    monkeypatch.setattr(internal_profile, "QwenTtsBackend", FakeBackend)
    monkeypatch.setattr(internal_profile, "_set_seed", lambda seed: None)
    monkeypatch.setattr(
        internal_profile,
        "profile_qwen_tts_batch",
        lambda wrapper, **kwargs: {"batch_size": len(kwargs["texts"])},
    )
    output_path = tmp_path / "pre-close-metrics.json"

    assert internal_profile.main(
        [
            "--model-path",
            "/models/tts",
            "--batch-sizes",
            "1",
            "--outer-active-prefix-talker-engine",
            "--no-warmup",
            "--out",
            str(output_path),
        ]
    ) == 0

    report = json.loads(output_path.read_text())
    assert events == ["snapshot", "close"]
    assert report["outer_engine"] == "active_prefix"
    assert report["runtime_metrics"] == {
        "code_predictor": {"graph_replays": 3},
        "outer_talker": {
            "allocated_kv_bytes": 4096,
            "backing_capacity_tokens": 512,
        },
    }


def test_internal_profile_keeps_batch_primary_before_snapshot_and_close_errors(
    monkeypatch, tmp_path
):
    events = []

    class BatchError(RuntimeError):
        pass

    class SnapshotError(RuntimeError):
        pass

    class CloseError(RuntimeError):
        pass

    batch_error = BatchError("batch failed")

    class FakeBackend:
        def __init__(self, **kwargs):
            def generate(**generate_kwargs):
                return generate_kwargs

            generate._qav_active_prefix_outer_talker = True
            generate._qav_static_outer_talker = True
            self._model = SimpleNamespace(
                generate_defaults={},
                model=SimpleNamespace(talker=SimpleNamespace(generate=generate)),
            )

        def runtime_metrics(self):
            events.append("snapshot")
            raise SnapshotError("snapshot failed")

        def close(self):
            events.append("close")
            raise CloseError("close failed")

    monkeypatch.setattr(internal_profile, "QwenTtsBackend", FakeBackend)
    monkeypatch.setattr(internal_profile, "_set_seed", lambda seed: None)
    monkeypatch.setattr(
        internal_profile,
        "profile_qwen_tts_batch",
        lambda wrapper, **kwargs: (_ for _ in ()).throw(batch_error),
    )
    output_path = tmp_path / "batch-snapshot-close-failure.json"

    with pytest.raises(BatchError) as raised:
        internal_profile.main(
            [
                "--model-path",
                "/models/tts",
                "--batch-sizes",
                "1",
                "--outer-active-prefix-talker-engine",
                "--no-warmup",
                "--out",
                str(output_path),
            ]
        )

    assert raised.value is batch_error
    assert events == ["snapshot", "close"]
    report = json.loads(output_path.read_text())
    assert report["outer_engine"] == "active_prefix"
    assert report["runtime_metrics"] == {}
    assert report["errors"] == [
        "BatchError: batch failed",
        "SnapshotError: snapshot failed",
        "CloseError: close failed",
    ]


def test_internal_profile_snapshot_failure_becomes_primary_and_still_closes(
    monkeypatch, tmp_path
):
    events = []

    class SnapshotError(RuntimeError):
        pass

    class CloseError(RuntimeError):
        pass

    snapshot_error = SnapshotError("snapshot failed after successful batches")

    class FakeBackend:
        def __init__(self, **kwargs):
            self._model = SimpleNamespace(
                generate_defaults={},
                model=SimpleNamespace(talker=SimpleNamespace(generate=lambda **kw: kw)),
            )

        def runtime_metrics(self):
            events.append("snapshot")
            raise snapshot_error

        def close(self):
            events.append("close")
            raise CloseError("close failed after snapshot")

    monkeypatch.setattr(internal_profile, "QwenTtsBackend", FakeBackend)
    monkeypatch.setattr(internal_profile, "_set_seed", lambda seed: None)
    monkeypatch.setattr(
        internal_profile,
        "profile_qwen_tts_batch",
        lambda wrapper, **kwargs: {"batch_size": len(kwargs["texts"])},
    )
    output_path = tmp_path / "snapshot-primary.json"

    with pytest.raises(SnapshotError) as raised:
        internal_profile.main(
            [
                "--model-path",
                "/models/tts",
                "--batch-sizes",
                "1",
                "--no-warmup",
                "--out",
                str(output_path),
            ]
        )

    assert raised.value is snapshot_error
    assert events == ["snapshot", "close"]
    report = json.loads(output_path.read_text())
    assert report["outer_engine"] == "upstream"
    assert report["runtime_metrics"] == {}
    assert report["errors"] == [
        "SnapshotError: snapshot failed after successful batches",
        "CloseError: close failed after snapshot",
    ]


def test_internal_profile_cli_captures_unique_batch_warnings(monkeypatch, tmp_path):
    class FakeBackend:
        def __init__(self, **kwargs):
            self._model = SimpleNamespace(
                generate_defaults={},
                model=SimpleNamespace(talker=SimpleNamespace(generate=lambda **kw: kw)),
            )

        def close(self):
            return None

    def profile_with_warning(wrapper, **kwargs):
        warnings.warn("batch profile warning", RuntimeWarning)
        warnings.warn("batch profile warning", RuntimeWarning)
        return {"batch_size": len(kwargs["texts"])}

    monkeypatch.setattr(internal_profile, "QwenTtsBackend", FakeBackend)
    monkeypatch.setattr(internal_profile, "_set_seed", lambda seed: None)
    monkeypatch.setattr(
        internal_profile,
        "profile_qwen_tts_batch",
        profile_with_warning,
    )

    output_path = tmp_path / "warnings.json"
    assert (
        internal_profile.main(
            [
                "--model-path",
                "/models/tts",
                "--batch-sizes",
                "1,2",
                "--no-warmup",
                "--out",
                str(output_path),
            ]
        )
        == 0
    )

    report = json.loads(output_path.read_text())
    assert report["status"] == "success"
    assert report["warnings"] == ["batch profile warning"]
    assert report["errors"] == []
    assert report["cache_overflow"] is False
    assert [item["batch_size"] for item in report["reports"]] == [1, 2]


def test_internal_profile_cli_writes_cache_overflow_failure_before_reraising(
    monkeypatch,
    tmp_path,
):
    class FakeBackend:
        def __init__(self, **kwargs):
            def generate(**generate_kwargs):
                return generate_kwargs

            generate._qav_static_outer_talker = True
            self._model = SimpleNamespace(
                generate_defaults={},
                model=SimpleNamespace(talker=SimpleNamespace(generate=generate)),
            )

        def close(self):
            return None

    def profile_then_overflow(wrapper, **kwargs):
        batch_size = len(kwargs["texts"])
        if batch_size == 1:
            return {"batch_size": 1}
        raise OuterTalkerEngineError(
            "requested prompt and decode tokens exceed static cache capacity"
        )

    monkeypatch.setattr(internal_profile, "QwenTtsBackend", FakeBackend)
    monkeypatch.setattr(internal_profile, "_set_seed", lambda seed: None)
    monkeypatch.setattr(
        internal_profile,
        "profile_qwen_tts_batch",
        profile_then_overflow,
    )

    output_path = tmp_path / "failure.json"
    with pytest.raises(OuterTalkerEngineError, match="static cache capacity"):
        internal_profile.main(
            [
                "--model-path",
                "/models/tts",
                "--texts",
                "same text",
                "--batch-sizes",
                "1,2",
                "--max-new-tokens",
                "64",
                "--outer-static-talker-engine",
                "--no-warmup",
                "--out",
                str(output_path),
            ]
        )

    report = json.loads(output_path.read_text())
    assert report["status"] == "failure"
    assert report["warnings"] == []
    assert report["errors"] == [
        "OuterTalkerEngineError: requested prompt and decode tokens exceed "
        "static cache capacity"
    ]
    assert report["cache_overflow"] is True
    assert report["reports"] == [{"batch_size": 1}]
    assert report["texts"] == ["same text"]
    assert report["max_new_tokens"] == 64
    assert report["outer_static_talker_engine"] is True


def test_internal_profile_cli_captures_constructor_warnings_and_closes_backend(
    monkeypatch,
    tmp_path,
):
    close_calls = []

    class FakeBackend:
        def __init__(self, **kwargs):
            assert kwargs["warmup"] is True
            warnings.warn("synchronous warmup warning", RuntimeWarning)
            self._model = SimpleNamespace(
                generate_defaults={},
                model=SimpleNamespace(talker=SimpleNamespace(generate=lambda **kw: kw)),
            )

        def close(self):
            close_calls.append(True)

    monkeypatch.setattr(internal_profile, "QwenTtsBackend", FakeBackend)
    monkeypatch.setattr(internal_profile, "_set_seed", lambda seed: None)
    monkeypatch.setattr(
        internal_profile,
        "profile_qwen_tts_batch",
        lambda wrapper, **kwargs: {"batch_size": len(kwargs["texts"])},
    )

    output_path = tmp_path / "constructor-warning.json"
    assert (
        internal_profile.main(
            [
                "--model-path",
                "/models/tts",
                "--batch-sizes",
                "1",
                "--out",
                str(output_path),
            ]
        )
        == 0
    )

    report = json.loads(output_path.read_text())
    assert report["status"] == "success"
    assert report["warnings"] == ["synchronous warmup warning"]
    assert report["errors"] == []
    assert close_calls == [True]


def test_internal_profile_cli_writes_constructor_cache_overflow_failure(
    monkeypatch,
    tmp_path,
):
    class FailingBackend:
        def __init__(self, **kwargs):
            assert kwargs["warmup"] is True
            warnings.warn("warmup reached static capacity", RuntimeWarning)
            raise OuterTalkerEngineError(
                "requested prompt and decode tokens exceed static cache capacity"
            )

    monkeypatch.setattr(internal_profile, "QwenTtsBackend", FailingBackend)

    output_path = tmp_path / "constructor-failure.json"
    with pytest.raises(OuterTalkerEngineError, match="static cache capacity"):
        internal_profile.main(
            [
                "--model-path",
                "/models/tts",
                "--outer-static-talker-engine",
                "--out",
                str(output_path),
            ]
        )

    report = json.loads(output_path.read_text())
    assert report["status"] == "failure"
    assert report["outer_engine"] == "static"
    assert report["warnings"] == ["warmup reached static capacity"]
    assert report["errors"] == [
        "OuterTalkerEngineError: requested prompt and decode tokens exceed "
        "static cache capacity"
    ]
    assert report["cache_overflow"] is True
    assert report["reports"] == []
    assert report["warmup"] is True


def test_internal_profile_cli_reports_close_failure_after_successful_batches(
    monkeypatch,
    tmp_path,
):
    class FakeBackend:
        def __init__(self, **kwargs):
            self._model = SimpleNamespace(
                generate_defaults={},
                model=SimpleNamespace(talker=SimpleNamespace(generate=lambda **kw: kw)),
            )

        def close(self):
            raise RuntimeError("backend close failed")

    monkeypatch.setattr(internal_profile, "QwenTtsBackend", FakeBackend)
    monkeypatch.setattr(internal_profile, "_set_seed", lambda seed: None)
    monkeypatch.setattr(
        internal_profile,
        "profile_qwen_tts_batch",
        lambda wrapper, **kwargs: {"batch_size": len(kwargs["texts"])},
    )

    output_path = tmp_path / "close-failure.json"
    with pytest.raises(RuntimeError, match="backend close failed"):
        internal_profile.main(
            [
                "--model-path",
                "/models/tts",
                "--batch-sizes",
                "1",
                "--no-warmup",
                "--out",
                str(output_path),
            ]
        )

    report = json.loads(output_path.read_text())
    assert report["status"] == "failure"
    assert report["errors"] == ["RuntimeError: backend close failed"]
    assert report["cache_overflow"] is False
    assert report["reports"] == [{"batch_size": 1}]


def test_internal_profile_cli_preserves_primary_batch_failure_when_close_also_fails(
    monkeypatch,
    tmp_path,
):
    constructed = []
    close_calls = []

    class PrimaryBatchError(RuntimeError):
        pass

    class SecondaryCloseError(OuterTalkerEngineError):
        pass

    primary_error = PrimaryBatchError("batch profile failed before shutdown")

    class FakeBackend:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self._model = SimpleNamespace(
                generate_defaults={},
                model=SimpleNamespace(talker=SimpleNamespace(generate=lambda **kw: kw)),
            )

        def close(self):
            close_calls.append(True)
            raise SecondaryCloseError(
                "requested prompt and decode tokens exceed static cache capacity"
            )

    def profile_fails(wrapper, **kwargs):
        raise primary_error

    monkeypatch.setattr(internal_profile, "QwenTtsBackend", FakeBackend)
    monkeypatch.setattr(internal_profile, "_set_seed", lambda seed: None)
    monkeypatch.setattr(internal_profile, "profile_qwen_tts_batch", profile_fails)

    output_path = tmp_path / "batch-and-close-failure.json"
    with pytest.raises(PrimaryBatchError) as raised:
        internal_profile.main(
            [
                "--model-path",
                "/models/tts",
                "--batch-sizes",
                "1",
                "--outer-static-talker-engine",
                "--no-warmup",
                "--out",
                str(output_path),
            ]
        )

    assert raised.value is primary_error
    assert len(constructed) == 1
    assert close_calls == [True]

    report = json.loads(output_path.read_text())
    assert report["status"] == "failure"
    assert report["errors"] == [
        "PrimaryBatchError: batch profile failed before shutdown",
        "SecondaryCloseError: requested prompt and decode tokens exceed "
        "static cache capacity",
    ]
    assert report["cache_overflow"] is False
    assert report["reports"] == []


def test_profile_copies_static_outer_engine_metrics_snapshot():
    wrapper = FakeWrapper()
    live_metrics = {"static_steps_by_batch": {1: 5}, "active_slots_per_step": [1]}
    engine = SimpleNamespace(metrics_snapshot=lambda: live_metrics)

    def static_generate(**kwargs):
        raise AssertionError("profile should use the model generation path")

    static_generate._qav_static_outer_talker = True
    static_generate._qav_outer_engine = engine
    wrapper.model.talker.generate = static_generate

    report = profile_qwen_tts_batch(
        wrapper,
        texts=["first fragment"],
        chunk_size=2,
        left_context_size=1,
    )
    live_metrics["static_steps_by_batch"][1] = 99
    live_metrics["active_slots_per_step"].append(0)

    assert report["outer_engine"] == "static"
    assert report["outer_engine_metrics"] == {
        "static_steps_by_batch": {1: 5},
        "active_slots_per_step": [1],
    }


def test_internal_profile_seed_reproduces_torch_sampling():
    import torch

    _set_seed(123)
    first = torch.rand(4)
    _set_seed(123)
    second = torch.rand(4)

    assert torch.equal(first, second)
