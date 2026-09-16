import sys
import types

import numpy as np
import pytest

from qwen_asr_vllm.agent.local_tts import (
    QwenTtsBackend,
    QwenTtsDependencyError,
    check_qwen_tts_runtime,
)
import qwen_asr_vllm.agent.local_tts as local_tts


def test_check_qwen_tts_runtime_reports_missing_imports_and_sox():
    def missing_spec(name):
        return None

    def missing_binary(name):
        return None

    try:
        check_qwen_tts_runtime(find_spec=missing_spec, which=missing_binary)
    except QwenTtsDependencyError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected QwenTtsDependencyError")

    assert "qwen-tts" in message
    assert "onnxruntime" in message
    assert "sox" in message
    assert "--no-deps" in message


def test_qwen_tts_backend_uses_local_model_and_encodes_wav(monkeypatch):
    class FakeModel:
        calls = []

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert path == "/models/tts"
            assert kwargs["device_map"] == "cpu"
            return cls()

        def get_supported_speakers(self):
            return ["alice"]

        def generate_custom_voice(self, text, speaker, language, **kwargs):
            assert speaker == "alice"
            assert language == "english"
            self.calls.append((text, kwargs))
            if isinstance(text, list):
                return [np.zeros(160, dtype=np.float32) for _ in text], 16000
            return [np.zeros(160, dtype=np.float32)], 16000

    fake_module = types.SimpleNamespace(Qwen3TTSModel=FakeModel)
    monkeypatch.setitem(sys.modules, "qwen_tts", fake_module)

    backend = QwenTtsBackend(
        model_path="/models/tts",
        device="cpu",
        language="english",
        warmup=False,
        check_runtime=False,
    )
    audio = backend.synthesize("hello")

    assert audio[:4] == b"RIFF"
    batch = backend.synthesize_batch(["hello", "world"])
    assert len(batch) == 2
    assert all(item[:4] == b"RIFF" for item in batch)
    streamed = backend.synthesize_with_mode("stream?", non_streaming_mode=False)
    assert streamed[:4] == b"RIFF"
    assert backend._model.calls[-1] == ("stream?", {"non_streaming_mode": False})


def test_qwen_tts_backend_rejects_negative_eos_before_model_load():
    with pytest.raises(ValueError, match="eos_token_id must be non-negative"):
        QwenTtsBackend(
            model_path="/models/tts",
            device="cpu",
            warmup=False,
            check_runtime=False,
            eos_token_id=-1,
        )


def test_qwen_tts_backend_applies_explicit_generation_controls(monkeypatch):
    class FakeModel:
        model = types.SimpleNamespace(talker=object())
        calls = []

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

        def get_supported_speakers(self):
            return ["alice"]

        def generate_custom_voice(self, text, speaker, language, **kwargs):
            self.calls.append(kwargs)
            count = len(text) if isinstance(text, list) else 1
            return [np.zeros(160, dtype=np.float32) for _ in range(count)], 16000

    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        types.SimpleNamespace(Qwen3TTSModel=FakeModel),
    )

    backend = QwenTtsBackend(
        model_path="/models/tts",
        device="cpu",
        warmup=False,
        check_runtime=False,
        do_sample=False,
        temperature=0.0,
        max_new_tokens=64,
        eos_token_id=2150,
    )
    backend.synthesize("hello")
    backend.synthesize_batch(["hello", "world"])
    backend.synthesize_with_mode("stream", non_streaming_mode=False)

    expected = {
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 64,
        "eos_token_id": 2150,
    }
    assert all(
        {key: call[key] for key in expected}
        == expected
        for call in FakeModel.calls
    )


def test_qwen_tts_backend_can_install_fast_code_predictor(monkeypatch):
    installed = []
    outer_installed = []

    class FakeModel:
        model = types.SimpleNamespace(talker=object())

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

        def get_supported_speakers(self):
            return ["alice"]

        def generate_custom_voice(self, text, speaker, language, **kwargs):
            return [np.zeros(160, dtype=np.float32)], 16000

    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        types.SimpleNamespace(Qwen3TTSModel=FakeModel),
    )
    monkeypatch.setattr(
        local_tts,
        "install_batched_fast_code_predictor",
        lambda talker, **kwargs: installed.append((talker, kwargs)) or True,
    )
    monkeypatch.setattr(
        local_tts,
        "install_explicit_talker_step_engine",
        lambda talker, **kwargs: outer_installed.append((talker, kwargs)) or True,
    )

    QwenTtsBackend(
        model_path="/models/tts",
        device="cpu",
        language="english",
        warmup=False,
        check_runtime=False,
        fast_code_predictor=True,
        fast_code_predictor_batch_window_ms=5,
        fast_code_predictor_max_batch_size=3,
        explicit_talker_step_engine=True,
        compile_step_engine=True,
        compile_step_engine_mode="max-autotune",
    )

    assert installed == [
        (
                FakeModel.model.talker,
            {
                "batch_window_ms": 5.0,
                "max_batch_size": 3,
                "compile_step": True,
                "compile_mode": "max-autotune",
            },
        )
    ]
    assert outer_installed == [
        (
            FakeModel.model.talker,
            {"compile_step": True, "compile_mode": "max-autotune"},
        )
    ]


def test_qwen_tts_backend_prefers_static_code_predictor(monkeypatch):
    static_installed = []
    fast_installed = []

    class FakeModel:
        model = types.SimpleNamespace(talker=object())

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

        def get_supported_speakers(self):
            return ["alice"]

    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        types.SimpleNamespace(Qwen3TTSModel=FakeModel),
    )
    monkeypatch.setattr(
        local_tts,
        "install_static_code_predictor",
        lambda talker: static_installed.append(talker) or True,
    )
    monkeypatch.setattr(
        local_tts,
        "install_batched_fast_code_predictor",
        lambda talker, **kwargs: fast_installed.append((talker, kwargs)) or True,
    )

    QwenTtsBackend(
        model_path="/models/tts",
        device="cpu",
        warmup=False,
        check_runtime=False,
        fast_code_predictor=True,
        static_code_predictor=True,
    )

    assert static_installed == [FakeModel.model.talker]
    assert fast_installed == []


def test_qwen_tts_backend_prefers_cuda_graph_code_predictor(monkeypatch):
    graph_installed = []
    static_installed = []

    class FakeModel:
        model = types.SimpleNamespace(talker=object())

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

        def get_supported_speakers(self):
            return ["alice"]

    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        types.SimpleNamespace(Qwen3TTSModel=FakeModel),
    )
    monkeypatch.setattr(
        local_tts,
        "install_cuda_graph_code_predictor",
        lambda talker, **kwargs: graph_installed.append((talker, kwargs)) or True,
    )
    monkeypatch.setattr(
        local_tts,
        "install_static_code_predictor",
        lambda talker: static_installed.append(talker) or True,
    )

    QwenTtsBackend(
        model_path="/models/tts",
        device="cpu",
        warmup=False,
        check_runtime=False,
        static_code_predictor=True,
        cuda_graph_code_predictor=True,
        cuda_graph_fixed_slots=2,
        cuda_graph_batch_window_ms=3,
    )

    assert graph_installed == [
        (
            FakeModel.model.talker,
            {"fixed_slot_count": 2, "batch_window_ms": 3.0},
        )
    ]
    assert static_installed == []


def test_qwen_tts_backend_installs_static_outer_engine_after_selected_predictor(
    monkeypatch,
):
    installed = []

    class FakeModel:
        model = types.SimpleNamespace(talker=object())
        warmup_calls = []

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

        def get_supported_speakers(self):
            return ["alice"]

        def generate_custom_voice(self, text, speaker, language, **kwargs):
            self.warmup_calls.append((text, kwargs))
            return [np.zeros(160, dtype=np.float32)], 16000

    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        types.SimpleNamespace(Qwen3TTSModel=FakeModel),
    )
    monkeypatch.setattr(
        local_tts,
        "install_cuda_graph_code_predictor",
        lambda talker, **kwargs: installed.append(("predictor", talker, kwargs))
        or True,
    )
    monkeypatch.setattr(
        local_tts,
        "install_static_outer_talker",
        lambda talker, **kwargs: installed.append(("outer", talker, kwargs)) or True,
    )

    QwenTtsBackend(
        model_path="/models/tts",
        device="cpu",
        warmup=True,
        check_runtime=False,
        cuda_graph_code_predictor=True,
        outer_static_talker_engine=True,
        outer_graph_max_cache_len=512,
    )

    assert installed == [
        (
            "predictor",
            FakeModel.model.talker,
            {"fixed_slot_count": 1, "batch_window_ms": 0.0},
        ),
        ("outer", FakeModel.model.talker, {"max_cache_len": 512}),
    ]
    assert FakeModel.warmup_calls[-1] == ("你好", {"max_new_tokens": 64})


def test_qwen_tts_backend_installs_outer_cuda_graph_with_fusion_options(monkeypatch):
    installed = []

    class FakeModel:
        model = types.SimpleNamespace(talker=object())

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

        def get_supported_speakers(self):
            return ["alice"]

        def generate_custom_voice(self, text, speaker, language, **kwargs):
            installed.append(("warmup", kwargs))
            return [np.zeros(160, dtype=np.float32)], 16000

    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        types.SimpleNamespace(Qwen3TTSModel=FakeModel),
    )
    monkeypatch.setattr(
        local_tts,
        "install_cuda_graph_outer_talker",
        lambda talker, **kwargs: installed.append(("outer", kwargs)) or True,
    )

    QwenTtsBackend(
        model_path="/models/tts",
        device="cpu",
        warmup=True,
        check_runtime=False,
        outer_cuda_graph_talker_engine=True,
        outer_graph_max_cache_len=512,
        outer_graph_fixed_slots=3,
        outer_graph_fuse_qkv=True,
        outer_graph_fuse_attention=True,
    )

    assert installed == [
        (
            "outer",
            {
                "max_cache_len": 512,
                "max_graph_batch_size": 3,
                "fuse_qkv": True,
                "fuse_attention": True,
                "fused_projection_kernel": False,
                "fusion_policy": "allow",
                "fusion_max_abs_error": 0.002,
                "fusion_max_relative_l2": 0.0002,
            },
        ),
        ("warmup", {"max_new_tokens": 64}),
    ]


@pytest.mark.parametrize(
    "conflicting_flag",
    ["outer_static_talker_engine", "explicit_talker_step_engine"],
)
def test_qwen_tts_backend_rejects_active_outer_conflicts_before_runtime_or_loading(
    monkeypatch, conflicting_flag
):
    runtime_checks = []
    model_loads = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            model_loads.append((path, kwargs))
            return cls()

    monkeypatch.setattr(
        local_tts,
        "check_qwen_tts_runtime",
        lambda: runtime_checks.append(True),
    )
    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        types.SimpleNamespace(Qwen3TTSModel=FakeModel),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        QwenTtsBackend(
            model_path="/models/tts",
            outer_active_prefix_talker_engine=True,
            **{conflicting_flag: True},
        )

    assert runtime_checks == []
    assert model_loads == []


def test_qwen_tts_backend_installs_active_prefix_after_predictor_before_warmup(
    monkeypatch,
):
    events = []

    class FakeModel:
        model = types.SimpleNamespace(talker=object())

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            events.append("load")
            return cls()

        def get_supported_speakers(self):
            return ["alice"]

        def generate_custom_voice(self, text, speaker, language, **kwargs):
            events.append(("warmup", text, kwargs))
            return [np.zeros(160, dtype=np.float32)], 16000

    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        types.SimpleNamespace(Qwen3TTSModel=FakeModel),
    )
    monkeypatch.setattr(
        local_tts,
        "install_cuda_graph_code_predictor",
        lambda talker, **kwargs: events.append(("predictor", kwargs)) or True,
    )
    monkeypatch.setattr(
        local_tts,
        "install_active_prefix_outer_talker",
        lambda talker, **kwargs: events.append(("active", kwargs)) or True,
        raising=False,
    )

    QwenTtsBackend(
        model_path="/models/tts",
        device="cpu",
        check_runtime=False,
        warmup=True,
        cuda_graph_code_predictor=True,
        outer_active_prefix_talker_engine=True,
        outer_graph_max_cache_len=512,
    )

    assert events == [
        "load",
        (
            "predictor",
            {"fixed_slot_count": 1, "batch_window_ms": 0.0},
        ),
        ("active", {"max_cache_len": 512}),
        ("warmup", "你好", {"max_new_tokens": 64}),
    ]


def test_qwen_tts_backend_close_attempts_predictor_and_outer_and_keeps_first_error():
    close_events = []

    class PredictorScheduler:
        def close(self):
            close_events.append("predictor")
            raise RuntimeError("predictor close failed")

    class OuterRuntime:
        def close(self):
            close_events.append("outer")
            raise RuntimeError("outer close failed")

    def predictor_generate(**kwargs):
        return kwargs

    predictor_generate._qav_scheduler = PredictorScheduler()

    def outer_generate(**kwargs):
        return kwargs

    outer_generate._qav_outer_engine = types.SimpleNamespace(
        step_runtime=OuterRuntime()
    )
    backend = object.__new__(QwenTtsBackend)
    backend._model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            talker=types.SimpleNamespace(
                generate=outer_generate,
                code_predictor=types.SimpleNamespace(generate=predictor_generate),
            )
        )
    )

    with pytest.raises(RuntimeError, match="predictor close failed"):
        backend.close()

    assert close_events == ["predictor", "outer"]


def test_qwen_tts_backend_runtime_metrics_are_deep_plain_snapshots():
    predictor_metrics = {"batches": [1], "nested": {"captures": 2}}
    outer_metrics = {"active_kv_tokens_per_step": [3], "errors": 0}
    scheduler = types.SimpleNamespace(
        metrics_snapshot=lambda: predictor_metrics,
    )
    outer_engine = types.SimpleNamespace(
        metrics_snapshot=lambda: outer_metrics,
        step_runtime=types.SimpleNamespace(close=lambda: None),
    )

    def predictor_generate(**kwargs):
        return kwargs

    predictor_generate._qav_scheduler = scheduler

    def outer_generate(**kwargs):
        return kwargs

    outer_generate._qav_outer_engine = outer_engine
    backend = object.__new__(QwenTtsBackend)
    backend._model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            talker=types.SimpleNamespace(
                generate=outer_generate,
                code_predictor=types.SimpleNamespace(generate=predictor_generate),
            )
        )
    )

    snapshot = backend.runtime_metrics()

    assert snapshot == {
        "code_predictor": {"batches": [1], "nested": {"captures": 2}},
        "outer_talker": {"active_kv_tokens_per_step": [3], "errors": 0},
    }
    snapshot["code_predictor"]["batches"].append(99)
    snapshot["outer_talker"]["active_kv_tokens_per_step"].append(99)
    assert backend.runtime_metrics() == {
        "code_predictor": {"batches": [1], "nested": {"captures": 2}},
        "outer_talker": {"active_kv_tokens_per_step": [3], "errors": 0},
    }
    assert all(
        value is not scheduler and value is not outer_engine
        for value in snapshot.values()
    )


def test_qwen_tts_backend_runtime_metrics_reject_resource_objects():
    resource = types.SimpleNamespace(close=lambda: None)
    scheduler = types.SimpleNamespace(
        metrics_snapshot=lambda: {"engine": resource},
    )

    def predictor_generate(**kwargs):
        return kwargs

    predictor_generate._qav_scheduler = scheduler
    backend = object.__new__(QwenTtsBackend)
    backend._model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            talker=types.SimpleNamespace(
                generate=lambda **kwargs: kwargs,
                code_predictor=types.SimpleNamespace(generate=predictor_generate),
            )
        )
    )

    with pytest.raises(TypeError, match="plain data"):
        backend.runtime_metrics()
