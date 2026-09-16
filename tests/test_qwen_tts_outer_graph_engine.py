from types import SimpleNamespace
from threading import RLock

import pytest
import torch

import qwen_asr_vllm.agent.qwen_tts_outer_graph_engine as graph_module

from qwen_asr_vllm.agent.qwen_tts_outer_graph_engine import (
    CUDAGraphOuterStepRuntime,
    OuterGraphEngineError,
    install_cuda_graph_outer_talker,
)
from qwen_asr_vllm.agent.qwen_tts_outer_static_engine import (
    OuterStepOutput,
    PreparedOuterDecode,
)


class FakeCache:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class FakeBundle:
    def __init__(self, events, kwargs):
        self.events = events
        self.kwargs = kwargs
        self.cache = FakeCache()
        self.lock = RLock()
        self.replays = 0

    def run(self, prepared):
        self.events.append("replay-main-model")
        self.replays += 1
        batch = prepared.codec_ids.shape[0]
        hidden = prepared.condition + 1
        return OuterStepOutput(
            logits=torch.zeros(batch, 1, 8),
            past_hidden=hidden,
            hidden_states=None,
        )


def _prepared(batch_size=2, marker=1):
    return PreparedOuterDecode(
        codec_ids=torch.full((batch_size, 3), marker, dtype=torch.long),
        condition=torch.zeros(batch_size, 1, 4),
        attention_mask=torch.zeros(batch_size, 1, 1, 8),
        position_ids=torch.zeros(3, batch_size, 1, dtype=torch.long),
        cache_position=torch.tensor([3], dtype=torch.long),
        output_hidden_states=False,
    )


def test_outer_graph_runtime_reuses_fixed_shape_bundle_and_keeps_predictor_outside():
    events = []
    created = []

    def factory(**kwargs):
        events.append("capture-main-model")
        created.append(kwargs)
        return FakeBundle(events, kwargs)

    runtime = CUDAGraphOuterStepRuntime(
        SimpleNamespace(config=SimpleNamespace()),
        bundle_factory=factory,
        allow_non_cuda_for_testing=True,
    )
    first = runtime.acquire(2, torch.device("cpu"), torch.float32, 8)
    first.run(_prepared(marker=1))
    first.release()
    second = runtime.acquire(2, torch.device("cpu"), torch.float32, 8)
    second.run(_prepared(marker=2))
    second.release()

    assert events == ["capture-main-model", "replay-main-model", "replay-main-model"]
    assert created[0]["batch_size"] == 2
    assert runtime.metrics_snapshot() == {
        "graph_captures": 0,
        "graph_replays": 2,
        "graph_errors": 0,
        "bundles": 1,
    }


def test_outer_graph_runtime_separates_batch_shape_bundles():
    created = []
    runtime = CUDAGraphOuterStepRuntime(
        SimpleNamespace(config=SimpleNamespace()),
        bundle_factory=lambda **kwargs: created.append(kwargs) or FakeBundle([], kwargs),
        allow_non_cuda_for_testing=True,
    )

    one = runtime.acquire(1, torch.device("cpu"), torch.float32, 8)
    two = runtime.acquire(2, torch.device("cpu"), torch.float32, 8)
    one.release()
    two.release()

    assert [item["batch_size"] for item in created] == [1, 2]


def test_outer_graph_runtime_uses_active_prefix_attention_mask():
    runtime = CUDAGraphOuterStepRuntime(
        SimpleNamespace(config=SimpleNamespace()),
        allow_non_cuda_for_testing=True,
    )

    mask = runtime.build_attention_mask(
        torch.tensor([[1, 1, 0]], dtype=torch.long),
        cache_position=3,
        max_cache_len=8,
        dtype=torch.float32,
    )

    assert mask.shape == (1, 1, 1, 4)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [[[[True, True, False, True]]]]


def test_outer_graph_replay_advances_active_prefix_host_length():
    bundle = object.__new__(graph_module._OuterGraphBundle)
    bundle.cache = SimpleNamespace(
        layers=[
            SimpleNamespace(active_length=3),
            SimpleNamespace(active_length=3),
        ]
    )

    bundle._set_active_cache_length(4)

    assert [layer.active_length for layer in bundle.cache.layers] == [4, 4]


def test_outer_graph_runtime_rejects_cpu_without_explicit_test_override():
    runtime = CUDAGraphOuterStepRuntime(SimpleNamespace(config=SimpleNamespace()))

    with pytest.raises(OuterGraphEngineError, match="CUDA"):
        runtime.acquire(1, torch.device("cpu"), torch.float32, 8)


def test_outer_graph_runtime_does_not_fallback_above_configured_batch_limit():
    runtime = CUDAGraphOuterStepRuntime(
        SimpleNamespace(config=SimpleNamespace()),
        max_graph_batch_size=1,
        allow_non_cuda_for_testing=True,
    )

    with pytest.raises(OuterGraphEngineError, match="batch size"):
        runtime.acquire(2, torch.device("cpu"), torch.float32, 8)


def test_outer_graph_lease_releases_bundle_after_run_error():
    class FailingBundle(FakeBundle):
        def run(self, prepared):
            raise RuntimeError("graph replay failed")

    runtime = CUDAGraphOuterStepRuntime(
        SimpleNamespace(config=SimpleNamespace()),
        bundle_factory=lambda **kwargs: FailingBundle([], kwargs),
        allow_non_cuda_for_testing=True,
    )
    lease = runtime.acquire(1, torch.device("cpu"), torch.float32, 8)
    with pytest.raises(RuntimeError, match="graph replay failed"):
        lease.run(_prepared(batch_size=1))
    lease.release()

    replacement = runtime.acquire(1, torch.device("cpu"), torch.float32, 8)
    replacement.release()


def test_outer_graph_runtime_closes_cached_bundles():
    bundles = []

    def factory(**kwargs):
        bundle = FakeBundle([], kwargs)
        bundle.closed = False
        bundle.close = lambda: setattr(bundle, "closed", True)
        bundles.append(bundle)
        return bundle

    runtime = CUDAGraphOuterStepRuntime(
        SimpleNamespace(config=SimpleNamespace()),
        bundle_factory=factory,
        allow_non_cuda_for_testing=True,
    )
    lease = runtime.acquire(1, torch.device("cpu"), torch.float32, 8)
    lease.release()

    runtime.close()

    assert bundles[0].closed is True
    assert runtime.metrics_snapshot()["bundles"] == 0


def test_install_cuda_graph_outer_talker_marks_engine_and_closes_runtime(monkeypatch):
    class Talker:
        def generate(self, **kwargs):
            return kwargs

    talker = Talker()
    closed = []

    class FakeRuntime:
        def __init__(self, *args, **kwargs):
            self._attention_patch = None

        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        "qwen_asr_vllm.agent.qwen_tts_outer_graph_engine.CUDAGraphOuterStepRuntime",
        FakeRuntime,
    )

    assert install_cuda_graph_outer_talker(talker, max_graph_batch_size=3) is True
    assert talker.generate._qav_cuda_graph_outer_talker is True
    assert talker.generate._qav_static_outer_talker is True
    assert talker.generate._qav_outer_engine.step_runtime is not None
    talker.generate._qav_outer_engine.step_runtime.close()
    assert closed == [True]
