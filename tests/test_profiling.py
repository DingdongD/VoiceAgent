"""Stage timing, batch phase attribution, and the frontend thread cap."""

from __future__ import annotations

import pytest
import torch

from qwen_asr_vllm.engine.engine import AsrEngine, _set_current_cuda_device
from qwen_asr_vllm.engine.model_runner import ModelRunner
from qwen_asr_vllm.engine.scheduler import ModelBatch
from qwen_asr_vllm.profiling import NULL_TIMER, StageTimer

from .helpers import make_request


class TestDisabledTimer:
    """The default timer sits in the per-batch path, so it must cost and do nothing."""

    def test_a_disabled_phase_records_nothing(self):
        timer = StageTimer(enabled=False)
        with timer.phase("anything"):
            pass
        assert timer.totals == {}
        assert timer.counts == {}

    def test_the_shared_null_timer_is_disabled(self):
        assert NULL_TIMER.enabled is False

    def test_a_disabled_phase_does_not_touch_cuda(self, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("a disabled timer must not create CUDA events")

        monkeypatch.setattr(torch.cuda, "Event", fail)
        with StageTimer(enabled=False).phase("anything"):
            pass

    def test_a_phase_propagates_exceptions(self):
        timer = StageTimer(enabled=False)
        with pytest.raises(ValueError, match="boom"), timer.phase("anything"):
            raise ValueError("boom")


class TestPhaseAttribution:
    """A batch's device time has to land on the stage that actually produced it."""

    def _batch(self, num_prefill: int, num_decode: int) -> ModelBatch:
        return ModelBatch(
            prefill=[make_request(prompt_len=64) for _ in range(num_prefill)],
            decode=[make_request(prompt_len=64) for _ in range(num_decode)],
        )

    def test_prefill_only(self):
        assert ModelRunner._phase_name(self._batch(2, 0)) == "llm_prefill"

    def test_decode_only(self):
        assert ModelRunner._phase_name(self._batch(0, 4)) == "llm_decode"

    def test_a_mixed_batch_is_neither(self):
        # Splitting shared kernel launches between the two would invent a number, and
        # it would flatter whichever stage got charged less.
        assert ModelRunner._phase_name(self._batch(1, 3)) == "llm_mixed"


class TestFrontendThreadCap:
    """``frontend_threads`` narrows torch's intra-op pool; see EngineConfig."""

    class Stub:
        def __init__(self, frontend_threads: int):
            self.config = type("Config", (), {"frontend_threads": frontend_threads})()

        limit = AsrEngine._limit_frontend_threads

    def _apply(self, wanted: int, current: int, monkeypatch) -> int | None:
        applied: list[int] = []
        monkeypatch.setattr(torch, "get_num_threads", lambda: current)
        monkeypatch.setattr(torch, "set_num_threads", applied.append)
        self.Stub(wanted).limit()
        return applied[0] if applied else None

    def test_a_wide_pool_is_narrowed(self, monkeypatch):
        assert self._apply(wanted=8, current=24, monkeypatch=monkeypatch) == 8

    def test_an_already_narrow_pool_is_left_alone(self, monkeypatch):
        # A caller who asked for fewer threads than we would has a reason.
        assert self._apply(wanted=8, current=4, monkeypatch=monkeypatch) is None

    def test_an_equal_pool_is_left_alone(self, monkeypatch):
        assert self._apply(wanted=8, current=8, monkeypatch=monkeypatch) is None

    def test_zero_opts_out(self, monkeypatch):
        assert self._apply(wanted=0, current=24, monkeypatch=monkeypatch) is None

    def test_the_default_avoids_the_pathological_width(self):
        from qwen_asr_vllm.config import EngineConfig

        # 24 threads made a 2s clip's mel extraction 100x slower than 8 did.
        assert EngineConfig.frontend_threads == 8


def test_set_current_cuda_device_uses_configured_index(monkeypatch):
    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)

    _set_current_cuda_device("cuda:3")

    assert str(selected) == "[device(type='cuda', index=3)]"


@pytest.mark.gpu
class TestEnabledTimer:
    def test_phases_accumulate_and_reset(self):
        timer = StageTimer(enabled=True)
        for _ in range(3):
            with timer.phase("work"):
                torch.randn(256, 256, device="cuda") @ torch.randn(256, 256, device="cuda")

        assert timer.counts["work"] == 3
        assert timer.totals["work"] > 0
        timer.reset()
        assert timer.totals == {}

    def test_separate_phases_stay_separate(self):
        small = torch.randn(64, 64, device="cuda")
        large = torch.randn(4096, 4096, device="cuda")
        # Warm up both shapes first: whichever phase runs a kernel for the first time
        # is charged that kernel's one-off load, which is larger than either matmul.
        for _ in range(3):
            small.sum()
            large @ large
        torch.cuda.synchronize()

        timer = StageTimer(enabled=True)
        with timer.phase("small"):
            small.sum()
        with timer.phase("large"):
            large @ large

        assert set(timer.totals) == {"small", "large"}
        assert timer.totals["large"] > timer.totals["small"]
