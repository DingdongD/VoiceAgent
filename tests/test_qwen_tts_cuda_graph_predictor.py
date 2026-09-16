import threading
from types import SimpleNamespace
from contextlib import contextmanager

import pytest
import torch

from qwen_asr_vllm.agent.qwen_tts_cuda_graph_predictor import (
    CUDAGraphCodePredictorEngine,
    FixedSlotCUDAGraphBatcher,
    _decode_generation_step,
    install_cuda_graph_code_predictor,
)
from qwen_asr_vllm.agent.qwen_tts_fast_predictor import FastCodePredictorError


class FakeCodePredictor:
    def generate(self, **kwargs):
        raise AssertionError("original generate should not be called")


class FakeGraphBundle:
    def __init__(self, token: int):
        self.token = token
        self.calls = 0
        self.closed = False

    def generate(self, **kwargs):
        assert not self.closed
        self.calls += 1
        batch_size = kwargs["inputs_embeds"].shape[0]
        steps = kwargs["max_new_tokens"]
        return SimpleNamespace(
            sequences=torch.full((batch_size, steps), self.token, dtype=torch.long)
        )

    def close(self):
        self.closed = True


def test_cuda_graph_engine_reuses_shape_specific_bundle():
    predictor = FakeCodePredictor()
    created = []

    def bundle_factory(*, code_predictor, inputs_embeds, max_new_tokens, output_hidden_states=False):
        assert code_predictor is predictor
        created.append((tuple(inputs_embeds.shape), max_new_tokens))
        return FakeGraphBundle(token=7)

    engine = CUDAGraphCodePredictorEngine(
        predictor,
        bundle_factory=bundle_factory,
        allow_non_cuda_for_testing=True,
    )

    first = engine.generate(
        inputs_embeds=torch.zeros(1, 2, 4),
        max_new_tokens=3,
        do_sample=False,
    )
    second = engine.generate(
        inputs_embeds=torch.ones(1, 2, 4),
        max_new_tokens=3,
        do_sample=False,
    )

    assert first.sequences.tolist() == [[7, 7, 7]]
    assert second.sequences.tolist() == [[7, 7, 7]]
    assert created == [((1, 2, 4), 3)]
    assert engine.graph_captures == 1
    assert engine.graph_replays == 2


def test_cuda_graph_engine_captures_output_hidden_states_mode():
    captured_modes = []

    def bundle_factory(*, code_predictor, inputs_embeds, max_new_tokens, output_hidden_states):
        captured_modes.append(bool(output_hidden_states))
        return FakeGraphBundle(token=7)

    engine = CUDAGraphCodePredictorEngine(
        FakeCodePredictor(),
        bundle_factory=bundle_factory,
        allow_non_cuda_for_testing=True,
    )

    engine.generate(
        inputs_embeds=torch.zeros(1, 2, 4),
        max_new_tokens=3,
        do_sample=False,
        output_hidden_states=True,
    )

    assert captured_modes == [True]


def test_cuda_graph_engine_captures_and_replays_in_input_device_context():
    state = {"active": False, "entries": 0}

    @contextmanager
    def device_context_factory(inputs_embeds):
        assert tuple(inputs_embeds.shape) == (1, 2, 4)
        state["active"] = True
        state["entries"] += 1
        try:
            yield
        finally:
            state["active"] = False

    class ContextCheckingBundle(FakeGraphBundle):
        def generate(self, **kwargs):
            assert state["active"] is True
            return super().generate(**kwargs)

    def bundle_factory(**kwargs):
        assert state["active"] is True
        return ContextCheckingBundle(token=6)

    engine = CUDAGraphCodePredictorEngine(
        FakeCodePredictor(),
        bundle_factory=bundle_factory,
        device_context_factory=device_context_factory,
        allow_non_cuda_for_testing=True,
    )

    result = engine.generate(
        inputs_embeds=torch.zeros(1, 2, 4),
        max_new_tokens=2,
        do_sample=False,
    )

    assert result.sequences.tolist() == [[6, 6]]
    assert state == {"active": False, "entries": 2}


def test_cuda_graph_engine_rejects_above_graph_batch_limit_without_fallback():
    engine = CUDAGraphCodePredictorEngine(
        FakeCodePredictor(),
        max_graph_batch_size=1,
        bundle_factory=lambda **kwargs: FakeGraphBundle(token=7),
        allow_non_cuda_for_testing=True,
    )

    with pytest.raises(FastCodePredictorError, match="exceeds max_graph_batch_size"):
        engine.generate(
            inputs_embeds=torch.zeros(2, 2, 4),
            max_new_tokens=3,
            do_sample=False,
        )
    assert engine.graph_captures == 0


def test_cuda_graph_engine_prewarms_fixed_slot_bundle_from_batch_one():
    created = []

    def bundle_factory(*, code_predictor, inputs_embeds, max_new_tokens, output_hidden_states=False):
        created.append((tuple(inputs_embeds.shape), max_new_tokens))
        return FakeGraphBundle(token=inputs_embeds.shape[0])

    engine = CUDAGraphCodePredictorEngine(
        FakeCodePredictor(),
        max_graph_batch_size=2,
        prewarm_batch_sizes=(2,),
        bundle_factory=bundle_factory,
        allow_non_cuda_for_testing=True,
    )

    first = engine.generate(
        inputs_embeds=torch.zeros(1, 2, 4),
        max_new_tokens=3,
        do_sample=False,
    )
    second = engine.generate(
        inputs_embeds=torch.zeros(2, 2, 4),
        max_new_tokens=3,
        do_sample=False,
    )

    assert first.sequences.tolist() == [[1, 1, 1]]
    assert second.sequences.tolist() == [[2, 2, 2], [2, 2, 2]]
    assert created == [((1, 2, 4), 3), ((2, 2, 4), 3)]
    assert engine.graph_captures == 2


def test_cuda_graph_engine_explicit_prewarm_makes_first_request_steady_state():
    created = []

    def bundle_factory(*, code_predictor, inputs_embeds, max_new_tokens, output_hidden_states=False):
        created.append(tuple(inputs_embeds.shape))
        return FakeGraphBundle(token=8)

    engine = CUDAGraphCodePredictorEngine(
        FakeCodePredictor(),
        bundle_factory=bundle_factory,
        allow_non_cuda_for_testing=True,
    )
    inputs = torch.zeros(1, 2, 4)

    engine.prewarm(inputs_embeds=inputs, max_new_tokens=3)
    engine.mark_steady_state()
    assert engine.graph_captures == 1
    result = engine.generate(
        inputs_embeds=inputs,
        max_new_tokens=3,
        do_sample=False,
    )

    assert result.sequences.tolist() == [[8, 8, 8]]
    assert created == [(1, 2, 4)]
    assert engine.metrics_snapshot()["lazy_captures"] == 0


def test_cuda_graph_engine_close_releases_bundles():
    bundles = []

    def bundle_factory(**kwargs):
        bundle = FakeGraphBundle(token=9)
        bundles.append(bundle)
        return bundle

    engine = CUDAGraphCodePredictorEngine(
        FakeCodePredictor(),
        bundle_factory=bundle_factory,
        allow_non_cuda_for_testing=True,
    )
    engine.generate(
        inputs_embeds=torch.zeros(1, 2, 4),
        max_new_tokens=2,
        do_sample=False,
    )
    engine.close()

    assert engine.metrics_snapshot()["bundles"] == 0
    assert bundles[0].closed is True


class RecordingGraphEngine:
    def __init__(self):
        self.batch_sizes = []

    def generate(self, **kwargs):
        inputs = kwargs["inputs_embeds"]
        self.batch_sizes.append(inputs.shape[0])
        values = inputs[:, 0, 0].to(dtype=torch.long).unsqueeze(1)
        sequences = values.repeat(1, kwargs["max_new_tokens"])
        if kwargs.get("return_dict_in_generate", True):
            return SimpleNamespace(sequences=sequences)
        return sequences


def test_fixed_slot_batcher_combines_two_concurrent_request_ids():
    engine = RecordingGraphEngine()
    batcher = FixedSlotCUDAGraphBatcher(
        engine,
        slot_count=2,
        batch_window_ms=50,
    )
    barrier = threading.Barrier(3)
    results = {}

    def run(request_id, value):
        barrier.wait()
        results[request_id] = batcher.generate(
            inputs_embeds=torch.full((1, 2, 4), value),
            max_new_tokens=3,
            do_sample=False,
        )

    threads = [
        threading.Thread(target=run, args=("a", 4.0)),
        threading.Thread(target=run, args=("b", 9.0)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)
    batcher.close()

    assert engine.batch_sizes == [2]
    assert results["a"].sequences.tolist() == [[4, 4, 4]]
    assert results["b"].sequences.tolist() == [[9, 9, 9]]
    assert batcher.request_batches == [2]
    assert batcher.slot_batches == [2]


def test_fixed_slot_batcher_flushes_single_request_without_padding():
    engine = RecordingGraphEngine()
    batcher = FixedSlotCUDAGraphBatcher(
        engine,
        slot_count=2,
        batch_window_ms=1,
    )

    result = batcher.generate(
        inputs_embeds=torch.full((1, 2, 4), 6.0),
        max_new_tokens=2,
        do_sample=False,
    )
    batcher.close()

    assert result.sequences.tolist() == [[6, 6]]
    assert engine.batch_sizes == [1]
    assert batcher.request_batches == [1]
    assert batcher.slot_batches == [1]


def test_fixed_slot_batcher_pads_partial_multi_request_batch_to_slot_count():
    engine = RecordingGraphEngine()
    batcher = FixedSlotCUDAGraphBatcher(
        engine,
        slot_count=4,
        batch_window_ms=20,
    )
    barrier = threading.Barrier(3)
    results = []

    def run(value):
        barrier.wait()
        results.append(
            batcher.generate(
                inputs_embeds=torch.full((1, 2, 4), value),
                max_new_tokens=2,
                do_sample=False,
            ).sequences.tolist()
        )

    threads = [threading.Thread(target=run, args=(value,)) for value in (2.0, 3.0)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)
    batcher.close()

    assert engine.batch_sizes == [4]
    assert sorted(results) == [[[2, 2]], [[3, 3]]]
    assert batcher.request_batches == [2]
    assert batcher.slot_batches == [4]
    assert batcher.padded_slots == 2


def test_install_cuda_graph_code_predictor_replaces_generate_once():
    predictor = FakeCodePredictor()
    talker = SimpleNamespace(code_predictor=predictor)
    factory = lambda **kwargs: FakeGraphBundle(token=5)

    assert install_cuda_graph_code_predictor(
        talker,
        bundle_factory=factory,
        allow_non_cuda_for_testing=True,
    ) is True
    assert install_cuda_graph_code_predictor(
        talker,
        bundle_factory=factory,
        allow_non_cuda_for_testing=True,
    ) is False

    result = predictor.generate(
        inputs_embeds=torch.zeros(1, 2, 4),
        max_new_tokens=2,
        do_sample=False,
    )

    assert result.sequences.tolist() == [[5, 5]]


def test_decode_generation_step_preserves_absolute_qwen_tts_mtp_head_index():
    assert [_decode_generation_step(15, step) for step in (1, 2, 3)] == [15, 16, 17]
    assert _decode_generation_step(1, 1) == 1
