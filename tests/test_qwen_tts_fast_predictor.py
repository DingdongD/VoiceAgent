import threading
import time
from types import SimpleNamespace

import pytest
import torch

from qwen_asr_vllm.agent.qwen_tts_fast_predictor import (
    FastCodePredictorError,
    install_batched_fast_code_predictor,
    fast_code_predictor_generate,
)


class FakeCodePredictor:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        raise AssertionError("original generate should not be called")

    def __call__(self, **kwargs):
        step = len(self.calls)
        self.calls.append(kwargs)
        batch_size = kwargs.get("inputs_embeds", kwargs.get("input_ids")).shape[0]
        logits = torch.zeros(batch_size, 1, 8)
        logits[:, :, step + 1] = 10.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=f"past-{step}",
            generation_steps=step + 1,
        )


def test_fast_code_predictor_generate_runs_fixed_step_greedy_loop():
    predictor = FakeCodePredictor()

    result = fast_code_predictor_generate(
        predictor,
        inputs_embeds=torch.zeros(2, 2, 4),
        max_new_tokens=3,
        do_sample=False,
        return_dict_in_generate=True,
    )

    assert result.sequences.tolist() == [[1, 2, 3], [1, 2, 3]]
    assert len(predictor.calls) == 3
    assert "inputs_embeds" in predictor.calls[0]
    assert predictor.calls[1]["input_ids"].tolist() == [[1], [1]]
    assert predictor.calls[1]["past_key_values"] == "past-0"
    assert predictor.calls[1]["generation_steps"] == 1


def test_batched_fast_code_predictor_can_use_compiled_predictor_callable():
    predictor = BatchedFakeCodePredictor()
    compiled = []

    def compiler(fn, **kwargs):
        compiled.append((fn, kwargs))

        def wrapped(**call_kwargs):
            return fn(**call_kwargs)

        return wrapped

    talker = SimpleNamespace(code_predictor=predictor)
    install_batched_fast_code_predictor(
        talker,
        compile_step=True,
        compile_mode="reduce-overhead",
        compiler=compiler,
    )

    result = predictor.generate(
        inputs_embeds=torch.zeros(1, 2, 4),
        max_new_tokens=2,
        do_sample=False,
    )

    assert result.sequences.tolist() == [[1, 2]]
    assert compiled == [(predictor, {"mode": "reduce-overhead"})]


def test_fast_code_predictor_rejects_unknown_generation_kwargs():
    with pytest.raises(FastCodePredictorError, match="num_beams"):
        fast_code_predictor_generate(
            FakeCodePredictor(),
            inputs_embeds=torch.zeros(1, 2, 4),
            max_new_tokens=2,
            num_beams=2,
        )


class BatchedFakeCodePredictor:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        raise AssertionError("original generate should not be called")

    def __call__(self, **kwargs):
        batch_source = kwargs.get("inputs_embeds", kwargs.get("input_ids"))
        batch_size = batch_source.shape[0]
        self.calls.append(batch_size)
        token_base = len(self.calls)
        logits = torch.zeros(batch_size, 1, 16)
        for index in range(batch_size):
            logits[index, :, token_base + index] = 10.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=_legacy_cache(batch_size, len(self.calls)),
            generation_steps=len(self.calls),
        )


def _legacy_cache(batch_size: int, value: int):
    key = torch.full((batch_size, 1, 1, 1), float(value))
    val = torch.full((batch_size, 1, 1, 1), float(value + 10))
    return ((key, val),)


def test_batched_fast_code_predictor_batches_concurrent_request_steps():
    predictor = BatchedFakeCodePredictor()
    talker = SimpleNamespace(code_predictor=predictor)
    install_batched_fast_code_predictor(
        talker,
        batch_window_ms=50,
        max_batch_size=4,
    )
    barrier = threading.Barrier(3)
    results = {}

    def run_request(name: str, offset: float):
        barrier.wait()
        time.sleep(offset)
        results[name] = predictor.generate(
            inputs_embeds=torch.zeros(1, 2, 4),
            max_new_tokens=3,
            do_sample=False,
        ).sequences.tolist()

    left = threading.Thread(target=run_request, args=("left", 0.0))
    right = threading.Thread(target=run_request, args=("right", 0.01))
    left.start()
    right.start()
    barrier.wait()
    left.join(timeout=2)
    right.join(timeout=2)

    assert sorted(results) == ["left", "right"]
    assert all(len(value[0]) == 3 for value in results.values())
    assert sorted(value[0][0] for value in results.values()) == [1, 2]
    assert predictor.calls == [2, 2, 2]
