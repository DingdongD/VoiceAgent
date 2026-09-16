import sys
from pathlib import Path

import torch


NANOVLLM_ROOT = Path("/home/nano-vllm")
if str(NANOVLLM_ROOT) not in sys.path:
    sys.path.append(str(NANOVLLM_ROOT))


def test_sampling_params_accept_zero_temperature_for_greedy():
    from nanovllm.sampling_params import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=8)

    assert params.temperature == 0.0


def test_sampler_uses_argmax_for_zero_temperature_rows():
    from nanovllm.layers.sampler import Sampler

    sampler = Sampler()
    logits = torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float32)

    outputs = [
        sampler(logits.clone(), torch.tensor([0.0])).item() for _ in range(20)
    ]

    assert outputs == [2] * 20


def test_voice_adapter_preserves_zero_temperature():
    from qwen_asr_vllm.agent.nano_llm import NanoVllmStepBatchingBackend

    captured = {}

    def sampling_params(**kwargs):
        captured.update(kwargs)
        return kwargs

    backend = object.__new__(NanoVllmStepBatchingBackend)
    backend._sampling_cls = sampling_params
    backend._temperature = 0.0
    backend._max_new_tokens = 8

    backend._sampling_params()

    assert captured["temperature"] == 0.0
