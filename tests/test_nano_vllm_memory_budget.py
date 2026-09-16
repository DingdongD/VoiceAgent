import sys
from types import SimpleNamespace

import torch


if "/home/nano-vllm" not in sys.path:
    sys.path.append("/home/nano-vllm")

from nanovllm.engine.model_runner import ModelRunner


def test_nano_vllm_allocator_preserves_explicit_kv_blocks(monkeypatch):
    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        num_kvcache_blocks=64,
        gpu_memory_utilization=0.5,
        kvcache_block_size=256,
    )
    runner.text_config = SimpleNamespace(
        num_key_value_heads=8,
        num_hidden_layers=28,
        hidden_size=1024,
        num_attention_heads=16,
        head_dim=128,
    )
    runner.world_size = 1
    runner.block_size = 256
    runner.model_dtype = torch.float16
    runner.model = SimpleNamespace(modules=lambda: [])
    allocated_shapes = []

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (30 * 2**30, 40 * 2**30))
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda: {
            "allocated_bytes.all.peak": 3 * 2**30,
            "allocated_bytes.all.current": 2 * 2**30,
        },
    )

    class FakeTensor:
        def __getitem__(self, _key):
            return self

    def fake_empty(*shape, **_kwargs):
        allocated_shapes.append(shape)
        return FakeTensor()

    monkeypatch.setattr(torch, "empty", fake_empty)

    runner.allocate_kv_cache()

    assert runner.config.num_kvcache_blocks == 64
    assert allocated_shapes[0][2] == 64
