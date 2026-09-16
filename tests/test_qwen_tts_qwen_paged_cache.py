import pytest
import torch

from qwen_asr_vllm.agent.qwen_tts_outer_paged_cache import PagedOuterCache
from qwen_asr_vllm.agent.qwen_tts_outer_varlen import (
    QwenPagedCache,
    VarlenMetadataError,
)


def _new_cache():
    pool = PagedOuterCache(num_layers=2, num_pages=8, page_size=2)
    pool.allocate("short", max_seq_len=6)
    pool.allocate("long", max_seq_len=6)
    return pool, QwenPagedCache(pool, num_layers=2)


def _prime(pool, request_id, length):
    keys = torch.arange(2 * 1 * 2 * length * 2, dtype=torch.float32).reshape(2, 1, 2, length, 2)
    pool.append(request_id, keys, keys + 100, torch.arange(length, dtype=torch.long))


def test_qwen_paged_cache_writes_each_layer_and_returns_padded_rows():
    pool, cache = _new_cache()
    cache.bind_rows(("short", "long"), torch.tensor([[0, 1], [0, 1]], dtype=torch.long))

    keys = torch.arange(2 * 2 * 2 * 2, dtype=torch.float32).reshape(2, 2, 2, 2)
    values = keys + 100
    returned = []
    for layer in range(2):
        returned.append(cache.update(keys + layer * 10, values + layer * 10, layer))

    short_keys, short_values = pool.read("short", layer_index=1, logical_length=2)
    long_keys, long_values = pool.read("long", layer_index=1, logical_length=2)
    assert returned[-1][0].shape == (2, 2, 2, 2)
    assert torch.equal(short_keys[0], keys[0] + 10)
    assert torch.equal(long_keys[0], keys[1] + 10)
    assert torch.equal(short_values[0], values[0] + 10)
    assert torch.equal(long_values[0], values[1] + 10)
    assert pool.snapshot()["live_requests"] == 2


def test_qwen_paged_cache_masks_independent_row_lengths():
    pool, cache = _new_cache()
    _prime(pool, "short", 2)
    _prime(pool, "long", 4)
    cache.bind_rows(("short", "long"), torch.tensor([[2], [4]], dtype=torch.long))
    keys = torch.ones((2, 2, 1, 2), dtype=torch.float32)
    cache.update(keys, keys, 0)

    mask = cache.attention_mask(dtype=torch.float32)
    assert mask.shape == (2, 1, 1, 5)
    assert torch.equal(mask[0, 0, 0, :3], torch.zeros(3))
    assert torch.equal(mask[0, 0, 0, 3:], torch.full((2,), torch.finfo(torch.float32).min))
    assert torch.equal(mask[1, 0, 0], torch.zeros(5))


def test_qwen_paged_cache_preserves_prompt_padding_for_later_decode():
    pool, cache = _new_cache()
    _prime(pool, "short", 2)
    cache.bind_rows(
        ("short",),
        torch.tensor([[2]], dtype=torch.long),
        key_valid_mask=torch.tensor([[False, True, True]], dtype=torch.bool),
    )
    keys = torch.ones((1, 2, 1, 2), dtype=torch.float32)
    cache.update(keys, keys, 0)

    mask = cache.attention_mask(dtype=torch.float32)
    assert torch.equal(mask[0, 0, 0, :], torch.tensor([torch.finfo(torch.float32).min, 0.0, 0.0]))


def test_qwen_paged_cache_requires_contiguous_row_mapping_and_position_shape():
    pool, cache = _new_cache()
    with pytest.raises(VarlenMetadataError, match="request IDs"):
        cache.bind_rows(("short", "short"), torch.tensor([[0], [0]], dtype=torch.long))
    with pytest.raises(VarlenMetadataError, match="shape"):
        cache.bind_rows(("short", "long"), torch.tensor([0, 1], dtype=torch.long))

    cache.bind_rows(("short", "long"), torch.tensor([[0], [0]], dtype=torch.long))
    keys = torch.ones((2, 2, 1, 2), dtype=torch.float32)
    with pytest.raises(VarlenMetadataError, match="batch rows"):
        cache.update(keys[:1], keys[:1], 0)


def test_kernel_metadata_is_staged_once_per_bound_decode_tick():
    pool = PagedOuterCache(num_layers=1, num_pages=16, page_size=2)
    pool.allocate("a", max_seq_len=8)
    cache = QwenPagedCache(pool, num_layers=1, use_paged_attention_kernel=True)

    cache.bind_rows(("a",), torch.tensor([[0]], dtype=torch.long))

    assert cache._kernel_page_table is not None
    assert cache._kernel_valid_mask is not None
    assert cache._kernel_page_table.shape == (1, 4)
    assert cache._kernel_valid_mask.shape == (1, 128)
