import pytest
import torch

from qwen_asr_vllm.agent.qwen_tts_outer_paged_cache import (
    PageTable,
    PagedCacheError,
    PagedOuterCache,
)


def _states(layers=2, batch=1, heads=2, steps=2, dim=3, offset=0):
    values = torch.arange(
        layers * batch * heads * steps * dim, dtype=torch.float32
    ).reshape(layers, batch, heads, steps, dim) + offset
    return values, values + 1000


def test_allocate_append_read_round_trip_across_page_boundary():
    cache = PagedOuterCache(num_layers=2, num_pages=4, page_size=2)
    table = cache.allocate("a", max_seq_len=4, batch_size=1)
    keys, values = _states(steps=4)

    cache.append("a", keys, values, torch.arange(4, dtype=torch.long))
    read_keys, read_values = cache.read("a", layer_index=1, logical_length=4)

    assert isinstance(table, PageTable)
    assert table.request_id == "a"
    assert read_keys.shape == (1, 2, 4, 3)
    assert torch.equal(read_keys, keys[1])
    assert torch.equal(read_values, values[1])
    assert cache.snapshot()["live_pages"] == 2


def test_pages_are_reused_only_after_release():
    cache = PagedOuterCache(num_layers=1, num_pages=2, page_size=2)
    first = cache.allocate("first", max_seq_len=3, batch_size=1)
    cache.release("first")
    second = cache.allocate("second", max_seq_len=3, batch_size=1)

    assert second.page_ids == first.page_ids
    assert cache.snapshot()["allocations"] == 2
    assert cache.snapshot()["releases"] == 1


def test_duplicate_request_and_capacity_overflow_are_rejected():
    cache = PagedOuterCache(num_layers=1, num_pages=2, page_size=2)
    cache.allocate("a", max_seq_len=4, batch_size=1)
    with pytest.raises(PagedCacheError, match="already allocated"):
        cache.allocate("a", max_seq_len=1, batch_size=1)
    with pytest.raises(PagedCacheError, match="insufficient free pages"):
        cache.allocate("b", max_seq_len=1, batch_size=1)


def test_append_rejects_non_contiguous_positions_and_cross_request_access():
    cache = PagedOuterCache(num_layers=1, num_pages=4, page_size=2)
    cache.allocate("a", max_seq_len=4, batch_size=1)
    cache.allocate("b", max_seq_len=2, batch_size=1)
    keys, values = _states(layers=1, steps=1)
    cache.append("a", keys, values, torch.tensor([0], dtype=torch.long))

    with pytest.raises(PagedCacheError, match="contiguous"):
        cache.append("a", keys, values, torch.tensor([2], dtype=torch.long))
    with pytest.raises(PagedCacheError, match="does not belong"):
        cache.read_page("b", page_id=cache._records["a"].table.page_ids[0], layer_index=0)


def test_release_is_idempotent_only_for_known_released_request():
    cache = PagedOuterCache(num_layers=1, num_pages=2, page_size=2)
    cache.allocate("a", max_seq_len=1, batch_size=1)
    cache.release("a")
    cache.release("a")
    with pytest.raises(PagedCacheError, match="unknown request"):
        cache.release("missing")


def test_commit_decode_advances_rows_after_fused_write():
    cache = PagedOuterCache(num_layers=1, num_pages=4, page_size=2)
    cache.allocate("a", max_seq_len=4)
    keys, values = _states(layers=1, steps=1)
    cache.append("a", keys, values, torch.tensor([0], dtype=torch.long))

    # The fused decode kernel writes position 1 before the host commits it.
    cache.commit_decode(("a",), torch.tensor([1], dtype=torch.long))

    assert cache.page_table("a").seq_len == 2
    with pytest.raises(PagedCacheError, match="continue request"):
        cache.commit_decode(("a",), torch.tensor([1], dtype=torch.long))
