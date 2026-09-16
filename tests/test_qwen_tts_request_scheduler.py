from types import SimpleNamespace

import pytest
import torch

from qwen_asr_vllm.agent.qwen_tts_outer_active_prefix_cache import ActivePrefixCache
from qwen_asr_vllm.agent.qwen_tts_request_scheduler import (
    _FixedSlotJoinedCachePool,
    _inner_compute_policy,
    joined_cache,
    RequestIdCodecScheduler,
)


def cohort(value, length=2):
    cache = ActivePrefixCache(1, 8)
    keys = torch.full((1, 2, length, 3), value)
    cache.update(keys, keys + 1, 0, {"cache_position": torch.arange(length)})
    return SimpleNamespace(cache=cache, cache_position=length)


def test_joined_cache_writes_each_request_and_preserves_row_order():
    a, b = cohort(2.0), cohort(8.0)
    cache = joined_cache([b, a])
    assert cache.get_seq_length() == 2
    assert cache.is_compileable is False
    keys = torch.stack([torch.full((2, 1, 3), 11.0), torch.full((2, 1, 3), 17.0)])
    result, _ = cache.update(keys, keys + 1, 0, {"cache_position": torch.tensor([2])})
    assert result.shape == (2, 2, 3, 3)
    assert torch.equal(result[0, :, :2], torch.full((2, 2, 3), 8.0))
    assert torch.equal(result[1, :, :2], torch.full((2, 2, 3), 2.0))
    assert torch.equal(a.cache.layers[0].keys[:, :, 2:3], keys[1:2])
    assert torch.equal(b.cache.layers[0].keys[:, :, 2:3], keys[0:1])
    assert a.cache.get_seq_length() == b.cache.get_seq_length() == 3


def test_joined_cache_copies_only_new_kv_after_cohort_initialization():
    a, b = cohort(2.0), cohort(8.0)
    cache = joined_cache([a, b])
    layer = cache.layers[0]
    keys = torch.stack(
        [torch.full((2, 1, 3), 11.0), torch.full((2, 1, 3), 17.0)]
    )

    cache.update(keys, keys + 1, 0, {"cache_position": torch.tensor([2])})
    first_copy_bytes = layer.copy_bytes
    cache.update(keys + 1, keys + 2, 0, {"cache_position": torch.tensor([3])})

    incremental_copy_bytes = layer.copy_bytes - first_copy_bytes
    expected = 2 * 2 * 1 * 3 * 2 * keys.element_size()
    assert incremental_copy_bytes == expected


def test_joined_cache_rejects_different_lengths_without_padding_or_fallback():
    with pytest.raises(ValueError, match="matching KV lengths"):
        joined_cache([cohort(2.0), cohort(8.0, length=3)])


def test_fixed_slot_pool_preserves_request_id_slots_when_membership_changes():
    first, second, newcomer = (
        cohort(2.0, length=3),
        cohort(8.0, length=3),
        cohort(14.0, length=3),
    )
    first.request_id = 1
    second.request_id = 2
    newcomer.request_id = 3
    pool = _FixedSlotJoinedCachePool([(1, first), (2, second)])
    keys = torch.stack(
        [torch.full((2, 1, 3), 11.0), torch.full((2, 1, 3), 17.0)]
    )
    pool.cache.update(keys, keys + 1, 0, {"cache_position": torch.tensor([3])})
    for layer in pool.cache.layers:
        layer.copy_bytes = 0

    ordered = pool.bind([(2, second), (3, newcomer)])
    # Request 2 retains physical row 1. Request 3 occupies the released row 0.
    assert [request_id for request_id, _cohort in ordered] == [3, 2]
    assert pool.request_to_slot == {3: 0, 2: 1}
    assert pool.last_reused_rows == 1
    assert pool.last_new_rows == 1
    pool.cache.update(keys + 1, keys + 2, 0, {"cache_position": torch.tensor([4])})

    layer = pool.cache.layers[0]
    # The old row is detached and the newcomer is materialized during bind;
    # the decode update itself writes directly through the aliases.
    assert layer.copy_bytes == 2 * 2 * (4 + 3) * 3 * keys.element_size()
    assert torch.equal(
        layer.joined_keys[0, :, :4], newcomer.cache.layers[0].keys[0, :, :4]
    )
    assert torch.equal(
        layer.joined_keys[1, :, :5], second.cache.layers[0].keys[0, :, :5]
    )


def test_fixed_slot_pool_aliases_active_cache_rows_and_materializes_only_new_rows():
    first, second, newcomer = (
        cohort(2.0, length=3),
        cohort(8.0, length=3),
        cohort(14.0, length=3),
    )
    pool = _FixedSlotJoinedCachePool([(1, first), (2, second)])
    layer = pool.cache.layers[0]
    assert layer.sources[0].keys.data_ptr() == layer.joined_keys.data_ptr()
    assert layer.sources[1].keys.data_ptr() == layer.joined_keys[1:].data_ptr()

    keys = torch.stack(
        [torch.full((2, 1, 3), 11.0), torch.full((2, 1, 3), 17.0)]
    )
    layer.copy_bytes = 0
    pool.cache.update(keys, keys + 1, 0, {"cache_position": torch.tensor([3])})
    assert layer.copy_bytes == 0

    pool.bind([(2, second), (3, newcomer)])
    layer.copy_bytes = 0
    pool.cache.update(keys + 1, keys + 2, 0, {"cache_position": torch.tensor([4])})
    assert layer.copy_bytes == 0
    assert layer.sources[1].keys.data_ptr() == layer.joined_keys[1:].data_ptr()
    assert first.cache.layers[0].keys.data_ptr() != layer.joined_keys.data_ptr()


def test_fixed_slot_pool_transfers_source_ownership_between_batch_shapes():
    first, second = cohort(2.0, length=3), cohort(8.0, length=3)
    two = _FixedSlotJoinedCachePool([(1, first), (2, second)])
    one = _FixedSlotJoinedCachePool([(2, second)])

    two_layer = two.cache.layers[0]
    one_layer = one.cache.layers[0]
    assert one_layer.sources[0] is second.cache.layers[0]
    assert one_layer.sources[0].keys.data_ptr() == one_layer.joined_keys.data_ptr()
    assert two_layer._aliased[1] is False
    assert two_layer._aliased[0] is True
    assert first.cache.layers[0]._qav_joined_owner is two_layer

    keys = torch.full((1, 2, 1, 3), 19.0)
    one.cache.update(keys, keys + 1, 0, {"cache_position": torch.tensor([3])})
    assert torch.equal(
        one_layer.joined_keys[0, :, :4], second.cache.layers[0].keys[0, :, :4]
    )


def test_scheduler_selects_same_cache_position_cohort_and_rotates_ties():
    scheduler = object.__new__(RequestIdCodecScheduler)
    scheduler.active = {
        1: SimpleNamespace(
            request_id=1,
            pending={
                "cohort": SimpleNamespace(cache_position=4),
                "output_hidden_states": True,
            },
        ),
        2: SimpleNamespace(
            request_id=2,
            pending={
                "cohort": SimpleNamespace(cache_position=4),
                "output_hidden_states": True,
            },
        ),
        3: SimpleNamespace(
            request_id=3,
            pending={
                "cohort": SimpleNamespace(cache_position=5),
                "output_hidden_states": True,
            },
        ),
    }
    scheduler._cohort_cursor = 0

    selected = scheduler._select_step_cohort()
    assert [request.request_id for request in selected] == [1, 2]

    scheduler.active = {
        1: scheduler.active[1],
        3: scheduler.active[3],
    }
    scheduler._cohort_cursor = 0
    first = scheduler._select_step_cohort()
    scheduler.active[1].pending["cohort"].cache_position = 5
    second = scheduler._select_step_cohort()
    assert [request.request_id for request in first] == [1]
    assert [request.request_id for request in second] == [1, 3]


def test_scheduler_catches_up_shorter_prefix_before_longer_cohort():
    scheduler = object.__new__(RequestIdCodecScheduler)
    scheduler.active = {
        1: SimpleNamespace(
            request_id=1,
            pending={
                "cohort": SimpleNamespace(cache_position=4),
                "output_hidden_states": True,
            },
        ),
        2: SimpleNamespace(
            request_id=2,
            pending={
                "cohort": SimpleNamespace(cache_position=5),
                "output_hidden_states": True,
            },
        ),
        3: SimpleNamespace(
            request_id=3,
            pending={
                "cohort": SimpleNamespace(cache_position=5),
                "output_hidden_states": True,
            },
        ),
    }
    scheduler._cohort_cursor = 0

    selected = scheduler._select_step_cohort()
    assert [request.request_id for request in selected] == [1]

    scheduler.active[1].pending["cohort"].cache_position = 5
    selected = scheduler._select_step_cohort()
    assert [request.request_id for request in selected] == [1, 2, 3]


def test_inner_compute_policy_restores_backend_flags():
    before = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )
    with _inner_compute_policy("highest", sdp_backend="math", deterministic=True):
        assert torch.get_float32_matmul_precision() == "highest"
        assert torch.backends.cuda.flash_sdp_enabled() is False
        assert torch.backends.cuda.mem_efficient_sdp_enabled() is False
        assert torch.backends.cuda.math_sdp_enabled() is True
        assert torch.are_deterministic_algorithms_enabled() is True
    after = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )
    assert after == before


def test_inner_compute_policy_rejects_invalid_backend_without_mutation():
    before = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
    )
    with pytest.raises(ValueError, match="SDPA backend"):
        with _inner_compute_policy("highest", sdp_backend="invalid"):
            pass
    after = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
    )
    assert after == before
