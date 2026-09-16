"""A request that cannot ever fit in the KV cache must not stall the scheduler.

The prefill loop stops at the first request it cannot place, which is the right
policy for transient pressure: blocks free up as decodes finish. But when nothing
is running there is nothing to wait for, and a request needing more blocks than the
cache holds in total would keep the scheduler reporting work forever while producing
empty batches -- a silent hang rather than an error.
"""
import pytest

from qwen_asr_vllm.engine.block_manager import BlockManager
from qwen_asr_vllm.engine.scheduler import Scheduler

from .helpers import BLOCK_SIZE, admit_for_prefill, make_config, make_request, start_decoding


@pytest.fixture
def scheduler_with_two_blocks():
    manager = BlockManager(num_blocks=2, block_size=BLOCK_SIZE)
    return Scheduler(make_config(), manager), manager


def test_unplaceable_request_is_aborted_not_stalled(scheduler_with_two_blocks):
    scheduler, _ = scheduler_with_two_blocks
    # Three blocks' worth of prompt against a two-block cache: waiting cannot help.
    request = admit_for_prefill(scheduler, make_request(prompt_len=3 * BLOCK_SIZE))

    assert not scheduler.schedule_model()

    assert scheduler.drain_aborted() == [request]
    assert request.finish_reason == "aborted:kv_cache_too_small"
    assert not scheduler.has_work


def test_transient_pressure_still_waits_instead_of_aborting(scheduler_with_two_blocks):
    scheduler, manager = scheduler_with_two_blocks
    # This one fits in the cache, but not while the other request holds a block.
    start_decoding(scheduler, manager, make_request(prompt_len=BLOCK_SIZE))
    blocked = admit_for_prefill(scheduler, make_request(prompt_len=2 * BLOCK_SIZE))

    scheduler.schedule_model()

    assert not scheduler.aborted
    assert list(scheduler.waiting_prefill) == [blocked]


def test_placeable_request_is_scheduled(scheduler_with_two_blocks):
    scheduler, _ = scheduler_with_two_blocks
    request = admit_for_prefill(scheduler, make_request(prompt_len=2 * BLOCK_SIZE))

    batch = scheduler.schedule_model()

    assert batch.prefill == [request]
    assert not scheduler.aborted
