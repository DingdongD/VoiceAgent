"""Preemption, abort and out-of-memory recovery.

These paths only fire under resource pressure, so a throughput benchmark on a big
card never reaches them -- every recorded sweep so far reports zero preemptions.
Driving the scheduler directly against a deliberately tiny cache is the only way to
hold them to any behaviour at all.
"""
import pytest

from qwen_asr_vllm.engine.block_manager import BlockManager
from qwen_asr_vllm.engine.request import RequestStage
from qwen_asr_vllm.engine.scheduler import ModelBatch, Scheduler

from .helpers import BLOCK_SIZE, admit_for_prefill, make_config, make_request, start_decoding


@pytest.fixture
def scheduler_and_manager():
    def build(num_blocks: int, **config_overrides):
        config = make_config(**config_overrides)
        manager = BlockManager(num_blocks=num_blocks, block_size=BLOCK_SIZE)
        return Scheduler(config, manager), manager

    return build


class TestPreemption:
    def test_preempts_the_newest_request_to_keep_the_oldest_going(self, scheduler_and_manager):
        scheduler, manager = scheduler_and_manager(num_blocks=2)
        # Both sit one token short of needing a second block, so only one can grow.
        first = start_decoding(scheduler, manager, make_request(prompt_len=BLOCK_SIZE))
        second = start_decoding(scheduler, manager, make_request(prompt_len=BLOCK_SIZE))
        assert manager.num_free_blocks == 0

        first.append_token(7)
        second.append_token(7)
        batch = scheduler.schedule_model()

        assert batch.decode == [first]
        assert scheduler.stats.num_preemptions == 1
        assert second.stage is RequestStage.WAITING_PREFILL
        assert second.block_table == []
        assert list(scheduler.waiting_prefill) == [second]

    def test_preemption_keeps_audio_embeddings(self, scheduler_and_manager):
        """A preempted request repeats its text prefill, never its audio encode."""
        scheduler, manager = scheduler_and_manager(num_blocks=2)
        start_decoding(scheduler, manager, make_request(prompt_len=BLOCK_SIZE)).append_token(7)
        victim = start_decoding(scheduler, manager, make_request(prompt_len=BLOCK_SIZE))
        victim.append_token(7)

        scheduler.schedule_model()

        assert victim.audio_embeds is not None
        assert victim.num_computed_tokens == 0

    def test_lone_request_that_cannot_grow_is_aborted(self, scheduler_and_manager):
        scheduler, manager = scheduler_and_manager(num_blocks=1)
        only = start_decoding(scheduler, manager, make_request(prompt_len=BLOCK_SIZE))
        only.append_token(7)

        batch = scheduler.schedule_model()

        assert not batch
        aborted = scheduler.drain_aborted()
        assert aborted == [only]
        assert only.finish_reason == "aborted:out_of_kv_blocks"
        assert only.audio_embeds is None
        assert not scheduler.has_work


class TestModelOom:
    def test_rollback_restores_prefill_requests_exactly(self, scheduler_and_manager):
        scheduler, manager = scheduler_and_manager(num_blocks=16)
        requests = [make_request(prompt_len=BLOCK_SIZE) for _ in range(4)]
        for request in requests:
            admit_for_prefill(scheduler, request)

        batch = scheduler.schedule_model()
        assert len(batch.prefill) == 4
        free_before_rollback = manager.num_free_blocks

        scheduler.on_model_oom(batch)

        assert manager.num_free_blocks > free_before_rollback
        assert list(scheduler.waiting_prefill) == requests
        for request in requests:
            assert request.block_table == []
            assert request.num_computed_tokens == 0
            assert request.audio_embeds is not None
            assert not request.is_finished

    def test_rollback_preserves_decode_order(self, scheduler_and_manager):
        scheduler, manager = scheduler_and_manager(num_blocks=16)
        running = [
            start_decoding(scheduler, manager, make_request(prompt_len=BLOCK_SIZE // 2))
            for _ in range(3)
        ]
        batch = scheduler.schedule_model()
        assert batch.decode == running

        scheduler.on_model_oom(batch)

        assert list(scheduler.running) == running
        assert all(request.stage is RequestStage.RUNNING_DECODE for request in running)

    def test_no_tokens_are_appended_by_a_failed_batch(self, scheduler_and_manager):
        scheduler, manager = scheduler_and_manager(num_blocks=16)
        request = start_decoding(scheduler, manager, make_request(prompt_len=BLOCK_SIZE // 2))
        length_before = len(request)

        scheduler.on_model_oom(scheduler.schedule_model())

        assert len(request) == length_before

    def test_cap_halves_then_recovers(self, scheduler_and_manager):
        scheduler, _ = scheduler_and_manager(num_blocks=32, max_num_seqs=8)
        for _ in range(8):
            admit_for_prefill(scheduler, make_request(prompt_len=BLOCK_SIZE))

        scheduler.on_model_oom(scheduler.schedule_model())
        assert scheduler.model_batch_limit == 4
        assert len(scheduler.schedule_model().requests) == 4

        scheduler.relax_limits()
        assert scheduler.model_batch_limit == 8

    def test_single_request_oom_gives_up_on_it(self, scheduler_and_manager):
        scheduler, manager = scheduler_and_manager(num_blocks=16)
        request = start_decoding(scheduler, manager, make_request(prompt_len=BLOCK_SIZE // 2))
        batch = scheduler.schedule_model()
        assert len(batch.requests) == 1

        scheduler.on_model_oom(batch)

        assert scheduler.drain_aborted() == [request]
        assert request.finish_reason == "aborted:out_of_memory"
        assert request.block_table == []
        assert not scheduler.has_work

    def test_empty_batch_rollback_is_a_no_op(self, scheduler_and_manager):
        scheduler, _ = scheduler_and_manager(num_blocks=16)
        scheduler.on_model_oom(ModelBatch())
        assert not scheduler.aborted
        assert not scheduler.has_work


class TestAudioOom:
    def test_rollback_requeues_in_order_and_drops_partial_embeddings(
        self, scheduler_and_manager
    ):
        scheduler, _ = scheduler_and_manager(num_blocks=16, max_audio_batch_frames=10_000)
        requests = [make_request(prompt_len=BLOCK_SIZE) for _ in range(3)]
        for request in requests:
            scheduler.add(request)

        selected = scheduler.schedule_audio()
        assert selected == requests

        scheduler.on_audio_oom(selected)

        assert list(scheduler.waiting_encode) == requests
        for request in requests:
            assert request.audio_embeds is None
            assert request.timings.encode_start is None
        assert scheduler.audio_frame_limit < 10_000

    def test_single_recording_oom_gives_up_on_it(self, scheduler_and_manager):
        scheduler, _ = scheduler_and_manager(num_blocks=16)
        request = make_request(prompt_len=BLOCK_SIZE)
        scheduler.add(request)

        scheduler.on_audio_oom(scheduler.schedule_audio())

        assert scheduler.drain_aborted() == [request]
        assert request.finish_reason == "aborted:out_of_memory"
        assert not scheduler.has_work

    def test_frame_cap_recovers_after_success(self, scheduler_and_manager):
        scheduler, _ = scheduler_and_manager(num_blocks=16, max_audio_batch_frames=8_000)
        scheduler.audio_frame_limit = 1_000

        scheduler.relax_limits()
        assert scheduler.audio_frame_limit == 2_000
        for _ in range(10):
            scheduler.relax_limits()
        assert scheduler.audio_frame_limit == 8_000
