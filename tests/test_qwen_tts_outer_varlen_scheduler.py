import torch

from qwen_asr_vllm.agent.qwen_tts_outer_paged_cache import PagedOuterCache
from qwen_asr_vllm.agent.qwen_tts_outer_varlen import (
    OuterVarlenRequest,
    OuterVarlenScheduler,
    PackedOuterOutput,
)


def _request(request_id, page_table, seq_len=1):
    return OuterVarlenRequest(
        request_id=request_id,
        page_table=page_table,
        seq_len=seq_len,
        max_seq_len=5,
        rope_delta=torch.zeros((1, 1), dtype=torch.long),
        past_hidden=torch.zeros((1, 1, 2)),
        trailing_text_hidden=torch.zeros((1, 2, 2)),
        tts_pad_embed=torch.zeros((1, 1, 2)),
    )


def test_scheduler_packs_ready_requests_and_emits_by_request_id():
    cache = PagedOuterCache(num_layers=1, num_pages=8, page_size=2)
    first = _request("first", cache.allocate("first", max_seq_len=5))
    second = _request("second", cache.allocate("second", max_seq_len=5), seq_len=2)
    packed_sizes = []
    emitted = []

    def decode(step):
        packed_sizes.append(step.request_ids)
        return PackedOuterOutput(
            request_ids=step.request_ids,
            logits=torch.zeros((len(step.requests), 4)),
            past_hidden=step.past_hidden + 1,
        )

    scheduler = OuterVarlenScheduler(
        cache=cache,
        prefill_fn=lambda request: None,
        decode_packed_fn=decode,
        emit_fn=lambda request_id, output: emitted.append(request_id),
    )
    scheduler.add_request(first)
    scheduler.add_request(second)
    scheduler.step()

    assert packed_sizes == [("first", "second")]
    assert emitted == ["first", "second"]
    assert first.seq_len == 2
    assert second.seq_len == 3
    assert first.past_hidden[0, 0, 0].item() == 1


def test_scheduler_removes_eos_request_on_next_tick_and_releases_pages():
    cache = PagedOuterCache(num_layers=1, num_pages=8, page_size=2)
    first = _request("first", cache.allocate("first", max_seq_len=4))
    second = _request("second", cache.allocate("second", max_seq_len=4))
    batches = []

    def decode(step):
        batches.append(step.request_ids)
        return PackedOuterOutput(
            request_ids=step.request_ids,
            logits=torch.zeros((len(step.requests), 4)),
            past_hidden=step.past_hidden,
            done_request_ids=("first",) if len(batches) == 1 else (),
        )

    scheduler = OuterVarlenScheduler(
        cache=cache,
        prefill_fn=lambda request: None,
        decode_packed_fn=decode,
    )
    scheduler.add_request(first)
    scheduler.add_request(second)
    scheduler.step()
    scheduler.step()

    assert batches == [("first", "second"), ("second",)]
    assert "first" not in scheduler.request_ids()
    assert cache.snapshot()["live_requests"] == 1
    scheduler.close()
    assert cache.snapshot()["live_pages"] == 0


def test_scheduler_cancel_and_callback_error_release_only_owned_requests():
    cache = PagedOuterCache(num_layers=1, num_pages=8, page_size=2)
    cancelled = _request("cancel", cache.allocate("cancel", max_seq_len=4))
    failed = _request("failed", cache.allocate("failed", max_seq_len=4))
    errors = []

    scheduler = OuterVarlenScheduler(
        cache=cache,
        prefill_fn=lambda request: None,
        decode_packed_fn=lambda step: (_ for _ in ()).throw(RuntimeError("tick failed")),
        emit_fn=lambda request_id, output: errors.append((request_id, output)),
    )
    scheduler.add_request(cancelled)
    scheduler.add_request(failed)
    assert scheduler.cancel("cancel") is True
    scheduler.step()

    assert cancelled.error is not None
    assert failed.error is not None
    assert scheduler.request_ids() == ()
    assert cache.snapshot()["live_pages"] == 0
