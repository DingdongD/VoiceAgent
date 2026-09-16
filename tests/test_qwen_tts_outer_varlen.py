import pytest
import torch

from qwen_asr_vllm.agent.qwen_tts_outer_paged_cache import PageTable
from qwen_asr_vllm.agent.qwen_tts_outer_varlen import (
    OuterVarlenRequest,
    VarlenMetadataError,
    build_packed_step,
)


def _request(request_id, seq_len, *, active=True, dtype=torch.float32):
    return OuterVarlenRequest(
        request_id=request_id,
        page_table=PageTable(request_id, (0, 1, 2), 1, 6, seq_len),
        seq_len=seq_len,
        max_seq_len=6,
        rope_delta=torch.tensor([[seq_len]], dtype=torch.long),
        past_hidden=torch.full((1, 1, 4), float(seq_len), dtype=dtype),
        trailing_text_hidden=torch.zeros((1, 3, 4), dtype=dtype),
        tts_pad_embed=torch.zeros((1, 1, 4), dtype=dtype),
        active=active,
    )


def test_build_packed_step_preserves_request_order_and_varlen_metadata():
    step = build_packed_step([_request("z", 5), _request("a", 3)])

    assert step.request_ids == ("z", "a")
    assert step.row_to_request_id == ("z", "a")
    assert step.seq_lens.tolist() == [5, 3]
    assert step.cu_seqlens.tolist() == [0, 5, 8]
    assert step.decode_positions.tolist() == [5, 3]
    assert step.page_table.tolist() == [[0, 1, 2], [0, 1, 2]]
    assert step.past_hidden[:, 0, 0].tolist() == [5.0, 3.0]


def test_build_packed_step_filters_inactive_requests_without_mutating_state():
    active = _request("active", 2)
    inactive = _request("done", 4, active=False)

    step = build_packed_step([active, inactive])

    assert step.request_ids == ("active",)
    assert active.active is True
    assert inactive.active is False


def test_packed_step_is_immutable_and_rejects_incompatible_dtype():
    step = build_packed_step([_request("a", 1)])
    with pytest.raises(AttributeError):
        step.request_ids = ("mutated",)
    with pytest.raises(VarlenMetadataError, match="dtype"):
        build_packed_step([_request("a", 1), _request("b", 2, dtype=torch.float16)])


def test_invalid_sequence_and_page_table_metadata_are_rejected():
    with pytest.raises(VarlenMetadataError, match="sequence"):
        build_packed_step([_request("bad", 0)])
    bad = _request("bad", 2)
    bad.page_table = PageTable("other", (0,), 1, 2, 2)
    with pytest.raises(VarlenMetadataError, match="page table"):
        build_packed_step([bad])
