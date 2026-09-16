import torch

from qwen_asr_vllm.agent.qwen_tts_outer_paged_cache import PagedOuterCache
from qwen_asr_vllm.agent.qwen_tts_outer_varlen import (
    OuterVarlenRequest,
    ReferencePagedAttentionAdapter,
    build_packed_step,
)


def _request(request_id, table, seq_len):
    return OuterVarlenRequest(
        request_id=request_id,
        page_table=table,
        seq_len=seq_len,
        max_seq_len=6,
        rope_delta=torch.zeros((1, 1), dtype=torch.long),
        past_hidden=torch.zeros((1, 1, 2)),
        trailing_text_hidden=torch.zeros((1, 2, 2)),
        tts_pad_embed=torch.zeros((1, 1, 2)),
    )


def test_reference_attention_matches_independent_variable_length_attention():
    cache = PagedOuterCache(num_layers=1, num_pages=8, page_size=2)
    table_a = cache.allocate("a", max_seq_len=6)
    table_b = cache.allocate("b", max_seq_len=6)
    keys_a = torch.tensor([[[[[1.0], [2.0], [3.0]]]]])
    values_a = keys_a * 10
    keys_b = torch.tensor([[[[[100.0], [200.0]]]]])
    values_b = keys_b * 10
    cache.append("a", keys_a, values_a, torch.arange(3))
    cache.append("b", keys_b, values_b, torch.arange(2))
    requests = [_request("a", table_a, 3), _request("b", table_b, 2)]
    step = build_packed_step(requests)
    queries = torch.tensor([[[[2.0]]], [[[150.0]]]])

    output = ReferencePagedAttentionAdapter(cache).decode_packed(step, queries)

    expected = []
    for request, query in zip(requests, queries):
        key, value = cache.read(request.request_id, layer_index=0, logical_length=request.seq_len)
        expected.append(
            torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=False
            )
        )
    assert torch.allclose(output.logits, torch.cat(expected, dim=0))
    assert output.request_ids == ("a", "b")
    assert output.metrics["attention_calls"] == 2
    assert output.metrics["gather_bytes"] > 0


def test_reference_adapter_keeps_page_boundaries_and_request_data_isolated():
    cache = PagedOuterCache(num_layers=1, num_pages=8, page_size=2)
    table_a = cache.allocate("a", max_seq_len=4)
    table_b = cache.allocate("b", max_seq_len=4)
    cache.append("a", torch.ones(1, 1, 1, 4, 1), torch.ones(1, 1, 1, 4, 1), torch.arange(4))
    cache.append("b", torch.full((1, 1, 1, 4, 1), 9.0), torch.full((1, 1, 1, 4, 1), 9.0), torch.arange(4))
    step = build_packed_step([_request("a", table_a, 4), _request("b", table_b, 4)])
    queries = torch.ones(2, 1, 1, 1)

    output = ReferencePagedAttentionAdapter(cache).decode_packed(step, queries)

    assert output.logits[0].item() == 1.0
    assert output.logits[1].item() == 9.0
