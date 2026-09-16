import pytest
import torch

from qwen_asr_vllm.agent.qwen_tts_paged_attention import (
    HAS_TRITON,
    PagedAttentionKernelError,
    fused_qkv_projection,
    fused_qkv_projection_kernel,
    calibrate_fused_qkv,
    install_qwen_paged_decode_attention,
    paged_attention_decode,
    projection_error_report,
)


def _inputs():
    # Two requests use disjoint physical pages and have different logical
    # lengths. The second row also contains a left-padded prompt token.
    page_table = torch.tensor([[3, 5, 0], [1, 7, 2]], dtype=torch.long)
    seq_lens = torch.tensor([3, 5], dtype=torch.long)
    valid = torch.tensor(
        [
            [True, True, True, False, False],
            [False, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    k_pool = torch.zeros((8, 2, 2, 4), dtype=torch.float32)
    v_pool = torch.zeros_like(k_pool)
    torch.manual_seed(17)
    for page in (3, 5, 1, 7, 2):
        k_pool[page].normal_()
        v_pool[page].normal_()
    q = torch.ones((2, 4, 1, 4), dtype=torch.float32)
    return q, k_pool, v_pool, page_table, seq_lens, valid


def test_paged_decode_attention_matches_request_local_reference():
    inputs = _inputs()

    output = paged_attention_decode(*inputs, use_triton=False)

    expected = []
    q, k_pool, v_pool, page_table, seq_lens, valid = inputs
    for row in range(2):
        positions = torch.arange(seq_lens[row])
        pages = page_table[row, positions // 2]
        offsets = positions % 2
        keys = k_pool[pages, :, offsets, :].permute(1, 0, 2).unsqueeze(0)
        values = v_pool[pages, :, offsets, :].permute(1, 0, 2).unsqueeze(0)
        keys = keys.repeat_interleave(2, dim=1)
        values = values.repeat_interleave(2, dim=1)
        scores = torch.matmul(q[row : row + 1], keys.transpose(-1, -2)) / 2.0
        scores = scores.masked_fill(~valid[row, : seq_lens[row]].view(1, 1, 1, -1), -torch.inf)
        expected.append(torch.softmax(scores, dim=-1) @ values)

    assert torch.allclose(output, torch.cat(expected, dim=0))


def test_paged_decode_attention_rejects_invalid_page_table():
    q, k_pool, v_pool, page_table, seq_lens, valid = _inputs()
    page_table = page_table.clone()
    page_table[0, 0] = 99

    with pytest.raises(PagedAttentionKernelError, match="page table"):
        paged_attention_decode(
            q,
            k_pool,
            v_pool,
            page_table,
            seq_lens,
            valid,
            use_triton=False,
        )


def test_qwen_attention_patch_is_idempotent_and_restores_modules():
    class Attention:
        def forward(self, hidden_states, **kwargs):
            return hidden_states, None

    class Layer:
        def __init__(self):
            self.self_attn = Attention()

    class Model:
        def __init__(self):
            self.layers = [Layer(), Layer()]

    class Talker:
        def __init__(self):
            self.model = Model()

    talker = Talker()
    originals = [layer.self_attn.forward for layer in talker.model.layers]

    handle = install_qwen_paged_decode_attention(talker)

    assert install_qwen_paged_decode_attention(talker) is handle
    assert all(
        layer.self_attn.forward is not original
        for layer, original in zip(talker.model.layers, originals)
    )
    handle.close()
    assert all(
        layer.self_attn.forward == original
        for layer, original in zip(talker.model.layers, originals)
    )


def test_qwen_attention_patch_restores_fused_qkv_attributes():
    class Attention:
        def __init__(self):
            self.q_proj = torch.nn.Linear(4, 4)
            self.k_proj = torch.nn.Linear(4, 2)
            self.v_proj = torch.nn.Linear(4, 2)

        def forward(self, hidden_states, **kwargs):
            return hidden_states, None

    class Layer:
        def __init__(self):
            self.self_attn = Attention()

    class Model:
        def __init__(self):
            self.layers = [Layer()]

    class Talker:
        def __init__(self):
            self.model = Model()

    talker = Talker()
    attention = talker.model.layers[0].self_attn
    handle = install_qwen_paged_decode_attention(talker, fuse_qkv=True)

    assert getattr(attention, "_qav_use_fused_qkv") is True
    assert attention._qav_fused_qkv_weight.shape == (8, 4)
    handle.close()
    assert not hasattr(attention, "_qav_use_fused_qkv")
    assert not hasattr(attention, "_qav_fused_qkv_weight")


def test_paged_decode_can_write_current_kv_inside_attention_launch():
    q, k_pool, v_pool, page_table, seq_lens, valid = _inputs()
    new_key = torch.randn((2, 2, 1, 4), dtype=torch.float32)
    new_value = torch.randn((2, 2, 1, 4), dtype=torch.float32)
    expected_k = k_pool.clone()
    expected_v = v_pool.clone()
    for row in range(2):
        page = page_table[row, (seq_lens[row] - 1) // 2]
        offset = (seq_lens[row] - 1) % 2
        expected_k[page, :, offset] = new_key[row, :, 0]
        expected_v[page, :, offset] = new_value[row, :, 0]

    candidate = paged_attention_decode(
        q,
        k_pool.clone(),
        v_pool.clone(),
        page_table,
        seq_lens,
        valid,
        new_key=new_key,
        new_value=new_value,
        use_triton=False,
    )
    expected = paged_attention_decode(
        q,
        expected_k,
        expected_v,
        page_table,
        seq_lens,
        valid,
        use_triton=False,
    )

    assert torch.allclose(candidate, expected)


def test_fused_qkv_projection_matches_three_linear_projections():
    class Projections:
        def __init__(self):
            self.q_proj = torch.nn.Linear(4, 6)
            self.k_proj = torch.nn.Linear(4, 2)
            self.v_proj = torch.nn.Linear(4, 2)

    module = Projections()
    hidden = torch.randn((2, 1, 4))
    q, k, v = fused_qkv_projection(module, hidden)

    assert torch.equal(q, module.q_proj(hidden))
    assert torch.equal(k, module.k_proj(hidden))
    assert torch.equal(v, module.v_proj(hidden))


def test_projection_error_report_exposes_numeric_and_argmax_drift():
    reference = torch.tensor([[1.0, 2.0, 3.0]])
    candidate = torch.tensor([[1.0, 2.01, 2.99]])

    report = projection_error_report(reference, candidate)

    assert report["max_abs"] == pytest.approx(0.01, abs=1e-6)
    assert report["relative_l2"] > 0
    assert report["top1_match"] is True


def test_custom_fused_qkv_projection_kernel_requires_cuda():
    class Projections:
        def __init__(self):
            self.q_proj = torch.nn.Linear(4, 4)
            self.k_proj = torch.nn.Linear(4, 2)
            self.v_proj = torch.nn.Linear(4, 2)

    with pytest.raises(PagedAttentionKernelError, match="CUDA"):
        fused_qkv_projection_kernel(
            Projections(), torch.randn(1, 1, 4), use_triton=True
        )


def test_calibrate_fused_qkv_reports_layer_metrics():
    class Attention:
        def __init__(self):
            self.q_proj = torch.nn.Linear(4, 4)
            self.k_proj = torch.nn.Linear(4, 2)
            self.v_proj = torch.nn.Linear(4, 2)

    talker = type(
        "Talker",
        (),
        {"model": type("Model", (), {"layers": [type("Layer", (), {"self_attn": Attention()})()]})()},
    )()

    report = calibrate_fused_qkv(talker)

    assert report["backend"] == "torch"
    assert report["top1_match"] is True
    assert len(report["layers"]) == 1
    assert report["max_abs"] < 1.0e-5


def test_static_cache_decode_can_use_fused_qkv_and_sdpa(monkeypatch):
    pytest.importorskip("qwen_tts")
    import qwen_tts.core.models.modeling_qwen3_tts as qwen_modeling

    monkeypatch.setattr(
        qwen_modeling,
        "apply_multimodal_rotary_pos_emb",
        lambda q, k, cos, sin, *args, **kwargs: (q, k),
    )

    class StaticCache:
        def __init__(self):
            self.updated = False

        def update(self, key, value, layer_idx, cache_kwargs):
            self.updated = True
            return key, value

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = 2
            self.num_key_value_groups = 2
            self.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.k_proj = torch.nn.Linear(4, 2, bias=False)
            self.v_proj = torch.nn.Linear(4, 2, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.o_proj = torch.nn.Linear(4, 4, bias=False)
            self.layer_idx = 0
            self.rope_scaling = {"mrope_section": [1, 1], "interleaved": False}
            self.config = type("Config", (), {"_attn_implementation": "eager"})()
            self.scaling = 2**-0.5
            self.sliding_window = None
            self.attention_dropout = 0.0

        def forward(self, hidden_states, **kwargs):
            return hidden_states, None

    class Layer:
        def __init__(self):
            self.self_attn = Attention()

    class Model:
        def __init__(self):
            self.layers = [Layer()]

    class Talker:
        def __init__(self):
            self.model = Model()

        def generate(self, **kwargs):
            return kwargs

    talker = Talker()
    handle = install_qwen_paged_decode_attention(
        talker,
        fuse_qkv=True,
        patch_static_cache=True,
        fuse_static_attention=True,
    )
    attention = talker.model.layers[0].self_attn
    cache = StaticCache()
    output, weights = attention.forward(
        torch.randn(1, 1, 4),
        position_embeddings=(torch.ones(1, 1, 2), torch.zeros(1, 1, 2)),
        attention_mask=None,
        past_key_values=cache,
        cache_position=torch.tensor([0]),
    )

    assert output.shape == (1, 1, 4)
    assert weights is None
    assert cache.updated is True
    handle.close()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="CUDA and Triton are required",
)
def test_custom_fused_qkv_projection_kernel_matches_torch_reference_on_cuda():
    class Projections:
        def __init__(self):
            self.q_proj = torch.nn.Linear(16, 24, bias=False).cuda().half()
            self.k_proj = torch.nn.Linear(16, 8, bias=False).cuda().half()
            self.v_proj = torch.nn.Linear(16, 8, bias=False).cuda().half()

    module = Projections()
    hidden = torch.randn((2, 1, 16), device="cuda", dtype=torch.float16)
    reference = fused_qkv_projection(module, hidden)
    candidate = fused_qkv_projection_kernel(module, hidden)

    assert all(
        torch.allclose(left, right, atol=2e-3, rtol=2e-3)
        for left, right in zip(reference, candidate)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_paged_decode_triton_matches_reference_on_cuda():
    inputs = tuple(value.cuda() for value in _inputs())
    reference = paged_attention_decode(*inputs, use_triton=False)
    candidate = paged_attention_decode(*inputs, use_triton=True)

    assert torch.allclose(candidate, reference, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="CUDA and Triton are required",
)
def test_paged_decode_triton_fused_write_matches_reference_on_cuda():
    q, k_pool, v_pool, page_table, seq_lens, valid = _inputs()
    q, k_pool, v_pool = q.cuda(), k_pool.cuda(), v_pool.cuda()
    page_table, seq_lens, valid = page_table.cuda(), seq_lens.cuda(), valid.cuda()
    new_key = torch.randn((2, 2, 1, 4), device="cuda", dtype=torch.float32)
    new_value = torch.randn((2, 2, 1, 4), device="cuda", dtype=torch.float32)
    expected_k = k_pool.clone()
    expected_v = v_pool.clone()
    for row in range(2):
        page = page_table[row, (seq_lens[row] - 1) // 2]
        offset = (seq_lens[row] - 1) % 2
        expected_k[page, :, offset] = new_key[row, :, 0]
        expected_v[page, :, offset] = new_value[row, :, 0]

    candidate = paged_attention_decode(
        q,
        k_pool,
        v_pool,
        page_table,
        seq_lens,
        valid,
        new_key=new_key,
        new_value=new_value,
        use_triton=True,
    )
    reference = paged_attention_decode(
        q,
        expected_k,
        expected_v,
        page_table,
        seq_lens,
        valid,
        use_triton=False,
    )

    assert torch.allclose(k_pool, expected_k)
    assert torch.allclose(v_pool, expected_v)
    assert torch.allclose(candidate, reference, atol=2e-3, rtol=2e-3)
