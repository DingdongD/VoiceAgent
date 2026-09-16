from types import SimpleNamespace

import pytest
import torch

qwen_configuration = pytest.importorskip("qwen_tts.core.models.configuration_qwen3_tts")
qwen_modeling = pytest.importorskip("qwen_tts.core.models.modeling_qwen3_tts")

from qwen_asr_vllm.agent.qwen_tts_outer_paged_cache import PagedOuterCache
from qwen_asr_vllm.agent.qwen_tts_outer_varlen import (
    OuterVarlenRequest,
    QwenOuterVarlenAdapter,
    QwenPagedCache,
    build_packed_step,
)


def _config():
    return qwen_configuration.Qwen3TTSTalkerConfig(
        vocab_size=32,
        text_vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=32,
        text_hidden_size=64,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [4, 2, 2],
            "interleaved": False,
        },
        num_code_groups=3,
        use_cache=True,
    )


def _position_ids(position):
    return torch.full((3, 1, 1), int(position), dtype=torch.long)


def _forward(model, hidden, cache, request_id, position):
    positions = torch.arange(position, position + hidden.shape[1], dtype=torch.long).view(1, -1)
    cache.bind_rows((request_id,), positions)
    return model(
        inputs_embeds=hidden,
        position_ids=positions.unsqueeze(0).expand(3, -1, -1),
        attention_mask=cache.attention_mask(dtype=hidden.dtype),
        past_key_values=cache,
        use_cache=True,
        cache_position=torch.arange(hidden.shape[1], dtype=torch.long),
    ).last_hidden_state


def test_qwen_model_forward_matches_dynamic_cache_for_packed_varlen_decode():
    torch.manual_seed(7)
    config = _config()
    config._attn_implementation = "eager"
    model = qwen_modeling.Qwen3TTSTalkerModel(config).eval()
    baseline = qwen_modeling.Qwen3TTSTalkerModel(config).eval()
    baseline.load_state_dict(model.state_dict())

    pool = PagedOuterCache(num_layers=2, num_pages=16, page_size=2)
    pool.allocate("short", max_seq_len=8)
    pool.allocate("long", max_seq_len=8)
    paged = QwenPagedCache(pool, num_layers=2)
    dynamic = {
        request_id: __import__("transformers").DynamicCache()
        for request_id in ("short", "long")
    }
    prompts = {"short": torch.randn(1, 2, 64), "long": torch.randn(1, 3, 64)}
    with torch.no_grad():
        paged_outputs = {}
        baseline_outputs = {}
        for request_id, hidden in prompts.items():
            paged_outputs[request_id] = _forward(model, hidden, paged, request_id, 0)
            # The helper only handles one query position, so use a normal
            # model call for the independent baseline prefill.
            baseline_outputs[request_id] = baseline(
                inputs_embeds=hidden,
                past_key_values=dynamic[request_id],
                use_cache=True,
                cache_position=torch.arange(hidden.shape[1]),
            ).last_hidden_state

        decode_inputs = torch.randn(2, 1, 64)
        paged.bind_rows(("short", "long"), torch.tensor([[2], [3]], dtype=torch.long))
        packed = model(
            inputs_embeds=decode_inputs,
            position_ids=torch.tensor([[[2], [3]], [[2], [3]], [[2], [3]]]),
            attention_mask=paged.attention_mask(dtype=decode_inputs.dtype),
            past_key_values=paged,
            use_cache=True,
            cache_position=torch.tensor([0]),
        ).last_hidden_state
        expected = torch.cat(
            [
                baseline(
                    inputs_embeds=decode_inputs[row : row + 1],
                    past_key_values=dynamic[request_id],
                    use_cache=True,
                    position_ids=_position_ids(position),
                    cache_position=torch.tensor([position]),
                ).last_hidden_state
                for row, (request_id, position) in enumerate((("short", 2), ("long", 3)))
            ],
            dim=0,
        )

    assert torch.allclose(packed, expected, atol=1e-5, rtol=1e-5)
    assert pool.snapshot()["live_pages"] == 8


def test_qwen_outer_varlen_runner_executes_packed_codec_decode():
    torch.manual_seed(11)
    config = _config()
    config._attn_implementation = "eager"
    model = qwen_modeling.Qwen3TTSTalkerModel(config).eval()
    talker = SimpleNamespace(
        config=config,
        model=model,
        get_input_embeddings=lambda: torch.nn.Embedding(config.vocab_size, config.hidden_size),
        code_predictor=SimpleNamespace(
            get_input_embeddings=lambda: [
                torch.nn.Embedding(config.vocab_size, config.hidden_size)
                for _ in range(config.num_code_groups - 1)
            ]
        ),
        codec_head=torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False),
    )
    adapter = QwenOuterVarlenAdapter(talker, max_cache_len=8, page_size=2, max_pages=16)
    runner = adapter.create_runner()
    runner.allocate_request("short", max_seq_len=8)
    runner.allocate_request("long", max_seq_len=8)

    with torch.no_grad():
        short_hidden = runner.prefill("short", inputs_embeds=torch.randn(1, 2, config.hidden_size))
        long_hidden = runner.prefill("long", inputs_embeds=torch.randn(1, 3, config.hidden_size))
        requests = [
            OuterVarlenRequest(
                request_id,
                adapter.cache.page_table(request_id),
                seq_len,
                8,
                torch.zeros((1, 1), dtype=torch.long),
                hidden[:, -1:],
                torch.zeros((1, 0, config.hidden_size)),
                torch.zeros((1, 1, config.hidden_size)),
            )
            for request_id, seq_len, hidden in (
                ("short", 2, short_hidden),
                ("long", 3, long_hidden),
            )
        ]
        step = build_packed_step(requests)
        output = runner.decode_packed(
            step,
            codec_ids=torch.ones((2, config.num_code_groups), dtype=torch.long),
            condition=torch.zeros((2, 1, config.hidden_size)),
        )

    assert output.request_ids == ("short", "long")
    assert output.logits.shape == (2, 1, config.vocab_size)
    assert output.past_hidden.shape == (2, 1, config.hidden_size)
    adapter.cache.release("short")
    adapter.cache.release("long")
    assert adapter.metrics()["cache"]["live_pages"] == 0
