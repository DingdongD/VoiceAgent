from types import SimpleNamespace

import torch

from qwen_asr_vllm.agent.qwen_tts_static_predictor import (
    PrefixStaticCache,
    StaticCodePredictorEngine,
    install_static_code_predictor,
)


class FakeStaticCache:
    def __init__(self, max_cache_len: int):
        self.max_cache_len = max_cache_len
        self.reset_calls = 0

    def reset(self):
        assert torch.is_inference_mode_enabled()
        self.reset_calls += 1


class FakeCodePredictor:
    config = SimpleNamespace()

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        raise AssertionError("original generate should not be called")

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        generation_step = kwargs.get("generation_steps")
        next_generation_step = 1 if generation_step is None else generation_step + 1
        batch_source = kwargs.get("inputs_embeds", kwargs.get("input_ids"))
        logits = torch.zeros(batch_source.shape[0], 1, 8)
        logits[:, :, next_generation_step] = 10.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=kwargs["past_key_values"],
            generation_steps=next_generation_step,
        )


def test_static_engine_reuses_cache_and_sets_explicit_positions():
    predictor = FakeCodePredictor()
    created = []

    def cache_factory(*, code_predictor, max_cache_len):
        assert code_predictor is predictor
        cache = FakeStaticCache(max_cache_len)
        created.append(cache)
        return cache

    engine = StaticCodePredictorEngine(predictor, cache_factory=cache_factory)

    first = engine.generate(
        inputs_embeds=torch.zeros(1, 2, 4),
        max_new_tokens=3,
        do_sample=False,
    )
    second = engine.generate(
        inputs_embeds=torch.zeros(1, 2, 4),
        max_new_tokens=3,
        do_sample=False,
    )

    assert first.sequences.tolist() == [[1, 2, 3]]
    assert second.sequences.tolist() == [[1, 2, 3]]
    assert len(created) == 1
    assert created[0].max_cache_len == 4
    assert created[0].reset_calls == 2
    assert [call["cache_position"].tolist() for call in predictor.calls] == [
        [0, 1],
        [2],
        [3],
        [0, 1],
        [2],
        [3],
    ]
    assert all(call["past_key_values"] is created[0] for call in predictor.calls)


def test_static_engine_separates_cache_pools_by_batch_shape():
    predictor = FakeCodePredictor()
    created = []

    def cache_factory(*, code_predictor, max_cache_len):
        cache = FakeStaticCache(max_cache_len)
        created.append(cache)
        return cache

    engine = StaticCodePredictorEngine(predictor, cache_factory=cache_factory)

    engine.generate(
        inputs_embeds=torch.zeros(1, 2, 4),
        max_new_tokens=2,
        do_sample=False,
    )
    engine.generate(
        inputs_embeds=torch.zeros(2, 2, 4),
        max_new_tokens=2,
        do_sample=False,
    )

    assert len(created) == 2


def test_install_static_code_predictor_replaces_generate_once():
    predictor = FakeCodePredictor()
    talker = SimpleNamespace(code_predictor=predictor)
    cache_factory = lambda **kwargs: FakeStaticCache(kwargs["max_cache_len"])

    assert install_static_code_predictor(talker, cache_factory=cache_factory) is True
    assert install_static_code_predictor(talker, cache_factory=cache_factory) is False
    result = predictor.generate(
        inputs_embeds=torch.zeros(1, 2, 4),
        max_new_tokens=2,
        do_sample=False,
    )

    assert result.sequences.tolist() == [[1, 2]]


def test_prefix_static_cache_exposes_dynamic_active_prefix_shape():
    config = SimpleNamespace(
        get_text_config=lambda decoder=True: SimpleNamespace(num_hidden_layers=1)
    )
    cache = PrefixStaticCache(config=config, max_cache_len=4)
    keys = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)

    cache.prepare_step(past_length=0, query_length=2)
    returned_keys, _ = cache.update(
        keys,
        keys,
        0,
        {"cache_position": torch.arange(2, dtype=torch.long)},
    )
    assert returned_keys.shape == (1, 2, 2, 3)
    assert cache.get_mask_sizes(torch.arange(2), 0) == (2, 0)
    assert cache.is_compileable is False

    cache.prepare_step(past_length=2, query_length=1)
    next_key = torch.ones((1, 2, 1, 3), dtype=torch.float32)
    returned_keys, _ = cache.update(
        next_key,
        next_key,
        0,
        {"cache_position": torch.tensor([2], dtype=torch.long)},
    )
    assert returned_keys.shape == (1, 2, 3, 3)
    assert torch.equal(returned_keys[..., 2:, :], next_key)
