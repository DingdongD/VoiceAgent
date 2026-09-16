from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest
import torch

import qwen_asr_vllm.agent.qwen_tts_outer_static_engine as outer_static_module

from qwen_asr_vllm.agent.qwen_tts_outer_static_engine import (
    EagerOuterStepRuntime,
    OuterTalkerEngineError,
    OuterCohortState,
    OuterRequestState,
    OuterRuntimeLease,
    OuterTalkerStaticEngine,
    build_decode_position_ids,
    install_static_outer_talker,
    select_text_condition,
)


class FakeStaticCache:
    def __init__(self, *, initial_seq_length=0, advance_extra=0, reset_seq_length=0):
        self.reset_calls = 0
        self.reset_inference_modes = []
        self.seq_length = initial_seq_length
        self.advance_extra = advance_extra
        self.reset_seq_length = reset_seq_length

    def reset(self):
        self.reset_calls += 1
        self.reset_inference_modes.append(torch.is_inference_mode_enabled())
        self.seq_length = self.reset_seq_length

    def get_seq_length(self):
        return self.seq_length

    def record_positions(self, cache_position):
        self.seq_length = int(cache_position[-1]) + 1 + self.advance_extra


class FakeStaticModel:
    def __init__(self):
        self.calls = []
        self.cache_positions = []
        self.decode_masks = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.cache_positions.append(tuple(kwargs["cache_position"].tolist()))
        kwargs["past_key_values"].record_positions(kwargs["cache_position"])
        if kwargs["inputs_embeds"].shape[1] == 1:
            self.decode_masks.append(kwargs["attention_mask"])
        hidden = kwargs["inputs_embeds"] + float(len(self.calls))
        return SimpleNamespace(
            last_hidden_state=hidden,
            hidden_states=(hidden,),
            attentions=None,
            past_key_values=kwargs["past_key_values"],
        )


class FakeCodecHead:
    def __init__(self, token_schedule=None):
        self.token_schedule = token_schedule or [1, 2, 3, 4, 5]
        self.calls = 0

    def __call__(self, hidden_states):
        scheduled = self.token_schedule[min(self.calls, len(self.token_schedule) - 1)]
        self.calls += 1
        batch_size = hidden_states.shape[0]
        if isinstance(scheduled, int):
            scheduled = [scheduled] * batch_size
        logits = torch.full(
            (batch_size, hidden_states.shape[1], 16),
            -100.0,
            device=hidden_states.device,
        )
        for row, token_id in enumerate(scheduled):
            logits[row, -1, token_id] = 100.0
        return logits


class FakeEmbedding:
    def __init__(self, offset=0, *, runtime=None):
        self.offset = offset
        self.runtime = runtime
        self.in_run = []

    def __call__(self, token_ids):
        if self.runtime is not None:
            self.in_run.append(self.runtime.in_run)
        values = token_ids.to(dtype=torch.float32).unsqueeze(-1) + self.offset
        return values.expand(-1, -1, 4)


class FakeCodePredictor:
    def __init__(self, runtime):
        self.runtime = runtime
        self.batch_sizes = []
        self.first_ids = []
        self._embeddings = [
            FakeEmbedding(10, runtime=runtime),
            FakeEmbedding(20, runtime=runtime),
        ]

    def generate(self, **kwargs):
        assert not self.runtime.in_run
        batch_size = kwargs["inputs_embeds"].shape[0]
        self.batch_sizes.append(batch_size)
        self.first_ids.append(kwargs["inputs_embeds"][:, 1, 0].to(torch.long).tolist())
        return SimpleNamespace(
            sequences=torch.tensor([[6, 7]], dtype=torch.long).expand(batch_size, -1)
        )

    def get_input_embeddings(self):
        return self._embeddings


class FakeOuterLease:
    def __init__(self, talker, runtime, fail_on_run=False):
        self.talker = talker
        self.runtime = runtime
        self.cache = FakeStaticCache()
        self.fail_on_run = fail_on_run
        self.release_calls = 0
        self.prepared_decodes = []

    def reset(self):
        self.cache.reset()

    def run(self, prepared_decode):
        self.runtime.in_run = True
        try:
            if self.fail_on_run:
                raise RuntimeError("decode failed")
            self.prepared_decodes.append(prepared_decode)
            codec_ids = prepared_decode.codec_ids
            codec_hiddens = [self.talker.get_input_embeddings()(codec_ids[:, :1])]
            predictor_embeddings = self.talker.code_predictor.get_input_embeddings()
            codec_hiddens.extend(
                predictor_embeddings[index](codec_ids[:, index + 1 : index + 2])
                for index in range(self.talker.config.num_code_groups - 1)
            )
            inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)
            inputs_embeds = inputs_embeds + prepared_decode.condition
            outputs = self.talker.model(
                inputs_embeds=inputs_embeds,
                attention_mask=prepared_decode.attention_mask,
                position_ids=prepared_decode.position_ids,
                past_key_values=self.cache,
                cache_position=prepared_decode.cache_position,
                use_cache=True,
                output_hidden_states=prepared_decode.output_hidden_states,
            )
            return SimpleNamespace(
                logits=self.talker.codec_head(outputs.last_hidden_state),
                past_hidden=outputs.last_hidden_state[:, -1:],
                hidden_states=outputs.hidden_states,
            )
        finally:
            self.runtime.in_run = False

    def release(self):
        self.release_calls += 1


class FakeOuterStepRuntime:
    def __init__(self, *, fail_on_run=False, cache=None):
        self.fail_on_run = fail_on_run
        self.cache = cache
        self.in_run = False
        self.acquire_calls = []
        self.leases = []

    def acquire(self, batch_size, device, dtype, max_cache_len):
        self.acquire_calls.append((batch_size, device, dtype, max_cache_len))
        lease = FakeOuterLease(self.talker, self, fail_on_run=self.fail_on_run)
        if self.cache is not None:
            lease.cache = self.cache
        self.leases.append(lease)
        return lease

    def build_attention_mask(
        self, prompt_attention_mask, cache_position, max_cache_len, dtype
    ):
        return EagerOuterStepRuntime.build_attention_mask(
            self,
            prompt_attention_mask,
            cache_position,
            max_cache_len,
            dtype,
        )


class FakeStaticTalker:
    def __init__(self, *, token_schedule=None, runtime=None):
        self.rope_deltas = None
        self.config = SimpleNamespace(codec_eos_token_id=9, num_code_groups=3)
        self.model = FakeStaticModel()
        self.codec_head = FakeCodecHead(token_schedule)
        self.runtime = runtime or FakeOuterStepRuntime()
        self.runtime.talker = self
        self._embedding = FakeEmbedding(runtime=self.runtime)
        self.code_predictor = FakeCodePredictor(self.runtime)

    def generate(self, **kwargs):
        raise AssertionError("original talker.generate should not be called")

    def get_input_embeddings(self):
        return self._embedding

    def get_rope_index(self, attention_mask):
        position_ids = attention_mask.float().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        max_positions = position_ids.max(0).values.max(-1, keepdim=True).values
        rope_deltas = max_positions + 1 - attention_mask.sum(-1, keepdim=True)
        return position_ids, rope_deltas


def fake_generation_kwargs(batch_size=2, max_new_tokens=3):
    attention_mask = torch.ones(batch_size, 3, dtype=torch.long)
    if batch_size > 1:
        attention_mask[1, 0] = 0
    return {
        "inputs_embeds": torch.zeros(batch_size, 3, 4),
        "attention_mask": attention_mask,
        "trailing_text_hidden": torch.full((batch_size, 2, 4), 3.0),
        "tts_pad_embed": torch.full((1, 1, 4), 9.0),
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": 1,
        "do_sample": False,
        "eos_token_id": 9,
        "output_hidden_states": True,
        "return_dict_in_generate": True,
    }


def test_outer_iterator_releases_lease_when_cancelled_at_step_boundary():
    talker = FakeStaticTalker()
    engine = OuterTalkerStaticEngine(talker, step_runtime=talker.runtime)
    iterator = engine.iterate(**fake_generation_kwargs(batch_size=1))
    pending = next(iterator)
    assert pending["cohort"].cache_position == 3
    assert talker.code_predictor.batch_sizes == []
    iterator.close()
    lease = talker.runtime.leases[0]
    assert lease.release_calls == 1
    assert lease.cache.reset_calls == 2
    assert engine.errors == 0


def test_outer_request_state_keeps_rope_and_text_condition_per_slot():
    first = OuterRequestState(
        request_id=11,
        rope_delta=torch.tensor([[2]]),
        past_hidden=torch.full((1, 1, 4), 1.0),
        generation_step=0,
        trailing_text_hidden=torch.full((1, 2, 4), 3.0),
        tts_pad_embed=torch.full((1, 1, 4), 9.0),
    )
    second = OuterRequestState(
        request_id=12,
        rope_delta=torch.tensor([[7]]),
        past_hidden=torch.full((1, 1, 4), 5.0),
        generation_step=3,
        trailing_text_hidden=torch.full((1, 1, 4), 6.0),
        tts_pad_embed=torch.full((1, 1, 4), 8.0),
    )

    positions = build_decode_position_ids(10, [first, second])
    conditions = select_text_condition([first, second])

    assert positions[:, 0, 0].tolist() == [12, 12, 12]
    assert positions[:, 1, 0].tolist() == [17, 17, 17]
    assert conditions[:, 0, :].tolist() == [[3.0] * 4, [8.0] * 4]


def test_outer_cohort_active_indices_follow_request_flags():
    requests = [
        OuterRequestState(1, None, None, 0, None, None),
        OuterRequestState(2, None, None, 0, None, None, active=False),
        OuterRequestState(3, None, None, 0, None, None),
    ]
    cohort = OuterCohortState(
        cache=None,
        cache_position=10,
        requests=requests,
        last_logits=None,
    )

    assert cohort.active_indices == [0, 2]

    requests[0].active = False
    requests[1].active = True

    assert cohort.active_indices == [1, 2]


def test_outer_request_history_defaults_are_not_shared():
    first = OuterRequestState(1, None, None, 0, None, None)
    second = OuterRequestState(2, None, None, 0, None, None)

    first.first_codebook_history.append("first-codebook")
    first.hidden_history.append("first-hidden")

    assert first.first_codebook_history is not second.first_codebook_history
    assert first.hidden_history is not second.hidden_history
    assert second.first_codebook_history == []
    assert second.hidden_history == []


def test_eager_runtime_preserves_fixed_four_dimensional_additive_mask():
    runtime = EagerOuterStepRuntime(FakeStaticTalker())
    prompt_mask = torch.tensor([[1, 1, 1], [0, 1, 1]])

    mask = runtime.build_attention_mask(
        prompt_mask,
        cache_position=3,
        max_cache_len=6,
        dtype=torch.float32,
    )

    masked = torch.finfo(torch.float32).min
    assert mask.shape == (2, 1, 1, 6)
    assert mask[:, 0, 0].tolist() == [
        [0.0, 0.0, 0.0, 0.0, masked, masked],
        [masked, 0.0, 0.0, 0.0, masked, masked],
    ]


def test_cache_position_validator_checks_the_complete_contiguous_range():
    cache = FakeStaticCache()
    validator = getattr(
        outer_static_module, "_validate_contiguous_cache_position", None
    )

    assert callable(validator), "engine-level cache-position validator is missing"

    with pytest.raises(OuterTalkerEngineError, match="cache position is not contiguous"):
        validator(cache, torch.tensor([0, 2]))


def test_static_outer_engine_rejects_prefill_position_start_mismatch_before_model():
    runtime = FakeOuterStepRuntime(
        cache=FakeStaticCache(initial_seq_length=1, reset_seq_length=1)
    )
    talker = FakeStaticTalker(runtime=runtime)
    engine = OuterTalkerStaticEngine(talker, max_cache_len=8, step_runtime=runtime)

    with pytest.raises(OuterTalkerEngineError, match="cache position is not contiguous"):
        engine.generate(**fake_generation_kwargs(batch_size=1, max_new_tokens=2))

    assert talker.model.calls == []


def test_static_outer_engine_rejects_decode_position_mismatch_before_lease_run():
    runtime = FakeOuterStepRuntime(cache=FakeStaticCache(advance_extra=1))
    talker = FakeStaticTalker(runtime=runtime)
    engine = OuterTalkerStaticEngine(talker, max_cache_len=8, step_runtime=runtime)

    with pytest.raises(OuterTalkerEngineError, match="cache position is not contiguous"):
        engine.generate(**fake_generation_kwargs(batch_size=1, max_new_tokens=2))

    assert len(talker.model.calls) == 1
    assert runtime.leases[0].prepared_decodes == []


def test_static_outer_engine_runs_prefill_then_explicit_decode_without_shared_rope_state():
    runtime = FakeOuterStepRuntime()
    talker = FakeStaticTalker(runtime=runtime)
    engine = OuterTalkerStaticEngine(talker, max_cache_len=8, step_runtime=runtime)

    result = engine.generate(**fake_generation_kwargs(batch_size=2, max_new_tokens=3))

    lease = runtime.leases[0]
    assert talker.rope_deltas is None
    assert talker.model.cache_positions == [(0, 1, 2), (3,), (4,)]
    assert talker.code_predictor.batch_sizes == [2, 2]
    assert all(mask.ndim == 4 and mask.shape[-1] == 8 for mask in talker.model.decode_masks)
    first_mask = talker.model.decode_masks[0][:, 0, 0]
    masked = torch.finfo(torch.float32).min
    assert first_mask[0].tolist() == [0.0, 0.0, 0.0, 0.0, masked, masked, masked, masked]
    assert first_mask[1].tolist() == [masked, 0.0, 0.0, 0.0, masked, masked, masked, masked]
    assert talker.model.calls[1]["position_ids"][:, :, 0].tolist() == [
        [3.0, 2.0],
        [3.0, 2.0],
        [3.0, 2.0],
    ]
    assert len(result.hidden_states) == 3
    assert result.hidden_states[0][-1] is None
    assert result.hidden_states[1][-1].shape == (2, 3)
    assert lease.cache.reset_calls == 2
    assert lease.cache.reset_inference_modes == [True, True]
    assert lease.release_calls == 1
    assert runtime.acquire_calls == [(2, torch.device("cpu"), torch.float32, 8)]

    metrics = engine.metrics_snapshot()
    assert metrics == {
        "prefill_calls": 1,
        "static_steps_by_batch": {2: 2},
        "active_slots_per_step": [2, 2],
        "slot_occupancy": 1.0,
        "errors": 0,
    }


def test_static_outer_engine_matches_generation_mixin_max_token_history_semantics(
    monkeypatch,
):
    runtime = FakeOuterStepRuntime()
    talker = FakeStaticTalker(runtime=runtime)
    engine = OuterTalkerStaticEngine(talker, max_cache_len=8, step_runtime=runtime)
    original_sample = outer_static_module._sample_next_token
    sampled_scores = []

    def record_sample(scores, *args, **kwargs):
        sampled_scores.append(scores.clone())
        return original_sample(scores, *args, **kwargs)

    monkeypatch.setattr(outer_static_module, "_sample_next_token", record_sample)

    result = engine.generate(**fake_generation_kwargs(batch_size=1, max_new_tokens=5))

    assert len(result.hidden_states) == 5
    assert len(sampled_scores) == 5
    assert talker.model.cache_positions == [(0, 1, 2), (3,), (4,), (5,), (6,)]
    assert talker.code_predictor.batch_sizes == [1, 1, 1, 1]
    assert engine.metrics_snapshot()["static_steps_by_batch"] == {1: 4}


def test_engine_metrics_snapshot_cannot_observe_a_partial_logical_step():
    runtime = FakeOuterStepRuntime()
    talker = FakeStaticTalker(runtime=runtime)
    engine = OuterTalkerStaticEngine(talker, max_cache_len=8, step_runtime=runtime)
    append_started = Event()
    allow_append_to_return = Event()
    snapshot_started = Event()
    snapshot_done = Event()

    class BlockingList(list):
        def append(self, value):
            super().append(value)
            append_started.set()
            assert allow_append_to_return.wait(timeout=2)

    engine.active_slots_per_step = BlockingList()

    def take_snapshot():
        snapshot_started.set()
        try:
            return engine.metrics_snapshot()
        finally:
            snapshot_done.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        mutation = executor.submit(engine._record_logical_step, 2, 2)
        assert append_started.wait(timeout=2)
        snapshot = executor.submit(take_snapshot)
        assert snapshot_started.wait(timeout=2)
        try:
            assert not snapshot_done.wait(timeout=0.1)
        finally:
            allow_append_to_return.set()

        mutation.result(timeout=2)
        observed = snapshot.result(timeout=2)

    assert observed["static_steps_by_batch"] == {2: 1}
    assert observed["active_slots_per_step"] == [2]
    assert observed["slot_occupancy"] == 1.0


def test_engine_metric_increments_are_exact_under_concurrency():
    runtime = FakeOuterStepRuntime()
    talker = FakeStaticTalker(runtime=runtime)
    engine = OuterTalkerStaticEngine(talker, max_cache_len=8, step_runtime=runtime)
    worker_count = 16
    start = Barrier(worker_count + 1)

    def record_one_session():
        start.wait(timeout=2)
        engine._record_prefill_call()
        engine._record_logical_step(2, 1)
        engine._record_error()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(record_one_session) for _ in range(worker_count)]
        start.wait(timeout=2)
        for future in futures:
            future.result(timeout=2)

    snapshot = engine.metrics_snapshot()
    assert snapshot["prefill_calls"] == worker_count
    assert snapshot["static_steps_by_batch"] == {2: worker_count}
    assert snapshot["active_slots_per_step"] == [1] * worker_count
    assert snapshot["slot_occupancy"] == 0.5
    assert snapshot["errors"] == worker_count


def test_static_outer_engine_defers_outer_embedding_assembly_to_runtime_lease():
    runtime = FakeOuterStepRuntime()
    talker = FakeStaticTalker(runtime=runtime)
    engine = OuterTalkerStaticEngine(talker, max_cache_len=8, step_runtime=runtime)

    engine.generate(**fake_generation_kwargs(batch_size=1, max_new_tokens=2))

    prepared = runtime.leases[0].prepared_decodes[0]
    assert prepared.codec_ids.tolist() == [[1, 6, 7]]
    assert prepared.condition.tolist() == [[[3.0, 3.0, 3.0, 3.0]]]
    assert not hasattr(prepared, "inputs_embeds")
    assert talker._embedding.in_run == [False, True]
    assert [embedding.in_run for embedding in talker.code_predictor._embeddings] == [
        [True],
        [True],
    ]


def test_production_eager_runtime_assembles_decode_and_owns_cache_lifecycle(
    monkeypatch,
):
    talker = FakeStaticTalker()
    created_caches = []
    released_leases = []

    def cache_factory(**kwargs):
        cache = FakeStaticCache()
        created_caches.append((cache, kwargs))
        return cache

    monkeypatch.setattr(
        OuterRuntimeLease,
        "release",
        lambda lease: released_leases.append(lease),
    )
    runtime = EagerOuterStepRuntime(talker, cache_factory=cache_factory)
    engine = OuterTalkerStaticEngine(talker, max_cache_len=8, step_runtime=runtime)

    engine.generate(**fake_generation_kwargs(batch_size=1, max_new_tokens=2))

    cache, factory_kwargs = created_caches[0]
    expected_decode = torch.full(
        (1, 1, 4),
        1 + (6 + 10) + (7 + 20) + 3,
    )
    assert torch.equal(talker.model.calls[1]["inputs_embeds"], expected_decode)
    assert talker.model.calls[0]["past_key_values"] is cache
    assert talker.model.calls[1]["past_key_values"] is cache
    assert factory_kwargs["max_cache_len"] == 8
    assert cache.reset_calls == 2
    assert cache.reset_inference_modes == [True, True]
    assert len(released_leases) == 1
    assert released_leases[0].cache is cache


def test_static_outer_engine_rejects_unsupported_kwargs_before_acquiring_cache():
    runtime = FakeOuterStepRuntime()
    talker = FakeStaticTalker(runtime=runtime)
    engine = OuterTalkerStaticEngine(talker, step_runtime=runtime)
    kwargs = fake_generation_kwargs(batch_size=1, max_new_tokens=1)
    kwargs["num_beams"] = 2

    with pytest.raises(OuterTalkerEngineError, match="unsupported.*num_beams"):
        engine.generate(**kwargs)

    assert runtime.acquire_calls == []


def test_static_outer_engine_requires_explicit_pad_token_to_match_eos():
    runtime = FakeOuterStepRuntime()
    talker = FakeStaticTalker(runtime=runtime)
    engine = OuterTalkerStaticEngine(talker, step_runtime=runtime)
    kwargs = fake_generation_kwargs(batch_size=1, max_new_tokens=1)
    kwargs["pad_token_id"] = 8

    with pytest.raises(OuterTalkerEngineError, match="pad_token_id must equal"):
        engine.generate(**kwargs)

    assert runtime.acquire_calls == []

    kwargs["pad_token_id"] = 9
    engine.generate(**kwargs)
    assert len(runtime.acquire_calls) == 1


def test_static_outer_engine_rejects_cache_overflow_before_acquiring_cache():
    runtime = FakeOuterStepRuntime()
    talker = FakeStaticTalker(runtime=runtime)
    engine = OuterTalkerStaticEngine(talker, max_cache_len=4, step_runtime=runtime)

    with pytest.raises(OuterTalkerEngineError, match="cache capacity"):
        engine.generate(**fake_generation_kwargs(batch_size=1, max_new_tokens=3))

    assert runtime.acquire_calls == []


def test_static_outer_engine_suppresses_eos_until_minimum_tokens_are_generated():
    runtime = FakeOuterStepRuntime()
    talker = FakeStaticTalker(token_schedule=[9, 9, 9, 4], runtime=runtime)
    engine = OuterTalkerStaticEngine(talker, max_cache_len=8, step_runtime=runtime)
    kwargs = fake_generation_kwargs(batch_size=1, max_new_tokens=4)
    kwargs["min_new_tokens"] = 2

    result = engine.generate(**kwargs)

    assert talker.code_predictor.first_ids == [[0], [0]]
    assert len(result.hidden_states) == 3


def test_static_outer_engine_masks_finished_rows_to_eos():
    runtime = FakeOuterStepRuntime()
    talker = FakeStaticTalker(
        token_schedule=[[9, 1], [5, 2], [4, 9]],
        runtime=runtime,
    )
    engine = OuterTalkerStaticEngine(talker, max_cache_len=8, step_runtime=runtime)
    kwargs = fake_generation_kwargs(batch_size=2, max_new_tokens=4)
    kwargs["min_new_tokens"] = 0

    result = engine.generate(**kwargs)

    assert talker.code_predictor.first_ids == [[9, 1], [9, 2]]
    assert len(result.hidden_states) == 3
    assert engine.metrics_snapshot()["active_slots_per_step"] == [2, 1]


def test_static_outer_engine_does_not_decode_terminal_all_eos_sample(monkeypatch):
    runtime = FakeOuterStepRuntime()
    talker = FakeStaticTalker(token_schedule=[9], runtime=runtime)
    engine = OuterTalkerStaticEngine(talker, max_cache_len=8, step_runtime=runtime)
    kwargs = fake_generation_kwargs(batch_size=1, max_new_tokens=5)
    kwargs["min_new_tokens"] = 0
    original_sample = outer_static_module._sample_next_token
    sampled_scores = []

    def record_sample(scores, *args, **kwargs):
        sampled_scores.append(scores.clone())
        return original_sample(scores, *args, **kwargs)

    monkeypatch.setattr(outer_static_module, "_sample_next_token", record_sample)

    result = engine.generate(**kwargs)

    assert len(sampled_scores) == 1
    assert len(result.hidden_states) == 1
    assert talker.model.cache_positions == [(0, 1, 2)]
    assert talker.code_predictor.first_ids == []
    assert engine.metrics_snapshot()["static_steps_by_batch"] == {}


def test_static_outer_engine_samples_final_logits_without_extra_decode(monkeypatch):
    runtime = FakeOuterStepRuntime()
    talker = FakeStaticTalker(runtime=runtime)
    engine = OuterTalkerStaticEngine(talker, max_cache_len=8, step_runtime=runtime)
    original_sample = outer_static_module._sample_next_token
    sampled_scores = []

    def record_sample(scores, *args, **kwargs):
        sampled_scores.append(scores.clone())
        return original_sample(scores, *args, **kwargs)

    monkeypatch.setattr(outer_static_module, "_sample_next_token", record_sample)

    result = engine.generate(**fake_generation_kwargs(batch_size=1, max_new_tokens=1))

    assert len(sampled_scores) == 1
    assert len(result.hidden_states) == 1
    assert talker.model.cache_positions == [(0, 1, 2)]
    assert talker.code_predictor.first_ids == []


def test_static_outer_engine_returns_hidden_state_tuple_when_requested():
    runtime = FakeOuterStepRuntime()
    talker = FakeStaticTalker(runtime=runtime)
    engine = OuterTalkerStaticEngine(talker, max_cache_len=8, step_runtime=runtime)
    kwargs = fake_generation_kwargs(batch_size=1, max_new_tokens=2)
    kwargs["return_dict_in_generate"] = False

    result = engine.generate(**kwargs)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[1][-1].shape == (1, 3)


def test_static_outer_engine_resets_and_releases_lease_after_decode_error():
    runtime = FakeOuterStepRuntime(fail_on_run=True)
    talker = FakeStaticTalker(runtime=runtime)
    engine = OuterTalkerStaticEngine(talker, max_cache_len=8, step_runtime=runtime)

    with pytest.raises(RuntimeError, match="decode failed"):
        engine.generate(**fake_generation_kwargs(batch_size=1, max_new_tokens=2))

    lease = runtime.leases[0]
    assert lease.cache.reset_calls == 2
    assert lease.release_calls == 1
    assert engine.metrics_snapshot()["errors"] == 1


def test_install_static_outer_talker_replaces_generate_once_without_fallback():
    runtime = FakeOuterStepRuntime()
    talker = FakeStaticTalker(runtime=runtime)

    assert install_static_outer_talker(talker, max_cache_len=8) is True
    assert install_static_outer_talker(talker, max_cache_len=8) is False

    static_generate = talker.generate
    kwargs = fake_generation_kwargs(batch_size=1, max_new_tokens=1)
    kwargs["num_beams"] = 2
    with pytest.raises(OuterTalkerEngineError, match="unsupported.*num_beams"):
        static_generate(**kwargs)

    assert static_generate._qav_static_outer_talker is True
    assert static_generate._qav_original_generate is not static_generate
    assert isinstance(static_generate._qav_outer_engine, OuterTalkerStaticEngine)
