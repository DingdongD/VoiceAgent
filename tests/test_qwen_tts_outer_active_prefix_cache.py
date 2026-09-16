import pytest
import torch
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import qwen_asr_vllm.agent.qwen_tts_outer_active_prefix_cache as active_cache_module

from qwen_asr_vllm.agent.qwen_tts_outer_active_prefix_cache import (
    ActivePrefixCache,
    ActivePrefixCacheError,
    ActivePrefixLayer,
)
from qwen_asr_vllm.agent.qwen_tts_outer_static_engine import PreparedOuterDecode


class RuntimeEmbedding:
    def __init__(self, offset=0):
        self.offset = offset

    def __call__(self, token_ids):
        values = token_ids.to(dtype=torch.float32).unsqueeze(-1) + self.offset
        return values.expand(-1, -1, 4)


class RuntimePredictor:
    def __init__(self):
        self.generate_calls = 0
        self._embeddings = [RuntimeEmbedding(10), RuntimeEmbedding(20)]

    def get_input_embeddings(self):
        return self._embeddings

    def generate(self, **kwargs):
        self.generate_calls += 1
        raise AssertionError("lease execution must not invoke predictor generation")


class RuntimeModel:
    def __init__(self, talker, *, fail_after_cache_update=False):
        self.talker = talker
        self.fail_after_cache_update = fail_after_cache_update
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        batch_size, query_length, _ = kwargs["inputs_embeds"].shape
        states = torch.ones((batch_size, 1, query_length, 2))
        for layer_index in range(self.talker.config.num_hidden_layers):
            kwargs["past_key_values"].update(
                states,
                states + 1,
                layer_idx=layer_index,
                cache_kwargs={"cache_position": kwargs["cache_position"]},
            )
        if self.fail_after_cache_update:
            raise RuntimeError("model failed after cache update")
        return SimpleNamespace(
            last_hidden_state=kwargs["inputs_embeds"] + 1,
            hidden_states=(kwargs["inputs_embeds"] + 1,),
        )


class RuntimeTalker:
    def __init__(self, *, fail_after_cache_update=False):
        self.config = SimpleNamespace(num_hidden_layers=2, num_code_groups=3)
        self._embedding = RuntimeEmbedding()
        self.code_predictor = RuntimePredictor()
        self.model = RuntimeModel(
            self, fail_after_cache_update=fail_after_cache_update
        )

    def get_input_embeddings(self):
        return self._embedding

    def codec_head(self, hidden_states):
        return hidden_states.sum(-1, keepdim=True)


def prepared_decode(batch_size=1, position=0, *, max_cache_len=4):
    return PreparedOuterDecode(
        codec_ids=torch.tensor([[1, 6, 7]], dtype=torch.long).expand(batch_size, -1),
        condition=torch.full((batch_size, 1, 4), 3.0),
        attention_mask=torch.ones((batch_size, position + 1), dtype=torch.long),
        position_ids=torch.full((3, batch_size, 1), position),
        cache_position=torch.tensor([position], dtype=torch.long),
        output_hidden_states=True,
    )


def active_runtime(talker, *, max_cached_leases=2):
    runtime_type = getattr(active_cache_module, "ActivePrefixOuterStepRuntime", None)
    assert callable(runtime_type), "active-prefix outer runtime is missing"
    return runtime_type(talker, max_cached_leases=max_cached_leases)


def test_active_prefix_installer_sets_exact_metadata_and_runtime(monkeypatch):
    events = []

    class FakeRuntime:
        def __init__(self, talker):
            self.talker = talker
            events.append(("runtime", talker))

    class FakeEngine:
        def __init__(self, talker, *, max_cache_len, step_runtime):
            self.talker = talker
            self.max_cache_len = max_cache_len
            self.step_runtime = step_runtime
            events.append(("engine", talker, max_cache_len, step_runtime))

        def generate(self, **kwargs):
            return kwargs

    monkeypatch.setattr(
        active_cache_module, "ActivePrefixOuterStepRuntime", FakeRuntime
    )
    monkeypatch.setattr(active_cache_module, "OuterTalkerStaticEngine", FakeEngine)

    def original_generate(**kwargs):
        return kwargs

    talker = SimpleNamespace(generate=original_generate)
    installer = getattr(active_cache_module, "install_active_prefix_outer_talker", None)
    assert callable(installer), "active-prefix outer installer is missing"

    assert installer(talker, max_cache_len=512) is True
    installed_generate = talker.generate
    assert installed_generate(test_value=7) == {"test_value": 7}
    assert {
        name for name in vars(installed_generate) if name.startswith("_qav_")
    } == {
        "_qav_active_prefix_outer_talker",
        "_qav_static_outer_talker",
        "_qav_original_generate",
        "_qav_outer_engine",
    }
    assert installed_generate._qav_active_prefix_outer_talker is True
    assert installed_generate._qav_static_outer_talker is True
    assert installed_generate._qav_original_generate is original_generate
    engine = installed_generate._qav_outer_engine
    assert engine.max_cache_len == 512
    assert isinstance(engine.step_runtime, FakeRuntime)
    assert engine.step_runtime.talker is talker
    assert events == [
        ("runtime", talker),
        ("engine", talker, 512, engine.step_runtime),
    ]

    assert installer(talker, max_cache_len=512) is False
    assert talker.generate is installed_generate
    assert len(events) == 2


@pytest.mark.parametrize(
    "metadata_name",
    ["_qav_static_outer_talker", "_qav_explicit_talker_step_engine"],
)
def test_active_prefix_installer_rejects_other_outer_engines(metadata_name):
    def installed_generate(**kwargs):
        return kwargs

    setattr(installed_generate, metadata_name, True)
    talker = SimpleNamespace(generate=installed_generate)
    installer = getattr(active_cache_module, "install_active_prefix_outer_talker", None)
    assert callable(installer), "active-prefix outer installer is missing"

    with pytest.raises(ActivePrefixCacheError, match="already installed"):
        installer(talker, max_cache_len=8)

    assert talker.generate is installed_generate


def test_active_prefix_runtime_builds_dynamic_two_dimensional_mask():
    runtime = active_runtime(RuntimeTalker(), max_cached_leases=2)
    prompt_mask = torch.tensor([[1, 1, 1], [0, 1, 1]])

    first = runtime.build_attention_mask(
        prompt_mask,
        cache_position=3,
        max_cache_len=8,
        dtype=torch.float32,
    )
    second = runtime.build_attention_mask(
        prompt_mask,
        cache_position=4,
        max_cache_len=8,
        dtype=torch.float32,
    )

    assert first.tolist() == [[1, 1, 1, 1], [0, 1, 1, 1]]
    assert second.tolist() == [[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]


def test_active_prefix_lease_runs_shared_outer_body_without_predictor_generation():
    talker = RuntimeTalker()
    runtime = active_runtime(talker)
    lease = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)
    prepared = prepared_decode()

    output = lease.run(prepared)

    assert talker.code_predictor.generate_calls == 0
    assert talker.model.calls[0]["past_key_values"] is lease.cache
    assert talker.model.calls[0]["attention_mask"] is prepared.attention_mask
    assert output.past_hidden.shape == (1, 1, 4)
    lease.release()


def test_active_prefix_pool_reuses_cache_and_backing_storage():
    runtime = active_runtime(RuntimeTalker(), max_cached_leases=1)
    first = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)
    first.run(prepared_decode())
    pointers = tuple(
        pointer
        for layer in first.cache.layers
        for pointer in (layer.keys.data_ptr(), layer.values.data_ptr())
    )
    cache = first.cache
    first.release()

    second = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)

    assert second.cache is cache
    assert second.cache.get_seq_length() == 0
    assert tuple(
        pointer
        for layer in second.cache.layers
        for pointer in (layer.keys.data_ptr(), layer.values.data_ptr())
    ) == pointers
    second.release()


def test_checked_out_active_prefix_leases_never_share_a_cache():
    runtime = active_runtime(RuntimeTalker(), max_cached_leases=2)

    first = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)
    second = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)

    assert first.cache is not second.cache
    first.release()
    second.release()


def test_active_prefix_exception_cleanup_returns_a_reset_cache():
    runtime = active_runtime(
        RuntimeTalker(fail_after_cache_update=True), max_cached_leases=1
    )
    first = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)
    cache = first.cache

    with pytest.raises(RuntimeError, match="model failed"):
        first.run(prepared_decode())
    first.release()
    second = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)

    assert second.cache is cache
    assert second.cache.get_seq_length() == 0
    assert all(
        torch.count_nonzero(layer.keys) == 0
        and torch.count_nonzero(layer.values) == 0
        for layer in second.cache.layers
    )
    second.release()


def test_active_prefix_release_is_idempotent_and_concurrency_safe():
    runtime = active_runtime(RuntimeTalker(), max_cached_leases=1)
    lease = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: lease.release(), range(8)))

    first = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)
    second = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)
    assert first.cache is lease.cache
    assert second.cache is not lease.cache
    first.release()
    second.release()


def test_active_prefix_pool_terminally_releases_excess_idle_cache():
    runtime = active_runtime(RuntimeTalker(), max_cached_leases=1)
    first = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)
    second = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)
    first.run(prepared_decode())
    second.run(prepared_decode())

    first.release()
    second.release()

    assert all(layer.keys is None for layer in second.cache.layers)
    with pytest.raises(ActivePrefixCacheError, match="released"):
        second.cache.update(
            torch.ones((1, 1, 1, 2)),
            torch.ones((1, 1, 1, 2)),
            layer_idx=0,
            cache_kwargs={"cache_position": torch.tensor([0])},
        )


def test_active_prefix_metrics_have_defined_counters_and_immutable_snapshots():
    runtime = active_runtime(RuntimeTalker(), max_cached_leases=1)
    lease = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)
    lease.run(prepared_decode())
    lease.run(prepared_decode(position=1))
    lease.release()

    snapshot = runtime.metrics_snapshot()

    assert snapshot == {
        "cache_allocations": 1,
        "allocated_kv_bytes": 128,
        "active_kv_tokens_per_step": [1, 2],
        "backing_capacity_tokens": 4,
        "active_capacity_ratio": 0.375,
        "cache_resets": 1,
        "cache_overflows": 0,
    }
    snapshot["active_kv_tokens_per_step"].append(99)
    snapshot["cache_allocations"] = 99
    assert runtime.metrics_snapshot()["active_kv_tokens_per_step"] == [1, 2]
    assert runtime.metrics_snapshot()["cache_allocations"] == 1


def test_active_prefix_runtime_counts_capacity_overflow_once_per_failed_run():
    runtime = active_runtime(RuntimeTalker())
    lease = runtime.acquire(1, torch.device("cpu"), torch.float32, 1)
    lease.run(prepared_decode(position=0, max_cache_len=1))

    with pytest.raises(ActivePrefixCacheError, match="capacity"):
        lease.run(prepared_decode(position=1, max_cache_len=1))

    assert runtime.metrics_snapshot()["cache_overflows"] == 1
    lease.release()


def test_active_prefix_runtime_close_releases_idle_and_rejects_acquire():
    runtime = active_runtime(RuntimeTalker(), max_cached_leases=1)
    lease = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)
    lease.run(prepared_decode())
    cache = lease.cache
    lease.release()

    runtime.close()
    runtime.close()

    assert all(layer.keys is None and layer.values is None for layer in cache.layers)
    snapshot = runtime.metrics_snapshot()
    assert snapshot["allocated_kv_bytes"] == 0
    assert snapshot["backing_capacity_tokens"] == 0
    with pytest.raises(ActivePrefixCacheError, match="closed"):
        runtime.acquire(1, torch.device("cpu"), torch.float32, 4)


@pytest.mark.parametrize("terminal_path", ["close", "excess", "discard"])
def test_terminal_runtime_release_runs_under_inference_mode(
    monkeypatch, terminal_path
):
    release_modes = []
    original_release = ActivePrefixCache.release

    def recording_release(cache):
        release_modes.append(torch.is_inference_mode_enabled())
        original_release(cache)

    monkeypatch.setattr(ActivePrefixCache, "release", recording_release)
    runtime = active_runtime(RuntimeTalker(), max_cached_leases=1)

    if terminal_path == "close":
        lease = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)
        lease.run(prepared_decode())
        lease.release()
        runtime.close()
    elif terminal_path == "excess":
        retained = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)
        excess = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)
        retained.run(prepared_decode())
        excess.run(prepared_decode())
        retained.release()
        excess.release()
    else:
        lease = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)
        lease.run(prepared_decode())

        def fail_reset():
            raise RuntimeError("reset failed")

        lease.cache.reset = fail_reset
        with pytest.raises(RuntimeError, match="reset failed"):
            lease.release()

    assert release_modes == [True]


def test_close_racing_checked_out_return_is_deadlock_free_and_terminal():
    runtime = active_runtime(RuntimeTalker(), max_cached_leases=1)
    lease = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)
    lease.run(prepared_decode())
    cache = lease.cache
    reset_barrier = Barrier(2)
    original_reset = cache.reset

    def barrier_reset():
        reset_barrier.wait(timeout=2)
        reset_barrier.wait(timeout=2)
        original_reset()

    cache.reset = barrier_reset
    with ThreadPoolExecutor(max_workers=2) as executor:
        returned = executor.submit(lease.release)
        reset_barrier.wait(timeout=2)
        closed = executor.submit(runtime.close)
        closed.result(timeout=2)
        reset_barrier.wait(timeout=2)
        returned.result(timeout=2)

    assert all(layer.keys is None and layer.values is None for layer in cache.layers)
    snapshot = runtime.metrics_snapshot()
    assert snapshot["allocated_kv_bytes"] == 0
    assert snapshot["backing_capacity_tokens"] == 0
    assert runtime._idle == {}
    with pytest.raises(ActivePrefixCacheError, match="closed"):
        runtime.acquire(1, torch.device("cpu"), torch.float32, 4)


def test_zero_idle_limit_does_not_create_empty_pool_keys():
    runtime = active_runtime(RuntimeTalker(), max_cached_leases=0)
    lease = runtime.acquire(1, torch.device("cpu"), torch.float32, 4)

    assert runtime._idle == {}
    lease.release()

    assert runtime._idle == {}


def test_active_prefix_layer_keeps_backing_address_and_returns_live_view():
    layer = ActivePrefixLayer(max_cache_len=8)
    first_keys = torch.full((2, 3, 3, 4), 1.0)
    first_values = torch.full_like(first_keys, 2.0)

    keys, values = layer.update(
        first_keys,
        first_values,
        {"cache_position": torch.arange(3)},
    )
    backing_ptrs = (layer.keys.data_ptr(), layer.values.data_ptr())

    next_keys = torch.full((2, 3, 1, 4), 5.0)
    next_values = torch.full_like(next_keys, 6.0)
    keys, values = layer.update(
        next_keys,
        next_values,
        {"cache_position": torch.tensor([3])},
    )

    assert keys.shape == values.shape == (2, 3, 4, 4)
    assert (layer.keys.data_ptr(), layer.values.data_ptr()) == backing_ptrs
    assert keys.data_ptr() == layer.keys.data_ptr()
    assert values.data_ptr() == layer.values.data_ptr()
    assert torch.equal(keys[:, :, :3], first_keys)
    assert torch.equal(keys[:, :, 3:4], next_keys)
    assert torch.count_nonzero(layer.keys[:, :, 4:]) == 0


def test_active_prefix_layer_exposes_dynamic_cache_sized_metadata():
    layer = ActivePrefixLayer(max_cache_len=5)
    sample = torch.ones((1, 2, 2, 3))

    layer.lazy_initialization(sample)

    assert layer.is_initialized
    assert layer.keys.shape == layer.values.shape == (1, 2, 5, 3)
    assert layer.get_seq_length() == 0
    assert layer.get_mask_sizes(torch.tensor([0, 1])) == (2, 0)
    assert layer.get_max_cache_shape() == 5
    assert layer.is_compileable is False

    keys, values = layer.update(
        sample,
        sample + 1,
        {"cache_position": torch.tensor([0, 1])},
    )

    assert keys.shape == values.shape == (1, 2, 2, 3)
    assert layer.get_seq_length() == 2
    assert layer.get_mask_sizes(torch.tensor([2])) == (3, 0)


def test_active_prefix_cache_updates_exactly_the_requested_layer():
    cache = ActivePrefixCache(num_hidden_layers=2, max_cache_len=4)
    key_states = torch.full((1, 2, 2, 3), 4.0)
    value_states = torch.full_like(key_states, 9.0)

    first_keys, first_values = cache.update(
        key_states,
        value_states,
        layer_idx=0,
        cache_kwargs={"cache_position": torch.tensor([0, 1])},
    )
    second_keys, second_values = cache.update(
        key_states + 1,
        value_states + 1,
        layer_idx=1,
        cache_kwargs={"cache_position": torch.tensor([0, 1])},
    )

    assert len(cache.layers) == 2
    assert torch.equal(first_keys, key_states)
    assert torch.equal(first_values, value_states)
    assert torch.equal(second_keys, key_states + 1)
    assert torch.equal(second_values, value_states + 1)
    assert cache.get_seq_length(0) == cache.get_seq_length(1) == 2


def test_reset_clears_only_active_prefix_and_preserves_inference_allocations():
    layer = ActivePrefixLayer(max_cache_len=5)
    key_states = torch.full((1, 1, 2, 2), 3.0)
    value_states = torch.full_like(key_states, 7.0)

    with torch.inference_mode():
        layer.update(
            key_states,
            value_states,
            {"cache_position": torch.tensor([0, 1])},
        )
        layer.keys[:, :, 4:].fill_(11.0)
        layer.values[:, :, 4:].fill_(13.0)
    key_ptr, value_ptr = layer.keys.data_ptr(), layer.values.data_ptr()

    layer.reset()

    assert (layer.keys.data_ptr(), layer.values.data_ptr()) == (key_ptr, value_ptr)
    assert layer.get_seq_length() == 0
    assert torch.count_nonzero(layer.keys[:, :, :2]) == 0
    assert torch.count_nonzero(layer.values[:, :, :2]) == 0
    assert torch.equal(layer.keys[:, :, 4:], torch.full((1, 1, 1, 2), 11.0))
    assert torch.equal(layer.values[:, :, 4:], torch.full((1, 1, 1, 2), 13.0))


@pytest.mark.parametrize(
"key_states,value_states,cache_kwargs",
[
    (torch.ones((1, 1, 1, 2)), torch.ones((1, 1, 1, 2)), {}),
    (
        torch.ones((1, 1, 1, 2)),
        torch.ones((1, 1, 1, 2)),
        {"cache_position": torch.zeros((1, 1), dtype=torch.long)},
    ),
    (
        torch.ones((1, 1, 1, 2)),
        torch.ones((1, 1, 1, 2)),
        {"cache_position": torch.tensor([0, 1])},
    ),
    (
        torch.ones((1, 1, 1, 2)),
        torch.ones((1, 1, 1, 3)),
        {"cache_position": torch.tensor([0])},
    ),
],
)
def test_invalid_updates_leave_existing_cache_contents_unchanged(
    key_states, value_states, cache_kwargs
):
    layer = ActivePrefixLayer(max_cache_len=3)
    initial_keys = torch.full((1, 1, 2, 2), 2.0)
    initial_values = torch.full_like(initial_keys, 6.0)
    layer.update(
        initial_keys,
        initial_values,
        {"cache_position": torch.tensor([0, 1])},
    )
    before_keys = layer.keys.clone()
    before_values = layer.values.clone()

    with pytest.raises(ActivePrefixCacheError):
        layer.update(key_states, value_states, cache_kwargs)

    assert layer.get_seq_length() == 2
    assert torch.equal(layer.keys, before_keys)
    assert torch.equal(layer.values, before_values)


def test_capacity_and_layout_failures_leave_existing_cache_contents_unchanged():
    layer = ActivePrefixLayer(max_cache_len=3)
    initial = torch.full((1, 1, 2, 2), 2.0)
    layer.update(initial, initial + 1, {"cache_position": torch.tensor([0, 1])})
    before_keys = layer.keys.clone()
    before_values = layer.values.clone()

    with pytest.raises(ActivePrefixCacheError, match="capacity"):
        layer.update(
            torch.ones((1, 1, 2, 2)),
            torch.ones((1, 1, 2, 2)),
            {"cache_position": torch.tensor([2, 3])},
        )
    with pytest.raises(ActivePrefixCacheError, match="layout"):
        layer.update(
            torch.ones((2, 1, 1, 2)),
            torch.ones((2, 1, 1, 2)),
            {"cache_position": torch.tensor([2])},
        )

    assert layer.get_seq_length() == 2
    assert torch.equal(layer.keys, before_keys)
    assert torch.equal(layer.values, before_values)


@pytest.mark.parametrize(
    "key_states,value_states,cache_position",
    [
        (
            object(),
            torch.ones((1, 1, 1, 2)),
            torch.tensor([0]),
        ),
        (
            torch.ones((1, 1, 1, 2)),
            object(),
            torch.tensor([0]),
        ),
        (
            torch.ones((1, 1, 2)),
            torch.ones((1, 1, 2)),
            torch.tensor([0]),
        ),
        (
            torch.ones((1, 1, 1, 2)),
            torch.ones((1, 2, 1, 2)),
            torch.tensor([0]),
        ),
        (
            torch.ones((1, 1, 1, 2), dtype=torch.float32),
            torch.ones((1, 1, 1, 2), dtype=torch.float64),
            torch.tensor([0]),
        ),
        (
            torch.ones((1, 1, 1, 2)),
            torch.ones((1, 1, 1, 2), device="meta"),
            torch.tensor([0]),
        ),
        (
            torch.ones((1, 1, 1, 2)).to_sparse(),
            torch.ones((1, 1, 1, 2)).to_sparse(),
            torch.tensor([0]),
        ),
        (
            torch.ones((1, 1, 1, 2)),
            torch.ones((1, 1, 1, 2)),
            [0],
        ),
        (
            torch.ones((1, 1, 1, 2)),
            torch.ones((1, 1, 1, 2)),
            torch.tensor([0], dtype=torch.int32),
        ),
        (
            torch.ones((1, 1, 1, 2)),
            torch.ones((1, 1, 1, 2)),
            torch.tensor([0], dtype=torch.long, device="meta"),
        ),
        (
            torch.ones((1, 1, 1, 2)),
            torch.ones((1, 1, 1, 2)),
            torch.tensor([[0]], dtype=torch.long),
        ),
        (
            torch.ones((1, 1, 2, 2)),
            torch.ones((1, 1, 2, 2)),
            torch.tensor([0], dtype=torch.long),
        ),
        (
            torch.ones((1, 1, 4, 2)),
            torch.ones((1, 1, 4, 2)),
            torch.arange(4, dtype=torch.long),
        ),
        (
            torch.ones((1, 1, 1, 2)),
            torch.ones((1, 1, 1, 2)),
            torch.tensor([3], dtype=torch.long),
        ),
    ],
)
def test_invalid_first_update_leaves_layer_uninitialized(
    key_states, value_states, cache_position
):
    layer = ActivePrefixLayer(max_cache_len=3)

    with pytest.raises(ActivePrefixCacheError):
        layer.update(
            key_states,
            value_states,
            {"cache_position": cache_position},
        )

    assert layer.is_initialized is False
    assert layer.keys is None
    assert layer.values is None
    assert layer.get_seq_length() == 0


@pytest.mark.parametrize(
    "key_states,value_states,cache_position",
    [
        (
            torch.full((1, 1, 1, 2), 11.0, dtype=torch.float32),
            torch.full((1, 1, 1, 2), 12.0, dtype=torch.float64),
            torch.tensor([2], dtype=torch.long),
        ),
        (
            torch.full((1, 1, 1, 2), 11.0, dtype=torch.float64),
            torch.full((1, 1, 1, 2), 12.0, dtype=torch.float64),
            torch.tensor([2], dtype=torch.long),
        ),
        (
            torch.full((1, 1, 1, 2), 11.0),
            torch.full((1, 1, 1, 2), 12.0),
            [2],
        ),
        (
            torch.full((1, 1, 1, 2), 11.0),
            torch.full((1, 1, 1, 2), 12.0),
            torch.tensor([2], dtype=torch.int32),
        ),
        (
            torch.full((1, 1, 1, 2), 11.0),
            torch.full((1, 1, 1, 2), 12.0),
            torch.tensor([4], dtype=torch.long),
        ),
    ],
)
def test_invalid_initialized_update_is_atomic(
    key_states, value_states, cache_position
):
    layer = ActivePrefixLayer(max_cache_len=4)
    initial_keys = torch.full((1, 1, 2, 2), 2.0)
    initial_values = torch.full_like(initial_keys, 6.0)
    layer.update(
        initial_keys,
        initial_values,
        {"cache_position": torch.tensor([0, 1])},
    )
    before_keys = layer.keys.clone()
    before_values = layer.values.clone()

    with pytest.raises(ActivePrefixCacheError):
        layer.update(
            key_states,
            value_states,
            {"cache_position": cache_position},
        )

    assert layer.get_seq_length() == 2
    assert torch.equal(layer.keys, before_keys)
    assert torch.equal(layer.values, before_values)


@pytest.mark.parametrize(
    "input_name,backing_name",
    [
        ("key_states", "keys"),
        ("key_states", "values"),
        ("value_states", "keys"),
        ("value_states", "values"),
    ],
)
def test_initialized_update_rejects_kv_backing_storage_alias_atomically(
    input_name, backing_name
):
    layer = ActivePrefixLayer(max_cache_len=4)
    initial_keys = torch.full((1, 1, 2, 2), 2.0)
    initial_values = torch.full_like(initial_keys, 6.0)
    layer.update(
        initial_keys,
        initial_values,
        {"cache_position": torch.tensor([0, 1])},
    )
    key_states = torch.full((1, 1, 1, 2), 11.0)
    value_states = torch.full_like(key_states, 12.0)
    aliased_input = getattr(layer, backing_name)[:, :, 2:3]
    if input_name == "key_states":
        key_states = aliased_input
    else:
        value_states = aliased_input
    before_keys = layer.keys.clone()
    before_values = layer.values.clone()

    with pytest.raises(ActivePrefixCacheError):
        layer.update(
            key_states,
            value_states,
            {"cache_position": torch.tensor([2])},
        )

    assert layer.get_seq_length() == 2
    assert torch.equal(layer.keys, before_keys)
    assert torch.equal(layer.values, before_values)


@pytest.mark.parametrize("backing_name", ["keys", "values"])
def test_initialized_update_rejects_cache_position_backing_storage_alias_atomically(
    backing_name,
):
    layer = ActivePrefixLayer(max_cache_len=4)
    initial_keys = torch.full((1, 1, 2, 2), 2, dtype=torch.long)
    initial_values = torch.full_like(initial_keys, 6)
    layer.update(
        initial_keys,
        initial_values,
        {"cache_position": torch.tensor([0, 1])},
    )
    cache_position = getattr(layer, backing_name)[:, :, 2:3, 0].reshape(-1)
    key_states = torch.full((1, 1, 1, 2), 11, dtype=torch.long)
    value_states = torch.full_like(key_states, 12)
    before_keys = layer.keys.clone()
    before_values = layer.values.clone()

    with pytest.raises(ActivePrefixCacheError):
        layer.update(
            key_states,
            value_states,
            {"cache_position": cache_position},
        )

    assert layer.get_seq_length() == 2
    assert torch.equal(layer.keys, before_keys)
    assert torch.equal(layer.values, before_values)


def test_device_mismatches_are_rejected_before_mutation():
    layer = ActivePrefixLayer(max_cache_len=4)
    initial = torch.ones((1, 1, 1, 2))
    layer.update(initial, initial, {"cache_position": torch.tensor([0])})
    before_keys = layer.keys.clone()
    before_values = layer.values.clone()
    meta_values = torch.ones((1, 1, 1, 2), device="meta")
    meta_position = torch.tensor([1], device="meta", dtype=torch.long)

    with pytest.raises(ActivePrefixCacheError):
        layer.update(initial, meta_values, {"cache_position": torch.tensor([1])})
    with pytest.raises(ActivePrefixCacheError):
        layer.update(initial, initial, {"cache_position": meta_position})

    assert layer.get_seq_length() == 1
    assert torch.equal(layer.keys, before_keys)
    assert torch.equal(layer.values, before_values)


def test_cache_reset_clears_all_initialized_layers_and_preserves_pointers():
    cache = ActivePrefixCache(num_hidden_layers=3, max_cache_len=4)
    first = torch.full((1, 1, 2, 2), 3.0)
    second = torch.full((1, 1, 1, 2), 5.0)
    with torch.inference_mode():
        cache.update(
            first,
            first + 1,
            layer_idx=0,
            cache_kwargs={"cache_position": torch.tensor([0, 1])},
        )
        cache.update(
            second,
            second + 1,
            layer_idx=2,
            cache_kwargs={"cache_position": torch.tensor([0])},
        )
    pointers = {
        index: (cache.layers[index].keys.data_ptr(), cache.layers[index].values.data_ptr())
        for index in (0, 2)
    }

    cache.reset()

    assert cache.layers[1].is_initialized is False
    for index in (0, 2):
        layer = cache.layers[index]
        assert layer.get_seq_length() == 0
        assert (layer.keys.data_ptr(), layer.values.data_ptr()) == pointers[index]
        assert torch.count_nonzero(layer.keys) == 0
        assert torch.count_nonzero(layer.values) == 0


def test_terminal_release_rejects_reuse_and_discards_storage():
    cache = ActivePrefixCache(num_hidden_layers=2, max_cache_len=3)
    states = torch.ones((1, 1, 1, 2))
    cache.update(
        states,
        states,
        layer_idx=0,
        cache_kwargs={"cache_position": torch.tensor([0])},
    )

    cache.release()

    assert all(layer.keys is None and layer.values is None for layer in cache.layers)
    assert all(layer.is_initialized is False for layer in cache.layers)
    with pytest.raises(ActivePrefixCacheError, match="released"):
        cache.update(
            states,
            states,
            layer_idx=0,
            cache_kwargs={"cache_position": torch.tensor([0])},
        )


@pytest.mark.parametrize("num_hidden_layers,max_cache_len", [(0, 4), (1, 0), (1, -1)])
def test_invalid_cache_configuration_is_rejected(num_hidden_layers, max_cache_len):
    with pytest.raises(ActivePrefixCacheError):
        ActivePrefixCache(
            num_hidden_layers=num_hidden_layers,
            max_cache_len=max_cache_len,
        )
