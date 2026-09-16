# Qwen-TTS Active-Prefix Outer Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failed fixed-full-extent outer `StaticCache` parity path with a preallocated active-prefix cache that preserves upstream `DynamicCache` attention semantics and exact greedy codec IDs.

**Architecture:** `ActivePrefixLayer` owns fixed-capacity K/V backing but returns only live-prefix views. `ActivePrefixOuterStepRuntime` reuses the existing explicit outer request/cohort engine while supplying DynamicCache-equivalent 2D masks and active cache lengths. Backend and profiler wiring remain opt-in, and all outer CUDA Graph work stays blocked until batch-1/batch-2 bitwise and full codec parity pass on CUDA 2.

**Tech Stack:** Python 3.11/3.13, PyTorch 2.5.1 CUDA, Transformers 4.57.6 public `Cache` APIs, local `qwen_tts`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-26-qwen-tts-active-prefix-cache-design.md`

## Global Constraints

- Do not modify files inside installed `qwen_tts` or `transformers` packages.
- Do not capture active-prefix decode with CUDA Graphs in this plan.
- Do not relax exact raw codec-ID parity to waveform-only or perceptual parity.
- Do not silently fall back to upstream generation when active-prefix mode is enabled.
- Use public Transformers 4.57.6 `Cache` and `CacheLayerMixin`; fail clearly on incompatible APIs.
- Validate contiguous cache positions once per outer model call, not once per decoder layer.
- Keep predictor generation outside the outer runtime lease.
- Do not add an implicit `.contiguous()` K/V copy to force attention parity.
  Record prefix-view shape/stride/contiguity and stop for a separate attention
  adapter design if fixed-capacity view layout changes numerical dispatch.
- Use CUDA 2 for real parity and stability tests; leave CUDA 3 contention out of acceptance timing.
- Do not create partial commits: the repository still has no baseline commit and all project sources are untracked. Record checkpoints in `progress.md` and the SDD ledger.

## File Structure

- Create `qwen_asr_vllm/agent/qwen_tts_outer_active_prefix_cache.py`: cache layer/cache, runtime pool/lease, installer, and cache metrics.
- Create `tests/test_qwen_tts_outer_active_prefix_cache.py`: cache contract, runtime, pooling, installer, and failure tests.
- Modify `qwen_asr_vllm/agent/qwen_tts_outer_static_engine.py`: runtime-owned mask strategy and one-call cache-position validation.
- Modify `tests/test_qwen_tts_outer_static_engine.py`: fixed-mask compatibility and active-runtime injection tests.
- Modify `qwen_asr_vllm/agent/local_tts.py`: mutually exclusive active-prefix flag, warmup, close, and immutable runtime metrics.
- Modify `tests/test_agent_local_tts.py`: backend precedence, warmup, shutdown, and metrics tests.
- Modify `bench/qwen_tts_internal_profile.py`: active-prefix CLI mode and report fields.
- Modify `tests/test_qwen_tts_internal_profile.py`: parser, factory, intended mode, and health-report tests.
- Create `bench/qwen_tts_active_prefix_parity_probe.py`: boundary and full-codec CUDA parity report.
- Create `tests/test_qwen_tts_active_prefix_parity_probe.py`: comparison, mismatch, serialization, and gate tests.
- Write CUDA artifacts under `results/qwen_tts_active_prefix_*.json` only from successful or explicitly failed controlled runs.

---

### Task 1: Implement The Active-Prefix Cache Contract

**Files:**
- Create: `qwen_asr_vllm/agent/qwen_tts_outer_active_prefix_cache.py`
- Create: `tests/test_qwen_tts_outer_active_prefix_cache.py`
- Modify: `progress.md`

**Interfaces:**
- Produces: `ActivePrefixCacheError`, `ActivePrefixLayer(max_cache_len)`, and `ActivePrefixCache(num_hidden_layers, max_cache_len)`.
- Consumes: Transformers 4.57.6 public `Cache` and `CacheLayerMixin`, plus PyTorch K/V tensors shaped `[batch, kv_heads, query, head_dim]`.

- [x] **Step 1: Write failing allocation and active-view tests**

```python
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
    assert torch.equal(keys[:, :, :3], first_keys)
    assert torch.equal(keys[:, :, 3:4], next_keys)
    assert torch.count_nonzero(layer.keys[:, :, 4:]) == 0
```

Add tests for `lazy_initialization`, `get_seq_length`, `get_mask_sizes`,
`get_max_cache_shape`, `is_compileable=False`, active-prefix-only return values,
and a multi-layer `ActivePrefixCache.update` call.

- [x] **Step 2: Run RED tests**

Run: `pytest tests/test_qwen_tts_outer_active_prefix_cache.py -q`

Expected: collection fails because the active-prefix module does not exist.

- [x] **Step 3: Implement the public Transformers cache types**

Implement the layer with the exact hot-path contract:

```python
class ActivePrefixLayer(CacheLayerMixin):
    is_compileable = False

    def __init__(self, max_cache_len: int):
        super().__init__()
        if int(max_cache_len) <= 0:
            raise ActivePrefixCacheError("max_cache_len must be positive")
        self.max_cache_len = int(max_cache_len)
        self.active_length = 0
        self._released = False

    def lazy_initialization(self, key_states):
        batch, heads, _, head_dim = key_states.shape
        shape = (batch, heads, self.max_cache_len, head_dim)
        self.keys = key_states.new_zeros(shape)
        self.values = key_states.new_zeros(shape)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if self._released:
            raise ActivePrefixCacheError("active-prefix layer is released")
        cache_position = (cache_kwargs or {}).get("cache_position")
        if cache_position is None or cache_position.ndim != 1:
            raise ActivePrefixCacheError("one-dimensional cache_position is required")
        if key_states.shape != value_states.shape or key_states.ndim != 4:
            raise ActivePrefixCacheError("key/value states must share a four-dimensional shape")
        if cache_position.numel() != key_states.shape[-2]:
            raise ActivePrefixCacheError("cache_position length must match query length")
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        if (
            key_states.shape[0], key_states.shape[1], key_states.shape[3]
        ) != (
            self.keys.shape[0], self.keys.shape[1], self.keys.shape[3]
        ):
            raise ActivePrefixCacheError("key/value layout changed after allocation")
        query_length = int(key_states.shape[-2])
        next_length = self.active_length + query_length
        if next_length > self.max_cache_len:
            raise ActivePrefixCacheError("active-prefix cache capacity exceeded")
        self.keys.index_copy_(2, cache_position, key_states)
        self.values.index_copy_(2, cache_position, value_states)
        self.active_length = next_length
        return self.keys[:, :, :next_length], self.values[:, :, :next_length]

    def get_seq_length(self):
        return self.active_length

    def get_mask_sizes(self, cache_position):
        return self.active_length + int(cache_position.shape[0]), 0

    def get_max_cache_shape(self):
        return self.max_cache_len
```

`ActivePrefixCache` creates exactly `num_hidden_layers` layer objects and calls
the public `Cache(layers=layers)` constructor. Import/version errors must be
wrapped in `ActivePrefixCacheError` and include `transformers.__version__`.

- [x] **Step 4: Add reset, terminal release, and failure tests**

Test that reset zeros only the previous active prefix, preserves data pointers,
sets all layer lengths to zero, and runs under inference mode. Test capacity
overflow, layout changes, missing/wrong-dimensional `cache_position`, and reuse
after terminal storage release. Assert failures leave `active_length` and
backing contents unchanged. Position-hole rejection belongs to the one-call
engine validator in Task 2, not to every layer update.

- [x] **Step 5: Run GREEN and compile checks**

Run:

```bash
pytest tests/test_qwen_tts_outer_active_prefix_cache.py -q
python -m py_compile qwen_asr_vllm/agent/qwen_tts_outer_active_prefix_cache.py
```

Expected: all Task 1 tests pass without warnings.

- [x] **Step 6: Record checkpoint**

Append public interfaces, test count, and the no-device-sync hot-path decision to `progress.md`.

---

### Task 2: Add Runtime-Owned Mask Strategy And Cache Leases

**Files:**
- Modify: `qwen_asr_vllm/agent/qwen_tts_outer_static_engine.py`
- Modify: `qwen_asr_vllm/agent/qwen_tts_outer_active_prefix_cache.py`
- Modify: `tests/test_qwen_tts_outer_static_engine.py`
- Modify: `tests/test_qwen_tts_outer_active_prefix_cache.py`
- Modify: `progress.md`

**Interfaces:**
- Produces: `ActivePrefixOuterStepRuntime`, active-prefix runtime leases, `build_attention_mask(...)`, `metrics_snapshot()`, and `close()`.
- Consumes: Task 1 cache types and the existing `OuterRuntimeLease`/`PreparedOuterDecode` contracts.

- [x] **Step 1: Write failing runtime-mask tests**

```python
def test_active_prefix_runtime_builds_dynamic_two_dimensional_mask():
    runtime = ActivePrefixOuterStepRuntime(FakeTalker(), max_cached_leases=2)
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
```

Add a compatibility test proving `EagerOuterStepRuntime` still returns the
existing fixed 4D additive mask and all current static-engine tests remain
unchanged.

- [x] **Step 2: Run RED tests**

Run:

```bash
pytest tests/test_qwen_tts_outer_active_prefix_cache.py tests/test_qwen_tts_outer_static_engine.py -q
```

Expected: active-prefix runtime symbols and runtime mask hook are missing.

- [x] **Step 3: Move mask construction behind the runtime interface**

Add this method to `EagerOuterStepRuntime`:

```python
def build_attention_mask(self, prompt_attention_mask, cache_position, max_cache_len, dtype):
    return _build_static_decode_mask(
        prompt_attention_mask,
        cache_position,
        max_cache_len,
        dtype=dtype,
    )
```

Change `_decode_step` to call `self.step_runtime.build_attention_mask(...)`
instead of `_build_static_decode_mask(...)` directly. Add one engine helper that
checks a one-dimensional `cache_position` equals the contiguous range beginning
at `lease.cache.get_seq_length()`. Invoke it once before the prefill model call
and once before each decode `lease.run`; raise
`OuterTalkerEngineError("cache position is not contiguous")` before model
execution when it differs. Do not repeat this check inside every cache layer.

- [x] **Step 4: Implement active-prefix runtime leases and reuse pool**

`ActivePrefixOuterStepRuntime.acquire(batch_size, device, dtype,
max_cache_len)` uses a lock-protected idle pool keyed by `(batch_size,
str(device), str(dtype), max_cache_len)`. It resets and returns an idle cache or
creates `ActivePrefixCache(num_hidden_layers=talker.config.num_hidden_layers,
max_cache_len=max_cache_len)`. Lease return resets the active prefix and returns
up to `max_cached_leases` entries per key; excess idle entries receive a
terminal storage release. A pooled cache remains reusable, while a terminally
released cache rejects all later updates.

The lease's `run(prepared)` must use the same codec embedding/model/head logic
as `_EagerOuterRuntimeLease.run`, passing `prepared.attention_mask` unchanged.
Do not call `talker.forward` or `code_predictor.generate` inside the lease.

- [x] **Step 5: Add runtime lifecycle and metrics tests**

Assert:

- predictor events occur before lease execution;
- prefill/decode use the same cache object;
- prefill and decode position holes fail before their model calls;
- a cache returned to the idle pool is reused with identical backing pointers;
- concurrent leases never share one cache;
- exception cleanup returns a reset cache;
- `cache_allocations`, `allocated_kv_bytes`, `active_kv_tokens_per_step`,
  `backing_capacity_tokens`, `active_capacity_ratio`, `cache_resets`, and
  `cache_overflows` report immutable copies;
- `close()` releases idle entries and rejects new acquisitions.

- [x] **Step 6: Run focused regressions**

Run:

```bash
pytest tests/test_qwen_tts_outer_active_prefix_cache.py tests/test_qwen_tts_outer_static_engine.py tests/test_qwen_tts_cuda_graph_predictor.py -q
python -m py_compile qwen_asr_vllm/agent/qwen_tts_outer_active_prefix_cache.py qwen_asr_vllm/agent/qwen_tts_outer_static_engine.py
```

Expected: active-prefix and historical fixed-static paths both pass.

- [x] **Step 7: Record checkpoint**

Record mask shapes, pool ownership, and focused test count in `progress.md`.

---

### Task 3: Install And Configure The Active-Prefix Mode

**Files:**
- Modify: `qwen_asr_vllm/agent/qwen_tts_outer_active_prefix_cache.py`
- Modify: `qwen_asr_vllm/agent/local_tts.py`
- Modify: `bench/qwen_tts_internal_profile.py`
- Modify: `tests/test_qwen_tts_outer_active_prefix_cache.py`
- Modify: `tests/test_agent_local_tts.py`
- Modify: `tests/test_qwen_tts_internal_profile.py`
- Modify: `progress.md`

**Interfaces:**
- Produces: `install_active_prefix_outer_talker`, backend flag `outer_active_prefix_talker_engine`, profiler flag `--outer-active-prefix-talker-engine`, and `QwenTtsBackend.runtime_metrics()`.
- Consumes: Task 2 runtime and current code-predictor/backend warmup ordering.

- [x] **Step 1: Write failing installer and exclusivity tests**

```python
def test_backend_installs_active_prefix_after_predictor_before_warmup(monkeypatch):
    events = []
    monkeypatch.setattr(local_tts, "install_cuda_graph_code_predictor", lambda *a, **k: events.append("predictor"))
    monkeypatch.setattr(local_tts, "install_active_prefix_outer_talker", lambda *a, **k: events.append(("active", k)))

    QwenTtsBackend(
        model_path="/models/tts",
        device="cpu",
        check_runtime=False,
        warmup=True,
        cuda_graph_code_predictor=True,
        outer_active_prefix_talker_engine=True,
        outer_graph_max_cache_len=512,
    )

    assert events[:2] == ["predictor", ("active", {"max_cache_len": 512})]
```

Add parameterized tests that active-prefix cannot be combined with the existing
fixed-static or historical explicit outer modes. Assert validation occurs
before model loading. Record that any future graph mode must join the same
exclusivity check when it is introduced; do not add an unused graph flag now.

- [x] **Step 2: Run RED tests**

Run:

```bash
pytest tests/test_qwen_tts_outer_active_prefix_cache.py tests/test_agent_local_tts.py tests/test_qwen_tts_internal_profile.py -q
```

Expected: missing installer/backend/parser interfaces.

- [x] **Step 3: Implement installer metadata and backend lifecycle**

The installer creates `OuterTalkerStaticEngine(step_runtime=
ActivePrefixOuterStepRuntime(talker), max_cache_len=max_cache_len)` and marks
`_qav_active_prefix_outer_talker`, `_qav_static_outer_talker`,
`_qav_original_generate`, and `_qav_outer_engine` on the replacement callable.

Backend warmup uses the same bounded `max_new_tokens` rule as fixed-static
mode. `close()` closes predictor and outer runtimes without masking a primary
error. `runtime_metrics()` returns deep copies under keys `code_predictor` and
`outer_talker`; it never returns engine or scheduler objects.

- [x] **Step 4: Wire profiler mode and health reporting**

Add `--outer-active-prefix-talker-engine`, pass it to `QwenTtsBackend` before
warmup, report `outer_engine="active_prefix"`, and persist the flag in both
success and failure JSON. `_intended_outer_engine(args)` must return active
prefix before fixed-static and explicit modes only after parser/backend
exclusivity validation.

- [x] **Step 5: Run wiring regressions**

Run:

```bash
pytest tests/test_qwen_tts_outer_active_prefix_cache.py tests/test_agent_local_tts.py tests/test_qwen_tts_internal_profile.py tests/test_agent_service_runners.py -q
python -m py_compile qwen_asr_vllm/agent/local_tts.py bench/qwen_tts_internal_profile.py
```

Expected: all pass with observed warning/error/cache-overflow fields intact.

- [x] **Step 6: Record checkpoint**

Record mode precedence, warmup, lifecycle, and test count in `progress.md`.

---

### Task 4: Build The Bitwise Boundary And Codec Parity Probe

**Files:**
- Create: `bench/qwen_tts_active_prefix_parity_probe.py`
- Create: `tests/test_qwen_tts_active_prefix_parity_probe.py`
- Modify: `progress.md`

**Interfaces:**
- Produces: `TensorComparison`, `compare_tensors`, `summarize_parity`, CLI `main`, and JSON schema `qwen_tts_active_prefix_parity`.
- Consumes: upstream DynamicCache generation, active-prefix installer, current codec hash helpers, and CUDA synchronization.

- [x] **Step 1: Write failing pure comparison tests**

```python
def test_compare_tensors_requires_bitwise_equality_and_reports_delta():
    reference = torch.tensor([1.0, 2.0], dtype=torch.float16)
    candidate = reference.clone()
    equal = compare_tensors("hidden", reference, candidate)
    candidate[1] = torch.nextafter(candidate[1], torch.tensor(float("inf"), dtype=torch.float16))
    different = compare_tensors("hidden", reference, candidate)

    assert equal.bitwise_equal is True
    assert equal.max_abs == 0.0
    assert different.bitwise_equal is False
    assert different.max_abs > 0.0
```

Add tests for shape/dtype mismatch, first codec divergence location,
batch-item hashes, EOS positions, finite/non-empty audio, failed gate reasons,
JSON serialization, and parser defaults.

- [x] **Step 2: Run RED tests**

Run: `pytest tests/test_qwen_tts_active_prefix_parity_probe.py -q`

Expected: import failure for the new probe.

- [x] **Step 3: Implement deterministic boundary capture**

The probe loads one model on CUDA 2, disables sampling for outer and predictor,
sets seed 7 before every run, and prepares each batch once. Capture upstream and
active-prefix runs from identical prepared tensors. Hooks record:

- prefill plus first four decode-step active K/V for every outer layer;
- last hidden state and processed outer logits;
- sampled first-codebook IDs and predictor sequences;
- complete raw codec tensors before decode.

For every captured K/V boundary also record shape, stride, contiguity, and the
available SDPA/attention dispatch evidence. Do not insert `.contiguous()` or a
copy solely to make the candidate match the compact DynamicCache layout.

Restore `talker.generate`, `talker.rope_deltas`, hooks, and cache state between
modes. Never compare tensors from different prepared inputs.

- [x] **Step 4: Implement report and strict gate**

Write exact config, environment, GPU identity, per-boundary comparisons,
codec hashes/shapes/EOS, audio sample counts, warnings/errors, runtime metrics,
and timings. `parity_passed` is true only when every required bitwise, codec,
EOS, and audio-count check passes. Exit code is 0 on pass and 2 on a completed
parity failure; runtime failures remain nonzero exceptions after writing a
failure JSON.

- [x] **Step 5: Run probe unit and compile tests**

Run:

```bash
pytest tests/test_qwen_tts_active_prefix_parity_probe.py -q
python -m py_compile bench/qwen_tts_active_prefix_parity_probe.py
```

Expected: all pass without requiring a model or GPU.

- [x] **Step 6: Record checkpoint**

Document the JSON schema and exit-code contract in `progress.md`.

---

### Task 5: Run Controlled CUDA 2 Exact-Parity Acceptance

**Files:**
- Create from probe: `results/qwen_tts_active_prefix_cuda2.json`
- Modify: `findings.md`
- Modify: `progress.md`
- Modify: `task_plan.md`

**Interfaces:**
- Consumes all Tasks 1-4.
- Produces the exact-parity decision that gates stability work and any future attention-kernel design.

- [x] **Step 1: Run focused verification before GPU use**

Run:

```bash
pytest tests/test_qwen_tts_outer_active_prefix_cache.py tests/test_qwen_tts_outer_static_engine.py tests/test_agent_local_tts.py tests/test_qwen_tts_internal_profile.py tests/test_qwen_tts_active_prefix_parity_probe.py -q
```

Expected: all pass without warnings.

- [x] **Step 2: Verify CUDA 2 is clean**

Run:

```bash
nvidia-smi -i 2 --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi -i 2 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

Expected: 16 MiB baseline usage, 0% utilization, and no compute process. If not
clean, do not kill unknown processes; wait or record the external blocker.

- [x] **Step 3: Run the controlled parity probe**

Run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
/opt/conda/envs/nano-vllm/bin/python bench/qwen_tts_active_prefix_parity_probe.py \
  --model-path /mnt/llm_data/voice_ckpt/qwen3_tts/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device cuda:0 \
  --texts "I'm here to help." \
  --batch-sizes 1,2 \
  --seed 7 \
  --max-new-tokens 64 \
  --max-cache-len 1024 \
  --out results/qwen_tts_active_prefix_cuda2.json
```

Expected: exit 0 and `parity_passed=true`.

- [x] **Step 4: Apply the strict gate**

If any bitwise boundary, raw codec, EOS, or audio sample-count check fails,
stop before Task 6. Preserve the JSON, identify the first failing boundary,
use systematic debugging, and do not authorize outer graph or kernel work.
If the first difference is attributable to prefix-view stride/contiguity and
attention dispatch, stop this implementation stage and open a separate
compact-layout attention-adapter design; do not add a hidden K/V copy.

If every check passes, repeat the same command in a fresh process and write
`results/qwen_tts_active_prefix_cuda2_repeat.json`. Both runs must pass with
identical codec hashes.

- [x] **Step 5: Record timing and decision**

Compare warmed active-prefix generation latency with upstream DynamicCache.
Require no more than 5% batch-1 regression. Record allocation count, KV bytes,
cache reuse, active-capacity ratio, exact hashes, GPU state, and decision in
`findings.md`, `progress.md`, and Phase 27 of `task_plan.md`.

---

### Task 6: Run Stability And Finalize The Active-Prefix Stage

**Files:**
- Create from probe: `results/qwen_tts_active_prefix_stability_cuda2.json`
- Modify: `findings.md`
- Modify: `progress.md`
- Modify: `task_plan.md`

**Interfaces:**
- Consumes a passed Task 5 exact-parity gate.
- Produces the final active-prefix acceptance and a separate go/no-go for designing a varlen/paged attention kernel.

- [x] **Step 1: Add a deterministic stability mode to the parity probe**

Write failing unit tests for `--requests`, `--duration-minutes`, mixed-text
rotation, left-padded batch-2 inputs, memory snapshots, and failure aggregation.
Implement the mode so it runs at least 100 requests or 10 minutes, whichever
takes longer, and records every request's hashes, latency, cache pointers,
allocated/reserved memory, and errors.

- [x] **Step 2: Run stability unit tests**

Run:

```bash
pytest tests/test_qwen_tts_active_prefix_parity_probe.py -q
python -m py_compile bench/qwen_tts_active_prefix_parity_probe.py
```

Expected: all pass.

- [x] **Step 3: Run real CUDA 2 stability**

Run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
/opt/conda/envs/nano-vllm/bin/python bench/qwen_tts_active_prefix_parity_probe.py \
  --model-path /mnt/llm_data/voice_ckpt/qwen3_tts/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --device cuda:0 \
  --texts "I'm here to help." "What can I do for you?" "Please send the next request." \
  --batch-sizes 1,2 \
  --seed 7 \
  --max-new-tokens 64 \
  --max-cache-len 1024 \
  --requests 100 \
  --duration-minutes 10 \
  --out results/qwen_tts_active_prefix_stability_cuda2.json
```

- [x] **Step 4: Run full repository verification**

Run:

```bash
pytest -q
python -m py_compile \
  qwen_asr_vllm/agent/qwen_tts_outer_active_prefix_cache.py \
  qwen_asr_vllm/agent/qwen_tts_outer_static_engine.py \
  qwen_asr_vllm/agent/local_tts.py \
  bench/qwen_tts_internal_profile.py \
  bench/qwen_tts_active_prefix_parity_probe.py
```

Expected: full suite and compilation pass without new warnings.

- [x] **Step 5: Final decision and cleanup**

Accept the stage only if all requests preserve exact codec hashes, backing
pointers remain stable per pooled lease, CUDA memory has no monotonic growth,
there is no cross-request state contamination, and GPU 2 returns to baseline.
Mark Phase 27 complete and record that Task 4 from the old graph plan remains
superseded, not resumed. Open a separate design cycle for a varlen/paged
attention kernel using the active-prefix implementation as the exact reference.

Do not delete the fixed-full-extent static prototype in this plan; retain it as
the documented failed control until the active-prefix stage is accepted and a
separate cleanup task is approved.
