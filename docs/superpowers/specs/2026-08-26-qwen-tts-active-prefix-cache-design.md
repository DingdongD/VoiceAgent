# Qwen-TTS Active-Prefix Outer Cache Design

## Context

The outer Qwen-TTS static-cache prototype preallocates stable KV storage and
removes shared `talker.rope_deltas`, but it does not preserve exact codec IDs.
Controlled CUDA 2 probes showed that stock Transformers `StaticCache` exposes
the full configured KV extent to attention. With SDPA this changes mask and GQA
dispatch relative to `DynamicCache`; with eager attention it still changes the
reduction extent. FP16 differences are then amplified by zero-margin decisions
inside the nested code predictor.

The approved exact-parity route therefore restores active-prefix attention
semantics before any outer CUDA Graph work. It retains preallocated KV backing
storage but presents only the live prefix to attention. The existing fixed-full-
extent outer graph design remains blocked until this parity stage passes.

## Goals

- Preallocate request-cohort KV storage without per-step `torch.cat` growth.
- Present the same active KV prefix and mask contract as upstream
  `DynamicCache` on every prefill and decode call.
- Preserve request-owned RoPE delta, hidden state, text condition, codec
  history, EOS state, and error isolation.
- Establish exact raw codec-ID parity for greedy batch-1 and batch-2 generation
  before attempting another outer attention or CUDA Graph optimization.
- Expose cache/runtime metrics sufficient to distinguish allocation savings
  from attention compute and predictor time.

## Non-Goals

- Do not patch files inside the installed `qwen_tts` or `transformers`
  packages.
- Do not capture active-prefix decode with CUDA Graphs in this stage.
- Do not claim fixed-shape attention: returned K/V views grow with the active
  sequence length.
- Do not relax exact codec-ID parity to waveform-only or perceptual parity.
- Do not implement arbitrary mid-cohort slot refill or paged attention in this
  stage.
- Do not silently fall back to upstream generation when the active-prefix mode
  is enabled.

## Selected Architecture

### 1. Preallocated Active-Prefix Cache

Add `qwen_asr_vllm.agent.qwen_tts_outer_active_prefix_cache` with these
principal types:

- `ActivePrefixCacheError`: configuration, capacity, and lifecycle failures.
- `ActivePrefixLayer`: one preallocated K/V backing pair for one decoder layer.
- `ActivePrefixCache`: a Transformers-compatible `Cache` composed of active-
  prefix layers.
- `ActivePrefixOuterStepRuntime`: creates cache-bound runtime leases for the
  existing explicit outer generation engine.

Each `ActivePrefixLayer` lazily allocates backing tensors from the first real K/V
update so device, dtype, KV-head count, and head dimension exactly match the
loaded model. Storage shape is:

```text
[batch_size, num_kv_heads, max_cache_len, head_dim]
```

`update(key_states, value_states, cache_kwargs)` requires an explicit one-
dimensional `cache_position`. The outer engine validates once per model call
that positions are contiguous and start at the cohort's active length. Each
layer writes with `index_copy_` along the sequence dimension, advances
`active_length` by the query length, and returns views of both tensors
restricted to `:active_length`. This avoids a device-to-host scalar sync in
every decoder layer. The unused tail is never exposed to attention.

A prefix view into fixed-capacity backing can have different batch/head strides
and contiguity from the compact tensor produced by `DynamicCache`. That layout
may select a different SDPA kernel even when shape and values match. The parity
probe must therefore record K/V shape, stride, contiguity, and the observed
attention dispatch at the first failing boundary. The implementation must not
hide this issue with an undocumented `.contiguous()` copy: that would restore a
per-step allocation/copy and defeat the cache objective. If layout alone breaks
bitwise parity, this stage stops for a separate compact-layout attention-adapter
design.

The engine rejects writes beyond `max_cache_len` and non-contiguous positions
before every prefill or decode model execution. The layer rejects capacity and
mismatched batch/KV/head shapes without repeating the contiguity check in each
decoder layer. A terminal storage release rejects reuse; returning a lease to
the runtime pool is not a terminal release. `reset()` zeros the previously
active prefix, preserves the allocation, and sets `active_length` to zero under
inference mode.

### 2. Transformers Cache Contract

`ActivePrefixCache` follows the installed Transformers 4.57.6 `Cache`
interface. `ActivePrefixLayer` derives from the public `CacheLayerMixin` and
implements all required abstract methods:

```python
is_compileable = False

def lazy_initialization(self, key_states) -> None:
    batch, heads, _, head_dim = key_states.shape
    shape = (batch, heads, self.max_cache_len, head_dim)
    self.keys = key_states.new_zeros(shape)
    self.values = key_states.new_zeros(shape)
    self.is_initialized = True

def get_seq_length(self) -> int:
    return active_length

def get_mask_sizes(self, cache_position) -> tuple[int, int]:
    return self.active_length + cache_position.shape[0], 0

def get_max_cache_shape(self) -> int:
    return max_cache_len
```

For decode, mask sizing therefore matches `DynamicCache`: existing active
length plus the current query token. Marking the cache non-compileable permits
Transformers' normal SDPA mask skip when the same upstream conditions apply.
The adapter must not report the backing capacity as the attention KV length.

The implementation uses the public `Cache` and `CacheLayerMixin` APIs from the
installed Transformers 4.57.6 runtime. If those required interfaces are absent
or incompatible, installation raises `ActivePrefixCacheError` with the detected
version; no local compatibility fallback or private package copy is retained.

### 3. Runtime Mask Strategy

The existing `OuterTalkerStaticEngine` already owns sampling, predictor calls,
request state, and lease lifetime. Its decode-mask construction becomes a
runtime strategy rather than a hard-coded fixed 4D mask:

```python
class OuterStepRuntimeProtocol:
    def acquire(batch_size, device, dtype, max_cache_len) -> OuterRuntimeLease:
        raise NotImplementedError

    def build_attention_mask(prompt_attention_mask, cache_position, dtype):
        raise NotImplementedError
```

- `EagerOuterStepRuntime` retains the fixed 4D additive mask used by the
  historical static prototype.
- `ActivePrefixOuterStepRuntime` returns the live two-dimensional mask:
  original prompt mask followed by one valid position for every generated
  token through the current decode step.

The active-prefix runtime passes this 2D mask, explicit `cache_position`, and
request-owned position IDs into `talker.model`. Transformers then creates or
skips its causal mask exactly as it does for `DynamicCache`.

The predictor remains outside `lease.run`. Codec embedding lookup, embedding
reduction, text-condition addition, outer model execution, and codec head stay
inside the lease execution boundary, preserving the existing separation for
future runtime experiments.

### 4. Installer And Configuration

Add an opt-in installer:

```python
install_active_prefix_outer_talker(
    talker,
    *,
    max_cache_len: int = 1024,
) -> bool
```

Installer metadata:

```python
generate._qav_active_prefix_outer_talker = True
generate._qav_static_outer_talker = True
generate._qav_original_generate = original_generate
generate._qav_outer_engine = engine
```

Add configuration to `QwenTtsBackend` and the internal profiler:

- `outer_active_prefix_talker_engine: bool = False`
- `outer_graph_max_cache_len: int = 1024` remains the shared capacity setting.

Outer implementations are mutually exclusive:

1. active-prefix outer engine;
2. fixed-full-extent static outer prototype;
3. future CUDA Graph outer engine;
4. historical explicit outer step engine;
5. upstream `talker.generate` when all flags are disabled.

An enabled active-prefix engine never catches its own failure and routes the
same request through upstream generation.

### 5. State, Cancellation, And Errors

- One runtime lease owns one cache for the full cohort lifetime.
- Prefill and every decode step use the same cache instance.
- `talker.rope_deltas` is never assigned; adjusted deltas remain request-owned.
- Cancellation is observed at outer step boundaries and inactive rows remain
  isolated until the cohort finishes.
- Cache reset and release occur in `torch.inference_mode()` on success and all
  exception paths.
- Capacity and unsupported-generation-argument errors are raised before cache
  mutation when possible.
- A failed lease cannot be returned to the runtime until reset completes.

### 6. Observability

The active-prefix runtime extends the existing outer snapshot with:

- `cache_allocations`
- `allocated_kv_bytes`
- `active_kv_tokens_per_step`
- `backing_capacity_tokens`
- `active_capacity_ratio`
- `cache_resets`
- `cache_overflows`
- existing `prefill_calls`, `static_steps_by_batch`, `active_slots_per_step`,
  `slot_occupancy`, and `errors`

Metrics are returned as immutable copies through `QwenTtsBackend.runtime_metrics`
and profiler JSON.

## Data Flow

1. The backend installs the selected code predictor engine.
2. It installs the active-prefix outer engine before warmup.
3. The outer engine validates generation arguments and acquires a lease.
4. Eager prefill writes K/V into preallocated backing storage and receives only
   the live prefix from each cache layer.
5. The engine samples the first codebook, runs the predictor outside the lease,
   and prepares codec IDs plus the per-request text condition.
6. The runtime builds a live 2D mask and executes the outer model against the
   active K/V views.
7. Request state and metrics advance once; EOS/cancellation updates active rows.
8. Completion or error resets and releases the lease.

## Verification

### Unit Tests

- Layer storage allocates once and keeps stable backing addresses.
- Prefill and decode writes land at exact `cache_position` indices.
- Returned K/V sequence lengths equal the active prefix, never capacity.
- Unused backing tails remain zero and are not exposed.
- Reset preserves allocation while clearing active length and data.
- Layer capacity/shape mismatches and engine-level position holes fail before
  partial mutation.
- `get_seq_length`, `get_mask_sizes`, and `is_compileable` match the active-
  prefix contract.
- Active-prefix runtime emits the same 2D masks as upstream generation for
  unpadded and left-padded batch rows.
- Lease cleanup, installer idempotence, mode exclusivity, CLI propagation, and
  metric snapshots are covered.

### Controlled CUDA 2 Parity

Use one loaded model and identical prepared inputs to compare upstream
`DynamicCache` and `ActivePrefixCache`:

1. Greedy generation, seed 7, `max_new_tokens=64`, batch sizes 1 and 2.
2. Compare prefill and the first four decode steps layer-by-layer.
3. Record K/V shape, stride, contiguity, and observed attention dispatch for
   both paths at every captured boundary.
4. Require bitwise-equal active K/V, last hidden states, and processed outer
   logits at each captured boundary.
5. Require identical first-codebook and all predictor codebook IDs.
6. Require identical complete raw codec tensors, shapes, EOS positions, and
   SHA-256 hashes.
7. Decode both outputs and require finite, non-empty audio with identical
   sample counts.
8. Repeat in fresh processes to exclude state leakage.

### Stability

- Run at least 100 sequential requests or 10 minutes, whichever is longer.
- Include mixed prompt lengths and left-padding in batch-2.
- Verify no backing reallocation, cross-request KV contamination, memory growth,
  warning, cache overflow, or unreleased CUDA process.

## Acceptance Criteria

- Full focused and repository test suites pass without new warnings.
- Batch-1 and batch-2 controlled CUDA 2 runs satisfy every bitwise boundary and
  complete codec-ID parity check.
- Decoded audio is finite/non-empty with matching sample counts.
- Backing storage addresses remain stable through every request and reset.
- No shared RoPE, hidden, condition, codec history, or cache state leaks between
  requests.
- Active-prefix runtime does not regress batch-1 generation latency by more
  than 5% relative to upstream `DynamicCache` after warmup.
- Task 4 outer CUDA Graph work remains blocked until all criteria above pass.

## Follow-Up Kernel Decision

After exact parity passes, profile allocation, cache update, attention, outer
model, and predictor costs again. A new design may then evaluate a varlen or
paged attention kernel that accepts preallocated backing plus live lengths.
That kernel must establish its own quality/parity contract; it is not implicitly
authorized by this active-prefix design.
