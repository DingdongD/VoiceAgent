# Qwen-TTS Outer Talker Static/Graph Engine Design

## Context

The current Qwen-TTS path has already moved the nested code predictor to a static-cache CUDA Graph engine. In a controlled batch-2 profile, `code_predictor.generate` fell to 3.4% of generation time while `talker.forward` accounted for 92.9% and `talker.model.forward` accounted for 53.9%. The remaining optimization target is therefore the outer autoregressive talker decode loop.

The installed Qwen-TTS model accepts Transformers `Cache` objects and explicit `cache_position`, but its default generation path creates a `DynamicCache` and stores `rope_deltas` on the shared talker module. The stock `StaticCache` writes one shared sequence position across all batch rows, so it cannot support arbitrary continuous slot refill for sessions at different decode positions without a custom cache and attention path.

## Goals

- Replace outer talker decode-time `DynamicCache` growth with stable static KV storage.
- Replay fixed-shape batch-1 and batch-2 outer model decode steps with CUDA Graphs.
- Preserve the existing code predictor CUDA Graph engine without nested graph capture.
- Keep all mutable generation state request- or cohort-owned; no shared `talker.rope_deltas` state.
- Compact a cohort from batch-2 to batch-1 when a sufficiently long single-request tail makes KV copying worthwhile.
- Measure real 1/2/4/8-session throughput, first-audio latency, total latency, graph hit rate, slot occupancy, queue delay, and GPU memory.

## Non-Goals

- Do not patch files inside the installed `qwen_tts` package.
- Do not capture variable-length prefill in CUDA Graphs.
- Do not implement arbitrary mid-cohort slot refill in the first version.
- Do not add a silent eager fallback inside the enabled graph engine. The existing upstream path remains available only when the feature flag is disabled.
- Do not reintroduce `torch.compile`; its startup and recompilation costs were already measured as unsuitable.

## Selected Architecture

### 1. Explicit Static Outer State

Add `qwen_asr_vllm.agent.qwen_tts_outer_static_engine` with these principal types:

- `OuterTalkerStaticEngine`: implements the `talker.generate(**kwargs)` contract.
- `OuterCohortState`: owns one static KV cache, current cache position, active-slot mask, per-slot rope delta, past hidden state, generation step, trailing text hidden state, generated first-codebook history, and codec hidden-state output history.
- `OuterGraphBundle`: owns fixed input/output buffers and one captured outer decode graph for a specific batch size, device, dtype, hidden size, KV capacity, and attention backend.
- `OuterGraphPool`: owns batch-1 and batch-2 bundles and serializes access to each bundle.
- `OuterRuntimeLease`: exposes the cache and decode runner owned by one eager runtime or graph bundle for the lifetime of a cohort.

Before prefill, the engine acquires an `OuterRuntimeLease` for the cohort batch size. Prefill remains eager but uses `talker.model` directly rather than `talker.forward`. The engine computes position IDs and rope deltas from the incoming attention mask, passes explicit position IDs into the model, writes prefill KV into the lease's `StaticCache`, computes codec logits through `talker.codec_head`, and stores rope delta in `OuterCohortState`.

The lease is required because a captured CUDA Graph is bound to the fixed addresses of its bundle-owned cache. Static prefill and graph replay must therefore operate on the same cache object; the static engine must not allocate an independent cache in graph mode.

This avoids the shared `talker.rope_deltas` field and makes concurrent cohort state independent.

### 2. Decode Step Split

One outer codec step is divided into four operations:

1. Sample the next first-codebook token from the prior outer logits.
2. Run the existing code predictor CUDA Graph to obtain the remaining codebook tokens.
3. Build codec embeddings and add either the indexed trailing-text hidden state or `tts_pad_embed`.
4. Replay the outer model graph with the prepared embedding, explicit cache position, position IDs, causal mask, and static KV cache; then compute the next outer logits and past hidden state.

The outer graph must not capture operation 2 because it would contain a replay of the existing predictor graph. The outer graph begins after all codec IDs are available.

`OuterGraphBundle` captures codec embedding lookup, embedding reduction, `talker.model`, `talker.codec_head`, and the update of output buffers. Sampling stays outside the graph because its measured cost is small and stochastic RNG behavior must remain explicit.

### 3. Fixed-Shape Attention Inputs

Each graph bundle owns:

- Static code ID input `[slots, num_code_groups]`.
- Static conditional embedding input `[slots, 1, hidden_size]`.
- Static cache-position input `[1]`.
- Explicit position IDs `[3, slots, 1]`.
- Preallocated 4D causal mask `[slots, 1, 1, max_cache_len]`.
- A `StaticCache` initialized for the same slot count and maximum cache length.

The engine updates the fixed buffers before replay. Passing a 4D mask bypasses generic Transformers mask construction and prevents Python-side shape changes during decode. `OuterGraphPool.acquire(batch_size)` returns the bundle as a lease before prefill and holds its lock until cohort completion or compaction.

All rows in a cohort share one physical cache position. Variable prompt lengths are represented by padded eager prefill plus per-row attention masks and rope deltas, matching native batch generation.

### 4. Cohort Slot Scheduling

The existing process TTS service remains the request-id admission boundary. Requests arriving within the admission window form a cohort of at most two sessions and enter one `OuterCohortState`.

The first implementation supports these slot transitions:

- `0 -> 1`: run batch-1.
- `0 -> 2`: run batch-2.
- `2 -> 1`: mark the completed row inactive. If the survivor remains active for two grace steps, copy its KV rows and request state into the batch-1 bundle and continue there.
- `1 -> 0` or `2 -> 0`: release/reset the bundle for the next cohort.

The scheduler does not place a new request into a free row of an active cohort. Stock static cache cannot safely write different sequence positions per row. Pending requests wait for a free bundle or form another cohort if a separate bundle is available.

Compaction statistics must record attempted compactions, copied KV bytes, copy duration, saved padded steps, and whether the decision improved estimated work. The policy stays configurable and can be disabled when real timing shows no gain.

### 5. Cancellation And Errors

- Cancellation marks the slot inactive at the next step boundary and prevents further audio emission.
- Cache reset, graph replay, and compaction execute under the owning bundle lock.
- A capture or parity failure raises `OuterTalkerGraphError` during backend warmup. It does not silently route the same request to another engine.
- Unsupported generation arguments fail before mutating cache state.
- Bundle state is reset after exceptions so a failed cohort cannot contaminate later requests.

## Runtime Configuration

Add these opt-in settings to `QwenTtsBackend`, timing CLI, UI CLI, and internal profiler:

- `outer_static_talker_engine: bool = False`
- `outer_cuda_graph_talker_engine: bool = False`
- `outer_graph_fixed_slots: int = 2`
- `outer_graph_max_cache_len: int = 1024`
- `outer_graph_compaction_grace_steps: int = 2`

Precedence is CUDA Graph outer engine, then static outer engine, then the existing upstream `talker.generate`. Existing experimental `explicit_talker_step_engine` is removed once the new static engine passes parity and timing because retaining two explicit loops would create an unused historical path.

## Observability

Expose a snapshot from the outer engine containing:

- `prefill_calls`
- `graph_captures`
- `graph_replays_by_batch`
- `static_steps_by_batch`
- `active_slots_per_step`
- `slot_occupancy`
- `compactions`
- `compaction_copy_bytes`
- `compaction_copy_ms`
- `padded_tail_steps`
- `errors`

The process service load probe writes these alongside per-request timing rather than relying only on aggregate wall time.

## Verification

### Unit Tests

- Static prefill and decode advance cache positions exactly once per codec frame.
- Two concurrent cohort rows retain independent rope deltas, trailing text, past hidden state, EOS state, and output history.
- The outer graph wrapper reuses shape-specific bundles and updates fixed inputs before replay.
- Predictor graph calls occur outside outer graph capture.
- Batch-2 completion triggers grace steps and then copies only the surviving row into batch-1 state.
- Cancellation and graph errors reset bundle state and wake all request waiters.
- Installer precedence and all CLI/backend settings are covered.

### Real CUDA Tests

Use controlled greedy generation and identical `max_new_tokens` values.

1. Compare upstream eager, outer static eager, and outer graph for batch-1 and batch-2.
2. Require identical codec IDs between static eager and graph for the same batch shape.
3. Compare decoded audio duration and verify finite, non-empty waveform output.
4. Run mixed short/medium/long texts with 1/2/4/8 concurrent callers and both simultaneous and staggered arrivals.
5. Run at least 100 requests or 10 minutes, whichever is longer, for stability.

CUDA 2 is the primary performance device because it is currently clean. CUDA 3 is a secondary consistency run because its pre-existing orphan context keeps utilization near 100% and distorts absolute timing.

## Acceptance Criteria

- Full unit suite passes with no new runtime warnings.
- Static eager and graph produce identical greedy codec IDs for each tested batch shape.
- Batch-1 graph first-audio and total latency regress by no more than 5% versus the current predictor-graph path after warmup.
- Batch-2 aggregate codec throughput improves by at least 15% over the current predictor-only graph path.
- No cross-session cache, rope delta, text-condition, or audio contamination occurs.
- Slot occupancy, graph hit rate, p50/p95 latency, queue delay, and peak GPU memory are reported for every load level.
- Arbitrary paged slot refill is considered only if staggered-load occupancy is below 70% or padded-tail work exceeds 20% after cohort compaction.

## Deferred Paged Slot Adapter

If the acceptance gate requires arbitrary refill, a separate design will introduce a per-slot KV cache with row-specific write positions, a block table or equivalent slot map, and a custom attention/mask adapter. That work is intentionally excluded here because stock `StaticCache` and Transformers causal masking accept one shared `cache_position` for the batch.
