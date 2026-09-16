# Qwen-TTS Varlen/Paged-Attention Adapter Design

**Status:** Proposed

**Goal:** Allow multiple Qwen-TTS outer talker sessions with different logical decode positions to share request-id level decode scheduling while preserving exact codec behavior.

## Context

The accepted ActivePrefixCache path is exact against the upstream Qwen-TTS DynamicCache path, but its cache position and decode mask are cohort-wide. A stock Transformers cache update receives one shared `cache_position` vector, so sessions at different decode positions cannot safely share one outer model call merely by adding padding or masking finished rows.

The current environment provides Triton but does not provide `flashinfer`, `flash-attn`, or `xformers`. The first implementation therefore needs an explicit PyTorch reference path that establishes the cache/page-table contract before introducing a custom kernel.

## Non-Goals

- Do not change the default Qwen-TTS backend path until exact parity is established.
- Do not claim a speedup from page metadata or Python scheduling alone.
- Do not implement arbitrary multi-GPU execution or cross-process page pools in this stage.
- Do not silently fall back from the adapter to an unrelated engine; unsupported shapes must return a clear capability error.

## Architecture

### Request state

Each scheduled outer request owns:

- stable `request_id`;
- `seq_len` and `max_seq_len` in logical token units;
- `page_ids` mapping logical pages to physical page-pool rows;
- `rope_delta`, `past_hidden`, `trailing_text_hidden`, and `tts_pad_embed`;
- generated first-codebook history and active/EOS state.

The request state is independent of the scheduler slot. A finished request can release its pages while a new request occupies the slot without changing other request mappings.

### Paged KV storage

`PagedOuterCache` owns fixed-size physical pages for every transformer layer. Its public operations are:

```python
allocate(request_id: str, batch_size: int, device, dtype) -> PageTable
append(request_id: str, key_states, value_states, logical_positions) -> None
read(request_id: str, layer_index: int, logical_length: int) -> tuple[key, value]
release(request_id: str) -> None
snapshot() -> dict[str, Any]
```

The allocator validates page bounds, dtype/device/layout, duplicate ownership, and release/reuse ordering. It never exposes another request's pages. `snapshot()` reports live pages, free pages, allocations, releases, and overflow counters for tests and profiling.

The reference implementation stores pages in ordinary strided PyTorch tensors. It may gather a request's logical prefix into a contiguous attention input because this is a correctness reference; the gather must be explicit in the adapter and must not be mistaken for a kernel-level performance result.

### Packed decode metadata

`PackedOuterStep` contains the ready request IDs, row-to-request mapping, per-request logical lengths, page tables, `cu_seqlens`, decode position IDs, per-row attention metadata, and packed `past_hidden`/first-codebook input tensors. Metadata is immutable during one model call.

Requests are packed only when they share model dtype, device, attention head geometry, and decode work shape. Logical lengths may differ. EOS-inactive rows are removed from the next packed step; they are not allowed to write or read pages after completion.

### Scheduler

`OuterVarlenScheduler` maintains a request registry and a ready queue. Each tick:

1. admits pending requests with prepared outer inputs;
2. performs eager prefill per request or in compatible prefill cohorts;
3. groups all ready requests into one packed outer decode call;
4. advances each request's logical length independently;
5. emits per-request codec/audio events and releases completed requests.

The scheduler has no hidden thread fallback. Its execution callback is injected so CPU tests can use a deterministic fake model and CUDA tests can use Qwen-TTS. Cancellation removes a request at a tick boundary and releases its pages exactly once.

### Reference attention adapter

The adapter exposes one model-facing operation:

```python
decode_packed(step: PackedOuterStep) -> PackedOuterOutput
```

It builds per-row Q/K/V views from the page table and invokes the public PyTorch SDPA/eager attention path with per-request causal lengths. The first version may execute compatible rows as a small loop over gathered prefixes; this is intentionally the reference implementation. It must produce the same logits, codec IDs, hidden states, and cache contents as independent ActivePrefixCache requests.

The adapter is opt-in and installed through a dedicated backend option. The existing ActivePrefixCache engine remains the exact control and is used for A/B parity and timing.

## Error And Lifecycle Rules

- Page allocation failure is a request-level capability/resource error with request ID and required page count.
- Invalid page table, logical position, or cross-request page access is a hard adapter error and increments no success metric.
- A model exception cancels only the affected request when the scheduler can prove page ownership remains isolated; otherwise the whole packed tick is failed and all affected requests are restored/released.
- Runtime close drains the ready queue, releases every live request exactly once, and reports outstanding pages as a failure.
- No request is reported complete until codec output, cache release, and event delivery have all succeeded.

## Exact-Parity Acceptance

For synthetic and real Qwen-TTS tests, the adapter must compare against independent ActivePrefixCache runs for:

- prefill and at least four decode boundaries;
- every layer K/V shape, stride, dtype, device, and values;
- last hidden states, raw and processed codec logits;
- sampled first codebook IDs, predictor sequences, raw codec tensors, EOS positions;
- decoded audio sample counts and finite/nonempty status.

The multi-request test must include at least two requests with different prompt lengths and deliberately different decode progress. It must prove no cross-request page access and exact per-request output ordering.

## Performance Measurement

Measure separately:

- independent batch-1 ActivePrefixCache latency;
- position-bucketed batch latency;
- packed reference adapter latency;
- later Triton kernel latency.

Report first audio, total generation, decode compute, aggregate frames/s, and scheduler overhead. The reference adapter is considered successful only for correctness and lifecycle stability; a performance regression is expected until the Triton kernel stage.

## Rollout

Stage A adds the page allocator, packed metadata, scheduler, and PyTorch reference adapter behind an explicit opt-in flag. Stage B adds a Triton paged-attention kernel using the same metadata contract, with exact parity against Stage A before any default-path consideration.
