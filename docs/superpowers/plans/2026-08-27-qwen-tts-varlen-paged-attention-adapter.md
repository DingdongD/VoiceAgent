# Qwen-TTS Varlen/Paged-Attention Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in request-id level varlen outer-talker scheduler with a paged KV reference adapter, prove exact parity, and establish the contract for a later Triton paged-attention kernel.

**Architecture:** Introduce a focused paged-cache module that owns physical KV pages and request page tables, a packed-step module that builds immutable per-tick metadata, and a scheduler that advances requests independently. The first model adapter uses explicit PyTorch gather/SDPA reference execution and never changes the default ActivePrefixCache path; Triton is enabled only after the reference contract passes parity.

**Tech Stack:** Python 3.11, PyTorch 2.5, Transformers 4.57 public cache APIs, Qwen-TTS 0.1.1, pytest, CUDA/Triton 3.3 for the later kernel stage.

**Spec:** `docs/superpowers/specs/2026-08-27-qwen-tts-varlen-paged-attention-adapter-design.md`

## Global Constraints

- Preserve the existing ActivePrefixCache path as the exact reference and default behavior.
- No hidden fallback to a different engine; unsupported dtype/device/head geometry must raise a capability error.
- Every page access must be ownership checked; finished or cancelled requests cannot read or write pages.
- CPU tests must not load Qwen-TTS or require CUDA.
- Real acceptance requires exact codec/KV/logit/audio parity and a two-request different-progress case before any performance claim.
- Do not delete the fixed-full static prototype or existing fixed-slot predictor scheduler; they remain documented controls.

---

### Task 1: Define Paged KV Allocator And Ownership Contract

**Files:**
- Create: `qwen_asr_vllm/agent/qwen_tts_outer_paged_cache.py`
- Create: `tests/test_qwen_tts_outer_paged_cache.py`

**Interfaces:**
- Consumes: PyTorch tensors and request IDs.
- Produces: `PageTable`, `PagedOuterCache`, `PagedCacheError` with `allocate`, `append`, `read`, `release`, `snapshot`.

- [x] **Step 1: Write failing allocator tests**

Add tests that require fixed-size page allocation, multiple request ownership, append/read round trips, page reuse after release, overflow rejection, duplicate request rejection, invalid logical positions, and cross-request read rejection. Assert `snapshot()` counters for live pages, allocations, releases, and overflows.

- [x] **Step 2: Run allocator tests to verify failure**

Run:

```bash
CUDA_VISIBLE_DEVICES='' pytest tests/test_qwen_tts_outer_paged_cache.py -q
```

Expected: collection or assertion failures because the new module and contract are absent.

- [x] **Step 3: Implement the minimal CPU/GPU-neutral page pool**

Use fixed tensors shaped `[num_pages, num_heads, page_size, head_dim]` for keys and values. Store request metadata as immutable page IDs plus logical length. Validate device, dtype, rank, page boundaries, monotonic append positions, and ownership before writes. `read()` must return only the requested logical prefix and must never expose the pool tensor directly.

- [x] **Step 4: Run allocator tests to verify pass**

Run the focused command above and verify all allocator tests pass.

---

### Task 2: Add Packed Varlen Decode Metadata

**Files:**
- Create: `qwen_asr_vllm/agent/qwen_tts_outer_varlen.py`
- Create: `tests/test_qwen_tts_outer_varlen.py`

**Interfaces:**
- Consumes: `PageTable`, request states, prompt/device/dtype metadata.
- Produces: frozen `OuterVarlenRequest`, `PackedOuterStep`, `PackedOuterOutput`, and `build_packed_step(requests)`.

- [x] **Step 1: Write failing packed-metadata tests**

Test stable row ordering by request ID, independent logical lengths, `cu_seqlens`, page-table shape, per-request decode positions, rope delta packing, active-only filtering, incompatible dtype/device rejection, and immutable metadata after construction.

- [x] **Step 2: Run the metadata tests to verify failure**

Run:

```bash
CUDA_VISIBLE_DEVICES='' pytest tests/test_qwen_tts_outer_varlen.py -q
```

Expected: failures because the packed types and builder are absent.

- [x] **Step 3: Implement immutable packed metadata**

Define dataclasses with tuples or cloned tensors for metadata. Sort rows deterministically by scheduler insertion order, compute `cu_seqlens` from each request logical length, pack per-row `position_ids`, and reject requests whose device/dtype/model geometry differs. Do not mutate request state while building a step.

- [x] **Step 4: Run metadata tests to verify pass**

Run the focused command and verify all metadata tests pass.

---

### Task 3: Implement Request-ID Varlen Scheduler

**Files:**
- Modify: `qwen_asr_vllm/agent/qwen_tts_outer_varlen.py`
- Create: `tests/test_qwen_tts_outer_varlen_scheduler.py`

**Interfaces:**
- Consumes: `PagedOuterCache`, `OuterVarlenRequest`, an injected `prefill_fn`, `decode_packed_fn`, and `emit_fn`.
- Produces: `OuterVarlenScheduler.add_request`, `cancel`, `step`, `run_until_idle`, `close`, and scheduler metrics.

- [x] **Step 1: Write failing scheduler tests**

Cover admission of two requests, different decode progress, one packed tick containing both ready requests, per-request result demultiplexing, EOS removal on the following tick, cancellation releasing pages exactly once, request failure isolation, close draining live requests, and deterministic output order.

- [x] **Step 2: Run scheduler tests to verify failure**

Run:

```bash
CUDA_VISIBLE_DEVICES='' pytest tests/test_qwen_tts_outer_varlen_scheduler.py -q
```

Expected: failures because the scheduler is absent.

- [x] **Step 3: Implement explicit tick-boundary scheduling**

Keep a request registry and ready queue under one lock. Each `step()` admits pending requests, invokes prefill for new requests, builds one compatible packed step, calls the injected decode callback, advances only successful requests, emits outputs by request ID, and releases completed/cancelled pages. A callback exception must produce a request-specific failure only when page ownership is still isolated; otherwise fail the packed tick and release every request in that tick.

- [x] **Step 4: Run scheduler tests to verify pass**

Run the focused command and verify all lifecycle and demultiplexing tests pass.

---

### Task 4: Add PyTorch Reference Attention Adapter

**Files:**
- Modify: `qwen_asr_vllm/agent/qwen_tts_outer_varlen.py`
- Create: `tests/test_qwen_tts_outer_varlen_attention.py`

**Interfaces:**
- Consumes: `PackedOuterStep`, `PagedOuterCache`, Q/K/V projection callbacks, and public PyTorch SDPA/eager attention.
- Produces: `ReferencePagedAttentionAdapter.decode_packed(step) -> PackedOuterOutput` and explicit gather/attention metrics.

- [x] **Step 1: Write failing synthetic attention parity tests**

Create two requests with different logical lengths and distinct K/V values. Assert each output equals independent attention, no request can observe another request's pages, causal positions are respected, page boundaries work, and metadata/gather overhead is reported separately.

- [x] **Step 2: Run synthetic attention tests to verify failure**

Run:

```bash
CUDA_VISIBLE_DEVICES='' pytest tests/test_qwen_tts_outer_varlen_attention.py -q
```

Expected: failures because the adapter is absent.

- [x] **Step 3: Implement explicit reference gather path**

Gather each request's logical K/V prefix from its page table into a temporary strided tensor, apply the public PyTorch attention operation with per-request causal length, and return rows in the packed request order. Expose counters for gather bytes, attention calls, and scheduler metadata time. Do not add a fallback engine or hide the gather in the cache object.

- [x] **Step 4: Run synthetic attention tests to verify pass**

Run the focused command and verify exact synthetic parity and isolation tests pass.

---

### Task 5: Integrate The Adapter With Qwen Outer Talker

**Files:**
- Modify: `qwen_asr_vllm/agent/qwen_tts_outer_varlen.py`
- Modify: `qwen_asr_vllm/agent/local_tts.py`
- Create: `tests/test_qwen_tts_outer_varlen_qwen_adapter.py`
- Modify: `bench/qwen_tts_active_prefix_parity_probe.py`

**Interfaces:**
- Consumes: prepared Qwen talker inputs and the existing ActivePrefixCache exact control.
- Produces: explicit `outer_varlen_paged_attention` backend option and parity probe mode for two request IDs with independent progress.

- [x] **Step 1: Write failing integration/option tests**

Assert the backend does not install the adapter by default, explicit opt-in installs exactly one adapter instance, incompatible configurations raise a capability error, and the probe records request IDs, page tables, logical lengths, per-request codec hashes, and output ordering.

- [x] **Step 2: Run integration tests to verify failure**

Run:

```bash
CUDA_VISIBLE_DEVICES='' pytest tests/test_qwen_tts_outer_varlen_qwen_adapter.py -q
```

Expected: failures because the option and probe mode are absent.

- [ ] **Step 3: Implement the opt-in Qwen integration**

Move request-local rope/past-hidden/text-condition state into `OuterVarlenRequest`. Keep Qwen prefill eager. For each packed decode tick, build codec IDs and conditions per row, run the reference adapter, update only the corresponding request state, apply EOS independently, and demultiplex audio/codec output. Preserve all original generation kwargs and keep ActivePrefixCache unchanged.

  - [x] Model-facing substep: add `QwenPagedCache` and `QwenOuterVarlenRunner` for explicit per-row positions, padded 4D causal masks, Qwen outer-model prefill, and packed codec-embedding decode. Full codec sampling/audio demultiplexing remains gated until checkpoint parity is completed.

- [ ] **Step 4: Run integration tests to verify pass**

Run the focused command plus the existing outer-engine and local-TTS suites:

```bash
CUDA_VISIBLE_DEVICES='' pytest -q \
  tests/test_qwen_tts_outer_varlen_qwen_adapter.py \
  tests/test_qwen_tts_outer_static_engine.py \
  tests/test_agent_local_tts.py
```

- [ ] **Step 5: Run real CUDA exact parity**

Use CUDA 2 and the local Qwen3-TTS checkpoint to compare two requests with different prompt lengths and deliberately staggered decode progress against independent ActivePrefixCache. Require exact KV/logit/codec/audio parity before proceeding to kernel work.

---

### Task 6: Benchmark Reference Adapter And Define Triton Kernel Gate

**Files:**
- Create: `bench/qwen_tts_outer_varlen_probe.py`
- Create: `tests/test_qwen_tts_outer_varlen_probe.py`
- Modify: `docs/superpowers/plans/2026-08-27-qwen-tts-varlen-paged-attention-adapter.md`
- Modify: `findings.md`
- Modify: `progress.md`

**Interfaces:**
- Consumes: existing ActivePrefixCache timing controls and the integrated reference adapter.
- Produces: reproducible A/B timing JSON and a go/no-go decision for the Triton kernel stage.

- [x] **Step 1: Write failing probe/report tests**

Require the probe to report per-request first audio, total generation, decode compute, scheduler overhead, aggregate frames/s, page allocations/releases, and exact parity status. Add a failure test for any missing request, page leak, cross-request mismatch, or non-finite metric.

- [x] **Step 2: Implement the reference timing probe**

Measure independent batch-1 ActivePrefixCache, position-bucketed batch, and packed reference adapter with the same seed/texts. Keep warmup separate from online timing and capture CUDA memory before/after. The reference adapter may be slower; report that result honestly and do not enable it by default based on metadata alone.

- [ ] **Step 3: Run probe tests and real CUDA timing**

Run CPU probe tests, then CUDA 2 with at least two different-progress sessions. Save the JSON result under `results/qwen_tts_outer_varlen_cuda2.json` and verify GPU baseline after cleanup.

- [ ] **Step 4: Record Triton kernel readiness**

Document the exact packed metadata contract, tensor layouts, page size, head geometry, and parity evidence. Mark Triton implementation ready only if the reference path is exact and page lifecycle is leak-free; otherwise record the blocking mismatch and leave the adapter opt-in.

- [ ] **Step 5: Run full verification**

Run:

```bash
CUDA_VISIBLE_DEVICES='' pytest -q
python -m py_compile \
  qwen_asr_vllm/agent/qwen_tts_outer_paged_cache.py \
  qwen_asr_vllm/agent/qwen_tts_outer_varlen.py \
  bench/qwen_tts_outer_varlen_probe.py
```

Expected: all existing tests and new adapter tests pass without new warnings.
