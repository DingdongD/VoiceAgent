# ASR Dual-Stream Encode∥Decode Overlap

Date: 2026-08-20  
Status: implemented (Phase 1 bench + Phase 2 opt-in engine path)  
Repo: `qwen-asr-vllm`  
Inspired by: `vla_lib` `DualStreamRuntime` / CUDA-stream encode∥decode  
Does not depend on importing `vla_lib`.

## Goal

Prove that when a scheduler step has **both** a non-empty audio-encode batch and a
non-empty model (prefill/decode) batch, running them on **two CUDA streams** can
beat serial execution by **≥ 1.15× wall clock**, with **bit-identical** (or
string-identical for full transcripts) outputs versus serial.

If the gate passes, a later Phase 2 may wire the same primitive into
`AsrEngine.step` behind a flag. Phase 1 does **not** change default engine
behavior.

## Decisions (locked)

| decision | choice |
|---|---|
| Rollout | Phase 1 independent bench → Phase 2 merge only if gate passes |
| Gate | wall speedup ≥ **1.15×** and correctness OK |
| Workloads | **both** synthetic mixed steps and real LibriSpeech-style replay |
| Execution model | **same-thread dual CUDA streams** (not DualStream-style worker threads) |
| CUDA graphs in Phase 1 | overlap path uses **eager** decode (`enforce_eager` or equivalent) |

## Non-goals (Phase 1)

- Dual-GPU / device split
- Changing streaming session / `recompute_overlap_seconds` semantics
- Changing mel frontend scheduling (already overlapped via `AsyncAsrEngine`)
- Changing continuous-batching admission policy
- Importing or depending on `vla_lib`

## Why this can help (and when it cannot)

Today `AsrEngine.step` is:

1. `schedule_audio` → `audio_runner.encode` → `cuda.synchronize` → `admit_prefill`
2. `schedule_model` → `model_runner.run` → postprocess

Encode and decode never share a launch window. Under multi-request load, request
B's encode can run while request A is decoding — same structural opportunity as
VLA trunk∥action-expert overlap.

Long-form **single-request** profiles show decode ≈ 97% of device time, so
encode∥decode will not move that needle. The interesting regime is **mixed
queues** (serving / high concurrency), where encode work and decode work coexist.

## Approach comparison

| approach | summary | verdict |
|---|---|---|
| **A. Same-thread dual CUDA streams** | Launch encode on `encode_stream`, model on `decode_stream`; sync both before admit/postprocess | **Chosen** — matches future `step` integration; keeps single-threaded scheduler |
| B. Dual threads + streams | Mirror `vla_lib.DualStreamRuntime` vision/policy threads | Rejected for Phase 1 — fights "one thread owns scheduler" invariant |
| C. Synthetic kernel-only microbench | Fake GEMMs to estimate upper bound | Insufficient alone — no ASR correctness signal |

## Architecture (Phase 1)

```
qwen-asr-vllm/
├── qwen_asr_vllm/engine/dual_stream.py   # reusable stream pair + serial/overlap runners
├── bench/dual_stream_overlap.py          # CLI: synthetic + real replay
└── results/dual_stream_*.json            # machine-readable report
```

### Primitive: `dual_stream.py`

Thin wrapper (no scheduler / block manager):

- Owns `encode_stream`, `decode_stream`, and an optional `encode_done` event.
- `run_serial(encode_fn, decode_fn)`: default stream; encode then decode; synchronize.
- `run_overlap(encode_fn, decode_fn)`: launch both; decode must **not** wait on
  this step's encode (see invariant below); synchronize both streams at end.

Callables close over runners and already-scheduled request lists. The primitive
only orders CUDA work.

### Invariant (data dependence)

Overlap is valid only when the model batch does **not** consume embeddings
produced by the same step's encode batch.

Current `step` already preserves this: encode → admit_prefill → schedule_model,
so newly encoded requests are not in the same model batch. Instrumented overlap
must:

1. `schedule_audio()` and `schedule_model()` **first** (same decisions as serial)
2. Run encode ∥ model via streams
3. **Then** `admit_prefill(encode_batch)` and `postprocess(...)`

Do not admit mid-flight encode results into the concurrent model batch.

### Synthetic workload

1. Warm up: encode + prefill enough requests so some are in `RUNNING_DECODE`.
2. Hold another set in `WAITING_ENCODE`.
3. Each measured round: one encode batch A and one decode batch B; time
   `run_serial` vs `run_overlap`.
4. Correctness: decode token ids equal; `audio_embeds` `allclose` (or equal)
   between modes for batch A.

### Real replay workload

1. LibriSpeech (or existing bench fixtures), e.g. 32–64 clips, high `max_num_seqs`,
   to produce mixed steps.
2. Two instrumented loops sharing the same scheduling decisions:
   - **serial**: current ordering (encode sync then model)
   - **overlap**: `run_overlap` with eager decode
3. Metrics focus on steps where **both** batches were non-empty; also report full
   wall time for the run.
4. Correctness: normalized full transcripts match serial for the same inputs.

### Report schema

```json
{
  "mode": "synthetic|real",
  "model": "...",
  "speedup_wall": 1.0,
  "speedup_mixed_steps": 1.0,
  "correctness_ok": true,
  "gate_1_15": false,
  "n_mixed_steps": 0,
  "notes": ""
}
```

`gate_1_15` is true only if `correctness_ok` and `speedup_mixed_steps >= 1.15`
(primary) — wall speedup is secondary context when mixed steps are few.

## Risks and mitigations

| risk | mitigation |
|---|---|
| CUDA graphs bound to default stream | Phase 1 overlap path **eager only** |
| Accidental encode→same-step decode dependence | Schedule both batches before launch; admit after sync |
| Timing under-counts CPU | Wall clock via `perf_counter` around full serial/overlap region |
| OOM on either stream | Mark run failed; no speedup claim |
| Allocator / SM contention → slowdown | Still a valid negative result; do not merge |
| Mel frontend dominates TTFT | Out of scope; already overlapped in `AsyncAsrEngine` |

## Phase 2 (conditional)

Only if Phase 1 report has `gate_1_15: true`:

- Wire `dual_stream` into `AsrEngine.step` behind `enable_dual_stream` / CLI flag
  (default **off**).
- Revisit CUDA graph capture on `decode_stream`.
- Reuse existing OOM halve-and-retry after both streams sync.
- Add a focused unit/integration test: mixed step token parity serial vs dual.

Phase 1 may leave a short comment in `engine.py` pointing at the primitive; it
must not change default behavior.

## Success criteria

1. Synthetic: correctness pass; report speedup (informational if < 1.15).
2. Real mixed: correctness pass; `speedup_mixed_steps >= 1.15` → approve Phase 2.
3. If gate fails: keep bench + JSON results documenting the finding; **no**
   production `step` change.

## Open questions (resolved)

- Rollout order → Phase 1 then conditional Phase 2.
- Gate threshold → 1.15× + correctness.
- Workloads → synthetic and real.
- Thread model → same-thread dual streams.
