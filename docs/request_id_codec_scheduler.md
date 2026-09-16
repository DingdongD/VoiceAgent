# Request-ID codec scheduler

Status: implemented and selected by the `cuda0-throughput` production profile.
The scheduler is strict: it fails service startup when the Qwen-TTS capability
contract is not present. It does not downgrade to the legacy stream-batch or
scalar process path.

## Execution

`VOICE_TTS_REQUEST_STEP_SCHEDULER=1` selects the scheduler in the concurrent
TTS service. It requires codec-step streaming, active-prefix outer runtime,
inner CUDA Graph, process isolation, and greedy outer/inner generation.
`activate_runtime_profile_environment()` sets this variable for
`cuda0-throughput` and sets `VOICE_TTS_LEGACY_STREAM_BATCH=0`. The legacy
`UnifiedTtsStreamBatchScheduler` is available only when the explicit `compat`
profile sets `VOICE_TTS_LEGACY_STREAM_BATCH=1`; it is never an implicit error
path from the production scheduler.

One service thread owns request preparation, outer prefill, codec generation,
and chunk decode. Outer generation is a resumable iterator holding its own KV
lease. Admission and cancellation happen between codec steps. A scheduler tick
batches inner predictor inputs across active request IDs. Outer steps are
grouped by KV length; joined cache layers write each request's KV separately
and concatenate the active prefixes for batched attention. Each fixed-size
cohort now owns a slot-backed joined cache: a surviving request keeps its
physical row, a newly admitted request takes a released row, and only the new
row materializes its prefix. Predictor inputs, outer outputs, and iterator
responses follow the request-id-to-slot map. This is not a paged-attention or
zero-copy adapter. Active-prefix source layers alias their current slot row,
so steady-state decode writes K/V once; admission and cross-pool ownership
transfers still materialize a prefix. Different KV lengths remain separate
outer groups.

The inner engine is called directly by this owner, bypassing its independent
queue. Strict parity is enabled by default: each request uses its own inner
predictor graph and only the resulting codebooks are assembled for the outer
cohort. This preserves scalar codec tokens while retaining outer scheduling.
Setting `VOICE_TTS_STRICT_INNER_PARITY=0` enables experimental inner batch GEMM;
it is expected to drift on FP16 reduction boundaries and is not production
eligible. Closing a request iterator resets and releases its KV lease.

The inner compute policy can be profiled explicitly with
`VOICE_TTS_INNER_SDP_BACKEND=math` and
`VOICE_TTS_INNER_DETERMINISTIC=1`. This disables Flash/MemEfficient SDPA and
enables PyTorch deterministic algorithms for the predictor scope. It does not
fall back to scalar execution. The flags remain disabled by default because a
full-system run must still produce audio chunks before its timing or parity is
considered valid. `VOICE_TTS_INNER_MATMUL_PRECISION` controls the matmul
precision setting (`highest`, `high`, or `medium`).

## Reproduce

Run the strict production profile as follows:

```bash
CUDA_VISIBLE_DEVICES=0 LLM_TEMPERATURE=0 VOICE_TTS_REQUEST_STEP_SCHEDULER=1 \
/opt/conda/envs/nano-vllm/bin/python bench/voice_agent_concurrency_probe.py \
  --runtime-profile cuda0-throughput --mode real \
  --audio results/cuda0_librispeech_sample.wav --llm-trigger final \
  --sessions 1 2 --tts-max-new-tokens 64 --timeout 120 \
  --output results/request_step_system_strict.json
```

These are real ASR/LLM/TTS measurements, excluding service startup and playback.
Audio is fed without real-time sleeps. Each level was measured once; two-session
p50/p95 describe sessions in that run, not a distribution of repeated trials.

## Historical A/B

| Sessions | Control first audio p50 | Candidate first audio p50 | Control total p50 | Candidate total p50 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1900.8 ms | 2342.9 ms | 4655.3 ms | 4490.2 ms |
| 2 | 3587.9 ms | 3440.8 ms | 7405.9 ms | 7227.4 ms |

The earlier candidate run (before strict parity) had 22 inner batch-2 ticks and
9 batch-1 ticks; all 53 outer executions were batch-1 because the two fragments
had different KV lengths. The two-session level added 63 inner batch-2 and 63
outer batch-2 executions. Those measurements are retained as the batch-GEMM
diagnostic result, not as a valid output-preserving speedup. Current strict
parity reports `inner_scalar_steps`; `slot_batches` still describes admitted
request IDs and `outer_slot_batches` describes outer execution groups.

All paired ASR hypotheses, LLM outputs and TTS input texts match. One-session
audio sizes differ (199988 vs 203828 bytes). Two-session sizes match but audio
SHA-256 differs. Thus the raw total reductions (3.5% / 2.4%) are **ineligible**
as exact-output speedups. The existing probe's `parity.matched` compares to its
own one-session signature, not to the control file: it is not an A/B gate.

Artifacts: `results/request_step_system_control.json` and
`results/request_step_system_candidate.json`.

The first candidate's `queue_wait_ms` included preparation/prefill. Current code
separates queue wait and `prefill_ms`; do not directly compare those earlier
values with pure control queue wait. Startup `resident_services.backend` metrics
are snapshots, not end-of-run engine counters. Use live `scheduler.slot_batches`
and `scheduler.outer_slot_batches` for this experiment.

## Remaining Optimization Work

- Localize batch-shape numerical drift using per-step codec/logit comparisons.
- Replace active KV concatenation with a validated variable-length adapter.
- Schedule first-audio work explicitly; single-session first audio regressed.
- Repeat matched-output trials and test late arrival, cancellation and mixed
  prompt lengths on GPU before making a speedup claim.
