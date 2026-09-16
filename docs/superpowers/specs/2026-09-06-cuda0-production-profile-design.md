# CUDA 0 Voice-Agent Production Profile Design

## Goal

Make the process-isolated ASR -> nano-vLLM -> Qwen-TTS runtime deterministic on
one 40 GiB CUDA device, remove first-cohort graph-capture latency, and enable the
already parity-validated TTS inner-predictor graph and compatible native batching.

This phase does not make experimental outer-talker graph, QKV fusion, or varlen
paged attention the default. Those paths require a separate performance and parity
gate after the production baseline is stable.

## Configuration Contract

The voice application exposes two optional fixed-cache settings:

- `ASR_NUM_KVCACHE_BLOCKS`: positive integer, or `-1` for automatic sizing.
- `LLM_NUM_KVCACHE_BLOCKS`: positive integer, or `-1` for automatic sizing.

Both values propagate through the process factories into their model engines.
Invalid values (`0` or values below `-1`) fail before loading model weights.

The CUDA 0 profile starts with ASR 128 blocks and LLM 64 blocks. These are capacity
floors for four 8192-token ASR sequences and four 4096-token LLM sequences with a
256-token block size. The profile may reserve modest additional blocks after real
resident-memory testing, but it must remain an explicit block count rather than a
free-memory fraction.

## LLM Graph Warmup

`NanoVllmStepBatchingBackend` accepts a sequence of startup batch sizes. For each
size, warmup submits that many one-token requests together through the existing
`add_request`/`step` scheduler. Common production shapes are `1, 2, 4`, clipped to
`max_num_seqs`, deduplicated, and validated as positive integers.

Warmup runs before the backend reports ready and resets chat history afterward.
Runtime metrics distinguish warmup steps from online steps so service statistics
remain meaningful. A warmup failure fails service startup; silently falling back
to request-path graph capture would reintroduce the latency being removed.

## TTS Production Profile

The UI and timing CLI gain a named `--runtime-profile` with `compat` and
`cuda0-throughput` choices. Existing defaults remain `compat` unless the profile
is explicitly selected, avoiding an unannounced output-policy change.

`cuda0-throughput` resolves to:

- all three services on logical `cuda:0` through environment configuration;
- codec-step TTS streaming;
- inner code-predictor CUDA Graph with two fixed slots;
- two unified TTS request workers and a 10 ms admission window;
- native outer batching only when the existing parity gate marks the cohort safe;
- greedy TTS controls in benchmark mode for reproducible timing.

Explicit CLI options override profile-derived values. Exact scalar parity remains
available and forces one worker/no outer native batching as it does today.

## Memory Admission

Before starting all services, the CUDA 0 launcher computes declared ASR and LLM KV
bytes from model metadata when available and records the final resident memory
after each service starts. If any service cannot initialize within its explicit
budget, startup fails with the service name and requested cache size. There is no
automatic retry with a smaller cache because that would make capacity dependent on
startup order and hide configuration errors.

## Measurement And Acceptance

CPU tests cover validation, factory propagation, warmup shape scheduling, profile
resolution, and explicit-option precedence. Existing exact-parity tests must stay
green.

On a clean physical CUDA 0, acceptance requires:

1. ASR, LLM, and TTS services remain resident concurrently without OOM.
2. A second four-request LLM cohort has no lazy graph capture and TTFT no worse than
   the clean-card warm baseline plus 10%.
3. Two-request greedy TTS output has identical per-request PCM byte counts between
   graph and non-graph native/service paths.
4. Full-chain timing records input-end, ASR final, LLM first token/sentence, TTS
   first chunk, total drain, queue wait, and audio duration separately.
5. Speedup is reported only for matching ASR text, LLM text, TTS rendered input,
   and TTS PCM duration.

## Deferred Work

After this profile passes, outer talker optimization proceeds against the fixed
baseline: first heterogeneous active-prefix/varlen slot batching, then shared-memory
PCM transfer. Per-position lazy outer CUDA Graph capture and non-parity QKV fusion
remain disabled because previous measurements showed regressions or codec drift.
