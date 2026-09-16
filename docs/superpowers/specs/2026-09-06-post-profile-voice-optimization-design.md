# Post-Profile Voice Optimization Design

## Scope

Complete the five approved follow-ups without replacing the resident
ASR/LLM/TTS process architecture: deterministic timing lifecycle, production
outer-talker integration, request-id slot scheduling, shared-memory PCM IPC,
and `1/2/4/8` full-session load measurement.

## Lifecycle And Evidence

An agent `done` event terminates one response turn, not prerecorded input.
`run_real_async()` must continue collecting ASR events until input feeding and
`session.close()` finish. Reports retain first-turn latency while exposing the
final ASR transcript separately. A comparison is eligible for a speedup only
when ASR final text, LLM text, rendered TTS inputs, PCM byte count, and PCM
duration match.

## Outer Talker Modes

`compat` remains unchanged. `cuda0-throughput` enables the existing
exact-parity ActivePrefix outer engine, inner predictor CUDA Graph, and native
request batching. The paged/varlen adapter remains a named explicit mode until
a real-model gate confirms codec equality. QKV fusion, per-position outer CUDA
Graphs, and custom fused projection stay disabled because existing reports
show parity or latency regressions.

Installation metrics must distinguish a merely attached adapter from a model
path that executed it. Unsupported combinations fail before model loading;
runtime exceptions are surfaced rather than silently switching engines.

## Dynamic Request Scheduling

The resident TTS scheduler owns request IDs and fixed admission slots. It
groups compatible requests during a bounded admission window, runs one native
batch, immediately refills available slots from the pending queue, and records
batch size, queue wait, active slots, padded slots, cancellations, and refill
count. Compatibility remains defined by the backend parity gate. This phase
does not claim arbitrary mid-kernel admission; paged outer decode remains the
mechanism for a later codec-step refill engine.

## Shared-Memory PCM IPC

Small control messages remain on multiprocessing queues. Byte payloads above
a configurable threshold use one-owner shared-memory descriptors. The sender
creates and writes the segment; the receiver copies bytes and unlinks it in a
`finally` block. Cancellation, timeout, process exit, and normal shutdown all
clean outstanding segments. The public TTS API continues returning `bytes`.

## Concurrency Probe

A new probe creates one shared set of resident services and independent
coordinators for `1/2/4/8` sessions. It reports per-session first audio and
completion latency, aggregate throughput, p50/p95, scheduler occupancy,
fallbacks, output hashes, errors, and resident memory. Fake mode is mandatory
for CPU regression; real mode requires a clean-card preflight and deterministic
LLM/TTS generation.

## Acceptance

- No early `done` may truncate final ASR reporting.
- `compat` behavior and public bytes APIs remain unchanged.
- Production outer mode must expose execution metrics and preserve controlled
  codec/PCM parity.
- Shared-memory segments must be absent after normal, cancelled, and failed
  requests.
- The concurrency probe must complete `1/2/4/8` fake sessions and write valid
  JSON; real CUDA claims require an idle device and matching outputs.

