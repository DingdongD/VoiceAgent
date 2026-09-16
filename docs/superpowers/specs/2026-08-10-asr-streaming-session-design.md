# ASR Streaming Session Design

Date: 2026-08-10  
Status: phase A + phase B implemented (2026-08-10)  
Repo: `qwen-asr-vllm`

## Goal

Build a real streaming session API on top of the existing offline engine so an
agent flow (streaming ASR → local LLM + tools → streaming TTS) can consume
incremental transcripts with a clear commit protocol — then accelerate the hot
path without lying about what the measurements allow.

Approved choices:

- Phasing: **C** — session/events/cancel first, speculative decode second
- Events: **B** — `partial` / `committed` / `final` with `commit_lag_words` (default 0)
- Delivery surface for phase A: **A** — Python API only (no HTTP/WS, no
  `voice_assistant_app` wiring in this slice)

## Performance expectations (read this first)

**Phase A alone does not improve throughput or chunk latency.**  
It wraps today's re-transcribe baseline in a session object and event protocol.
Wall-clock cost per chunk stays the same order as
`bench/profile_streaming.py`. The win is integration: cancel, partials, and a
commit hook for the agent.

**Phase B is where performance moves.**  
Wire the previous chunk's `output_token_ids` through `verify_draft` / accept /
continue decode. Measured ceiling on TED audio (2s chunks):

| utterance | corpus draft accept | decode steps saved |
|---|---|---|
| 120s | ~0.35 | ~34% |
| 300s | ~0.33 | ~32% |

Acceptance is bimodal (many chunks near 0, many near 1). Word-level stability
(~99%) does **not** imply token-prefix accept. Early punctuation edits (e.g.
`"TED,"` → `"TED"`) collapse LCP.

**Out of scope for both phases (deliberately):**  
Perfect audio-KV reuse across chunks. Prefill is 2–5% of device time; the
structural layout (`[prefix][audio…][suffix][transcript…]`) plus RoPE shift
makes this the wrong first bet. Ceiling ≈ **2–5%** end-to-end even if perfect.

So: streaming *management* enables the agent and unlocks phase-B acceleration;
it is not by itself a 62× story. Offline continuous-batching gains remain for
multi-utterance batch workloads, separate from single-session streaming latency.

## Chunk model (important)

Sessions use a **growing audio prefix**, not non-overlapping windows:

```text
feed #1: audio[0:2s]           → re-transcribe → partial A
feed #2: audio[0:4s]           → re-transcribe → partial B
feed #3: audio[0:6s]           → re-transcribe → partial C
```

Consequences:

- Later requests **contain** earlier audio; decode **repeats** most of the
  transcript work (measured ~76× decode tokens over a 300s session vs the final
  transcript length).
- Previous-chunk words usually reappear in the next partial (same utterance
  re-recognized) but are **not guaranteed** token-identical; revisions happen.
- Phase-B transcript draft only makes sense under this cumulative model.
  Non-overlapping “transcribe only the new 2s” is **out of scope** and would
  need a different architecture (incremental state + explicit revision policy).

## Non-goals (phase A)

- HTTP / WebSocket endpoints
- Changes to `/home/voice_assistant_app`
- Audio KV cross-chunk reuse
- Scheduler rewrite or CUDA-graph changes
- Non-overlapping chunk transcription
- Guaranteeing that `committed` never disagrees with a later revision (data
  shows deep revisions; lag commit is a protocol, not a correctness proof)

## Phase A design

### Public API

```python
from qwen_asr_vllm.engine.async_engine import AsyncAsrEngine
from qwen_asr_vllm.engine.streaming import StreamEvent

engine = AsyncAsrEngine(model=...)
session = engine.open_stream(
    language="en",
    commit_lag_words=0,
    chunk_policy="retranscribe",  # phase B adds "speculate"
)

for event in session.feed(pcm_chunk):  # sync iterator over this chunk's events
    assert event.kind in ("partial", "committed", "final")

final_events = session.close()  # last re-transcribe → final; cancel in-flight
```

Async variant: `async for event in session.feed_async(pcm_chunk)` if the
session is owned by `AsyncAsrEngine` (preferred implementation path).

### Types

`StreamEvent`:

| field | meaning |
|---|---|
| `kind` | `partial` \| `committed` \| `final` |
| `text` | full hypothesis for this emission (for `committed`, the locked prefix only) |
| `committed_text` | current committed prefix (cumulative) |
| `audio_seconds` | audio duration visible to this emission |
| `output_token_ids` | token ids for the full hypothesis when `kind=partial`/`final` (for phase-B draft) |
| `chunk_index` | 0-based chunk that produced this emission |
| `commit_violation` | optional bool: new hypothesis disagrees inside already-committed region |

### Event semantics

1. **`partial`** — full transcript of audio accumulated so far. May revise
   earlier words on later chunks.
2. **`committed`** — `words(partial)[:-commit_lag_words]` joined, but never
   shorter than the previous committed string in *intent*. Implementation:
   - Compute candidate = drop last N words from partial.
   - If candidate is a prefix extension of previous committed (or equal), emit
     `committed` with the new text.
   - If candidate conflicts with previous committed (not a prefix match under
     the same normalization used for WER), set `commit_violation=True`, **do
     not rewrite history in the event stream**, keep `committed_text` at the
     last good value, and still emit `partial` with the truth. Callers that
     need strict lock can ignore later partials inside the locked region; we
     refuse to silently pretend the model agreed.
3. **`final`** — emitted once from `close()` after a last transcription of all
   buffered audio (or reuse the last partial if `close(reuse_last=True)`).

`commit_lag_words=0`: do not emit `committed` events (partial + final only).

### Session behaviour

- Buffer float32 PCM @ 16 kHz (or resample at feed boundary via existing
  frontend assumptions — session requires 16 kHz like the engine).
- Each `feed()`:
  1. Append samples.
  2. Cancel any in-flight transcription for this session.
  3. Submit full buffer via existing `AsyncAsrEngine.submit` /
     `SamplingParams.for_audio`.
  4. On result: emit `partial`; maybe `committed`.
- `close()`: cancel in-flight, optional final submit, emit `final`, free buffer.
- One session ↔ at most one in-flight request id (tracked for cancel).

### Module layout

| path | responsibility |
|---|---|
| `qwen_asr_vllm/engine/streaming.py` | `StreamEvent`, `StreamingSession`, commit helpers |
| `qwen_asr_vllm/engine/async_engine.py` | `open_stream(...)` factory |
| `tests/test_streaming_session.py` | event order, cancel, lag=0/N, final ≡ offline transcribe |
| `README.md` | short “Streaming session (phase A/B)” section |

### Success criteria (phase A)

- Growing-prefix session emits a `partial` per `feed` that produced speech;
  exactly one `final` on `close`
- `final.text` matches one-shot `transcribe` on the same audio after
  `normalize_text` (or English normalizer if language=en and we choose parity
  with longform — default: `normalize_text` for unit speed)
- Feeding a new chunk cancels the previous in-flight request (no leaked
  tracked requests after settle)
- `commit_lag_words>0`: committed word count is non-decreasing; violations
  observable via `commit_violation`
- Existing suite including `tests/test_longform_wer.py` still passes

## Phase B design (follow-on, not phase A code)

### Draft source: previous transcript — not a draft model, not n-gram

Speculative decoding here means: take the **previous chunk's greedy
`output_token_ids`** as the draft for the new (longer) audio, run one verify
forward (`ModelRunner.verify` / `AsrEngine.verify_draft`), accept the leading
token prefix that still matches argmax, then decode only the residual.

| draft style | used? | why |
|---|---|---|
| Small draft LM (classic speculative decoding) | **No (phase B)** | Needs a second model on-GPU, training/alignment for ASR transcripts, and does not exploit the fact that consecutive chunks already share ~99% words |
| N-gram / prompt lookup draft | **No (phase B)** | ASR output is open vocabulary + punctuation; n-gram hit rate on spontaneous speech is a guess we have not measured, and it still would not reuse the already-computed previous hypothesis |
| Previous-chunk transcript tokens | **Yes** | Already measured on TED: ~33% corpus token-prefix accept, ~32% decode steps saved; `verify` ≡ LCP at temperature 0; self-draft on same audio accepts 100% |

This is closer to “prompt lookup / recycling the last hypothesis” than to Medusa/EAGLE.
A small draft model or n-gram remains a later experiment if transcript-draft
acceptance proves too bimodal in product traffic — it is not the phase-B plan.

### Steps

1. Extend `chunk_policy="speculate"`:
   - Seed draft from previous partial's `output_token_ids` (strip stop ids).
   - Verify + continue decode from the accepted prefix (scheduler-integrated
     variant; today's `verify_draft` only measures and tears down).
2. Fallback to full re-transcribe decode when draft is empty or accept length is 0.
3. Gate with longform WER + streaming latency bench vs phase A.
4. Document bimodal accept; do not capacity-plan on the mean alone.

## Risks

| risk | mitigation |
|---|---|
| Users expect phase A latency win | This doc; README warning |
| Lag commit unsafe (deep revisions) | default lag=0; violations visible |
| Cancel races with completing step | reuse existing `cancel_requested` path; session waits/drains one request |
| Final ≠ last partial if close re-runs | allow `close(reuse_last=True)` default True when last partial covers full buffer |

## Approval

- Product shape: approved in chat (C + events B + Python-only A)
- Performance narrative: phase A = API; phase B ≈ 30% decode-step reduction —
  confirmed as the intended path to “提升性能”
