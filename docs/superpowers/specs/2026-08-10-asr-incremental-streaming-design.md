# ASR Incremental Streaming State Machine

Date: 2026-08-10  
Status: implemented (2026-08-10)  
Repo: `qwen-asr-vllm`  
Depends on: `2026-08-10-asr-streaming-session-design.md` (phase A/B session API)

## Goal

Add `chunk_policy="incremental"` to `StreamingSession`: most chunks transcribe only
the **unlocked audio tail** and append to committed text; periodically (or on
stress) **re-transcribe a sliding window** to allow bounded revision. Same
`partial` / `committed` / `final` events as today.

Approved: revision strategy **B** (bounded recompute), implementation approach
**1** (tail incremental + window recompute). Not true cross-chunk audio KV reuse.

## Why not full re-transcribe / pure lock

- Full growing-prefix re-transcribe repeats decode ~76× over a 300s session.
- Transcript speculation recovers ~32% decode steps but still sees full audio.
- Hard-locking committed text (strategy A) maximises latency but freezes errors.
- Deep unrestricted revision (strategy C) collapses back to full re-transcribe.

Bounded window recompute is the product compromise for agent dialogue.

## State

Per session, in addition to existing buffer / committed_text / last tokens:

| field | meaning |
|---|---|
| `committed_audio_end` | sample index; audio before this is considered covered by `committed_text` |
| `last_recompute_at` | sample index when the last window recompute finished |
| `tail_text` | unstable hypothesis for audio after committed region (may be rewritten) |

Invariant (soft): `text ≈ committed_text + sep + tail_text` for partials when no
violation handling has left them intentionally divergent.

## Algorithm (each `feed`)

1. Append PCM to `buffer`.
2. If `buffer` shorter than `tail_min_seconds` of new audio since last emit, return `[]`.
3. **Window recompute** if
   `(len(buffer) - last_recompute_at) / sample_rate >= recompute_seconds`:
   - `start = max(0, committed_audio_end - recompute_overlap_seconds * sr)`
   - Transcribe `buffer[start:]` via existing submit / `transcribe_with_draft`
     (draft = previous window hypothesis tokens when available).
   - Map result to text spanning from `start`; replace `tail_text` and, if the
     overlap region disagrees with the suffix of `committed_text`, set
     `commit_violation` and **keep** committed (phase-1 `on_violation="keep"`).
   - Set `last_recompute_at = len(buffer)`.
4. **Else tail-only**:
   - Transcribe `buffer[committed_audio_end:]` (must be ≥ `tail_min_seconds`).
   - `tail_text =` result text; partial text = join committed + tail.
5. Advance commit with `commit_lag_words` on the partial text; when committed
   grows by K words, advance `committed_audio_end` by a **proportional** estimate
   `K / max(words(partial),1) * len(buffer)` clamped so it never moves past
   `len(buffer) - lag_audio_floor`. Document that audio/text alignment is
   approximate; window recompute corrects drift.
6. Emit `partial` / optional `committed`.

`close(reuse_last=True)`: emit `final` from last partial; optional full-buffer
pass when `reuse_last=False`.

## Parameters

| name | default | notes |
|---|---|---|
| `commit_lag_words` | `16` | incremental recommends >0 |
| `recompute_seconds` | `6.0` | window period |
| `recompute_overlap_seconds` | `2.0` | overlap into committed audio |
| `tail_min_seconds` | `0.5` | skip tiny tails |
| `on_violation` | `"keep"` | only `keep` in v1 |

## API

```python
session = engine.open_stream(
    language="en",
    chunk_policy="incremental",
    commit_lag_words=16,
    recompute_seconds=6.0,
    recompute_overlap_seconds=2.0,
    tail_min_seconds=0.5,
)
```

Extra kwargs accepted by `open_stream` / `StreamingSession.__init__` and ignored
by other policies.

## Success criteria

- Unit tests: tail join, periodic recompute path invoked, violation keep semantics
- GPU: short growing session runs; cumulative transcribed-audio seconds
  (sum of lengths sent to the engine) ≪ ∑ growing-prefix durations for the same feeds
- `retranscribe` / `speculate` unchanged
- README states bounded correctness (not longform WER-equivalent)

## Non-goals (v1)

- Audio KV reuse across feeds
- `on_violation="rollback_window"`
- Exact force-alignment of committed words to sample indices
- HTTP/WS / voice_assistant_app wiring
