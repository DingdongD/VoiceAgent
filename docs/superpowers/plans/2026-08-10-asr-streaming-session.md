# ASR Streaming Session Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Python `StreamingSession` API (phase A) that feeds a growing audio prefix, emits `partial` / optional `committed` / `final` events, and cancels in-flight work — then land phase-B previous-transcript speculation as a follow-on for ~30% decode-step savings.

**Architecture:** Session layer over `AsyncAsrEngine.submit` / `cancel`. Each `feed` appends PCM, cancels the prior request, re-transcribes the full buffer (growing prefix). Commit lag is a pure text protocol. Phase B later swaps the inner transcribe for verify-draft + residual decode without changing the event contract.

**Tech Stack:** Python 3.11, existing `qwen_asr_vllm` engine, pytest, numpy float32 PCM @ 16 kHz.

**Spec:** `docs/superpowers/specs/2026-08-10-asr-streaming-session-design.md`

## Global Constraints

- Growing-prefix chunks only (not non-overlapping windows)
- Phase A: `chunk_policy="retranscribe"` only; no HTTP/WS; no `voice_assistant_app` changes
- Draft for phase B = previous `output_token_ids`, not a small LM, not n-gram
- `commit_lag_words` default 0 (no `committed` events)
- Do not silently rewrite committed history on revision; set `commit_violation`
- Phase A must not claim latency wins in README
- Keep longform WER gate green after engine-touching tasks
- Prefer absolute imports consistent with the package; follow existing test markers (`gpu`, `checkpoint`, `slow`)

---

## File map

| path | role |
|---|---|
| `qwen_asr_vllm/engine/streaming.py` | **Create** — `StreamEvent`, commit helpers, `StreamingSession` |
| `qwen_asr_vllm/engine/async_engine.py` | **Modify** — `open_stream(...)` |
| `qwen_asr_vllm/engine/__init__.py` | **Modify** — export session types if other engine symbols are exported |
| `qwen_asr_vllm/__init__.py` | **Modify** — optional public export of `StreamEvent` / `StreamingSession` |
| `tests/test_streaming_session.py` | **Create** — unit + GPU session tests |
| `README.md` | **Modify** — short Streaming session section (A/B, performance honesty) |
| Phase B only (later tasks): `engine.py` / `model_runner.py` / `streaming.py` | integrate verify→continue decode; bench |

---

### Task 1: Commit helpers + StreamEvent (CPU)

**Files:**
- Create: `qwen_asr_vllm/engine/streaming.py`
- Test: `tests/test_streaming_session.py`

**Interfaces:**
- Produces:
  - `StreamEvent(kind, text, committed_text, audio_seconds, output_token_ids, chunk_index, commit_violation=False)`
  - `split_words(text: str) -> list[str]`
  - `join_words(words: list[str]) -> str`
  - `commit_candidate(partial_text: str, lag_words: int) -> str`
  - `advance_committed(previous: str, candidate: str) -> tuple[str, bool]`  
    → `(new_or_kept_committed, violation)`  
    Prefix check on whitespace-split words (same as lag). Violation if candidate is not equal to previous and not an extension of previous word list.

- [ ] **Step 1: Write failing tests for commit helpers**

```python
from qwen_asr_vllm.engine.streaming import (
    advance_committed,
    commit_candidate,
    join_words,
    split_words,
)

def test_commit_candidate_drops_lag():
    assert commit_candidate("one two three four", 2) == "one two"

def test_commit_candidate_lag_zero_is_full():
    assert commit_candidate("one two", 0) == "one two"

def test_advance_extends_prefix():
    text, violation = advance_committed("one two", "one two three")
    assert text == "one two three" and not violation

def test_advance_conflict_keeps_previous():
    text, violation = advance_committed("one two", "one dos three")
    assert text == "one two" and violation
```

- [ ] **Step 2: Run tests — expect fail (module missing)**

```bash
cd /home/qwen-asr-vllm && /opt/conda/envs/nano-vllm/bin/python -m pytest tests/test_streaming_session.py -q -k "commit_ or advance_" --no-header
```

- [ ] **Step 3: Implement helpers + `StreamEvent` dataclass in `streaming.py`**

Keep `StreamingSession` as a stub or omit until Task 2; only helpers needed here.

- [ ] **Step 4: Run tests — expect pass**

```bash
cd /home/qwen-asr-vllm && /opt/conda/envs/nano-vllm/bin/python -m pytest tests/test_streaming_session.py -q -k "commit_ or advance_" --no-header
```

- [ ] **Step 5: Commit** (only if the user asked to commit)

```bash
git add qwen_asr_vllm/engine/streaming.py tests/test_streaming_session.py
git commit -m "$(cat <<'EOF'
Add streaming commit helpers and StreamEvent.

EOF
)"
```

---

### Task 2: StreamingSession retranscribe + open_stream

**Files:**
- Modify: `qwen_asr_vllm/engine/streaming.py`
- Modify: `qwen_asr_vllm/engine/async_engine.py`
- Test: `tests/test_streaming_session.py`

**Interfaces:**
- Consumes: `AsyncAsrEngine.submit`, `RequestHandle.cancel` / `result`, `SamplingParams.for_audio`, commit helpers from Task 1
- Produces:
  - `AsyncAsrEngine.open_stream(language=None, context="", commit_lag_words=0, chunk_policy="retranscribe", sample_rate=16000) -> StreamingSession`
  - `StreamingSession.feed(pcm: np.ndarray) -> list[StreamEvent]`
  - `StreamingSession.close(reuse_last: bool = True) -> list[StreamEvent]`
  - Properties: `audio_seconds`, `committed_text`, `closed`

Behaviour notes:
- Reject `chunk_policy` other than `"retranscribe"` in phase A with `ValueError`
- `feed` on empty/too-short buffer (<160 samples): return `[]` (no event)
- `feed` cancels prior in-flight handle before submit
- `close(reuse_last=True)`: if last partial already covers full buffer, emit `final` from cache without resubmit; else one last submit
- After `close`, further `feed` raises `RuntimeError`
- Word lag: if `commit_lag_words > 0`, after each partial compute candidate; call `advance_committed`; emit `committed` only when committed text **changed** or on violation emit a `committed` event with `commit_violation=True` and unchanged `text=committed_text` so the flag is visible (document this in docstring)

- [ ] **Step 1: Write failing session tests with a stub engine**

Use a fake engine object that records submits/cancels and returns canned `AsrOutput` (pattern from `tests/test_async_engine.py` stubs). Cover:
- growing feeds → one partial each
- close → one final; `reuse_last` avoids extra submit
- lag>0 emits committed; conflict sets violation
- feed after close raises
- cancel called when a second feed arrives while first unfinished (simulate slow future)

- [ ] **Step 2: Run — expect fail**

```bash
cd /home/qwen-asr-vllm && /opt/conda/envs/nano-vllm/bin/python -m pytest tests/test_streaming_session.py -q -k "Session or session" --no-header
```

- [ ] **Step 3: Implement `StreamingSession` + `open_stream`**

- [ ] **Step 4: Run stub tests — expect pass**

- [ ] **Step 5: Commit** (only if asked)

---

### Task 3: GPU integration parity

**Files:**
- Test: `tests/test_streaming_session.py` (add marked class)

**Interfaces:**
- Consumes: real `AsyncAsrEngine`, LibriSpeech or short TED clip via `bench.data`

- [ ] **Step 1: Write GPU test**

```python
@pytest.mark.gpu
@pytest.mark.checkpoint
@pytest.mark.slow
def test_final_matches_oneshot_transcribe(model_dir):
    # 4–6s clip, 2s feeds, close(reuse_last=True)
    # normalize_text(final) == normalize_text(engine.transcribe([full])[0].text)
```

Also: after session closes, `engine.engine.scheduler.num_in_flight == 0` (or equivalent health/tracked empty).

- [ ] **Step 2: Run GPU test**

```bash
cd /home/qwen-asr-vllm && CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/nano-vllm/bin/python -m pytest tests/test_streaming_session.py -q -m "gpu" --no-header
```

- [ ] **Step 3: Fix mismatches (sampling caps, language, stop-token stripping)**

- [ ] **Step 4: Commit** (only if asked)

---

### Task 4: README + package exports

**Files:**
- Modify: `README.md`
- Modify: `qwen_asr_vllm/__init__.py` and/or `engine/__init__.py` as needed

- [ ] **Step 1: Document Streaming session**

Must state explicitly:
- Phase A = growing-prefix re-transcribe + events; **no latency claim**
- Phase B = previous-transcript draft (~33% accept, ~32% decode steps); not small-LM / not n-gram
- Non-overlapping chunks unsupported
- Point to spec + `profile_streaming.py` / `profile_speculation.py`

- [ ] **Step 2: Export `StreamEvent`, `StreamingSession` from a stable import path**

- [ ] **Step 3: Lint + full relevant tests**

```bash
cd /home/qwen-asr-vllm && /opt/conda/envs/nano-vllm/bin/python -m ruff check qwen_asr_vllm/engine/streaming.py qwen_asr_vllm/engine/async_engine.py tests/test_streaming_session.py
CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/nano-vllm/bin/python -m pytest tests/test_streaming_session.py tests/test_async_engine.py -q --no-header
```

- [ ] **Step 4: Commit** (only if asked)

---

### Task 5 (Phase B follow-on): Speculate-then-decode path

**Status: implemented 2026-08-10** (`transcribe_with_draft`, `chunk_policy="speculate"`).

**Do not start until Tasks 1–4 are done and the user asks for phase B.**

**Files:**
- Modify: `qwen_asr_vllm/engine/engine.py` — `transcribe_with_draft(...)` or integrate verify into request lifecycle so accepted tokens stay in KV and decode continues
- Modify: `qwen_asr_vllm/engine/streaming.py` — `chunk_policy="speculate"`
- Modify: `bench/profile_speculation.py` or new `bench/profile_streaming_speculate.py` — latency vs retranscribe
- Test: `tests/test_speculate.py` + streaming GPU test that accept path WER-matches retranscribe on same feeds

**Interfaces:**
- Produces: same `StreamEvent` stream; internal path uses previous `output_token_ids` as draft
- Fallback: accept=0 → identical to retranscribe
- Gate: `tests/test_longform_wer.py` still passes; streaming final ≡ retranscribe final on a short growing session

Key design constraint from measurements: verify must include the last prompt token in the query (or full recompute) so prediction for draft[0] is valid — already handled in `verify_draft` by forcing `num_computed_tokens=0`. Continuing decode after accept must **not** deallocate; today's measure-only API must grow a “keep blocks / append residual” mode.

- [ ] **Step 1: Design spike (½ day max)** — choose integrate-into-`step` vs session-private runner path; write short ADR note under `docs/superpowers/specs/` if API changes
- [ ] **Step 2: TDD for “self-draft full accept then one more token is EOS/stop” on real audio**
- [ ] **Step 3: Wire `chunk_policy="speculate"`**
- [ ] **Step 4: Bench 120s TED chunk latency vs phase A; record accept rate**
- [ ] **Step 5: Longform WER gate**
- [ ] **Step 6: README update with measured delta**

---

## Execution order

```text
Task 1 (helpers) → Task 2 (session API) → Task 3 (GPU parity) → Task 4 (docs/exports)
        ↓
   user checkpoint
        ↓
Task 5 (phase B speculation)  # optional, separate approval
```

## Done definition (phase A)

- Stub + GPU tests green
- `open_stream` / `feed` / `close` usable from Python
- README states performance honesty
- Spec chunk model reflected in code docstrings
- No WS/assistant coupling
