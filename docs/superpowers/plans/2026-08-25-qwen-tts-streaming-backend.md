# Qwen TTS Streaming Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the Qwen-TTS codec-step streaming probe into an opt-in production backend path and measure real ASR -> LLM -> TTS timing.

**Architecture:** Keep ASR, LLM, and TTS as resident runner/service components. Add `QwenTtsBackend.synthesize_stream()` using the existing codec-frame hook and chunk decoder, then expose it through the existing `ProcessTtsBackend` IPC path. Gate it with CLI flags so stable full-utterance synthesis remains available for comparison.

**Tech Stack:** Python, FastAPI/uvicorn latency UI, local `qwen_tts`, numpy WAV encoding, multiprocessing process runners, pytest.

**Spec:** Chat-approved C-stage design from 2026-08-25: productionize Stage B, adaptive chunks, reduce fragmentation where possible, verify audio quality/timing, then run real timing.

## Global Constraints

- Do not replace the stable TTS path by default until real timing has been measured.
- Do not add external services or remote model calls; use the local Qwen-TTS model.
- Preserve the existing process-isolated ASR, LLM, and TTS runner architecture.
- Prefer existing `synthesize_stream` IPC support over introducing a new protocol.
- Real timing output must distinguish non-streaming and codec-step streaming TTS.

---

### Task 1: Backend Streaming API

**Files:**
- Modify: `qwen_asr_vllm/agent/local_tts.py`
- Modify: `qwen_asr_vllm/agent/qwen_tts_streaming.py`
- Test: `tests/test_local_tts_streaming.py`

**Interfaces:**
- Consumes: `stream_decode_codec_frames(run_generate, speech_tokenizer, chunk_size, left_context_size, eos_token_id)`
- Produces: `QwenTtsBackend.synthesize_stream(text: str) -> Iterable[bytes | None]`

- [ ] Write tests for opt-in streaming support and adaptive first/rest chunk sizes.
- [ ] Run the new tests and verify they fail before implementation.
- [ ] Add constructor options `streaming`, `stream_chunk_size`, `stream_first_chunk_size`, and `stream_left_context_size`.
- [ ] Implement `synthesize_stream()` using the codec-frame hook when streaming is enabled; otherwise yield the non-streaming result.
- [ ] Run the targeted tests and verify they pass.

### Task 2: CLI Wiring

**Files:**
- Modify: `bench/voice_agent_timing.py`
- Modify: `bench/serve_voice_agent_ui.py`
- Test: `tests/test_voice_agent_timing_streaming_flags.py`

**Interfaces:**
- Consumes: `create_real_tts_agent(streaming: bool, stream_chunk_size: int, stream_first_chunk_size: int | None, stream_left_context_size: int)`
- Produces: CLI flags `--tts-streaming-engine`, `--tts-stream-chunk-size`, `--tts-stream-first-chunk-size`, and `--tts-stream-left-context-size`

- [ ] Write parser/factory tests for timing and UI CLIs.
- [ ] Run the tests and verify they fail before implementation.
- [ ] Add the flags and pass them into `QwenTtsBackend`.
- [ ] Ensure process-isolated timing and UI factories use the same TTS config.
- [ ] Run the targeted tests and verify they pass.

### Task 3: Real Timing

**Files:**
- Modify only if needed: `bench/voice_agent_timing.py`
- Output: `results/voice_agent_timing_*`

**Interfaces:**
- Consumes: `bench/voice_agent_timing.py --mode real --real-target async`
- Produces: JSON timing with `details.tts_streaming == true` for streaming runs.

- [ ] Run non-streaming async timing on an existing dataset/audio sample.
- [ ] Run streaming async timing on the same input with chunk8/first4.
- [ ] Compare first-audio, TTS wait, total time, and event offsets.
- [ ] Restart the latency UI with streaming enabled for live testing.
