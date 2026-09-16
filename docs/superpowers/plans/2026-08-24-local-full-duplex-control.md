# Local Full-Duplex Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local full-duplex control to the voice agent while preserving resident local model services.

**Architecture:** Keep `AsyncVoiceAgentCoordinator` as the orchestrator and add focused helper modules for segmentation and turn decisions. Extend backend protocols and process runners additively so local Qwen TTS and nano-vLLM continue to work without cloud-style dependencies.

**Tech Stack:** Python 3, asyncio, multiprocessing queues, pytest, FastAPI WebSocket.

**Spec:** `docs/superpowers/specs/2026-08-24-local-full-duplex-control-design.md`

## Global Constraints

- Do not add xtalk, LangChain, Deepgram, ElevenLabs, or cloud API dependencies.
- Do not replace `ProcessAsrEngine`, `ProcessNanoLlmBackend`, `ProcessTtsBackend`, or `NanoVllmStepBatchingBackend`.
- Do not move ASR back to utterance-level batch mode.
- Existing `asr_*`, `llm_start`, `llm_chunk`, `llm_done`, `tts_chunk`, and `done` events remain compatible.

---

### Task 1: Text Segmentation

**Files:**
- Create: `qwen_asr_vllm/agent/text_segmenter.py`
- Test: `tests/test_agent_text_segmenter.py`
- Modify: `qwen_asr_vllm/agent/async_coordinator.py`

**Interfaces:**
- Produces: `TextSegmenter.add(token: str) -> list[str]`, `TextSegmenter.flush() -> str | None`
- Produces: `TextSegmenter(min_chars: int = 1, flush_chars: int | None = None)`

- [ ] Write failing tests for abbreviation, decimal, Chinese punctuation, and flush threshold behavior.
- [ ] Run `pytest tests/test_agent_text_segmenter.py -q` and confirm failures are from missing module.
- [ ] Implement `TextSegmenter`.
- [ ] Replace coordinator's private sentence split logic with `TextSegmenter`.
- [ ] Run `pytest tests/test_agent_text_segmenter.py tests/test_agent_async_coordinator.py -q`.

### Task 2: Turn State and Interruption

**Files:**
- Create: `qwen_asr_vllm/agent/turn.py`
- Test: `tests/test_agent_turn.py`
- Modify: `qwen_asr_vllm/agent/async_coordinator.py`

**Interfaces:**
- Produces: `VoiceTurnStateMachine.on_user_audio() -> list[AgentEvent]`
- Produces: `VoiceTurnStateMachine.on_agent_start() -> list[AgentEvent]`
- Produces: `VoiceTurnStateMachine.on_agent_done() -> list[AgentEvent]`

- [ ] Write failing tests for listening/thinking/speaking transitions and interruption while speaking.
- [ ] Run `pytest tests/test_agent_turn.py -q` and confirm failures are from missing module.
- [ ] Implement `VoiceTurnStateMachine`.
- [ ] Wire coordinator to emit turn events on audio feed, LLM start, first TTS, done, and interruption.
- [ ] Run `pytest tests/test_agent_turn.py tests/test_agent_async_coordinator.py -q`.

### Task 3: Streaming TTS Fallback

**Files:**
- Modify: `qwen_asr_vllm/agent/backends.py`
- Modify: `qwen_asr_vllm/agent/async_coordinator.py`
- Modify: `qwen_asr_vllm/agent/service_runners.py`
- Test: `tests/test_agent_async_coordinator.py`
- Test: `tests/test_agent_service_runners.py`

**Interfaces:**
- Adds optional `TtsBackend.synthesize_stream(text: str) -> Iterable[bytes | None]`
- Adds `ProcessTtsBackend.synthesize_stream(text: str)`

- [ ] Write failing tests proving streaming TTS emits two `tts_chunk` events and fallback marks `tts_streaming=false`.
- [ ] Run targeted tests and confirm failures.
- [ ] Implement optional streaming TTS protocol and process runner command.
- [ ] Run targeted coordinator and service runner tests.

### Task 4: LLM Cancel and Playback Ack

**Files:**
- Modify: `qwen_asr_vllm/agent/async_coordinator.py`
- Modify: `qwen_asr_vllm/agent/service_runners.py`
- Modify: `qwen_asr_vllm/server.py`
- Modify: `qwen_asr_vllm/agent/latency_ui.py`
- Test: `tests/test_agent_async_coordinator.py`
- Test: `tests/test_agent_service_runners.py`
- Test: `tests/test_agent_server.py`

**Interfaces:**
- Adds `AsyncVoiceAgentCoordinator.interrupt()`
- Adds optional backend `cancel(request_id: int | None = None)`
- Adds WebSocket control messages `interrupt` and `tts_played`

- [ ] Write failing tests for interrupt cancelling pending output and server accepting playback ack.
- [ ] Run targeted tests and confirm failures.
- [ ] Implement best-effort local cancellation and playback ack event flow.
- [ ] Run targeted tests.

### Task 5: Cleanup and Verification

**Files:**
- Modify: `qwen_asr_vllm/agent/async_coordinator.py`
- Modify only proven-unused compatibility code if tests and search show no current runtime dependency.

- [ ] Search for legacy/fallback references before deleting.
- [ ] Remove coordinator-local sentence splitting once `TextSegmenter` is wired.
- [ ] Keep synchronous coordinator and threaded runner exports if tests or factory still depend on them.
- [ ] Run `pytest -q -m "not gpu and not checkpoint"`.

