# Voice Agent Streaming Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a testable ASR -> LLM -> TTS streaming runtime around the existing ASR engine.

**Architecture:** Keep ASR inference internals unchanged and add a separate `qwen_asr_vllm.agent` package. The coordinator consumes `AsyncAsrEngine.open_stream()` events, drives an injectable streaming LLM backend, chunks text for an injectable TTS backend, and exposes events to a WebSocket route.

**Tech Stack:** Python 3.10, dataclasses, asyncio, FastAPI WebSocket, pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-voice-agent-streaming-runtime-design.md`

## Global Constraints

- Do not require `nano-vllm`, `qwen_tts`, GPU, or model checkpoints in default tests.
- Do not change ASR scheduler, model runner, or existing `/v1/audio/transcriptions` behavior.
- Use `final` ASR text as the default LLM trigger; allow `committed` via configuration.
- Keep old voice assistant robot/device/wake-word concerns out of this repo.

---

### Task 1: Event Protocol

**Files:**
- Create: `qwen_asr_vllm/agent/__init__.py`
- Create: `qwen_asr_vllm/agent/events.py`
- Test: `tests/test_agent_events.py`

**Interfaces:**
- Produces: `AgentEvent(type: str, data: dict)`, `AgentEvent.to_json()`, `agent_event(type: str, **data) -> AgentEvent`

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_events.py` with assertions that `agent_event("llm_chunk", text="hi").to_json()` serializes to `{"type": "llm_chunk", "text": "hi"}`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_events.py -q`
Expected: FAIL because `qwen_asr_vllm.agent` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `qwen_asr_vllm/agent/events.py` with a dataclass and JSON serializer.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_events.py -q`
Expected: PASS.

### Task 2: Coordinator With Stub Backends

**Files:**
- Create: `qwen_asr_vllm/agent/backends.py`
- Create: `qwen_asr_vllm/agent/coordinator.py`
- Test: `tests/test_agent_coordinator.py`

**Interfaces:**
- Consumes: `agent_event(...)`
- Produces: `VoiceAgentCoordinator.feed(pcm) -> list[AgentEvent]`, `VoiceAgentCoordinator.close() -> list[AgentEvent]`, `LlmBackend.chat_stream(text)`, `TtsBackend.synthesize(text)`

- [ ] **Step 1: Write failing coordinator tests**

Tests should define fake ASR session events, fake LLM chunks `["hello", "."]`, fake TTS bytes `b"wav:hello."`, and assert ASR final triggers LLM/TTS output.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_agent_coordinator.py -q`
Expected: FAIL because coordinator modules do not exist.

- [ ] **Step 3: Implement coordinator**

Implement a synchronous coordinator that re-emits ASR events, starts LLM after trigger text, emits `llm_chunk`, and emits `tts_chunk` on sentence boundary.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_agent_events.py tests/test_agent_coordinator.py -q`
Expected: PASS.

### Task 3: Voice WebSocket Route

**Files:**
- Modify: `qwen_asr_vllm/server.py`
- Test: `tests/test_agent_server.py`

**Interfaces:**
- Consumes: `create_app(engine, voice_factory=None)`
- Produces: `WS /v1/voice/sessions`

- [ ] **Step 1: Write failing server test**

Test should pass a fake `voice_factory`, open `/v1/voice/sessions`, send JSON `{"type": "start"}`, send binary PCM, send JSON `{"type": "close"}`, and assert JSON events are returned.

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_agent_server.py -q`
Expected: FAIL because `create_app` does not accept `voice_factory` and route is missing.

- [ ] **Step 3: Implement route**

Add optional `voice_factory` argument to `create_app`; if omitted, do not create a voice route. If provided, create session per WebSocket.

- [ ] **Step 4: Run existing server tests and new server test**

Run: `pytest tests/test_server.py tests/test_agent_server.py -q`
Expected: PASS.

### Task 4: Documentation

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: implemented agent runtime APIs.
- Produces: short usage section for Python and WebSocket voice runtime.

- [ ] **Step 1: Add concise README section**

Document `VoiceAgentCoordinator` and `create_app(..., voice_factory=...)`.

- [ ] **Step 2: Run targeted tests**

Run: `pytest tests/test_agent_events.py tests/test_agent_coordinator.py tests/test_agent_server.py tests/test_server.py -q`
Expected: PASS.
