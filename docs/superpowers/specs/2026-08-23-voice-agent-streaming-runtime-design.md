# Voice Agent Streaming Runtime Design

Date: 2026-08-23
Status: approved for initial migration
Repo: `qwen-asr-vllm`

## Goal

Move the prior ASR/LLM/TTS voice agent concept into this repo as a streaming runtime that can overlap ASR, LLM, and TTS work without contaminating the ASR engine scheduler.

## Scope

This migration adds an agent runtime layer:

```text
PCM chunks -> ASR stream events -> LLM text deltas -> TTS audio chunks
```

The runtime consumes the existing `AsyncAsrEngine.open_stream()` API. It exposes injectable LLM and TTS backend interfaces so real `nano-vllm` and Qwen3-TTS adapters can be used in production while tests use deterministic stubs.

## Non-Goals

- No wake-word, robot microphone, speaker device, or systemd migration from `/home/voice_assistant_app`.
- No changes to ASR scheduler internals.
- No requirement that Qwen3-TTS itself supports model-native audio-token streaming in this phase.
- No dependency on `nano-vllm` or `qwen_tts` for the default test suite.

## Architecture

Add `qwen_asr_vllm.agent` as a separate package:

- `events.py`: typed runtime event classes and JSON serialization helpers.
- `backends.py`: `LlmBackend`, `TtsBackend`, optional adapter factories.
- `coordinator.py`: a session-level orchestrator that accepts PCM chunks, forwards ASR events, starts LLM on committed or final transcript text, chunks LLM output for TTS, and emits audio chunks.

Extend `server.py` with an optional WebSocket route:

- `WS /v1/voice/sessions`: accepts JSON control messages and binary PCM chunks, emits JSON events and binary audio.

The existing `/v1/audio/transcriptions` endpoint remains unchanged.

## Data Flow

1. Client sends binary 16 kHz float32 PCM chunks or WAV bytes decoded by the caller-selected endpoint contract.
2. Coordinator feeds chunks to `StreamingSession.feed()`.
3. ASR `partial`, `committed`, and `final` are re-emitted as agent events.
4. LLM starts when configured trigger text is available. The initial migration uses `final` by default for correctness; `committed` can be selected for latency.
5. LLM text deltas are emitted immediately.
6. Sentence or phrase boundaries are queued to TTS.
7. TTS audio bytes are emitted as soon as each chunk is synthesized.

## Cancellation And Backpressure

Each voice session owns one coordinator. Closing or resetting a session cancels the active ASR session and prevents future LLM/TTS output from being emitted. TTS work is bounded by a small queue in the coordinator; production deployments can map this to device-specific GPU queues later.

## Testing

Unit tests use fake ASR sessions, fake streaming LLM, and fake TTS. Tests pin:

- Event JSON shape.
- LLM starts after `final` by default.
- LLM can start from `committed` when configured.
- TTS fires at sentence boundaries and emits audio chunks before final done.
- Server keeps existing transcription endpoint behavior while adding the voice route.
