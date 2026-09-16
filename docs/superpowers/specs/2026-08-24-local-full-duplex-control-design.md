# Local Full-Duplex Control Design

## Goal

Move xtalk's full-duplex control ideas into the existing local runner/service voice runtime without replacing local Qwen ASR/TTS, nano-vLLM step batching, or process isolation.

## Non-Goals

- Do not add xtalk, LangChain, Deepgram, ElevenLabs, or cloud API dependencies.
- Do not replace `ProcessAsrEngine`, `ProcessNanoLlmBackend`, `ProcessTtsBackend`, or `NanoVllmStepBatchingBackend`.
- Do not move ASR back to utterance-level batch mode.
- Do not delete public synchronous APIs unless tests prove they are unused by the current app.

## Architecture

The existing `AsyncVoiceAgentCoordinator` remains the session orchestrator. It gains three local control primitives:

- `TextSegmenter`: turns LLM text deltas into stable TTS text segments with abbreviation/decimal aware boundaries, flush thresholds, and coalescing compatibility.
- `VoiceTurnStateMachine`: records listening/thinking/speaking state and emits explicit interruption decisions.
- Optional streaming TTS protocol: `synthesize_stream(text)` is preferred when a backend supports it; `synthesize(text)` remains the local fallback and the event stream marks the mode.

IPC runners stay model-resident. Cancellation is best-effort: the coordinator cancels local async tasks immediately and calls backend `cancel()` when present. The process LLM wrapper exposes request-level cancellation to prevent stale chunks from reaching the UI after interruption.

## Events

New events are additive JSON messages:

- `turn_listening`, `turn_thinking`, `turn_speaking`
- `turn_interrupted`
- `llm_first_chunk`
- `llm_sentence_ready`
- `tts_first_chunk`
- `tts_playback_ack`

Existing `asr_*`, `llm_start`, `llm_chunk`, `llm_done`, `tts_chunk`, and `done` remain.

## Cleanup Policy

Clean only unused fallback code that is not referenced by runtime entrypoints or tests. Keep synchronous coordinator and threaded runners if they are still exported or covered by tests. Prefer marking legacy paths as compatibility code over deleting them when current public imports depend on them.

