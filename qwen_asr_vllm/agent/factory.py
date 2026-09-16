from __future__ import annotations

import threading
from typing import Any, Callable

from qwen_asr_vllm.agent.async_coordinator import AsyncVoiceAgentCoordinator
from qwen_asr_vllm.agent.backends import LlmBackend, TtsBackend
from qwen_asr_vllm.agent.coordinator import VoiceAgentCoordinator
from qwen_asr_vllm.agent.service_runners import (
    MultiplexedThreadedLlmBackend,
    MultiplexedThreadedTtsBackend,
    ThreadedAsrEngine,
)


def build_voice_factory(
    *,
    llm: LlmBackend,
    tts: TtsBackend,
    llm_trigger: str = "final",
    asr_kwargs: dict[str, Any] | None = None,
    async_mode: bool = False,
    tts_concurrency: int = 2,
    tts_flush_chars: int | None = None,
    tts_flush_after_ms: int | None = None,
    tts_flush_min_chars: int = 12,
    tts_coalesce_chars: int | None = None,
    tts_coalesce_wait_ms: int | None = None,
    tts_first_sentence_immediate: bool = False,
    tts_defer_short_segments_chars: int | None = None,
    tts_defer_short_segments_ms: int | None = None,
    tts_stream_first_segment_only: bool = False,
    min_committed_words: int = 1,
    min_committed_audio_seconds: float = 0.0,
    max_committed_audio_seconds: float | None = None,
    defer_tts_audio_until_asr_final: bool = False,
    tts_playback_preroll_ms: float = 0.0,
    barge_in_policy: str = "auto",
    barge_in_rms_threshold: float = 1e-4,
) -> Callable[[Any], VoiceAgentCoordinator]:
    """Return a WebSocket-session factory for `server.create_app`.

    A fresh coordinator is created per connected voice session; the LLM and TTS
    backends may be shared if their implementations are thread-safe.
    """
    shared_llm = llm
    if async_mode and not getattr(llm, "resident_runner", False):
        shared_llm = MultiplexedThreadedLlmBackend(llm)
    shared_tts = (
        MultiplexedThreadedTtsBackend(tts) if async_mode else tts
    )
    asr_runner_lock = threading.Lock()
    asr_runner: ThreadedAsrEngine | None = None
    asr_runner_source_id: int | None = None

    def factory(engine: Any) -> VoiceAgentCoordinator:
        if async_mode:
            nonlocal asr_runner, asr_runner_source_id
            engine_id = id(engine)
            with asr_runner_lock:
                if asr_runner is None or asr_runner_source_id != engine_id:
                    asr_runner = ThreadedAsrEngine(engine)
                    asr_runner_source_id = engine_id
            return AsyncVoiceAgentCoordinator(
                asr_runner,
                shared_llm,
                shared_tts,
                llm_trigger=llm_trigger,
                asr_kwargs=dict(asr_kwargs or {}),
                tts_concurrency=tts_concurrency,
                tts_flush_chars=tts_flush_chars,
                tts_flush_after_ms=tts_flush_after_ms,
                tts_flush_min_chars=tts_flush_min_chars,
                tts_coalesce_chars=tts_coalesce_chars,
                tts_coalesce_wait_ms=tts_coalesce_wait_ms,
                tts_first_sentence_immediate=tts_first_sentence_immediate,
                tts_defer_short_segments_chars=tts_defer_short_segments_chars,
                tts_defer_short_segments_ms=tts_defer_short_segments_ms,
                tts_stream_first_segment_only=tts_stream_first_segment_only,
                min_committed_words=min_committed_words,
                min_committed_audio_seconds=min_committed_audio_seconds,
                max_committed_audio_seconds=max_committed_audio_seconds,
                defer_tts_audio_until_asr_final=defer_tts_audio_until_asr_final,
                tts_playback_preroll_ms=tts_playback_preroll_ms,
                barge_in_policy=barge_in_policy,
                barge_in_rms_threshold=barge_in_rms_threshold,
            )
        return VoiceAgentCoordinator(
            engine,
            llm,
            tts,
            llm_trigger=llm_trigger,
            asr_kwargs=dict(asr_kwargs or {}),
        )

    def close_runners() -> None:
        nonlocal asr_runner
        if asr_runner is not None:
            asr_runner.close()
            asr_runner = None
        if async_mode:
            close_llm = getattr(shared_llm, "close", None)
            if callable(close_llm):
                close_llm()
            close_tts = getattr(shared_tts, "close", None)
            if callable(close_tts):
                close_tts()

    factory.close = close_runners  # type: ignore[attr-defined]
    return factory
