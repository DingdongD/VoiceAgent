from __future__ import annotations

import re
from typing import Any, Iterable
from uuid import uuid4

import numpy as np

from qwen_asr_vllm.agent.backends import LlmBackend, TtsBackend
from qwen_asr_vllm.agent.events import AgentEvent, agent_event

SENTENCE_ENDS = re.compile(r"([。！？.!?\n])")


class VoiceAgentCoordinator:
    """Session-level ASR -> LLM -> TTS streaming coordinator.

    The coordinator is intentionally outside the ASR engine. It owns one streaming
    ASR session and drives injectable LLM/TTS backends, which makes the whole voice
    runtime testable without GPU model dependencies.
    """

    def __init__(
        self,
        asr_engine: Any,
        llm: LlmBackend,
        tts: TtsBackend,
        *,
        llm_trigger: str = "final",
        asr_kwargs: dict[str, Any] | None = None,
    ):
        if llm_trigger not in {"final", "committed"}:
            raise ValueError("llm_trigger must be 'final' or 'committed'")
        self.llm = llm
        self.tts = tts
        self.llm_trigger = llm_trigger
        self._llm_session_id = (
            uuid4().hex if getattr(self.llm, "supports_session_history", False) else None
        )
        if self._llm_session_id is not None:
            reset = getattr(self.llm, "reset", None)
            if callable(reset):
                reset(session_id=self._llm_session_id)
        self.session = asr_engine.open_stream(**(asr_kwargs or {}))
        self._llm_started = False
        self._closed = False

    def feed(self, pcm: np.ndarray) -> list[AgentEvent]:
        if self._closed:
            raise RuntimeError("voice session is closed")
        audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
        return self._handle_asr_events(self.session.feed(audio))

    def close(self, reuse_last: bool = True) -> list[AgentEvent]:
        if self._closed:
            raise RuntimeError("voice session is closed")
        self._closed = True
        try:
            return self._handle_asr_events(self.session.close(reuse_last=reuse_last))
        finally:
            if self._llm_session_id is not None:
                reset = getattr(self.llm, "reset", None)
                if callable(reset):
                    reset(session_id=self._llm_session_id)

    def _handle_asr_events(self, stream_events: Iterable[Any]) -> list[AgentEvent]:
        out: list[AgentEvent] = []
        for event in stream_events:
            out.append(self._asr_event(event))
            trigger_text = self._trigger_text(event)
            if trigger_text:
                out.extend(self._run_llm_and_tts(trigger_text))
        return out

    def _asr_event(self, event: Any) -> AgentEvent:
        return agent_event(
            f"asr_{event.kind}",
            text=event.text,
            committed_text=event.committed_text,
            audio_seconds=event.audio_seconds,
            chunk_index=event.chunk_index,
            commit_violation=event.commit_violation,
        )

    def _trigger_text(self, event: Any) -> str | None:
        if self._llm_started:
            return None
        if event.kind != self.llm_trigger:
            return None
        text = (event.text or event.committed_text or "").strip()
        return text or None

    def _run_llm_and_tts(self, prompt: str) -> list[AgentEvent]:
        self._llm_started = True
        events = [agent_event("llm_start", text=prompt)]
        full_reply = ""
        sentence_buffer = ""

        stream = (
            self.llm.chat_stream(prompt, session_id=self._llm_session_id)
            if self._llm_session_id is not None
            else self.llm.chat_stream(prompt)
        )
        for chunk in stream:
            if not chunk:
                continue
            full_reply += chunk
            sentence_buffer += chunk
            events.append(agent_event("llm_chunk", text=chunk))
            emitted, sentence_buffer = self._emit_ready_tts(sentence_buffer)
            events.extend(emitted)

        remainder = sentence_buffer.strip()
        if remainder:
            audio = self.tts.synthesize(remainder)
            if audio:
                events.append(agent_event("tts_chunk", text=remainder, audio=audio))
        events.append(agent_event("llm_done", text=full_reply))
        events.append(agent_event("done", text=full_reply))
        return events

    def _emit_ready_tts(self, text: str) -> tuple[list[AgentEvent], str]:
        events: list[AgentEvent] = []
        rest = text
        while True:
            match = SENTENCE_ENDS.search(rest)
            if match is None:
                return events, rest
            end = match.end()
            sentence = rest[:end].strip()
            rest = rest[end:]
            if not sentence:
                continue
            audio = self.tts.synthesize(sentence)
            if audio:
                events.append(agent_event("tts_chunk", text=sentence, audio=audio))
