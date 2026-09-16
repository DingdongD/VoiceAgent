from __future__ import annotations

import asyncio
import io
import re
import wave
from contextlib import suppress
from typing import Any, Iterable
from uuid import uuid4

import numpy as np

from qwen_asr_vllm.agent.backends import LlmBackend, TtsBackend
from qwen_asr_vllm.agent.events import AgentEvent, agent_event
from qwen_asr_vllm.agent.text_segmenter import TextSegmenter
from qwen_asr_vllm.agent.turn import VoiceTurnStateMachine


def _wav_duration_ms(audio: bytes) -> float | None:
    """Duration of a WAV chunk, or None when the bytes are not readable as WAV.

    TTS backends in tests emit opaque markers rather than WAV, and a malformed
    chunk must not be mistaken for silence, so callers treat None as unknown.
    """
    try:
        with wave.open(io.BytesIO(audio), "rb") as handle:
            rate = handle.getframerate()
            if not rate:
                return None
            return handle.getnframes() / rate * 1000.0
    except (EOFError, wave.Error, OSError):
        return None

# How incoming user audio may cancel a reply that is already being spoken.
#
# ``auto``            any frame above the RMS threshold interrupts.
# ``after-asr-final`` energy barge-in waits until the utterance that triggered
#                     the reply has been finalized. Needed once first audio is
#                     fast enough to start before the user stops speaking,
#                     otherwise the rest of the same utterance cancels the turn.
# ``explicit-only``   energy barge-in is disabled; only ``interrupt()`` cancels.
BARGE_IN_POLICIES = frozenset({"auto", "after-asr-final", "explicit-only"})


class AsyncVoiceAgentCoordinator:
    """Async ASR -> LLM -> TTS pipeline with local full-duplex control."""

    def __init__(
        self,
        asr_engine: Any,
        llm: LlmBackend,
        tts: TtsBackend,
        *,
        llm_trigger: str = "final",
        asr_kwargs: dict[str, Any] | None = None,
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
        reset_llm_on_start: bool = True,
        barge_in_policy: str = "auto",
        barge_in_rms_threshold: float = 1e-4,
    ):
        if llm_trigger not in {"final", "committed"}:
            raise ValueError("llm_trigger must be 'final' or 'committed'")
        if barge_in_policy not in BARGE_IN_POLICIES:
            raise ValueError(
                f"barge_in_policy={barge_in_policy!r}; expected one of "
                f"{sorted(BARGE_IN_POLICIES)}"
            )
        self.llm = llm
        self.tts = tts
        self.llm_trigger = llm_trigger
        self._llm_session_id = (
            uuid4().hex if getattr(self.llm, "supports_session_history", False) else None
        )
        self._llm_session_released = False
        reset_llm = getattr(self.llm, "reset", None)
        if reset_llm_on_start and callable(reset_llm):
            if self._llm_session_id is None:
                reset_llm()
            else:
                reset_llm(session_id=self._llm_session_id)
        self.session = asr_engine.open_stream(**(asr_kwargs or {}))
        self._events: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self._llm_started = False
        self._closed = False
        self._llm_task: asyncio.Task | None = None
        self._tts_tasks: list[asyncio.Task] = []
        self._tts_sem = asyncio.Semaphore(max(1, int(tts_concurrency)))
        self.tts_flush_chars = (
            max(1, int(tts_flush_chars)) if tts_flush_chars else None
        )
        self.tts_flush_after_seconds = (
            max(1, int(tts_flush_after_ms)) / 1000 if tts_flush_after_ms else None
        )
        self.tts_flush_min_chars = max(1, int(tts_flush_min_chars))
        self.tts_coalesce_chars = (
            max(1, int(tts_coalesce_chars)) if tts_coalesce_chars else None
        )
        self.tts_coalesce_wait_seconds = (
            max(0, int(tts_coalesce_wait_ms)) / 1000
            if tts_coalesce_wait_ms is not None
            else None
        )
        self.tts_first_sentence_immediate = bool(tts_first_sentence_immediate)
        self.tts_defer_short_segments_chars = (
            max(1, int(tts_defer_short_segments_chars))
            if tts_defer_short_segments_chars
            else None
        )
        self.tts_defer_short_segments_seconds = (
            max(1, int(tts_defer_short_segments_ms)) / 1000
            if tts_defer_short_segments_ms
            else None
        )
        self.tts_stream_first_segment_only = bool(tts_stream_first_segment_only)
        self.min_committed_words = max(1, int(min_committed_words))
        self.min_committed_audio_seconds = max(0.0, float(min_committed_audio_seconds))
        self.max_committed_audio_seconds = (
            max(0.0, float(max_committed_audio_seconds))
            if max_committed_audio_seconds is not None
            else None
        )
        self.defer_tts_audio_until_asr_final = bool(defer_tts_audio_until_asr_final)
        # Playback that starts on the first chunk starves whenever TTS generates
        # slower than real time. Holding the opening chunks until this much audio
        # exists trades a later start for a reply without gaps.
        self.tts_playback_preroll_ms = float(tts_playback_preroll_ms)
        self._tts_queue: asyncio.Queue[tuple[int, str] | None] | None = None
        self._tts_worker_task: asyncio.Task | None = None
        self._asr_lock = asyncio.Lock()
        self._turns = VoiceTurnStateMachine()
        self._generation_id = 0
        self._cancelled_generations: set[int] = set()
        self._first_llm_chunk_seen: set[int] = set()
        self._first_tts_chunk_seen: set[int] = set()
        self._first_tts_audio_ready_seen: set[int] = set()
        self._streamed_tts_generation_seen: set[int] = set()
        self._asr_final_seen = False
        self._asr_final_event: asyncio.Event | None = None
        self._deferred_tts_audio: list[tuple[int, str, bytes, bool]] = []
        self._preroll_audio: list[tuple[int, str, bytes, bool]] = []
        self._preroll_buffered_ms = 0.0
        self._preroll_open: set[int] = set()
        self._immediate_tts_generation_seen: set[int] = set()
        self.barge_in_policy = barge_in_policy
        self._speech_rms_threshold = max(0.0, float(barge_in_rms_threshold))
        self.barge_in_suppressed = 0

    async def feed(self, pcm: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("voice session is closed")
        audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if self._has_user_speech(audio):
            # Only the interrupting transition is gated. Other transitions, such
            # as idle -> listening, must still run under every policy.
            if self._turns.state == "speaking" and not self._energy_barge_in_allowed():
                self.barge_in_suppressed += 1
            else:
                turn_events = self._turns.on_user_audio()
                if any(event.type == "turn_interrupted" for event in turn_events):
                    await self._cancel_active_generation()
                for event in turn_events:
                    await self._emit(event)
        async with self._asr_lock:
            events = await asyncio.to_thread(self.session.feed, audio)
        await self._handle_asr_events(events)

    async def close(self, reuse_last: bool = True) -> None:
        if self._closed:
            raise RuntimeError("voice session is closed")
        self._closed = True
        try:
            async with self._asr_lock:
                events = await asyncio.to_thread(
                    self.session.close, reuse_last=reuse_last
                )
            await self._handle_asr_events(events)
            if self._llm_task is not None:
                with suppress(asyncio.CancelledError):
                    await self._llm_task
        finally:
            await self._release_llm_session()

    async def interrupt(self) -> None:
        turn_events = self._turns.on_user_audio()
        if not any(event.type == "turn_interrupted" for event in turn_events):
            turn_events.insert(
                0, agent_event("turn_interrupted", from_state=self._turns.state)
            )
        await self._cancel_active_generation()
        for event in turn_events:
            await self._emit(event)
        await self._emit(agent_event("cancelled", stage="agent"))

    async def playback_ack(self, payload: dict[str, Any] | None = None) -> None:
        await self._emit(agent_event("tts_playback_ack", payload=payload or {}))

    async def next_event(self, timeout: float | None = None) -> AgentEvent:
        if timeout is None:
            return await self._events.get()
        return await asyncio.wait_for(self._events.get(), timeout=timeout)

    async def stop(self) -> None:
        await self._cancel_active_generation()
        await self._release_llm_session()

    async def _cancel_active_generation(self) -> None:
        self._generation_id += 1
        self._cancelled_generations.add(self._generation_id - 1)
        self._llm_started = False
        cancel = getattr(self.llm, "cancel", None)
        if callable(cancel) and not getattr(self.llm, "resident_runner", False):
            await asyncio.to_thread(cancel, None)
        if self._llm_task is not None and not self._llm_task.done():
            self._llm_task.cancel()
        for task in self._tts_tasks:
            if not task.done():
                task.cancel()
        self._tts_tasks.clear()
        if self._tts_worker_task is not None and not self._tts_worker_task.done():
            self._tts_worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._tts_worker_task
        self._tts_worker_task = None
        self._tts_queue = None
        self._deferred_tts_audio.clear()
        self._preroll_audio.clear()
        self._preroll_buffered_ms = 0.0

    async def _handle_asr_events(self, stream_events: Iterable[Any]) -> None:
        for event in stream_events:
            await self._emit(self._asr_event(event))
            if event.kind == "final":
                self._asr_final_seen = True
                if self._asr_final_event is not None:
                    self._asr_final_event.set()
                await self._release_deferred_tts_audio()
            trigger_text = self._trigger_text(event)
            if trigger_text:
                self._llm_started = True
                self._generation_id += 1
                generation_id = self._generation_id
                self._llm_task = asyncio.create_task(
                    self._run_llm_and_tts(trigger_text, generation_id)
                )

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
        if event.kind == "committed" and not self._committed_trigger_ready(text, event):
            return None
        return text or None

    def _committed_trigger_ready(self, text: str, event: Any) -> bool:
        audio_seconds = float(getattr(event, "audio_seconds", 0.0) or 0.0)
        if (
            self.max_committed_audio_seconds is not None
            and audio_seconds >= self.max_committed_audio_seconds
        ):
            return True
        if audio_seconds < self.min_committed_audio_seconds:
            return False
        return len(text.split()) >= self.min_committed_words

    async def _run_llm_and_tts(self, prompt: str, generation_id: int) -> None:
        for event in self._turns.on_agent_start():
            await self._emit(event)
        await self._emit(agent_event("llm_start", text=prompt))

        chunk_queue: asyncio.Queue[Any] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        sentinel = object()

        def consume_llm() -> None:
            try:
                stream = (
                    self.llm.chat_stream(prompt, session_id=self._llm_session_id)
                    if self._llm_session_id is not None
                    else self.llm.chat_stream(prompt)
                )
                for chunk in stream:
                    if self._is_cancelled(generation_id):
                        break
                    if chunk:
                        loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
            except BaseException as exc:  # noqa: BLE001 - delivered through event stream
                if not self._is_cancelled(generation_id):
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, exc)
            finally:
                if not self._is_cancelled(generation_id):
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, sentinel)

        worker = loop.run_in_executor(None, consume_llm)
        full_reply = ""
        segmenter = TextSegmenter(min_chars=1, flush_chars=self.tts_flush_chars)
        flush_deadline: float | None = None
        pending_tts_fragment: str | None = None
        pending_tts_deadline: float | None = None

        async def submit_tts_fragment(fragment: str, *, force: bool = False) -> None:
            nonlocal pending_tts_fragment, pending_tts_deadline
            text = self._clean_tts_text(fragment)
            if not text:
                return
            if pending_tts_fragment:
                text = self._merge_tts_text(pending_tts_fragment, text)
                pending_tts_fragment = None
                pending_tts_deadline = None
            if self._should_defer_tts_fragment(text, force=force):
                pending_tts_fragment = text
                if pending_tts_deadline is None:
                    pending_tts_deadline = self._new_tts_defer_deadline()
                return
            await self._emit(agent_event("llm_sentence_ready", text=text))
            await self._submit_tts(text, generation_id)

        async def flush_pending_tts_fragment() -> None:
            nonlocal pending_tts_fragment, pending_tts_deadline
            if not pending_tts_fragment:
                return
            text = pending_tts_fragment
            pending_tts_fragment = None
            pending_tts_deadline = None
            await self._emit(agent_event("llm_sentence_ready", text=text))
            await self._submit_tts(text, generation_id)

        try:
            while True:
                if self._tts_defer_deadline_expired(pending_tts_deadline):
                    await flush_pending_tts_fragment()
                    continue
                if self._flush_deadline_expired(segmenter, flush_deadline):
                    fragment = segmenter.flush_if_at_least(self.tts_flush_min_chars)
                    if fragment:
                        await submit_tts_fragment(fragment, force=True)
                    flush_deadline = self._next_flush_deadline(segmenter)
                    continue
                timeout = self._next_llm_wait_timeout(
                    segmenter,
                    flush_deadline,
                    pending_tts_deadline,
                )
                try:
                    item = await asyncio.wait_for(chunk_queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    if self._tts_defer_deadline_expired(pending_tts_deadline):
                        await flush_pending_tts_fragment()
                        continue
                    fragment = segmenter.flush_if_at_least(self.tts_flush_min_chars)
                    if fragment:
                        await submit_tts_fragment(fragment, force=True)
                    flush_deadline = self._next_flush_deadline(segmenter)
                    continue
                if item is sentinel:
                    break
                if self._is_cancelled(generation_id):
                    return
                if isinstance(item, BaseException):
                    await self._emit(agent_event("error", stage="llm", text=str(item)))
                    break
                chunk = str(item)
                full_reply += chunk
                if generation_id not in self._first_llm_chunk_seen:
                    self._first_llm_chunk_seen.add(generation_id)
                    await self._emit(agent_event("llm_first_chunk", text=chunk))
                await self._emit(agent_event("llm_chunk", text=chunk))
                for sentence in segmenter.add(chunk):
                    await submit_tts_fragment(sentence)
                flush_deadline = self._next_flush_deadline(segmenter, flush_deadline)

            await worker
            if self._is_cancelled(generation_id):
                return
            remainder = segmenter.flush()
            if remainder:
                await submit_tts_fragment(remainder, force=True)
            await flush_pending_tts_fragment()
            await self._emit(agent_event("llm_done", text=full_reply))
            await self._wait_for_tts()
            if self._is_cancelled(generation_id):
                return
            await self._wait_for_asr_final_if_deferred(generation_id)
            await self._release_deferred_tts_audio()
            # A reply shorter than the preroll target never trips the gate, so it
            # opens here rather than leaving the audio unspoken.
            await self._release_preroll_audio()
            if self._is_cancelled(generation_id):
                return
            await self._emit(agent_event("done", text=full_reply))
            for event in self._turns.on_agent_done():
                await self._emit(event)
        except asyncio.CancelledError:
            self._cancelled_generations.add(generation_id)
            raise

    async def _release_llm_session(self) -> None:
        if self._llm_session_id is None or self._llm_session_released:
            return
        self._llm_session_released = True
        reset = getattr(self.llm, "reset", None)
        if callable(reset):
            await asyncio.to_thread(reset, session_id=self._llm_session_id)

    def _flush_deadline_expired(
        self, segmenter: TextSegmenter, flush_deadline: float | None
    ) -> bool:
        if flush_deadline is None or not segmenter.has_pending():
            return False
        return asyncio.get_running_loop().time() >= flush_deadline

    def _chunk_wait_timeout(
        self, segmenter: TextSegmenter, flush_deadline: float | None
    ) -> float | None:
        if self.tts_flush_after_seconds is None or not segmenter.has_pending():
            return None
        if flush_deadline is None:
            return self.tts_flush_after_seconds
        return max(0.0, flush_deadline - asyncio.get_running_loop().time())

    def _next_llm_wait_timeout(
        self,
        segmenter: TextSegmenter,
        flush_deadline: float | None,
        tts_defer_deadline: float | None,
    ) -> float | None:
        timeouts: list[float] = []
        chunk_timeout = self._chunk_wait_timeout(segmenter, flush_deadline)
        if chunk_timeout is not None:
            timeouts.append(chunk_timeout)
        if tts_defer_deadline is not None:
            timeouts.append(
                max(0.0, tts_defer_deadline - asyncio.get_running_loop().time())
            )
        if not timeouts:
            return None
        return min(timeouts)

    def _next_flush_deadline(
        self, segmenter: TextSegmenter, previous: float | None = None
    ) -> float | None:
        if self.tts_flush_after_seconds is None or not segmenter.has_pending():
            return None
        return previous or asyncio.get_running_loop().time() + self.tts_flush_after_seconds

    def _should_defer_tts_fragment(self, text: str, *, force: bool) -> bool:
        if force or self.tts_defer_short_segments_chars is None:
            return False
        return len(text) < self.tts_defer_short_segments_chars

    def _new_tts_defer_deadline(self) -> float | None:
        if self.tts_defer_short_segments_seconds is None:
            return None
        return asyncio.get_running_loop().time() + self.tts_defer_short_segments_seconds

    def _tts_defer_deadline_expired(self, deadline: float | None) -> bool:
        if deadline is None:
            return False
        return asyncio.get_running_loop().time() >= deadline

    def _merge_tts_text(self, left: str, right: str) -> str:
        return (left.rstrip() + " " + right.lstrip()).strip()

    async def _submit_tts(self, text: str, generation_id: int) -> None:
        text = self._clean_tts_text(text)
        if not text:
            return
        if self.tts_coalesce_chars is None:
            self._tts_tasks.append(
                asyncio.create_task(self._synthesize(text, generation_id))
            )
            return
        if self._tts_queue is None:
            self._tts_queue = asyncio.Queue()
        if self._tts_worker_task is None:
            self._tts_worker_task = asyncio.create_task(self._run_tts_worker())
        await self._tts_queue.put((generation_id, text))

    def _clean_tts_text(self, text: str) -> str:
        cleaned = str(text).replace("\ufffd", "").strip()
        cleaned = re.sub(r"\s+([,.!?;:。！？])", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    async def _wait_for_tts(self) -> None:
        if self.tts_coalesce_chars is None:
            if self._tts_tasks:
                await asyncio.gather(*self._tts_tasks, return_exceptions=True)
            return
        if self._tts_worker_task is None:
            return
        assert self._tts_queue is not None
        await self._tts_queue.put(None)
        with suppress(asyncio.CancelledError):
            await self._tts_worker_task

    async def _run_tts_worker(self) -> None:
        assert self._tts_queue is not None
        pending: tuple[int, str] | None = None
        closing = False
        while True:
            if pending is None:
                item = await self._tts_queue.get()
            else:
                item = pending
                pending = None
            if item is None:
                return

            generation_id, text = item
            first_sentence_immediate = (
                self.tts_first_sentence_immediate
                and generation_id not in self._immediate_tts_generation_seen
            )
            if first_sentence_immediate:
                self._immediate_tts_generation_seen.add(generation_id)
            elif self.tts_coalesce_wait_seconds is not None:
                deadline = (
                    asyncio.get_running_loop().time()
                    + self.tts_coalesce_wait_seconds
                )
                while pending is None and not closing:
                    timeout = deadline - asyncio.get_running_loop().time()
                    if timeout <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(
                            self._tts_queue.get(),
                            timeout=timeout,
                        )
                    except asyncio.TimeoutError:
                        break
                    pending, closing, text = self._merge_tts_queue_item(
                        item, generation_id, text, pending, closing
                    )
            while pending is None and not closing:
                try:
                    item = self._tts_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                pending, closing, text = self._merge_tts_queue_item(
                    item, generation_id, text, pending, closing
                )
                if closing or pending is not None:
                    break

            await self._synthesize(text, generation_id)
            if closing and pending is None:
                return

    def _merge_tts_queue_item(
        self,
        item: tuple[int, str] | None,
        generation_id: int,
        text: str,
        pending: tuple[int, str] | None,
        closing: bool,
    ) -> tuple[tuple[int, str] | None, bool, str]:
        if item is None:
            return pending, True, text
        next_generation_id, next_text = item
        if next_generation_id != generation_id:
            return item, closing, text
        merged = (text.rstrip() + " " + next_text.lstrip()).strip()
        if len(merged) <= int(self.tts_coalesce_chars):
            return pending, closing, merged
        return item, closing, text

    async def _synthesize(self, text: str, generation_id: int) -> None:
        if self._is_cancelled(generation_id):
            return
        synthesize_stream = getattr(self.tts, "synthesize_stream", None)
        supports_streaming = getattr(
            self.tts, "supports_streaming_tts", callable(synthesize_stream)
        )
        if supports_streaming and callable(synthesize_stream):
            if self.tts_stream_first_segment_only:
                if generation_id in self._streamed_tts_generation_seen:
                    supports_streaming = False
                else:
                    self._streamed_tts_generation_seen.add(generation_id)
        if supports_streaming and callable(synthesize_stream):
            await self._synthesize_streaming(text, generation_id, synthesize_stream)
            return
        async with self._tts_sem:
            audio = await asyncio.to_thread(self.tts.synthesize, text)
        if audio and not self._is_cancelled(generation_id):
            await self._emit_tts_audio(text, audio, generation_id, streaming=False)

    async def _synthesize_streaming(self, text: str, generation_id: int, synthesize_stream) -> None:
        sentinel = object()

        def next_item(iterator):
            return next(iterator, sentinel)

        async with self._tts_sem:
            iterator = iter(synthesize_stream(text))
            while not self._is_cancelled(generation_id):
                audio = await asyncio.to_thread(next_item, iterator)
                if audio is sentinel:
                    return
                if audio:
                    await self._emit_tts_audio(
                        text, audio, generation_id, streaming=True
                    )

    async def _emit_tts_audio(
        self, text: str, audio: bytes, generation_id: int, *, streaming: bool
    ) -> None:
        if generation_id not in self._first_tts_audio_ready_seen:
            self._first_tts_audio_ready_seen.add(generation_id)
            await self._emit(agent_event("tts_audio_ready", text=text))
        if self.defer_tts_audio_until_asr_final and not self._asr_final_seen:
            self._deferred_tts_audio.append((generation_id, text, audio, streaming))
            return
        if self._preroll_holds(audio, generation_id):
            self._preroll_audio.append((generation_id, text, audio, streaming))
            return
        # The gate has opened, so anything held must go out ahead of this chunk.
        await self._release_preroll_audio()
        await self._emit_tts_audio_now(text, audio, generation_id, streaming=streaming)

    def _preroll_holds(self, audio: bytes, generation_id: int) -> bool:
        """Hold this chunk if playback should not start yet.

        Audio whose duration cannot be read is never held: the gate exists to
        guarantee a buffer, and it cannot reason about a chunk it cannot measure.
        """
        if self.tts_playback_preroll_ms <= 0 or generation_id in self._preroll_open:
            return False
        duration = _wav_duration_ms(audio)
        if duration is None:
            self._preroll_open.add(generation_id)
            return False
        self._preroll_buffered_ms += duration
        if self._preroll_buffered_ms >= self.tts_playback_preroll_ms:
            self._preroll_open.add(generation_id)
            return False
        return True

    async def _release_preroll_audio(self) -> None:
        """Open the gate and flush, for a reply that never reached the target."""
        while self._preroll_audio:
            generation_id, text, audio, streaming = self._preroll_audio.pop(0)
            self._preroll_open.add(generation_id)
            if not self._is_cancelled(generation_id):
                await self._emit_tts_audio_now(
                    text, audio, generation_id, streaming=streaming
                )

    async def _emit_tts_audio_now(
        self, text: str, audio: bytes, generation_id: int, *, streaming: bool
    ) -> None:
        if generation_id not in self._first_tts_chunk_seen:
            self._first_tts_chunk_seen.add(generation_id)
            await self._emit(agent_event("tts_first_chunk", text=text))
            for event in self._turns.on_agent_audio():
                await self._emit(event)
        await self._emit(
            agent_event(
                "tts_chunk",
                text=text,
                audio=audio,
                tts_streaming=streaming,
            )
        )

    async def _release_deferred_tts_audio(self) -> None:
        while self._deferred_tts_audio:
            generation_id, text, audio, streaming = self._deferred_tts_audio.pop(0)
            if not self._is_cancelled(generation_id):
                await self._emit_tts_audio_now(
                    text, audio, generation_id, streaming=streaming
                )

    async def _wait_for_asr_final_if_deferred(self, generation_id: int) -> None:
        if not self.defer_tts_audio_until_asr_final or self._asr_final_seen:
            return
        if self._is_cancelled(generation_id):
            return
        if self._asr_final_event is None:
            self._asr_final_event = asyncio.Event()
        await self._asr_final_event.wait()

    def _is_cancelled(self, generation_id: int) -> bool:
        return generation_id in self._cancelled_generations

    def _has_user_speech(self, audio: np.ndarray) -> bool:
        if audio.size == 0:
            return False
        return float(np.sqrt(np.mean(np.square(audio)))) > self._speech_rms_threshold

    def _energy_barge_in_allowed(self) -> bool:
        if self.barge_in_policy == "auto":
            return True
        if self.barge_in_policy == "explicit-only":
            return False
        return self._asr_final_seen

    async def _emit(self, event: AgentEvent) -> None:
        await self._events.put(event)
