"""Growing-prefix and incremental streaming sessions.

Policies:

* ``retranscribe`` — each feed re-runs the full buffer.
* ``speculate`` — same, but previous transcript tokens draft the residual decode.
* ``incremental`` — transcribe only the unlocked audio tail; periodically
  re-transcribe a sliding window for bounded revision (see incremental design spec).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qwen_asr_vllm.engine.async_engine import AsyncAsrEngine, RequestCancelled, RequestHandle
from qwen_asr_vllm.engine.engine import AsrOutput
from qwen_asr_vllm.engine.request import SamplingParams

SAMPLE_RATE = 16000
MIN_SAMPLES = 160  # matches engine frontend lower bound for a meaningful clip
POLICIES = frozenset({"retranscribe", "speculate", "incremental"})


@dataclass
class StreamEvent:
    kind: str  # "partial" | "committed" | "final"
    text: str
    committed_text: str
    audio_seconds: float
    output_token_ids: list[int] = field(default_factory=list)
    chunk_index: int = 0
    commit_violation: bool = False


def split_words(text: str) -> list[str]:
    return str(text).split()


def join_words(words: list[str]) -> str:
    return " ".join(words)


def join_committed_tail(committed: str, tail: str) -> str:
    if not committed:
        return tail
    if not tail:
        return committed
    return f"{committed} {tail}"


def commit_candidate(partial_text: str, lag_words: int) -> str:
    """Drop the trailing ``lag_words`` words from a partial transcript."""
    words = split_words(partial_text)
    if lag_words <= 0:
        return join_words(words)
    if lag_words >= len(words):
        return ""
    return join_words(words[:-lag_words])


def advance_committed(previous: str, candidate: str) -> tuple[str, bool]:
    """Extend the committed prefix, or keep it and flag a violation.

    Returns ``(committed_text, violation)``. A violation means ``candidate`` is
    neither equal to ``previous`` nor a word-list extension of it; history is not
    rewritten.
    """
    prev_words = split_words(previous)
    cand_words = split_words(candidate)
    if cand_words == prev_words:
        return previous, False
    if len(cand_words) >= len(prev_words) and cand_words[: len(prev_words)] == prev_words:
        return join_words(cand_words), False
    return previous, True


class StreamingSession:
    """One streaming transcription session.

    Not safe for concurrent ``feed`` from multiple threads; one caller drives it.
    """

    def __init__(
        self,
        engine: AsyncAsrEngine,
        *,
        language: str | None = None,
        context: str = "",
        commit_lag_words: int = 0,
        chunk_policy: str = "speculate",
        sample_rate: int = SAMPLE_RATE,
        recompute_seconds: float = 6.0,
        recompute_overlap_seconds: float = 2.0,
        tail_min_seconds: float = 0.5,
        on_violation: str = "keep",
    ):
        if chunk_policy not in POLICIES:
            raise ValueError(
                f"chunk_policy={chunk_policy!r}; expected one of {sorted(POLICIES)}"
            )
        if on_violation != "keep":
            raise ValueError('on_violation only supports "keep" in v1')
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"streaming session requires {SAMPLE_RATE} Hz PCM, got {sample_rate}")

        self._engine = engine
        self.language = language
        self.context = context
        self.commit_lag_words = commit_lag_words
        self.chunk_policy = chunk_policy
        self.sample_rate = sample_rate
        self.recompute_seconds = recompute_seconds
        self.recompute_overlap_seconds = recompute_overlap_seconds
        self.tail_min_seconds = tail_min_seconds
        self.on_violation = on_violation

        self.last_draft_tokens = 0
        self.last_draft_accepted = 0
        self.total_draft_tokens = 0
        self.total_draft_accepted = 0
        # Sum of waveform samples actually sent to the engine (for incremental ROI).
        self.total_transcribed_samples = 0
        self.window_recomputes = 0

        self._buffer = np.zeros(0, dtype=np.float32)
        self._chunk_index = -1
        self._committed_text = ""
        self._tail_text = ""
        self._last_partial_text = ""
        self._last_token_ids: list[int] = []
        self._last_audio_seconds = 0.0
        self._in_flight: RequestHandle | None = None
        self._closed = False
        self._committed_audio_end = 0
        self._last_recompute_at = 0

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def committed_text(self) -> str:
        return self._committed_text

    @property
    def audio_seconds(self) -> float:
        return float(self._buffer.size) / self.sample_rate

    @property
    def total_transcribed_seconds(self) -> float:
        return self.total_transcribed_samples / self.sample_rate

    def feed(self, pcm: np.ndarray) -> list[StreamEvent]:
        """Append PCM and emit events for this chunk."""
        if self._closed:
            raise RuntimeError("streaming session is closed")
        if pcm is None or pcm.size == 0:
            return []

        audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
        self._buffer = np.concatenate([self._buffer, audio]) if self._buffer.size else audio
        if self._buffer.size < MIN_SAMPLES:
            return []

        if self.chunk_policy == "incremental":
            return self._feed_incremental()

        self._cancel_in_flight()
        self._chunk_index += 1
        output = self._submit_waveform(self._buffer)
        self.total_transcribed_samples += int(self._buffer.size)
        return self._events_from_output(output, kind_partial=True)

    def close(self, reuse_last: bool = True) -> list[StreamEvent]:
        """Finish the session and emit a single ``final`` event."""
        if self._closed:
            raise RuntimeError("streaming session is closed")
        self._cancel_in_flight()

        events: list[StreamEvent] = []
        if self._buffer.size >= MIN_SAMPLES:
            if reuse_last and self._last_partial_text and self._chunk_index >= 0:
                events.append(
                    StreamEvent(
                        kind="final",
                        text=self._last_partial_text,
                        committed_text=self._committed_text,
                        audio_seconds=self._last_audio_seconds,
                        output_token_ids=list(self._last_token_ids),
                        chunk_index=self._chunk_index,
                    )
                )
            else:
                self._chunk_index += 1
                if self.chunk_policy == "incremental":
                    output = self._submit_waveform(self._buffer)
                    self.total_transcribed_samples += int(self._buffer.size)
                    partial = output.text
                    self._last_partial_text = partial
                    self._last_token_ids = list(output.output_token_ids)
                    self._last_audio_seconds = self.audio_seconds
                    events.extend(
                        self._emit_commit_events(
                            partial, output.output_token_ids, kind_partial=False
                        )
                    )
                else:
                    output = self._submit_waveform(self._buffer)
                    self.total_transcribed_samples += int(self._buffer.size)
                    events.extend(self._events_from_output(output, kind_partial=False))
                events.append(
                    StreamEvent(
                        kind="final",
                        text=self._last_partial_text,
                        committed_text=self._committed_text,
                        audio_seconds=self._last_audio_seconds,
                        output_token_ids=list(self._last_token_ids),
                        chunk_index=self._chunk_index,
                    )
                )
        self._closed = True
        self._buffer = np.zeros(0, dtype=np.float32)
        return events

    def _feed_incremental(self) -> list[StreamEvent]:
        sr = self.sample_rate
        tail_min = max(MIN_SAMPLES, int(self.tail_min_seconds * sr))
        new_samples = len(self._buffer) - self._committed_audio_end
        if new_samples < tail_min and self._chunk_index >= 0:
            return []

        self._cancel_in_flight()
        self._chunk_index += 1

        recompute_every = int(self.recompute_seconds * sr)
        due = (len(self._buffer) - self._last_recompute_at) >= recompute_every
        # Always recompute once we have a full window's worth of audio the first time.
        first_window = self._last_recompute_at == 0 and len(self._buffer) >= recompute_every

        violation = False
        if due or first_window:
            overlap = int(self.recompute_overlap_seconds * sr)
            start = max(0, self._committed_audio_end - overlap)
            waveform = self._buffer[start:]
            output = self._submit_waveform(waveform)
            self.total_transcribed_samples += int(waveform.size)
            self._last_recompute_at = len(self._buffer)
            self.window_recomputes += 1

            window_seconds = max(waveform.size / sr, 1e-6)
            overlap_seconds = max(0.0, (self._committed_audio_end - start) / sr)
            w_words = split_words(output.text)
            n_overlap = int(len(w_words) * (overlap_seconds / window_seconds)) if w_words else 0
            n_overlap = min(n_overlap, len(w_words))
            if n_overlap > 0 and self._committed_text:
                window_prefix = join_words(w_words[:n_overlap])
                committed_suffix = join_words(split_words(self._committed_text)[-n_overlap:])
                if window_prefix != committed_suffix:
                    violation = True
            self._tail_text = join_words(w_words[n_overlap:])
            token_ids = list(output.output_token_ids)
        else:
            waveform = self._buffer[self._committed_audio_end :]
            if waveform.size < MIN_SAMPLES:
                self._chunk_index -= 1
                return []
            output = self._submit_waveform(waveform)
            self.total_transcribed_samples += int(waveform.size)
            self._tail_text = output.text
            token_ids = list(output.output_token_ids)

        partial = join_committed_tail(self._committed_text, self._tail_text)
        self._last_partial_text = partial
        self._last_token_ids = token_ids
        self._last_audio_seconds = self.audio_seconds
        return self._emit_commit_events(
            partial, token_ids, kind_partial=True, forced_violation=violation
        )

    def _cancel_in_flight(self) -> None:
        handle = self._in_flight
        self._in_flight = None
        if handle is None or handle.done():
            return
        handle.cancel()
        try:
            handle.result(timeout=30.0)
        except RequestCancelled:
            pass
        except Exception as exc:  # noqa: BLE001 - must not strand the session
            import logging

            logging.getLogger(__name__).debug(
                "ignoring error while draining cancelled stream request: %s", exc
            )

    def _submit_waveform(self, waveform: np.ndarray) -> AsrOutput:
        seconds = max(len(waveform) / self.sample_rate, 1e-3)
        sampling = SamplingParams.for_audio(seconds)
        use_draft = (
            self.chunk_policy == "speculate"
            and bool(self._last_token_ids)
            and self._chunk_index > 0
        )
        if use_draft:
            output = self._engine.transcribe_with_draft(
                waveform,
                draft_token_ids=self._last_token_ids,
                sample_rate=self.sample_rate,
                context=self.context,
                language=self.language,
                sampling=sampling,
            )
            self.last_draft_tokens = output.draft_tokens
            self.last_draft_accepted = output.draft_accepted
            self.total_draft_tokens += output.draft_tokens
            self.total_draft_accepted += output.draft_accepted
            return output

        handle = self._engine.submit(
            waveform,
            sample_rate=self.sample_rate,
            context=self.context,
            language=self.language,
            sampling=sampling,
        )
        self._in_flight = handle
        try:
            output = handle.result()
        finally:
            if self._in_flight is handle:
                self._in_flight = None
        assert isinstance(output, AsrOutput)
        self.last_draft_tokens = 0
        self.last_draft_accepted = 0
        return output

    def _sync_committed_audio_end(self, partial_text: str) -> None:
        """Approximate sample index covered by committed words (proportional)."""
        p_words = split_words(partial_text)
        c_words = split_words(self._committed_text)
        if not p_words:
            return
        unlocked = min(self.commit_lag_words, len(p_words))
        max_committed_words = len(p_words) - unlocked
        n = min(len(c_words), max_committed_words)
        ratio = n / len(p_words)
        new_end = int(len(self._buffer) * ratio)
        floor = max(0, len(self._buffer) - max(MIN_SAMPLES, int(self.tail_min_seconds * self.sample_rate)))
        new_end = min(new_end, floor)
        self._committed_audio_end = max(self._committed_audio_end, new_end)

    def _emit_commit_events(
        self,
        partial_text: str,
        token_ids: list[int],
        *,
        kind_partial: bool,
        forced_violation: bool = False,
    ) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        violation = forced_violation
        if kind_partial:
            events.append(
                StreamEvent(
                    kind="partial",
                    text=partial_text,
                    committed_text=self._committed_text,
                    audio_seconds=self._last_audio_seconds,
                    output_token_ids=list(token_ids),
                    chunk_index=self._chunk_index,
                    commit_violation=violation,
                )
            )

        if self.commit_lag_words > 0:
            candidate = commit_candidate(partial_text, self.commit_lag_words)
            previous = self._committed_text
            new_committed, adv_violation = advance_committed(previous, candidate)
            violation = violation or adv_violation
            changed = new_committed != previous
            if changed and not adv_violation:
                self._committed_text = new_committed
                self._sync_committed_audio_end(partial_text)
            if changed or violation:
                events.append(
                    StreamEvent(
                        kind="committed",
                        text=self._committed_text,
                        committed_text=self._committed_text,
                        audio_seconds=self._last_audio_seconds,
                        chunk_index=self._chunk_index,
                        commit_violation=violation,
                    )
                )
            if kind_partial and events:
                events[0].committed_text = self._committed_text
                events[0].commit_violation = violation
        return events

    def _events_from_output(self, output, *, kind_partial: bool) -> list[StreamEvent]:
        self._last_partial_text = output.text
        self._last_token_ids = list(output.output_token_ids)
        self._last_audio_seconds = output.audio_seconds
        return self._emit_commit_events(
            output.text, output.output_token_ids, kind_partial=kind_partial
        )
