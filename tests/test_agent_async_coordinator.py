import asyncio
import io
import time
import wave

import numpy as np

from qwen_asr_vllm.agent.async_coordinator import AsyncVoiceAgentCoordinator
from qwen_asr_vllm.engine.streaming import StreamEvent


def wav_bytes(duration_ms: float, *, sample_rate: int = 24000) -> bytes:
    """A silent WAV chunk of a known duration, shaped like real TTS output."""
    frames = int(sample_rate * duration_ms / 1000.0)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)
    return buffer.getvalue()


class WavStreamingTts:
    """Streaming TTS that emits WAV chunks so preroll duration can be asserted."""

    def __init__(self, chunk_ms=200.0, chunks=4):
        self.chunk_ms = chunk_ms
        self.chunks = chunks
        self.texts = []

    def synthesize_stream(self, text):
        self.texts.append(text)
        for _ in range(self.chunks):
            yield wav_bytes(self.chunk_ms)

    def synthesize(self, text):
        raise AssertionError("streaming TTS path should be preferred")


class FakeAsrSession:
    def __init__(self, feed_events=None, close_events=None):
        self.feed_events = list(feed_events or [])
        self.close_events = list(close_events or [])

    def feed(self, pcm):
        return list(self.feed_events)

    def close(self, reuse_last=True):
        return list(self.close_events)


class BlockingAsrSession:
    def __init__(self):
        self.calls = 0

    def feed(self, pcm):
        self.calls += 1
        if self.calls == 1:
            return [
                StreamEvent(
                    kind="committed",
                    text="status",
                    committed_text="status",
                    audio_seconds=0.5,
                )
            ]
        time.sleep(0.2)
        return []

    def close(self, reuse_last=True):
        return []


class FakeAsrEngine:
    def __init__(self, session):
        self.session = session

    def open_stream(self, **kwargs):
        return self.session


class SlowStreamingLlm:
    def __init__(self, chunks, delay):
        self.chunks = list(chunks)
        self.delay = delay
        self.finished_at = None

    def chat_stream(self, text):
        for chunk in self.chunks:
            time.sleep(self.delay)
            yield chunk
        self.finished_at = time.perf_counter()


class SlowTts:
    def __init__(self, delay):
        self.delay = delay
        self.starts = []

    def synthesize(self, text):
        self.starts.append((text, time.perf_counter()))
        time.sleep(self.delay)
        return ("wav:" + text).encode()


class StreamingTts:
    def __init__(self):
        self.texts = []

    def synthesize_stream(self, text):
        self.texts.append(text)
        yield b"audio-1:" + text.encode()
        yield b"audio-2:" + text.encode()

    def synthesize(self, text):
        raise AssertionError("streaming TTS path should be preferred")


class DualModeTts:
    supports_streaming_tts = True

    def __init__(self):
        self.streamed = []
        self.synthesized = []

    def synthesize_stream(self, text):
        self.streamed.append(text)
        yield b"stream:" + text.encode()

    def synthesize(self, text):
        self.synthesized.append(text)
        return b"sync:" + text.encode()


class CancelAwareLlm:
    def __init__(self):
        self.cancelled = False

    def chat_stream(self, text):
        yield "partial response without sentence end "
        while not self.cancelled:
            time.sleep(0.01)

    def cancel(self, request_id=None):
        self.cancelled = True


class ResidentCancelAwareLlm(CancelAwareLlm):
    resident_runner = True


class SessionAwareLlm:
    supports_session_history = True

    def __init__(self):
        self.calls = []
        self.resets = []

    def chat_stream(self, text, *, session_id=None):
        self.calls.append((session_id, text))
        yield "Ready."

    def reset(self, *, session_id=None):
        self.resets.append(session_id)


async def collect_until_done(coordinator, timeout=2.0):
    events = []
    while True:
        event = await coordinator.next_event(timeout=timeout)
        events.append(event.as_dict())
        if event.type == "done":
            return events


def test_async_coordinator_overlaps_tts_with_remaining_llm_stream():
    asyncio.run(_async_coordinator_overlaps_tts_with_remaining_llm_stream())


async def _async_coordinator_overlaps_tts_with_remaining_llm_stream():
    session = FakeAsrSession(
        feed_events=[
            StreamEvent(
                kind="committed",
                text="status",
                committed_text="status",
                audio_seconds=0.5,
            )
        ]
    )
    llm = SlowStreamingLlm(["First.", " Second."], delay=0.05)
    tts = SlowTts(delay=0.18)
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(session),
        llm,
        tts,
        llm_trigger="committed",
        tts_concurrency=2,
    )

    started = time.perf_counter()
    await coordinator.feed(np.zeros(1600, dtype=np.float32))
    events = await collect_until_done(coordinator)
    elapsed = time.perf_counter() - started

    core_types = [
        event["type"]
        for event in events
        if event["type"] in {
            "asr_committed",
            "llm_start",
            "llm_chunk",
            "llm_done",
            "tts_chunk",
            "done",
        }
    ]
    assert core_types == [
        "asr_committed",
        "llm_start",
        "llm_chunk",
        "llm_chunk",
        "llm_done",
        "tts_chunk",
        "tts_chunk",
        "done",
    ]
    assert tts.starts[0][0] == "First."
    assert llm.finished_at is not None
    assert tts.starts[0][1] < llm.finished_at
    assert elapsed < 0.35


def test_async_coordinator_close_triggers_final_text_and_waits_for_tts():
    asyncio.run(_async_coordinator_close_triggers_final_text_and_waits_for_tts())


async def _async_coordinator_close_triggers_final_text_and_waits_for_tts():
    session = FakeAsrSession(
        close_events=[
            StreamEvent(kind="final", text="lights", committed_text="", audio_seconds=0.8)
        ]
    )
    llm = SlowStreamingLlm(["Done."], delay=0.01)
    tts = SlowTts(delay=0.01)
    coordinator = AsyncVoiceAgentCoordinator(FakeAsrEngine(session), llm, tts)

    await coordinator.close()
    events = await collect_until_done(coordinator)

    assert events[0]["type"] == "asr_final"
    assert events[-1] == {"type": "done", "text": "Done."}


def test_async_coordinator_runs_asr_feed_off_event_loop():
    asyncio.run(_async_coordinator_runs_asr_feed_off_event_loop())


async def _async_coordinator_runs_asr_feed_off_event_loop():
    session = BlockingAsrSession()
    llm = SlowStreamingLlm(["Ok."], delay=0.01)
    tts = SlowTts(delay=0.01)
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(session),
        llm,
        tts,
        llm_trigger="committed",
    )

    await coordinator.feed(np.zeros(1600, dtype=np.float32))
    blocking_feed = asyncio.create_task(coordinator.feed(np.zeros(1600, dtype=np.float32)))
    event_types = []
    while "llm_start" not in event_types:
        event = await coordinator.next_event(timeout=1.0)
        event_types.append(event.type)

    assert "llm_start" in event_types
    assert not blocking_feed.done()
    await blocking_feed
    await collect_until_done(coordinator)


def test_async_coordinator_can_flush_tts_fragment_before_sentence_end():
    asyncio.run(_async_coordinator_can_flush_tts_fragment_before_sentence_end())


async def _async_coordinator_can_flush_tts_fragment_before_sentence_end():
    session = FakeAsrSession(
        feed_events=[
            StreamEvent(
                kind="committed",
                text="status",
                committed_text="status",
                audio_seconds=0.5,
            )
        ]
    )
    llm = SlowStreamingLlm(["alpha ", "beta ", "gamma "], delay=0.04)
    tts = SlowTts(delay=0.01)
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(session),
        llm,
        tts,
        llm_trigger="committed",
        tts_flush_chars=10,
    )

    await coordinator.feed(np.zeros(1600, dtype=np.float32))
    events = await collect_until_done(coordinator)

    assert tts.starts[0][0] == "alpha beta"
    assert llm.finished_at is not None
    assert tts.starts[0][1] < llm.finished_at
    assert [event["text"] for event in events if event["type"] == "tts_chunk"] == [
        "alpha beta",
        "gamma",
    ]


def test_async_coordinator_can_flush_tts_fragment_after_time_budget():
    asyncio.run(_async_coordinator_can_flush_tts_fragment_after_time_budget())


async def _async_coordinator_can_flush_tts_fragment_after_time_budget():
    session = FakeAsrSession(
        feed_events=[
            StreamEvent(
                kind="committed",
                text="status update",
                committed_text="status update",
                audio_seconds=0.5,
            )
        ]
    )
    llm = SlowStreamingLlm(["alpha beta ", "gamma"], delay=0.08)
    tts = SlowTts(delay=0.01)
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(session),
        llm,
        tts,
        llm_trigger="committed",
        tts_flush_after_ms=30,
        tts_flush_min_chars=8,
    )

    await coordinator.feed(np.zeros(1600, dtype=np.float32))
    events = await collect_until_done(coordinator)

    assert tts.starts[0][0] == "alpha beta"
    assert llm.finished_at is not None
    assert tts.starts[0][1] < llm.finished_at
    assert [event["text"] for event in events if event["type"] == "tts_chunk"] == [
        "alpha beta",
        "gamma",
    ]


def test_async_coordinator_gates_short_committed_text_before_starting_llm():
    asyncio.run(_async_coordinator_gates_short_committed_text_before_starting_llm())


async def _async_coordinator_gates_short_committed_text_before_starting_llm():
    session = FakeAsrSession(
        feed_events=[
            StreamEvent(
                kind="committed",
                text="too short",
                committed_text="too short",
                audio_seconds=0.4,
            )
        ]
    )
    llm = SlowStreamingLlm(["Should not run."], delay=0.01)
    tts = SlowTts(delay=0.01)
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(session),
        llm,
        tts,
        llm_trigger="committed",
        min_committed_words=3,
        min_committed_audio_seconds=0.8,
    )

    await coordinator.feed(np.zeros(1600, dtype=np.float32))

    assert (await coordinator.next_event(timeout=1.0)).type == "asr_committed"
    try:
        await coordinator.next_event(timeout=0.05)
    except asyncio.TimeoutError:
        pass
    else:
        raise AssertionError("short committed text should not start LLM")
    assert llm.finished_at is None
    assert tts.starts == []


def _preroll_asr_session():
    """An ASR session that finalizes one utterance, enough to drive one full turn."""
    return FakeAsrSession(
        close_events=[
            StreamEvent(
                kind="final",
                text="tell me something",
                committed_text="tell me something",
                audio_seconds=1.0,
            )
        ]
    )


def test_async_coordinator_prerolls_playback_until_the_buffer_is_full():
    """Starting playback on the first chunk starves when TTS runs slower than real time.

    Measured on this host at TTS RTF 1.34: playback began with 160 ms buffered and
    starved 809.7 ms later, ending 1141.8 ms in deficit. Holding the first chunks
    until a target buffer exists converts that stutter into a later, gapless start.
    """
    asyncio.run(_async_coordinator_prerolls_playback_until_the_buffer_is_full())


async def _async_coordinator_prerolls_playback_until_the_buffer_is_full():
    tts = WavStreamingTts(chunk_ms=200.0, chunks=4)
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(_preroll_asr_session()),
        SlowStreamingLlm(["Ready."], delay=0.0),
        tts,
        tts_playback_preroll_ms=500.0,
    )

    await coordinator.feed(np.zeros(1600, dtype=np.float32))
    await coordinator.close()
    events = []
    while "done" not in [event.type for event in events]:
        events.append(await coordinator.next_event(timeout=2.0))

    chunks = [event for event in events if event.type == "tts_chunk"]
    assert len(chunks) == 4, "no audio may be dropped by the preroll gate"

    types = [event.type for event in events]
    # 200 + 200 = 400 ms is under the 500 ms target, so the gate opens on the
    # third chunk and the two held chunks flush ahead of it.
    first_chunk_index = types.index("tts_chunk")
    assert types.index("tts_first_chunk") < first_chunk_index
    third_arrival = [
        index for index, event in enumerate(events) if event.type == "tts_chunk"
    ][2]
    assert third_arrival - first_chunk_index == 2, (
        "the held chunks must flush contiguously once the buffer target is met"
    )


def test_async_coordinator_preroll_flushes_a_reply_shorter_than_the_buffer():
    """A short reply never reaches the target, so the gate must open at turn end."""
    asyncio.run(_async_coordinator_preroll_flushes_a_reply_shorter_than_the_buffer())


async def _async_coordinator_preroll_flushes_a_reply_shorter_than_the_buffer():
    tts = WavStreamingTts(chunk_ms=100.0, chunks=2)
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(_preroll_asr_session()),
        SlowStreamingLlm(["Hi."], delay=0.0),
        tts,
        tts_playback_preroll_ms=5000.0,
    )

    await coordinator.feed(np.zeros(1600, dtype=np.float32))
    await coordinator.close()
    events = []
    while "done" not in [event.type for event in events]:
        events.append(await coordinator.next_event(timeout=2.0))

    assert len([event for event in events if event.type == "tts_chunk"]) == 2


def test_async_coordinator_preroll_is_off_by_default():
    asyncio.run(_async_coordinator_preroll_is_off_by_default())


async def _async_coordinator_preroll_is_off_by_default():
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(_preroll_asr_session()),
        SlowStreamingLlm(["Ready."], delay=0.0),
        WavStreamingTts(chunk_ms=200.0, chunks=2),
    )

    assert coordinator.tts_playback_preroll_ms == 0.0


def test_async_coordinator_preroll_does_not_withhold_unmeasurable_audio():
    """Audio whose duration cannot be read must never be held hostage by the gate."""
    asyncio.run(_async_coordinator_preroll_does_not_withhold_unmeasurable_audio())


async def _async_coordinator_preroll_does_not_withhold_unmeasurable_audio():
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(_preroll_asr_session()),
        SlowStreamingLlm(["Ready."], delay=0.0),
        StreamingTts(),
        tts_playback_preroll_ms=5000.0,
    )

    await coordinator.feed(np.zeros(1600, dtype=np.float32))
    await coordinator.close()
    events = []
    while "done" not in [event.type for event in events]:
        events.append(await coordinator.next_event(timeout=2.0))

    assert len([event for event in events if event.type == "tts_chunk"]) == 2


def test_async_coordinator_can_defer_tts_audio_until_asr_final():
    asyncio.run(_async_coordinator_can_defer_tts_audio_until_asr_final())


async def _async_coordinator_can_defer_tts_audio_until_asr_final():
    session = FakeAsrSession(
        feed_events=[
            StreamEvent(
                kind="committed",
                text="status update",
                committed_text="status update",
                audio_seconds=0.8,
            )
        ],
        close_events=[
            StreamEvent(
                kind="final",
                text="status update complete",
                committed_text="status update",
                audio_seconds=1.2,
            )
        ],
    )
    llm = SlowStreamingLlm(["Ready."], delay=0.01)
    tts = SlowTts(delay=0.01)
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(session),
        llm,
        tts,
        llm_trigger="committed",
        defer_tts_audio_until_asr_final=True,
    )

    await coordinator.feed(np.zeros(1600, dtype=np.float32))
    seen = []
    while "llm_done" not in seen:
        seen.append((await coordinator.next_event(timeout=1.0)).type)

    started_wait = time.perf_counter()
    while not tts.starts and time.perf_counter() - started_wait < 1.0:
        await asyncio.sleep(0.01)
    assert tts.starts
    while "tts_audio_ready" not in seen:
        seen.append((await coordinator.next_event(timeout=1.0)).type)
    close_started = time.perf_counter()
    try:
        event = await coordinator.next_event(timeout=0.05)
    except asyncio.TimeoutError:
        pass
    else:
        raise AssertionError(f"TTS audio should wait for ASR final, got {event.type}")

    await coordinator.close()
    events = []
    while "done" not in [event["type"] for event in events]:
        events.append((await coordinator.next_event(timeout=1.0)).as_dict())

    event_types = [event["type"] for event in events]
    assert event_types.index("asr_final") < event_types.index("tts_first_chunk")
    assert event_types.index("tts_first_chunk") < event_types.index("tts_chunk")
    assert tts.starts[0][1] < close_started


def test_async_coordinator_coalesces_queued_tts_fragments_when_tts_is_busy():
    asyncio.run(_async_coordinator_coalesces_queued_tts_fragments_when_tts_is_busy())


async def _async_coordinator_coalesces_queued_tts_fragments_when_tts_is_busy():
    session = FakeAsrSession(
        feed_events=[
            StreamEvent(
                kind="committed",
                text="status",
                committed_text="status",
                audio_seconds=0.5,
            )
        ]
    )
    llm = SlowStreamingLlm(["A.", " B.", " C.", " D."], delay=0.01)
    tts = SlowTts(delay=0.12)
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(session),
        llm,
        tts,
        llm_trigger="committed",
        tts_coalesce_chars=20,
    )

    await coordinator.feed(np.zeros(1600, dtype=np.float32))
    events = await collect_until_done(coordinator)

    assert [start[0] for start in tts.starts] == ["A.", "B. C. D."]
    assert [event["text"] for event in events if event["type"] == "tts_chunk"] == [
        "A.",
        "B. C. D.",
    ]


def test_async_coordinator_waits_briefly_to_coalesce_initial_tts_fragments():
    asyncio.run(_async_coordinator_waits_briefly_to_coalesce_initial_tts_fragments())


async def _async_coordinator_waits_briefly_to_coalesce_initial_tts_fragments():
    session = FakeAsrSession(
        feed_events=[
            StreamEvent(
                kind="committed",
                text="status",
                committed_text="status",
                audio_seconds=0.5,
            )
        ]
    )
    llm = SlowStreamingLlm(["A.", " B.", " C."], delay=0.01)
    tts = SlowTts(delay=0.01)
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(session),
        llm,
        tts,
        llm_trigger="committed",
        tts_coalesce_chars=20,
        tts_coalesce_wait_ms=50,
    )

    await coordinator.feed(np.zeros(1600, dtype=np.float32))
    events = await collect_until_done(coordinator)

    assert [start[0] for start in tts.starts] == ["A. B. C."]
    assert [event["text"] for event in events if event["type"] == "tts_chunk"] == [
        "A. B. C.",
    ]


def test_async_coordinator_can_submit_first_tts_segment_immediately():
    asyncio.run(_async_coordinator_can_submit_first_tts_segment_immediately())


async def _async_coordinator_can_submit_first_tts_segment_immediately():
    session = FakeAsrSession(
        feed_events=[
            StreamEvent(
                kind="committed",
                text="status",
                committed_text="status",
                audio_seconds=0.5,
            )
        ]
    )
    llm = SlowStreamingLlm(["A.", " B.", " C."], delay=0.01)
    tts = SlowTts(delay=0.03)
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(session),
        llm,
        tts,
        llm_trigger="committed",
        tts_coalesce_chars=20,
        tts_coalesce_wait_ms=100,
        tts_first_sentence_immediate=True,
    )

    await coordinator.feed(np.zeros(1600, dtype=np.float32))
    events = await collect_until_done(coordinator)

    assert [start[0] for start in tts.starts] == ["A.", "B. C."]
    assert [event["text"] for event in events if event["type"] == "tts_chunk"] == [
        "A.",
        "B. C.",
    ]


def test_async_coordinator_defers_short_tts_segments_until_merged():
    asyncio.run(_async_coordinator_defers_short_tts_segments_until_merged())


async def _async_coordinator_defers_short_tts_segments_until_merged():
    session = FakeAsrSession(
        feed_events=[
            StreamEvent(
                kind="committed",
                text="status",
                committed_text="status",
                audio_seconds=0.5,
            )
        ]
    )
    llm = SlowStreamingLlm(["I'm here to help!", " What can I do for you?"], delay=0.01)
    tts = SlowTts(delay=0.01)
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(session),
        llm,
        tts,
        llm_trigger="committed",
        tts_defer_short_segments_chars=48,
        tts_defer_short_segments_ms=80,
    )

    await coordinator.feed(np.zeros(1600, dtype=np.float32))
    events = await collect_until_done(coordinator)

    assert [start[0] for start in tts.starts] == [
        "I'm here to help! What can I do for you?"
    ]
    assert [event["text"] for event in events if event["type"] == "tts_chunk"] == [
        "I'm here to help! What can I do for you?"
    ]


def test_async_coordinator_removes_replacement_characters_from_tts_input():
    asyncio.run(_async_coordinator_removes_replacement_characters_from_tts_input())


async def _async_coordinator_removes_replacement_characters_from_tts_input():
    session = FakeAsrSession(
        feed_events=[
            StreamEvent(
                kind="committed",
                text="status",
                committed_text="status",
                audio_seconds=0.5,
            )
        ]
    )
    llm = SlowStreamingLlm(["Ready \ufffd."], delay=0.01)
    tts = SlowTts(delay=0.01)
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(session),
        llm,
        tts,
        llm_trigger="committed",
    )

    await coordinator.feed(np.zeros(1600, dtype=np.float32))
    await collect_until_done(coordinator)

    assert [start[0] for start in tts.starts] == ["Ready."]



def test_async_coordinator_prefers_streaming_tts_chunks():
    asyncio.run(_async_coordinator_prefers_streaming_tts_chunks())


async def _async_coordinator_prefers_streaming_tts_chunks():
    session = FakeAsrSession(
        feed_events=[
            StreamEvent(
                kind="committed",
                text="status",
                committed_text="status",
                audio_seconds=0.5,
            )
        ]
    )
    llm = SlowStreamingLlm(["Ready."], delay=0.01)
    tts = StreamingTts()
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(session),
        llm,
        tts,
        llm_trigger="committed",
    )

    await coordinator.feed(np.zeros(1600, dtype=np.float32))
    events = await collect_until_done(coordinator)

    tts_events = [event for event in events if event["type"] == "tts_chunk"]
    assert [event["audio"] for event in tts_events] == [b"audio-1:Ready.", b"audio-2:Ready."]
    assert [event["tts_streaming"] for event in tts_events] == [True, True]


def test_async_coordinator_can_stream_only_first_tts_segment():
    asyncio.run(_async_coordinator_can_stream_only_first_tts_segment())


async def _async_coordinator_can_stream_only_first_tts_segment():
    session = FakeAsrSession(
        feed_events=[
            StreamEvent(
                kind="committed",
                text="status",
                committed_text="status",
                audio_seconds=0.5,
            )
        ]
    )
    llm = SlowStreamingLlm(["First.", " Second."], delay=0.01)
    tts = DualModeTts()
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(session),
        llm,
        tts,
        llm_trigger="committed",
        tts_stream_first_segment_only=True,
    )

    await coordinator.feed(np.zeros(1600, dtype=np.float32))
    events = await collect_until_done(coordinator)

    assert tts.streamed == ["First."]
    assert tts.synthesized == ["Second."]
    tts_events = [event for event in events if event["type"] == "tts_chunk"]
    assert [event["audio"] for event in tts_events] == [
        b"stream:First.",
        b"sync:Second.",
    ]
    assert [event["tts_streaming"] for event in tts_events] == [True, False]


def test_async_coordinator_marks_non_streaming_tts_fallback():
    asyncio.run(_async_coordinator_marks_non_streaming_tts_fallback())


async def _async_coordinator_marks_non_streaming_tts_fallback():
    session = FakeAsrSession(
        feed_events=[
            StreamEvent(
                kind="committed",
                text="status",
                committed_text="status",
                audio_seconds=0.5,
            )
        ]
    )
    llm = SlowStreamingLlm(["Ready."], delay=0.01)
    tts = SlowTts(delay=0.01)
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(session),
        llm,
        tts,
        llm_trigger="committed",
    )

    await coordinator.feed(np.zeros(1600, dtype=np.float32))
    events = await collect_until_done(coordinator)

    tts_events = [event for event in events if event["type"] == "tts_chunk"]
    assert tts_events == [
        {"type": "tts_chunk", "text": "Ready.", "audio": b"wav:Ready.", "tts_streaming": False}
    ]


def test_async_coordinator_interrupt_cancels_generation_without_tts_output():
    asyncio.run(_async_coordinator_interrupt_cancels_generation_without_tts_output())


async def _async_coordinator_interrupt_cancels_generation_without_tts_output():
    session = FakeAsrSession(
        feed_events=[
            StreamEvent(
                kind="committed",
                text="status",
                committed_text="status",
                audio_seconds=0.5,
            )
        ]
    )
    llm = CancelAwareLlm()
    tts = SlowTts(delay=0.01)
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(session),
        llm,
        tts,
        llm_trigger="committed",
        tts_flush_chars=None,
    )

    await coordinator.feed(np.zeros(1600, dtype=np.float32))
    seen = []
    while "llm_chunk" not in seen:
        seen.append((await coordinator.next_event(timeout=1.0)).type)

    await coordinator.interrupt()
    interrupted = []
    while "cancelled" not in interrupted:
        interrupted.append((await coordinator.next_event(timeout=1.0)).type)

    assert "turn_interrupted" in interrupted
    assert llm.cancelled is True
    assert tts.starts == []


def test_async_coordinator_scopes_shared_llm_history_per_session():
    asyncio.run(_async_coordinator_scopes_shared_llm_history_per_session())


async def _async_coordinator_scopes_shared_llm_history_per_session():
    llm = SessionAwareLlm()
    tts = SlowTts(delay=0)
    coordinators = [
        AsyncVoiceAgentCoordinator(
            FakeAsrEngine(
                FakeAsrSession(
                    feed_events=[
                        StreamEvent(
                            kind="committed",
                            text=text,
                            committed_text=text,
                            audio_seconds=0.5,
                        )
                    ]
                )
            ),
            llm,
            tts,
            llm_trigger="committed",
        )
        for text in ("alpha", "beta")
    ]

    await asyncio.gather(
        *(
            coordinator.feed(np.zeros(1600, dtype=np.float32))
            for coordinator in coordinators
        )
    )
    await asyncio.gather(
        *(collect_until_done(coordinator) for coordinator in coordinators)
    )

    session_ids = [session_id for session_id, _ in llm.calls]
    assert len(set(session_ids)) == 2
    assert None not in session_ids
    assert set(llm.resets) == set(session_ids)


def test_async_coordinator_does_not_globally_cancel_shared_resident_llm():
    asyncio.run(_async_coordinator_does_not_globally_cancel_shared_resident_llm())


async def _async_coordinator_does_not_globally_cancel_shared_resident_llm():
    llm = ResidentCancelAwareLlm()
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(FakeAsrSession()),
        llm,
        SlowTts(delay=0),
    )

    await coordinator.interrupt()

    assert llm.cancelled is False


def test_async_coordinator_energy_barge_in_interrupts_by_default():
    asyncio.run(_async_coordinator_energy_barge_in_interrupts_by_default())


async def _async_coordinator_energy_barge_in_interrupts_by_default():
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(FakeAsrSession()),
        CancelAwareLlm(),
        SlowTts(delay=0),
    )
    coordinator._turns.state = "speaking"

    await coordinator.feed(np.full(1600, 0.2, dtype=np.float32))

    seen = []
    while True:
        try:
            seen.append((await coordinator.next_event(timeout=0.2)).type)
        except asyncio.TimeoutError:
            break
    assert "turn_interrupted" in seen


def test_async_coordinator_after_asr_final_policy_ignores_same_utterance_audio():
    asyncio.run(_async_coordinator_after_asr_final_policy_ignores_same_utterance_audio())


async def _async_coordinator_after_asr_final_policy_ignores_same_utterance_audio():
    """Audio still arriving for the utterance that produced the reply must not barge in.

    Once first audio is fast enough to play before the user stops speaking, the
    remainder of the same utterance would otherwise cancel the turn.
    """
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(FakeAsrSession()),
        CancelAwareLlm(),
        SlowTts(delay=0),
        barge_in_policy="after-asr-final",
    )
    coordinator._turns.state = "speaking"

    await coordinator.feed(np.full(1600, 0.2, dtype=np.float32))

    seen = []
    while True:
        try:
            seen.append((await coordinator.next_event(timeout=0.2)).type)
        except asyncio.TimeoutError:
            break
    assert "turn_interrupted" not in seen
    assert coordinator._turns.state == "speaking"


def test_async_coordinator_after_asr_final_policy_allows_barge_in_once_final_seen():
    asyncio.run(_async_coordinator_after_asr_final_policy_allows_barge_in_once_final_seen())


async def _async_coordinator_after_asr_final_policy_allows_barge_in_once_final_seen():
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(FakeAsrSession()),
        CancelAwareLlm(),
        SlowTts(delay=0),
        barge_in_policy="after-asr-final",
    )
    coordinator._asr_final_seen = True
    coordinator._turns.state = "speaking"

    await coordinator.feed(np.full(1600, 0.2, dtype=np.float32))

    seen = []
    while True:
        try:
            seen.append((await coordinator.next_event(timeout=0.2)).type)
        except asyncio.TimeoutError:
            break
    assert "turn_interrupted" in seen


def test_async_coordinator_explicit_only_policy_keeps_manual_interrupt():
    asyncio.run(_async_coordinator_explicit_only_policy_keeps_manual_interrupt())


async def _async_coordinator_explicit_only_policy_keeps_manual_interrupt():
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(FakeAsrSession()),
        CancelAwareLlm(),
        SlowTts(delay=0),
        barge_in_policy="explicit-only",
    )
    coordinator._asr_final_seen = True
    coordinator._turns.state = "speaking"

    await coordinator.feed(np.full(1600, 0.2, dtype=np.float32))
    seen = []
    while True:
        try:
            seen.append((await coordinator.next_event(timeout=0.2)).type)
        except asyncio.TimeoutError:
            break
    assert "turn_interrupted" not in seen

    await coordinator.interrupt()
    manual = []
    while "cancelled" not in manual:
        manual.append((await coordinator.next_event(timeout=1.0)).type)
    assert "turn_interrupted" in manual


def test_async_coordinator_barge_in_rms_threshold_is_configurable():
    asyncio.run(_async_coordinator_barge_in_rms_threshold_is_configurable())


async def _async_coordinator_barge_in_rms_threshold_is_configurable():
    coordinator = AsyncVoiceAgentCoordinator(
        FakeAsrEngine(FakeAsrSession()),
        CancelAwareLlm(),
        SlowTts(delay=0),
        barge_in_rms_threshold=0.5,
    )
    coordinator._turns.state = "speaking"

    await coordinator.feed(np.full(1600, 0.2, dtype=np.float32))

    seen = []
    while True:
        try:
            seen.append((await coordinator.next_event(timeout=0.2)).type)
        except asyncio.TimeoutError:
            break
    assert "turn_interrupted" not in seen


def test_async_coordinator_rejects_unknown_barge_in_policy():
    try:
        AsyncVoiceAgentCoordinator(
            FakeAsrEngine(FakeAsrSession()),
            CancelAwareLlm(),
            SlowTts(delay=0),
            barge_in_policy="whenever",
        )
    except ValueError as exc:
        assert "barge_in_policy" in str(exc)
    else:
        raise AssertionError("unknown barge_in_policy must be rejected")
