import numpy as np

from qwen_asr_vllm.agent.coordinator import VoiceAgentCoordinator
from qwen_asr_vllm.engine.streaming import StreamEvent


class FakeAsrSession:
    def __init__(self, feed_events=None, close_events=None):
        self.feed_events = list(feed_events or [])
        self.close_events = list(close_events or [])
        self.closed = False
        self.received = []

    def feed(self, pcm):
        self.received.append(np.asarray(pcm, dtype=np.float32))
        return list(self.feed_events)

    def close(self, reuse_last=True):
        self.closed = True
        return list(self.close_events)


class FakeAsrEngine:
    def __init__(self, session):
        self.session = session
        self.open_kwargs = None

    def open_stream(self, **kwargs):
        self.open_kwargs = kwargs
        return self.session


class FakeLlm:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.prompts = []

    def chat_stream(self, text):
        self.prompts.append(text)
        yield from self.chunks


class SessionAwareLlm:
    supports_session_history = True

    def __init__(self):
        self.calls = []
        self.resets = []

    def chat_stream(self, text, *, session_id=None):
        self.calls.append((session_id, text))
        yield "Done."

    def reset(self, *, session_id=None):
        self.resets.append(session_id)


class FakeTts:
    def __init__(self):
        self.texts = []

    def synthesize(self, text):
        self.texts.append(text)
        return ("wav:" + text).encode()


def event_dicts(events):
    return [event.as_dict() for event in events]


def test_default_trigger_waits_for_final_before_starting_llm_and_tts():
    session = FakeAsrSession(
        feed_events=[
            StreamEvent(
                kind="committed",
                text="turn on",
                committed_text="turn on",
                audio_seconds=1.0,
            )
        ],
        close_events=[
            StreamEvent(
                kind="final",
                text="turn on the light",
                committed_text="turn on",
                audio_seconds=1.4,
            )
        ],
    )
    llm = FakeLlm(["Done", "."])
    tts = FakeTts()
    coordinator = VoiceAgentCoordinator(FakeAsrEngine(session), llm, tts)

    feed_events = coordinator.feed(np.zeros(1600, dtype=np.float32))
    assert [event["type"] for event in event_dicts(feed_events)] == ["asr_committed"]
    assert llm.prompts == []

    close_events = event_dicts(coordinator.close())

    assert llm.prompts == ["turn on the light"]
    assert tts.texts == ["Done."]
    assert close_events == [
        {
            "type": "asr_final",
            "text": "turn on the light",
            "committed_text": "turn on",
            "audio_seconds": 1.4,
            "chunk_index": 0,
            "commit_violation": False,
        },
        {"type": "llm_start", "text": "turn on the light"},
        {"type": "llm_chunk", "text": "Done"},
        {"type": "llm_chunk", "text": "."},
        {"type": "tts_chunk", "text": "Done.", "audio": b"wav:Done."},
        {"type": "llm_done", "text": "Done."},
        {"type": "done", "text": "Done."},
    ]


def test_committed_trigger_starts_llm_during_feed():
    session = FakeAsrSession(
        feed_events=[
            StreamEvent(
                kind="committed",
                text="hello",
                committed_text="hello",
                audio_seconds=0.8,
            )
        ]
    )
    llm = FakeLlm(["Hi", "!"])
    tts = FakeTts()
    coordinator = VoiceAgentCoordinator(
        FakeAsrEngine(session), llm, tts, llm_trigger="committed"
    )

    events = event_dicts(coordinator.feed(np.zeros(1600, dtype=np.float32)))

    assert llm.prompts == ["hello"]
    assert tts.texts == ["Hi!"]
    assert [event["type"] for event in events] == [
        "asr_committed",
        "llm_start",
        "llm_chunk",
        "llm_chunk",
        "tts_chunk",
        "llm_done",
        "done",
    ]


def test_sync_coordinator_scopes_shared_llm_history_per_session():
    llm = SessionAwareLlm()
    coordinators = [
        VoiceAgentCoordinator(
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
            FakeTts(),
            llm_trigger="committed",
        )
        for text in ("alpha", "beta")
    ]

    for coordinator in coordinators:
        coordinator.feed(np.zeros(1600, dtype=np.float32))

    session_ids = [session_id for session_id, _ in llm.calls]
    assert len(set(session_ids)) == 2
    assert None not in session_ids
    assert set(llm.resets) == set(session_ids)


def test_tts_flushes_remainder_without_sentence_boundary_on_close():
    session = FakeAsrSession(
        close_events=[
            StreamEvent(kind="final", text="status", committed_text="", audio_seconds=0.5)
        ]
    )
    llm = FakeLlm(["Working"])
    tts = FakeTts()
    coordinator = VoiceAgentCoordinator(FakeAsrEngine(session), llm, tts)

    events = event_dicts(coordinator.close())

    assert tts.texts == ["Working"]
    assert {"type": "tts_chunk", "text": "Working", "audio": b"wav:Working"} in events
