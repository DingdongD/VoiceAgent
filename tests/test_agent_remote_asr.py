import numpy as np

from qwen_asr_vllm.agent.remote_asr import RemoteAsrEngine, serve_asr_engine
from qwen_asr_vllm.engine.streaming import StreamEvent


class FakeRemoteAsrSession:
    def __init__(self, kwargs):
        self.kwargs = dict(kwargs)
        self.feeds = []

    def feed(self, pcm):
        self.feeds.append(np.asarray(pcm, dtype=np.float32).copy())
        samples = int(self.feeds[-1].size)
        return [
            StreamEvent(
                kind="committed",
                text=f"words {samples}",
                committed_text=f"words {samples}",
                audio_seconds=samples / 16000.0,
                chunk_index=len(self.feeds),
            )
        ]

    def close(self, reuse_last=True):
        total = sum(int(chunk.size) for chunk in self.feeds)
        return [
            StreamEvent(
                kind="final",
                text="the whole utterance",
                committed_text="the whole utterance",
                audio_seconds=total / 16000.0,
                chunk_index=len(self.feeds),
            )
        ]


class FakeRemoteAsrEngine:
    def __init__(self):
        self.sessions = []

    def open_stream(self, **kwargs):
        session = FakeRemoteAsrSession(kwargs)
        self.sessions.append(session)
        return session

    def runtime_metrics(self):
        return {"backend": "fake-edge"}


def test_remote_asr_roundtrips_pcm_and_events_over_localhost():
    backend = FakeRemoteAsrEngine()
    listener = serve_asr_engine(backend, "127.0.0.1", 0)
    host, port = listener.getsockname()
    client = RemoteAsrEngine(host, port, timeout=5.0)
    try:
        assert client.runtime_metrics()["backend"] == "fake-edge"
        session = client.open_stream(language="en", commit_lag_words=2)
        pcm = np.linspace(-0.2, 0.2, 1600, dtype=np.float32)
        events = session.feed(pcm)
        assert [event.kind for event in events] == ["committed"]
        assert events[0].text == "words 1600"
        finals = session.close()
        assert finals[0].kind == "final"
        assert finals[0].text == "the whole utterance"
        np.testing.assert_allclose(backend.sessions[0].feeds[0], pcm)
        assert backend.sessions[0].kwargs["language"] == "en"
        assert backend.sessions[0].kwargs["commit_lag_words"] == 2
    finally:
        client.close()
        listener.close()


def test_remote_asr_unknown_session_is_an_error():
    listener = serve_asr_engine(FakeRemoteAsrEngine(), "127.0.0.1", 0)
    host, port = listener.getsockname()
    client = RemoteAsrEngine(host, port, timeout=5.0)
    try:
        session = client.open_stream()
        session._session_id = 999
        try:
            session.feed(np.zeros(160, dtype=np.float32))
        except RuntimeError as exc:
            assert "unknown session" in str(exc)
        else:
            raise AssertionError("unknown session should fail the RPC")
    finally:
        client.close()
        listener.close()
