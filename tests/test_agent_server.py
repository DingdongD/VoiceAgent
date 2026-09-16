import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from qwen_asr_vllm.agent.events import agent_event
from qwen_asr_vllm.agent import latency_data
from qwen_asr_vllm.engine.async_engine import AsyncAsrEngine
from qwen_asr_vllm.server import create_app

from .test_async_engine import StubEngine


class FakeVoiceSession:
    def __init__(self):
        self.received = []
        self.closed = False

    def feed(self, pcm):
        self.received.append(np.asarray(pcm, dtype=np.float32))
        return [
            agent_event("asr_partial", text="hello"),
            agent_event("tts_chunk", text="hi.", audio=b"wav:hi."),
        ]

    def close(self):
        self.closed = True
        return [agent_event("done", text="hi.")]


class FakeAsyncVoiceSession:
    def __init__(self):
        import asyncio

        self.events = asyncio.Queue()
        self.received = []
        self.closed = False

    async def feed(self, pcm):
        self.received.append(np.asarray(pcm, dtype=np.float32))
        await self.events.put(agent_event("asr_partial", text="async hello"))
        await self.events.put(agent_event("tts_chunk", text="async hi.", audio=b"wav:async"))

    async def close(self):
        self.closed = True
        await self.events.put(agent_event("done", text="async hi."))

    async def next_event(self):
        return await self.events.get()

    async def playback_ack(self, payload=None):
        await self.events.put(agent_event("tts_playback_ack", payload=payload or {}))

    async def interrupt(self):
        await self.events.put(agent_event("turn_interrupted", from_state="speaking"))


def test_voice_websocket_streams_json_events_and_binary_audio():
    created = []

    def voice_factory(engine):
        session = FakeVoiceSession()
        created.append((engine, session))
        return session

    engine = AsyncAsrEngine(engine=StubEngine(), frontend_workers=1)
    try:
        with TestClient(create_app(engine, voice_factory=voice_factory)) as client:
            with client.websocket_connect("/v1/voice/sessions") as ws:
                ws.send_json({"type": "start"})
                assert ws.receive_json() == {"type": "session_started"}

                pcm = np.zeros(8, dtype=np.float32)
                ws.send_bytes(pcm.tobytes())

                assert ws.receive_json() == {"type": "asr_partial", "text": "hello"}
                assert ws.receive_json() == {"type": "tts_chunk", "text": "hi."}
                assert ws.receive_bytes() == b"wav:hi."

                ws.send_json({"type": "close"})
                assert ws.receive_json() == {"type": "done", "text": "hi."}

        assert created[0][1].closed is True
        np.testing.assert_array_equal(created[0][1].received[0], pcm)
    finally:
        engine.close()


def test_voice_websocket_supports_async_voice_sessions():
    created = []

    def voice_factory(engine):
        session = FakeAsyncVoiceSession()
        created.append(session)
        return session

    engine = AsyncAsrEngine(engine=StubEngine(), frontend_workers=1)
    try:
        with TestClient(create_app(engine, voice_factory=voice_factory)) as client:
            with client.websocket_connect("/v1/voice/sessions") as ws:
                ws.send_json({"type": "start"})
                assert ws.receive_json() == {"type": "session_started"}

                pcm = np.zeros(8, dtype=np.float32)
                ws.send_bytes(pcm.tobytes())
                assert ws.receive_json() == {"type": "asr_partial", "text": "async hello"}
                assert ws.receive_json() == {"type": "tts_chunk", "text": "async hi."}
                assert ws.receive_bytes() == b"wav:async"

                ws.send_json({"type": "close"})
                assert ws.receive_json() == {"type": "done", "text": "async hi."}

        assert created[0].closed is True
        np.testing.assert_array_equal(created[0].received[0], pcm)
    finally:
        engine.close()


def test_agent_latency_page_serves_realtime_voice_console():
    def voice_factory(engine):
        return FakeVoiceSession()

    engine = AsyncAsrEngine(engine=StubEngine(), frontend_workers=1)
    try:
        with TestClient(create_app(engine, voice_factory=voice_factory)) as client:
            response = client.get("/agent-latency")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        body = response.text
        assert "Agent Latency Console" in body
        assert "/v1/voice/sessions" in body
        assert "firstAudioMs" in body
        assert "llmFirstChunkMs" in body
        assert "firstTtsTextMs" in body
        assert "ttsFirstChunkMs" in body
        assert "sendPlaybackAck" in body
        assert "tts_played" in body
        assert "totalMs" in body
        assert "streamUploadedAudioRealtime" in body
        assert "inputAudioMs" in body
        assert "asrWer" in body
        assert "goldenAsrText" in body
        assert "ttsInputText" in body
        assert "TTS Input From LLM" in body
        assert "llmOutputText" in body
        assert "LLM Generated Output" in body
        assert "updateLlmOutput(event)" in body
        assert "TTS Target Text" not in body
        assert "audioClips" in body
        assert "wavDurationSeconds" in body
        assert "appendAudioClip" in body
        assert "loadDatasetSamples" in body
        assert "updateAsrHypothesis(event)" in body
        assert "fullAsrText(event)" in body
        assert "event.committed_text || event.text" in body
        assert "Latest ASR Hypothesis" in body
        assert "Waiting for ASR events." in body
        assert "enqueueLocalPlayback" in body
        assert "playNextLocalClip" in body
        assert "browser-local" in body
        assert "loadTopology" in body
        assert "getUserMedia" in body
    finally:
        engine.close()


def test_agent_latency_dataset_endpoints_expose_golden_asr_and_wav(monkeypatch):
    class Sample:
        sample_id = "sample-1"
        duration = 0.25
        text = "HELLO WORLD"

    def fake_samples(split, limit):
        assert split == "test-clean"
        assert limit == 4
        return [Sample()]

    def fake_audio(index, split, limit):
        assert index == 0
        assert split == "test-clean"
        assert limit == 4
        return b"RIFFdemo-wav"

    monkeypatch.setattr(latency_data, "load_librispeech_samples", fake_samples)
    monkeypatch.setattr(latency_data, "get_librispeech_sample_wav", fake_audio)

    def voice_factory(engine):
        return FakeVoiceSession()

    engine = AsyncAsrEngine(engine=StubEngine(), frontend_workers=1)
    try:
        with TestClient(create_app(engine, voice_factory=voice_factory)) as client:
            listing = client.get("/agent-latency/datasets", params={"split": "test-clean", "limit": 4})
            audio = client.get(
                "/agent-latency/datasets/0/audio",
                params={"split": "test-clean", "limit": 4},
            )

        assert listing.status_code == 200
        assert listing.json() == {
            "split": "test-clean",
            "samples": [
                {
                    "index": 0,
                    "sample_id": "sample-1",
                    "duration_seconds": 0.25,
                    "golden_asr_text": "HELLO WORLD",
                }
            ],
        }
        assert audio.status_code == 200
        assert audio.headers["content-type"].startswith("audio/wav")
        assert audio.content == b"RIFFdemo-wav"
    finally:
        engine.close()



def test_voice_websocket_accepts_playback_ack_and_interrupt_controls():
    created = []

    def voice_factory(engine):
        session = FakeAsyncVoiceSession()
        created.append(session)
        return session

    engine = AsyncAsrEngine(engine=StubEngine(), frontend_workers=1)
    try:
        with TestClient(create_app(engine, voice_factory=voice_factory)) as client:
            with client.websocket_connect("/v1/voice/sessions") as ws:
                ws.send_json({"type": "start"})
                assert ws.receive_json() == {"type": "session_started"}

                ws.send_json({"type": "tts_played", "chunk_index": 1, "audio_ms": 240})
                assert ws.receive_json() == {
                    "type": "tts_playback_ack",
                    "payload": {"type": "tts_played", "chunk_index": 1, "audio_ms": 240},
                }

                ws.send_json({"type": "interrupt"})
                assert ws.receive_json() == {"type": "turn_interrupted", "from_state": "speaking"}
    finally:
        engine.close()
