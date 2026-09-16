"""HTTP contract: OpenAI compatibility, error mapping, health and metrics.

Run against the stub engine from the async tests, so this covers the parts a client
actually depends on -- status codes, field names, content types -- without needing a
checkpoint or a GPU.
"""
import io
import wave

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from qwen_asr_vllm.audio.decode import UnsupportedAudio, decode_audio
from qwen_asr_vllm.engine.async_engine import AsyncAsrEngine
from qwen_asr_vllm.server import create_app

from .test_async_engine import StubEngine


def wav_bytes(seconds: float = 1.0, sample_rate: int = 16000, channels: int = 1) -> bytes:
    samples = np.zeros(int(seconds * sample_rate) * channels, dtype=np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


@pytest.fixture
def client():
    stub = StubEngine()
    engine = AsyncAsrEngine(engine=stub, frontend_workers=2)
    with TestClient(create_app(engine)) as test_client:
        test_client.stub = stub
        yield test_client
    engine.close()


def upload(client, data=None, **fields):
    return client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", data if data is not None else wav_bytes(), "audio/wav")},
        data=fields,
    )


class TestTranscriptions:
    def test_json_is_the_openai_shape(self, client):
        response = upload(client)
        assert response.status_code == 200
        body = response.json()
        assert body["text"].startswith("text-")
        assert body["language"] == "en"

    def test_text_format_returns_plain_text(self, client):
        response = upload(client, response_format="text")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert response.text.strip().startswith("text-")

    def test_verbose_json_adds_usage_and_stage_timings(self, client):
        body = upload(client, response_format="verbose_json").json()
        assert body["task"] == "transcribe"
        assert body["usage"]["input_tokens"] == 16
        assert set(body["timings"]) >= {"queue_seconds", "encode_seconds", "total_seconds"}

    def test_unknown_response_format_is_rejected(self, client):
        assert upload(client, response_format="srt").status_code == 400

    def test_context_and_language_are_forwarded(self, client):
        assert upload(client, context="Ada Lovelace", language="en").status_code == 200

    def test_missing_file_is_a_validation_error(self, client):
        assert client.post("/v1/audio/transcriptions", data={}).status_code == 422

    def test_undecodable_upload_is_a_client_error(self, client):
        response = upload(client, data=b"this is not audio")
        assert response.status_code == 400
        assert "decode" in response.json()["error"]["message"]

    def test_empty_upload_is_a_client_error(self, client):
        assert upload(client, data=b"").status_code == 400

    def test_timeout_maps_to_gateway_timeout(self, client):
        # The engine never finishes this one, so the deadline is what resolves it.
        client.stub.stall = True
        response = upload(client, timeout=0.05)
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"]

    def test_concurrent_uploads_all_succeed(self, client):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(lambda _: upload(client), range(16)))

        assert all(response.status_code == 200 for response in responses)
        assert len({response.json()["text"] for response in responses}) == 16


class TestOperations:
    def test_health_is_ok_while_the_loop_lives(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["kv_cache_blocks"] == 64

    def test_health_reports_503_when_the_loop_thread_is_gone(self, client):
        client.app.state.engine.close()
        assert client.get("/health").status_code == 503

    def test_metrics_are_prometheus_text(self, client):
        upload(client)
        response = client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        body = response.text
        assert "asr_requests_received_total 1" in body
        assert "asr_request_duration_seconds_bucket" in body
        assert "asr_scheduler_num_preemptions" in body
        assert "asr_kv_cache_blocks_free 64" in body

    def test_models_lists_the_loaded_checkpoint(self, client):
        body = client.get("/v1/models").json()
        assert body["data"][0]["id"] == "stub"


class TestAudioDecoding:
    def test_wav_decodes_to_mono_float32_at_16k(self):
        waveform = decode_audio(wav_bytes(seconds=2.0))
        assert waveform.dtype == np.float32
        assert waveform.ndim == 1
        assert abs(len(waveform) - 32000) <= 1

    def test_stereo_is_averaged_to_mono(self):
        waveform = decode_audio(wav_bytes(seconds=1.0, channels=2))
        assert waveform.ndim == 1
        assert abs(len(waveform) - 16000) <= 1

    def test_resampled_to_16k(self):
        waveform = decode_audio(wav_bytes(seconds=1.0, sample_rate=8000))
        assert abs(len(waveform) - 16000) <= 100

    def test_garbage_raises_unsupported_audio(self):
        with pytest.raises(UnsupportedAudio):
            decode_audio(b"\x00\x01\x02not audio at all")
