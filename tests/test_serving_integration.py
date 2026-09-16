"""Serving stack against a real engine, over real HTTP.

The unit-level server tests run on a stub, which proves the HTTP contract but not
that the async wrapper and the real engine agree -- the stub cannot get the audio
frontend, the encoder or the KV cache wrong. This drives uvicorn on a real socket
with real LibriSpeech audio, and also checks that concurrent HTTP requests get the
same transcriptions as the synchronous engine on the same clips.
"""
from __future__ import annotations

import io
import socket
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_asr_vllm.engine.async_engine import AsyncAsrEngine
from qwen_asr_vllm.server import create_app

from .conftest import model_path

pytestmark = [pytest.mark.gpu, pytest.mark.checkpoint, pytest.mark.slow]

MODEL_PATH = model_path()
NUM_SAMPLES = 8
# Several engines coexist across the suite, so none may claim the whole card.
GPU_FRACTION = 0.25


def to_wav(waveform: np.ndarray, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((np.clip(waveform, -1, 1) * 32767).astype(np.int16).tobytes())
    return buffer.getvalue()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def samples():
    from bench.data import load_librispeech

    return load_librispeech(split="test-clean", num_samples=NUM_SAMPLES)


@pytest.fixture(scope="module")
def live_server(samples):
    import uvicorn

    engine = AsyncAsrEngine(
        model=MODEL_PATH,
        max_num_seqs=8,
        max_model_len=2048,
        gpu_memory_utilization=GPU_FRACTION,
    )
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(engine), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 60
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("uvicorn did not come up")

    yield f"http://127.0.0.1:{port}", engine

    server.should_exit = True
    thread.join(timeout=30)
    engine.close()


def test_transcribes_over_http(live_server, samples):
    import httpx

    base_url, _ = live_server
    sample = samples[0]
    response = httpx.post(
        f"{base_url}/v1/audio/transcriptions",
        files={"file": ("clip.wav", to_wav(sample.audio), "audio/wav")},
        data={"language": "en", "response_format": "verbose_json"},
        timeout=120,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["text"].strip()
    # The engine reports the canonical language name the model works in, matching
    # what OpenAI's verbose_json returns; "en" on the way in is normalized to it.
    assert body["language"] == "English"
    assert body["duration"] == pytest.approx(len(sample.audio) / 16000, abs=0.05)
    assert body["timings"]["total_seconds"] > 0


def test_concurrent_http_matches_the_synchronous_engine(live_server, samples):
    """Serving must not change the transcription, only when it is produced."""
    import httpx

    base_url, async_engine = live_server

    def transcribe(sample):
        response = httpx.post(
            f"{base_url}/v1/audio/transcriptions",
            files={"file": ("clip.wav", to_wav(sample.audio), "audio/wav")},
            data={"language": "en"},
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["text"]

    with ThreadPoolExecutor(max_workers=NUM_SAMPLES) as pool:
        served = list(pool.map(transcribe, samples))

    # Same engine, one request at a time: the batching is the only difference.
    direct = [
        async_engine.transcribe(sample.audio, language="en").text for sample in samples
    ]

    assert served == direct


def test_health_and_metrics_reflect_the_traffic(live_server, samples):
    import httpx

    base_url, _ = live_server

    health = httpx.get(f"{base_url}/health", timeout=30).json()
    assert health["status"] == "ok"
    assert health["kv_cache_blocks"] > 0
    assert health["model"] == MODEL_PATH

    metrics = httpx.get(f"{base_url}/metrics", timeout=30).text
    received = next(
        int(line.split()[1])
        for line in metrics.splitlines()
        if line.startswith("asr_requests_received_total ")
    )
    assert received > 0
    assert "asr_audio_seconds_total" in metrics
    assert "asr_finish_reason_total" in metrics
