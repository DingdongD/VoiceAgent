from __future__ import annotations

import io
import wave
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=16)
def load_librispeech_samples(split: str = "test-clean", limit: int = 8):
    from bench.data import load_librispeech

    return tuple(load_librispeech(split=split, num_samples=limit, sort_by_duration=True))


def get_librispeech_sample_wav(
    index: int,
    *,
    split: str = "test-clean",
    limit: int = 8,
) -> bytes:
    samples = load_librispeech_samples(split=split, limit=limit)
    if index < 0 or index >= len(samples):
        raise IndexError(f"sample index {index} out of range")
    sample = samples[index]
    audio = np.asarray(sample.audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return encode_pcm16_wav(audio, int(sample.sample_rate))


def encode_pcm16_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())
    return buffer.getvalue()
