"""Decode uploaded audio to the mono float32 the frontend expects.

soundfile handles WAV/FLAC/OGG natively; anything else goes through audioread via
librosa. Kept out of the frontend because it is an I/O concern: the engine's own
interface takes a waveform, and a serving layer is the only thing that has to deal
with whatever bytes a client uploaded.
"""
from __future__ import annotations

import io

import numpy as np

TARGET_SAMPLE_RATE = 16000


class UnsupportedAudio(ValueError):
    """The upload could not be decoded to a waveform."""


def to_mono_float32(waveform: np.ndarray) -> np.ndarray:
    """Average channels and drop to float32 without rescaling."""
    if waveform.ndim > 1:
        # soundfile gives (frames, channels); librosa may give (channels, frames).
        axis = 1 if waveform.shape[0] >= waveform.shape[1] else 0
        waveform = waveform.mean(axis=axis)
    return np.ascontiguousarray(waveform, dtype=np.float32)


def decode_audio(data: bytes, target_sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Decode container bytes to a mono 16 kHz waveform."""
    if not data:
        raise UnsupportedAudio("the uploaded file is empty")

    import soundfile

    try:
        waveform, sample_rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as soundfile_error:  # noqa: BLE001 - fall back, then report both
        try:
            import librosa

            waveform, sample_rate = librosa.load(
                io.BytesIO(data), sr=target_sample_rate, mono=True
            )
        except Exception as librosa_error:  # noqa: BLE001
            raise UnsupportedAudio(
                f"could not decode the audio: soundfile said {soundfile_error}; "
                f"librosa said {librosa_error}"
            ) from soundfile_error

    waveform = to_mono_float32(waveform)
    if sample_rate != target_sample_rate:
        import librosa

        waveform = librosa.resample(
            waveform, orig_sr=sample_rate, target_sr=target_sample_rate
        )
    if waveform.size == 0:
        raise UnsupportedAudio("the decoded audio contains no samples")
    return waveform
