"""Closed-form audio token accounting.

The number of audio tokens a waveform expands into is fully determined by its
mel frame count, so the scheduler can budget KV blocks *before* the audio
encoder runs. This is the property that lets the engine treat audio encoding as
an independent pipeline stage instead of something that must happen inside
prefill.
"""

from __future__ import annotations

CHUNK_FRAMES = 100
"""Mel frames per convolution chunk (``n_window * 2`` for every released config)."""


def conv_output_len(length: int) -> int:
    """Output length of one ``Conv2d(kernel=3, stride=2, padding=1)`` layer."""
    return (length - 1) // 2 + 1


def after_cnn_len(frames: int) -> int:
    """Frames surviving the three stacked stride-2 convolutions."""
    return conv_output_len(conv_output_len(conv_output_len(frames)))


TOKENS_PER_FULL_CHUNK = after_cnn_len(CHUNK_FRAMES)
"""13 tokens per 100 mel frames, i.e. 13 audio tokens per second of 16kHz audio."""


def num_audio_tokens(mel_frames: int, chunk_frames: int = CHUNK_FRAMES) -> int:
    """Audio tokens produced for ``mel_frames`` mel frames.

    Equivalent to the upstream ``_get_feat_extract_output_lengths`` but written
    in terms of the chunking that actually causes it.
    """
    full_chunks, tail = divmod(mel_frames, chunk_frames)
    return full_chunks * after_cnn_len(chunk_frames) + after_cnn_len(tail)


def chunk_lengths_for(mel_frames: int, chunk_frames: int = CHUNK_FRAMES) -> list[int]:
    """Split a request's mel frames into convolution chunks.

    A frame count that is an exact multiple of ``chunk_frames`` yields only full
    chunks; otherwise the final chunk holds the remainder.
    """
    if mel_frames <= 0:
        raise ValueError(f"mel_frames must be positive, got {mel_frames}")
    full_chunks, tail = divmod(mel_frames, chunk_frames)
    lengths = [chunk_frames] * full_chunks
    if tail:
        lengths.append(tail)
    return lengths
