"""Cross-request packing for the audio encoder.

The Qwen3-ASR audio tower is already written in packed/varlen form: it splits mel
input into fixed chunks, flattens them into one sequence, and isolates requests
through ``cu_seqlens``. That means several requests of different lengths can share
a single encoder call with zero padding waste. This module computes the metadata
that makes such a call well-defined.

Two quantities are derived from config constants rather than from the shape of
whatever happens to be in the batch:

* chunks are padded to exactly ``chunk_frames``, so the post-convolution length
  is always ``TOKENS_PER_FULL_CHUNK`` and the sinusoidal position slice is fixed;
* the attention window is ``TOKENS_PER_FULL_CHUNK * (n_window_infer //
  chunk_frames)``.

Upstream infers both from the widest chunk present, which makes the encoder
output depend on how requests were grouped whenever every chunk is shorter than
``chunk_frames`` (i.e. all audio under one second). Pinning them to the config
removes that dependency; for any batch containing at least one full chunk the two
formulations agree exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qwen_asr_vllm.audio.tokens import (
    CHUNK_FRAMES,
    TOKENS_PER_FULL_CHUNK,
    after_cnn_len,
    chunk_lengths_for,
    num_audio_tokens,
)


@dataclass
class PackedAudioBatch:
    """Everything the audio encoder needs for one cross-request call."""

    mel: torch.Tensor
    """Shape ``(num_mel_bins, sum(mel_frames))`` -- all requests concatenated."""
    chunk_lengths: torch.Tensor
    """Mel frames in each convolution chunk, flattened over requests."""
    chunk_valid_lens: torch.Tensor
    """Post-convolution valid length of each chunk (``<= TOKENS_PER_FULL_CHUNK``)."""
    cu_seqlens: torch.Tensor
    """int32 attention-window boundaries over the packed token sequence."""
    audio_token_lens: list[int]
    """Audio tokens belonging to each request, in batch order."""
    window_tokens: int
    """Attention window size implied by the config."""
    max_seqlen: int
    """Longest actual attention window in this batch, for kernel launch config."""

    @property
    def num_chunks(self) -> int:
        return int(self.chunk_lengths.numel())

    @property
    def total_tokens(self) -> int:
        return sum(self.audio_token_lens)

    def split_outputs(self, packed: torch.Tensor) -> list[torch.Tensor]:
        """Slice a packed ``(total_tokens, dim)`` encoder output back per request."""
        if packed.size(0) != self.total_tokens:
            raise ValueError(
                f"encoder returned {packed.size(0)} tokens, expected {self.total_tokens}"
            )
        return list(torch.split(packed, self.audio_token_lens, dim=0))


def attention_window_tokens(n_window_infer: int, chunk_frames: int = CHUNK_FRAMES) -> int:
    """Audio tokens covered by one encoder attention window."""
    return TOKENS_PER_FULL_CHUNK * (n_window_infer // chunk_frames)


def build_window_boundaries(audio_token_lens: list[int], window_tokens: int) -> list[int]:
    """Cumulative attention-window boundaries over the packed token sequence.

    Each request is cut into ``window_tokens``-sized windows so a token never
    attends across a request boundary, and long audio stays within the encoder's
    local attention span.
    """
    boundaries = [0]
    for token_len in audio_token_lens:
        full_windows, remainder = divmod(token_len, window_tokens)
        boundaries.extend([window_tokens] * full_windows)
        if remainder:
            boundaries.append(remainder)
    cumulative = []
    running = 0
    for step in boundaries:
        running += step
        cumulative.append(running)
    return cumulative


def pack_audio_batch(
    mels: list[torch.Tensor],
    n_window_infer: int,
    chunk_frames: int = CHUNK_FRAMES,
    device: torch.device | str = "cpu",
) -> PackedAudioBatch:
    """Pack per-request mel tensors into a single encoder batch.

    Args:
        mels: one ``(num_mel_bins, mel_frames)`` tensor per request, unpadded.
        n_window_infer: encoder inference window in mel frames.
        chunk_frames: mel frames per convolution chunk.
    """
    if not mels:
        raise ValueError("pack_audio_batch requires at least one request")

    chunk_lengths: list[int] = []
    audio_token_lens: list[int] = []
    for mel in mels:
        frames = int(mel.size(-1))
        chunk_lengths.extend(chunk_lengths_for(frames, chunk_frames))
        audio_token_lens.append(num_audio_tokens(frames, chunk_frames))

    window_tokens = attention_window_tokens(n_window_infer, chunk_frames)
    cumulative = build_window_boundaries(audio_token_lens, window_tokens)
    max_seqlen = max(
        (end - start for start, end in zip(cumulative[:-1], cumulative[1:])),
        default=cumulative[0],
    )

    return PackedAudioBatch(
        mel=torch.cat([m.to(device) for m in mels], dim=-1),
        chunk_lengths=torch.tensor(chunk_lengths, dtype=torch.long, device=device),
        chunk_valid_lens=torch.tensor(
            [after_cnn_len(length) for length in chunk_lengths], dtype=torch.long, device=device
        ),
        cu_seqlens=torch.tensor(cumulative, dtype=torch.int32, device=device),
        audio_token_lens=audio_token_lens,
        window_tokens=window_tokens,
        max_seqlen=max_seqlen,
    )
