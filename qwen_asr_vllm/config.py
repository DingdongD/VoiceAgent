"""Configuration objects read straight from a Qwen3-ASR checkpoint.

``transformers`` does not register the ``qwen3_asr`` model type, so
``AutoConfig.from_pretrained`` raises on these checkpoints. The fields we need
are a small, stable subset of ``config.json``, so we parse it ourselves and keep
the engine free of any dependency on the model type being registered.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import torch


def _resolve_dtype(value: str | torch.dtype | None) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value is None:
        return torch.bfloat16
    dtype = getattr(torch, str(value), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported dtype: {value!r}")
    return dtype


@dataclass
class AudioEncoderConfig:
    num_mel_bins: int = 128
    d_model: int = 896
    encoder_layers: int = 18
    encoder_attention_heads: int = 14
    encoder_ffn_dim: int = 3584
    downsample_hidden_size: int = 480
    output_dim: int = 1024
    n_window: int = 50
    n_window_infer: int = 800
    conv_chunksize: int = 500
    max_source_positions: int = 1500
    activation_function: str = "gelu"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.encoder_attention_heads

    @property
    def chunk_frames(self) -> int:
        """Mel frames per convolution chunk (the encoder splits input this way)."""
        return self.n_window * 2

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AudioEncoderConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class TextConfig:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 65536
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    hidden_act: str = "silu"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TextConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class Qwen3ASRConfig:
    audio: AudioEncoderConfig
    text: TextConfig
    audio_token_id: int
    audio_start_token_id: int
    audio_end_token_id: int
    dtype: torch.dtype = torch.bfloat16
    support_languages: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    """Ids that terminate generation, taken from ``generation_config.json``."""

    @staticmethod
    def _read_stop_token_ids(model_path: str) -> list[int]:
        generation_path = os.path.join(model_path, "generation_config.json")
        if not os.path.isfile(generation_path):
            return []
        with open(generation_path, "r", encoding="utf-8") as handle:
            eos = json.load(handle).get("eos_token_id")
        if eos is None:
            return []
        return [int(eos)] if isinstance(eos, int) else [int(token) for token in eos]

    @classmethod
    def from_pretrained(cls, model_path: str) -> Qwen3ASRConfig:
        config_path = os.path.join(model_path, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"config.json not found under {model_path}")
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)

        thinker = raw.get("thinker_config", raw)
        return cls(
            audio=AudioEncoderConfig.from_dict(thinker["audio_config"]),
            text=TextConfig.from_dict(thinker["text_config"]),
            audio_token_id=int(thinker["audio_token_id"]),
            audio_start_token_id=int(thinker["audio_start_token_id"]),
            audio_end_token_id=int(thinker["audio_end_token_id"]),
            dtype=_resolve_dtype(thinker.get("dtype") or raw.get("dtype")),
            support_languages=list(raw.get("support_languages", [])),
            stop_token_ids=cls._read_stop_token_ids(model_path),
        )


@dataclass
class EngineConfig:
    """Runtime knobs for :class:`~qwen_asr_vllm.engine.engine.AsrEngine`.

    ``max_audio_batch_frames`` bounds the audio encoder stage the same way
    ``max_num_batched_tokens`` bounds the text stage: it is the total number of
    mel frames packed into a single cross-request encoder call. 48000 frames is
    480 seconds of audio, which fits comfortably on a 40GB card for the 0.6B
    encoder while still giving the batcher room to combine many short requests.
    """

    model: str
    max_model_len: int = 4096
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 8192
    max_audio_batch_frames: int = 48000
    max_audio_batch_size: int = 32
    gpu_memory_utilization: float = 0.85
    """Fraction of memory *still free after loading weights* to spend on KV cache.

    Deliberately not a fraction of the card's total capacity: on a shared GPU that
    would make the setting's effect depend on what other processes hold.
    """
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    enable_prefix_cache: bool = True
    enforce_eager: bool = False
    """Skip CUDA graph capture for the decode path.

    Decode steps are dominated by CPU work: profiling the 0.6B model at batch 4
    measured 37ms of CPU dispatch against 7.9ms of GPU time per step, because the
    forward is a few hundred small kernels and the launch overhead dwarfs each one.
    Replaying a captured graph issues the whole step as a single launch. Set this to
    fall back to eager when debugging, or on a driver where capture misbehaves.
    """
    enable_dual_stream: bool = False
    """Overlap audio encode and text decode on separate CUDA streams.

    Uses overlap-eligible schedule order (schedule both batches, execute, then
    admit newly encoded requests). Default off. When enabled, ``enforce_eager`` is
    forced on until decode CUDA graphs are captured on the decode stream.
    """
    frontend_threads: int = 8
    """Intra-op threads for the CPU mel frontend, or 0 to leave torch's default.

    ``WhisperFeatureExtractor`` builds the spectrogram from a chain of small torch CPU
    ops. On a 48-core host torch defaults to 24 intra-op threads, and at that width
    the OpenMP barrier around each op costs far more than the op: extracting features
    for a 2-second clip took 122ms at 24 threads and 1.2ms at 8, a 100x difference on
    identical work. Longer clips are insensitive -- 1200s measured 574ms at 24 threads
    versus 548ms at 8 -- so a low cap wins across the range.

    This is applied process-wide, because torch's intra-op pool is global and cannot
    be scoped to the frontend's threads. Set 0 to opt out.
    """
    device: str = "cuda"
    dtype: torch.dtype | str | None = None
    model_config: Qwen3ASRConfig | None = None

    def __post_init__(self):
        if self.num_kvcache_blocks == 0 or self.num_kvcache_blocks < -1:
            raise ValueError("num_kvcache_blocks must be -1 or positive")
        if not os.path.isdir(self.model):
            raise NotADirectoryError(f"model path is not a directory: {self.model}")
        if self.kvcache_block_size % 256 != 0:
            raise ValueError("kvcache_block_size must be a multiple of 256")
        if self.model_config is None:
            self.model_config = Qwen3ASRConfig.from_pretrained(self.model)
        self.dtype = _resolve_dtype(self.dtype or self.model_config.dtype)
        self.max_model_len = min(self.max_model_len, self.model_config.text.max_position_embeddings)
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                "max_num_batched_tokens must be >= max_model_len so that a single request "
                "can always be prefilled"
            )
        if self.enable_dual_stream and not self.enforce_eager:
            # Graphs are captured on the default stream; dual-stream decode runs on
            # a side stream until capture is revisited.
            self.enforce_eager = True
