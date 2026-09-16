"""Waveform to mel frontend.

Qwen3-ASR uses a stock ``WhisperFeatureExtractor`` (128 mel bins, hop 160,
n_fft 400, 16kHz), so we drive it directly rather than going through
``Qwen3ASRProcessor``. That processor is unavailable here anyway: it needs the
``qwen3_asr`` model type registered in ``transformers``, and ``AutoProcessor``
silently degrades to returning only the tokenizer.

Features are extracted one request at a time. Batching the extractor would make
the frame count batch-dependent: the attention mask of a padded item reports
``ceil(samples / hop)`` frames while an unpadded item reports ``floor``, which
shifts the audio token count by one and would make results depend on how
requests happen to be grouped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

from qwen_asr_vllm.audio.tokens import num_audio_tokens

TARGET_SAMPLE_RATE = 16000


@dataclass
class AudioFeatures:
    """Mel features for one request, with its already-known token footprint."""

    mel: torch.Tensor
    """Shape ``(num_mel_bins, mel_frames)``, unpadded."""
    mel_frames: int
    num_audio_tokens: int
    audio_seconds: float


class AudioFrontend:
    def __init__(self, model_path: str, dtype: torch.dtype = torch.float32):
        self.extractor = WhisperFeatureExtractor.from_pretrained(model_path)
        self.dtype = dtype
        if self.extractor.sampling_rate != TARGET_SAMPLE_RATE:
            raise ValueError(
                f"expected a {TARGET_SAMPLE_RATE}Hz feature extractor, "
                f"got {self.extractor.sampling_rate}"
            )

    @property
    def num_mel_bins(self) -> int:
        return self.extractor.feature_size

    @property
    def hop_length(self) -> int:
        return self.extractor.hop_length

    def _to_mono_16k(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(waveform)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        audio = audio.astype(np.float32, copy=False)
        if sample_rate != TARGET_SAMPLE_RATE:
            import librosa

            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
            ).astype(np.float32, copy=False)
        return audio

    def __call__(self, waveform: np.ndarray, sample_rate: int) -> AudioFeatures:
        audio = self._to_mono_16k(waveform, sample_rate)
        if audio.size < self.hop_length:
            raise ValueError(
                f"audio is too short to produce a single mel frame "
                f"({audio.size} samples < hop {self.hop_length})"
            )
        extracted = self.extractor(
            [audio],
            sampling_rate=TARGET_SAMPLE_RATE,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        mel = extracted["input_features"][0]
        frames = int(extracted["attention_mask"][0].sum())
        mel = mel[:, :frames].to(self.dtype).contiguous()
        return AudioFeatures(
            mel=mel,
            mel_frames=frames,
            num_audio_tokens=num_audio_tokens(frames),
            audio_seconds=audio.size / TARGET_SAMPLE_RATE,
        )
