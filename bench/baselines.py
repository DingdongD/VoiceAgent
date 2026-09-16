"""Baseline transcribers for three-way comparison.

Both baselines are driven with *our* mel frontend and prompt construction so that
comparisons isolate the inference engine rather than re-measuring preprocessing
differences.

The reference implementation carries a telling comment above its audio encoder
loop -- "audio encoder do not support batch inference to keep precision" -- and so
encodes one request at a time. It does support packed batch inference; see
``tests/test_audio_encoder_parity.py`` for the measurement showing that batching
costs no more accuracy there than it does here.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_asr_vllm.audio.frontend import AudioFrontend
from qwen_asr_vllm.config import Qwen3ASRConfig
from qwen_asr_vllm.postprocess import parse_asr_output
from qwen_asr_vllm.prompt import PromptBuilder, normalize_language

DEFAULT_MAX_NEW_TOKENS = 440


@dataclass
class BaselineResult:
    texts: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    raw_texts: list[str] = field(default_factory=list)
    num_output_tokens: list[int] = field(default_factory=list)
    wall_seconds: float = 0.0


class ReferenceTranscriber:
    """Official ``modeling_qwen3_asr`` implementation via ``generate``."""

    def __init__(self, model_path: str, device: str = "cuda"):
        from reference.configuration_qwen3_asr import Qwen3ASRConfig as ReferenceConfig
        from reference.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration

        self.config = Qwen3ASRConfig.from_pretrained(model_path)
        self.frontend = AudioFrontend(model_path)
        self.prompt_builder = PromptBuilder(model_path, self.config)
        self.device = torch.device(device)

        reference_config = ReferenceConfig.from_pretrained(model_path)
        self.model = (
            Qwen3ASRForConditionalGeneration.from_pretrained(
                model_path,
                config=reference_config,
                dtype=self.config.dtype,
                attn_implementation="flash_attention_2",
            )
            .to(self.device)
            .eval()
        )
        self.stop_token_ids = sorted(self.prompt_builder.stop_token_ids)

    @torch.inference_mode()
    def transcribe(
        self,
        audios: list[np.ndarray],
        sample_rate: int = 16000,
        context: str = "",
        language: str | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> BaselineResult:
        canonical_language = normalize_language(language)
        result = BaselineResult()
        started = time.perf_counter()

        for waveform in audios:
            features = self.frontend(waveform, sample_rate)
            layout = self.prompt_builder.build(
                features.num_audio_tokens, context=context, language=canonical_language
            )
            input_ids = torch.tensor([layout.token_ids], dtype=torch.long, device=self.device)
            input_features = features.mel.unsqueeze(0).to(self.device, self.config.dtype)
            feature_mask = torch.ones(
                (1, features.mel_frames), dtype=torch.long, device=self.device
            )

            generated = self.model.generate(
                input_ids=input_ids,
                input_features=input_features,
                feature_attention_mask=feature_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.stop_token_ids,
                do_sample=False,
            )
            output_ids = generated.sequences[0][input_ids.size(1) :].tolist()
            if output_ids and output_ids[-1] in self.prompt_builder.stop_token_ids:
                output_ids = output_ids[:-1]

            raw_text = self.prompt_builder.decode(output_ids)
            detected_language, text = parse_asr_output(raw_text, user_language=canonical_language)
            result.texts.append(text)
            result.languages.append(detected_language)
            result.raw_texts.append(raw_text)
            result.num_output_tokens.append(len(output_ids))

        result.wall_seconds = time.perf_counter() - started
        return result


class NanoVllmAsrTranscriber:
    """The existing ``/home/nano-vllm`` ASR path, for a like-for-like comparison.

    That path reaches into the ``qwen_asr`` research package for its processor and
    output parser, and relies on ``AutoProcessor`` returning a real Qwen3-ASR
    processor. Whether both hold depends on which environment is active, so
    construction failures are surfaced rather than swallowed.
    """

    def __init__(
        self,
        model_path: str,
        nano_vllm_root: str = "/home/nano-vllm",
        qwen_asr_root: str = "/home/qwen_asr/Qwen3-ASR",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ):
        for root in (nano_vllm_root, qwen_asr_root):
            if root not in sys.path:
                sys.path.insert(0, root)
        from nanovllm.asr.qwen3 import NanoQwen3ASR

        self.backend = NanoQwen3ASR(model_path=model_path, max_new_tokens=max_new_tokens)

    def transcribe(
        self,
        audios: list[np.ndarray],
        sample_rate: int = 16000,
        context: str = "",
        language: str | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> BaselineResult:
        result = BaselineResult()
        started = time.perf_counter()
        for waveform in audios:
            (transcription,) = self.backend.transcribe(
                [(waveform, sample_rate)],
                languages=[language],
                contexts=[context],
            )
            text = getattr(transcription, "text", None) or ""
            result.texts.append(text)
            result.languages.append(getattr(transcription, "language", "") or "")
            result.raw_texts.append(getattr(transcription, "raw_text", text) or text)
            result.num_output_tokens.append(len(getattr(transcription, "token_ids", []) or []))
        result.wall_seconds = time.perf_counter() - started
        return result
