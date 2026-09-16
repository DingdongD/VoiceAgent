"""Prompt construction for Qwen3-ASR.

The chat template shipped with the checkpoint reduces to a fixed 16-token layout
with a single ``<|audio_pad|>`` placeholder that the processor expands to one
token per audio frame group:

    <|im_start|>system\\n{context}<|im_end|>\\n
    <|im_start|>user\\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\\n
    <|im_start|>assistant\\n

We build that string with a single placeholder, tokenize once, then expand the
placeholder in the id list. Because the placeholder is a special token it is
atomic under BPE, so this is identical to tokenizing a string that already holds
N copies -- without materialising a huge string for long audio.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from transformers import AutoTokenizer

from qwen_asr_vllm.config import Qwen3ASRConfig

ASR_TEXT_TAG = "<asr_text>"

LANGUAGE_ALIASES = {
    "en": "English",
    "english": "English",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "chinese": "Chinese",
    "zh-tw": "Cantonese",
    "cantonese": "Cantonese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "tr": "Turkish",
    "hi": "Hindi",
}


def normalize_language(language: str | None) -> str | None:
    """Map a user-supplied language to the canonical name the model expects."""
    if language is None:
        return None
    key = str(language).strip()
    if not key:
        return None
    return LANGUAGE_ALIASES.get(key.lower(), key[:1].upper() + key[1:].lower())


@dataclass
class PromptLayout:
    """Token ids for one request plus the location of its audio span.

    ``audio_offset`` lets the model scatter audio embeddings with a slice
    assignment instead of comparing every token against the placeholder id.
    """

    token_ids: list[int]
    audio_offset: int
    audio_length: int

    def __len__(self) -> int:
        return len(self.token_ids)


class PromptBuilder:
    def __init__(self, model_path: str, config: Qwen3ASRConfig):
        # Loaded exactly as the reference does, warts included: this checkpoint's
        # tokenizer_config carries a regex transformers now flags as wrong, but the
        # model was trained with it and "fixing" it would diverge from the
        # reference tokenization.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.config = config
        self.audio_token_id = config.audio_token_id
        self.stop_token_ids = set(config.stop_token_ids)
        if self.tokenizer.eos_token_id is not None:
            self.stop_token_ids.add(self.tokenizer.eos_token_id)
        # The template depends only on context and language, never on the audio, so
        # tokenizing it once per distinct pair covers a whole workload. The lock also
        # makes build() safe to call from several frontend worker threads.
        self._template_lock = threading.Lock()
        self._templates: dict[tuple[str, str | None], tuple[list[int], int]] = {}

    def _template(self, context: str, language: str | None) -> tuple[list[int], int]:
        key = (context, language)
        with self._template_lock:
            cached = self._templates.get(key)
            if cached is not None:
                return cached

        text = (
            f"<|im_start|>system\n{context or ''}<|im_end|>\n"
            f"<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        if language:
            # Forcing the language turns the assistant turn into a plain
            # transcription: the model no longer emits the language metadata line.
            text += f"language {language}{ASR_TEXT_TAG}"

        template_ids = self.tokenizer.encode(text)
        try:
            offset = template_ids.index(self.audio_token_id)
        except ValueError as exc:
            raise RuntimeError(
                "audio placeholder missing from tokenized prompt; the tokenizer may not "
                "treat <|audio_pad|> as a special token"
            ) from exc

        with self._template_lock:
            self._templates[key] = (template_ids, offset)
        return template_ids, offset

    def build(
        self,
        num_audio_tokens: int,
        context: str = "",
        language: str | None = None,
    ) -> PromptLayout:
        if num_audio_tokens < 1:
            raise ValueError(f"num_audio_tokens must be positive, got {num_audio_tokens}")

        template_ids, offset = self._template(context, language)
        token_ids = (
            template_ids[:offset]
            + [self.audio_token_id] * num_audio_tokens
            + template_ids[offset + 1 :]
        )
        return PromptLayout(token_ids=token_ids, audio_offset=offset, audio_length=num_audio_tokens)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
