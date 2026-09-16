"""Transcription quality metrics.

LibriSpeech references are uppercase and unpunctuated while the model emits
cased, punctuated text, so a raw comparison would report a large error rate that
says nothing about recognition quality. Both sides are normalised the same way
before scoring.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_PUNCTUATION = re.compile(r"[^\w\s']", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation, and collapse whitespace."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = _PUNCTUATION.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()

def compute_wer(references: list[str], hypotheses: list[str]) -> float:
    import jiwer

    refs = [normalize_text(reference) for reference in references]
    hyps = [normalize_text(hypothesis) for hypothesis in hypotheses]
    pairs = [(ref, hyp) for ref, hyp in zip(refs, hyps) if ref]
    if not pairs:
        return float("nan")
    return jiwer.wer([ref for ref, _ in pairs], [hyp for _, hyp in pairs])


def compute_cer(references: list[str], hypotheses: list[str]) -> float:
    import jiwer

    refs = [normalize_text(reference) for reference in references]
    hyps = [normalize_text(hypothesis) for hypothesis in hypotheses]
    pairs = [(ref, hyp) for ref, hyp in zip(refs, hyps) if ref]
    if not pairs:
        return float("nan")
    return jiwer.cer([ref for ref, _ in pairs], [hyp for _, hyp in pairs])


DEFAULT_SPELLING_MAP = Path("/mnt/llm_data/voice_ckpt/whisper-large-v3/normalizer.json")


def load_english_normalizer(spelling_map: Path | None = None) -> Callable[[str], str]:
    """Whisper's ``EnglishTextNormalizer``, the community standard for English WER.

    :func:`normalize_text` is enough for LibriSpeech, whose references are already
    unpunctuated words. It is not enough for anything read from a real transcript,
    which spells numbers out: a reference saying "twenty seven kilometers" against a
    hypothesis saying "27 kilometers" scores two errors on a correct recognition. This
    normalizer canonicalises both sides to "27 kilometers", expands contractions, and
    folds British spellings, so the result is comparable with published numbers rather
    than to a rule set invented here.
    """
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    path = spelling_map or DEFAULT_SPELLING_MAP
    try:
        mapping = json.loads(path.read_text())
    except OSError:
        # Costs only British/American spelling folding, so degrade rather than fail.
        logger.warning("no spelling map at %s; spelling variants will count as errors", path)
        mapping = {}
    return EnglishTextNormalizer(mapping)


@dataclass
class WordScore:
    """Corpus-level word error rate with its edit breakdown."""

    wer: float
    substitutions: int
    insertions: int
    deletions: int
    ref_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions


def score_words(
    references: list[str],
    hypotheses: list[str],
    normalizer: Callable[[str], str] = normalize_text,
) -> WordScore:
    """Pool edits across the corpus before dividing, and keep them itemised.

    Averaging per-utterance rates would let a short recording weigh as much as a
    20-minute one. The breakdown matters at long form specifically: a deletion spike is
    what truncation or attention collapse looks like, and it is invisible in the total.
    """
    import jiwer

    refs = [normalizer(reference) for reference in references]
    hyps = [normalizer(hypothesis) for hypothesis in hypotheses]
    pairs = [(ref, hyp) for ref, hyp in zip(refs, hyps) if ref]
    if not pairs:
        return WordScore(float("nan"), 0, 0, 0, 0)
    out = jiwer.process_words([ref for ref, _ in pairs], [hyp for _, hyp in pairs])
    ref_words = sum(len(ref.split()) for ref, _ in pairs)
    return WordScore(
        wer=out.wer,
        substitutions=out.substitutions,
        insertions=out.insertions,
        deletions=out.deletions,
        ref_words=ref_words,
    )


def exact_match_rate(left: list[str], right: list[str]) -> float:
    """Fraction of pairs that normalise to identical strings."""
    if not left:
        return float("nan")
    matches = sum(normalize_text(a) == normalize_text(b) for a, b in zip(left, right))
    return matches / len(left)
