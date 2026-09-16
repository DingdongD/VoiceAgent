"""Decode Qwen3-ASR raw output into (language, text).

Without a forced language the model emits a metadata line followed by the
transcription:

    language Chinese<asr_text>....

With a forced language the assistant turn is already primed with that prefix, so
whatever comes back is pure transcription. The repetition guard mirrors the
reference implementation because it changes the emitted text and therefore WER.
"""

from __future__ import annotations

ASR_TEXT_TAG = "<asr_text>"
LANGUAGE_PREFIX = "language "
DEFAULT_REPETITION_THRESHOLD = 20
MAX_REPEATED_PATTERN_LEN = 20


def normalize_language_name(language: str) -> str:
    """Canonical casing used by the model: ``cHINese`` -> ``Chinese``."""
    text = str(language).strip()
    if not text:
        raise ValueError("language is empty")
    return text[:1].upper() + text[1:].lower()


def _collapse_char_runs(text: str, threshold: int) -> str:
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        run = 1
        while index + run < length and text[index + run] == text[index]:
            run += 1
        out.append(text[index] if run > threshold else text[index : index + run])
        index += run
    return "".join(out)


def _collapse_pattern_runs(
    text: str,
    threshold: int,
    max_pattern_len: int = MAX_REPEATED_PATTERN_LEN,
) -> str:
    length = len(text)
    min_run_chars = threshold * 2
    if length < min_run_chars:
        return text

    out: list[str] = []
    index = 0
    found = False
    while index <= length - min_run_chars:
        for pattern_len in range(1, max_pattern_len + 1):
            if index + pattern_len * threshold > length:
                break
            pattern = text[index : index + pattern_len]
            repeats_cleanly = all(
                text[index + rep * pattern_len : index + rep * pattern_len + pattern_len] == pattern
                for rep in range(1, threshold)
            )
            if not repeats_cleanly:
                continue
            run_end = index + threshold * pattern_len
            while run_end + pattern_len <= length and text[run_end : run_end + pattern_len] == pattern:
                run_end += pattern_len
            out.append(pattern)
            out.append(_collapse_pattern_runs(text[run_end:], threshold, max_pattern_len))
            found = True
            break
        if found:
            break
        out.append(text[index])
        index += 1

    if not found:
        out.append(text[index:])
    return "".join(out)


def fix_repetitions(text: str, threshold: int = DEFAULT_REPETITION_THRESHOLD) -> str:
    """Collapse degenerate repetition loops in a decoded transcription."""
    return _collapse_pattern_runs(_collapse_char_runs(text, threshold), threshold)


def parse_asr_output(raw: str | None, user_language: str | None = None) -> tuple[str, str]:
    """Split raw model output into ``(language, text)``."""
    if raw is None:
        return "", ""
    text = str(raw).strip()
    if not text:
        return "", ""

    text = fix_repetitions(text)

    if user_language:
        return user_language, text

    if ASR_TEXT_TAG not in text:
        return "", text.strip()

    meta_part, text_part = text.split(ASR_TEXT_TAG, 1)

    # "language None" is how the model reports that it heard no speech.
    if "language none" in meta_part.lower():
        transcription = text_part.strip()
        return ("", "") if not transcription else ("", transcription)

    language = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith(LANGUAGE_PREFIX):
            value = line[len(LANGUAGE_PREFIX) :].strip()
            if value:
                language = normalize_language_name(value)
            break

    return language, text_part.strip()
