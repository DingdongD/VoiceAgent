from __future__ import annotations

import re


class TextSegmenter:
    """Incrementally split LLM text into stable TTS segments."""

    ABBREVIATIONS = {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "ave",
        "blvd",
        "rd",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
        "inc",
        "ltd",
        "corp",
        "co",
        "dept",
        "vs",
        "etc",
        "approx",
        "appt",
        "e.g",
        "i.e",
        "a.m",
        "p.m",
    }
    SENTENCE_END_RE = re.compile(r"[.!?。！？]")

    def __init__(self, *, min_chars: int = 1, flush_chars: int | None = None):
        self.min_chars = max(1, int(min_chars))
        self.flush_chars = max(1, int(flush_chars)) if flush_chars else None
        self._buffer = ""

    def add(self, token: str) -> list[str]:
        self._buffer += token
        segments = self._extract_sentences()
        if not segments and self.flush_chars is not None:
            fragment = self._pop_flush_fragment()
            if fragment:
                segments.append(fragment)
        return segments

    def flush(self) -> str | None:
        text = self._buffer.strip()
        self._buffer = ""
        return text or None

    def has_pending(self) -> bool:
        return bool(self._buffer.strip())

    def pop_fragment(self, min_chars: int) -> str | None:
        return self._pop_fragment(max(1, int(min_chars)))

    def flush_if_at_least(self, min_chars: int) -> str | None:
        if len(self._buffer.strip()) < max(1, int(min_chars)):
            return None
        return self.flush()

    def _extract_sentences(self) -> list[str]:
        segments: list[str] = []
        search_from = 0
        while True:
            match = self.SENTENCE_END_RE.search(self._buffer, search_from)
            if match is None:
                return segments
            end = match.end()
            if not self._boundary_confirmed(end):
                search_from = end
                continue
            candidate = self._buffer[:end].strip()
            if not self._is_sentence_boundary(candidate):
                search_from = end
                continue
            if len(candidate) < self.min_chars:
                return segments
            segments.append(candidate)
            self._buffer = self._buffer[end:].lstrip()
            search_from = 0

    def _boundary_confirmed(self, end: int) -> bool:
        if self._buffer[end - 1] in "。！？":
            return True
        if end >= len(self._buffer):
            return True
        return self._buffer[end].isspace() or self._buffer[end] in "\"')]}，、"

    def _is_sentence_boundary(self, candidate: str) -> bool:
        if not candidate:
            return False
        mark = candidate[-1]
        if mark in "。！？":
            return True
        without_mark = candidate[:-1].rstrip()
        if not without_mark:
            return False
        last = without_mark.rsplit(None, 1)[-1].lower().rstrip(".")
        return last not in self.ABBREVIATIONS

    def _pop_flush_fragment(self) -> str | None:
        assert self.flush_chars is not None
        return self._pop_fragment(self.flush_chars)

    def _pop_fragment(self, min_chars: int) -> str | None:
        text = self._buffer
        stripped = text.strip()
        if len(stripped) < min_chars:
            return None
        leading = len(text) - len(text.lstrip())
        body = text[leading:]
        split_at = body.rfind(" ", 0, min_chars + 1)
        min_split = max(1, min_chars // 2)
        if split_at < min_split:
            split_at = min_chars
        fragment = body[:split_at].strip()
        self._buffer = body[split_at:].lstrip()
        return fragment or None
