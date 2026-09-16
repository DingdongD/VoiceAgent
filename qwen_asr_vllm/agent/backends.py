from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Generator, Protocol


class LlmBackend(Protocol):
    def chat_stream(
        self, text: str, *, session_id: Any = None
    ) -> Generator[str, None, None]:
        """Yield text deltas for one assistant reply."""


class TtsBackend(Protocol):
    def synthesize(self, text: str) -> bytes | None:
        """Return an encoded audio chunk for a stable text segment."""

    def synthesize_stream(self, text: str) -> Iterable[bytes | None]:
        """Optionally yield encoded audio chunks as soon as they are ready."""
