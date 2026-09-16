"""Request state machine.

A request moves through three GPU stages. Audio encoding is its own stage rather
than something folded into prefill, which is what lets several requests share one
encoder call and stops one long recording from stalling the whole batch.

    WAITING_ENCODE -> WAITING_PREFILL -> RUNNING_DECODE -> FINISHED
"""

from __future__ import annotations

import enum
import itertools
import time
from dataclasses import dataclass, field

import torch

from qwen_asr_vllm.audio.frontend import AudioFeatures
from qwen_asr_vllm.prompt import PromptLayout


class RequestStage(enum.Enum):
    WAITING_ENCODE = enum.auto()
    WAITING_PREFILL = enum.auto()
    RUNNING_DECODE = enum.auto()
    FINISHED = enum.auto()


@dataclass
class SamplingParams:
    temperature: float = 0.0
    max_new_tokens: int = 440
    """Kept at the upstream default so parity tests compare like with like.

    It suits the ~30s clips it was chosen for and silently truncates anything longer:
    a 20-minute transcript needs about 4900 tokens. Callers that do not know the audio
    length in advance should use :meth:`for_audio`.
    """
    ignore_eos: bool = False

    @classmethod
    def for_audio(cls, audio_seconds: float, max_new_tokens: int = 0, **kwargs):
        """A cap scaled to the audio, so long input is not quietly cut short.

        Real speech emits 3.5-3.7 tokens per audio second across read and spontaneous
        material, so 8 per second leaves better than a 2x margin while still bounding a
        runaway repetition loop. ``max_new_tokens`` overrides the estimate when given.
        """
        if max_new_tokens > 0:
            return cls(max_new_tokens=max_new_tokens, **kwargs)
        return cls(max_new_tokens=max(440, int(audio_seconds * 8) + 64), **kwargs)


@dataclass
class RequestTimings:
    """Stage transition timestamps, for wall-clock decomposition in benchmarks."""

    arrival: float = field(default_factory=time.perf_counter)
    encode_start: float | None = None
    encode_end: float | None = None
    prefill_end: float | None = None
    finish: float | None = None

    def _span(self, start: float | None, end: float | None) -> float:
        return 0.0 if start is None or end is None else end - start

    @property
    def queue_seconds(self) -> float:
        return self._span(self.arrival, self.encode_start)

    @property
    def encode_seconds(self) -> float:
        return self._span(self.encode_start, self.encode_end)

    @property
    def prefill_seconds(self) -> float:
        return self._span(self.encode_end, self.prefill_end)

    @property
    def decode_seconds(self) -> float:
        return self._span(self.prefill_end, self.finish)

    @property
    def total_seconds(self) -> float:
        return self._span(self.arrival, self.finish)


class AsrRequest:
    """One transcription in flight.

    ``num_computed_tokens`` tracks how much of the prompt already has KV state,
    covering both prefix-cache hits and completed prefill, so the runner can
    compute a query length the same way for prefill and decode.
    """

    # itertools.count is atomic under CPython, so ids stay unique when requests are
    # prepared on several frontend worker threads.
    _ids = itertools.count()

    def __init__(
        self,
        features: AudioFeatures,
        layout: PromptLayout,
        sampling: SamplingParams,
        block_size: int,
        stop_token_ids: set[int],
        language: str | None = None,
    ):
        self.request_id = next(AsrRequest._ids)

        self.features = features
        self.layout = layout
        self.sampling = sampling
        self.block_size = block_size
        self.stop_token_ids = stop_token_ids
        self.language = language

        self.stage = RequestStage.WAITING_ENCODE
        self.timings = RequestTimings()
        self.audio_embeds: torch.Tensor | None = None

        self.token_ids: list[int] = list(layout.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.num_computed_tokens = 0
        self.num_cached_tokens = 0
        self.block_table: list[int] = []
        self.finish_reason: str | None = None
        # Set when cancellation arrives while this request is inside a batch already
        # being executed; honoured as soon as that step lands.
        self.cancel_requested = False

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def last_token_id(self) -> int:
        return self.token_ids[-1]

    @property
    def num_blocks(self) -> int:
        return (len(self.token_ids) + self.block_size - 1) // self.block_size

    def block(self, index: int) -> list[int]:
        return self.token_ids[index * self.block_size : (index + 1) * self.block_size]

    @property
    def num_cacheable_blocks(self) -> int:
        """Blocks that lie entirely before the audio span.

        Every audio position carries the same placeholder token id, so token ids
        alone cannot distinguish two different recordings. A block containing
        audio is therefore unsafe to reuse -- and so is every block after it,
        since prefix hashes chain. Only the fully-textual prefix, typically a
        long system prompt used for hotword biasing, can be shared.
        """
        return self.layout.audio_offset // self.block_size

    @property
    def num_tokens_to_compute(self) -> int:
        """Query length for the next forward pass; 1 once decoding."""
        return len(self.token_ids) - self.num_computed_tokens

    def mark_encoding(self) -> None:
        self.stage = RequestStage.WAITING_PREFILL
        self.timings.encode_end = time.perf_counter()

    def mark_prefilled(self) -> None:
        self.stage = RequestStage.RUNNING_DECODE
        self.timings.prefill_end = time.perf_counter()

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def check_finished(self) -> bool:
        if not self.sampling.ignore_eos and self.last_token_id in self.stop_token_ids:
            self.finish("stop")
            return True
        if self.num_output_tokens >= self.sampling.max_new_tokens:
            self.finish("length")
            return True
        return False

    def finish(self, reason: str) -> None:
        self.stage = RequestStage.FINISHED
        self.finish_reason = reason
        self.timings.finish = time.perf_counter()

    @property
    def is_finished(self) -> bool:
        return self.stage is RequestStage.FINISHED
