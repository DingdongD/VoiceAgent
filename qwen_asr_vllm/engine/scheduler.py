"""Three-stage scheduling with mixed prefill/decode batches.

Two departures from the usual nano-vLLM shape:

*Audio encoding is its own scheduling stage.* Several requests are packed into one
encoder call, so a batch is no longer held hostage by whichever recording happens
to be longest.

*Prefill and decode share a batch.* A step is not "all prefill" or "all decode".
Because the attention path is a single varlen kernel where decode is just a
query length of one, a newly arrived request can be prefilled in the same forward
pass that advances everything already decoding, instead of making them idle for a
step.

Requests already decoding are admitted first each step so that arrivals cannot
starve work in flight.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from qwen_asr_vllm.config import EngineConfig
from qwen_asr_vllm.engine.block_manager import BlockManager
from qwen_asr_vllm.engine.request import AsrRequest, RequestStage


@dataclass
class ModelBatch:
    """Requests sharing one decoder forward pass."""

    prefill: list[AsrRequest] = field(default_factory=list)
    decode: list[AsrRequest] = field(default_factory=list)

    @property
    def requests(self) -> list[AsrRequest]:
        return self.prefill + self.decode

    @property
    def num_tokens(self) -> int:
        return sum(request.num_tokens_to_compute for request in self.requests)

    def __bool__(self) -> bool:
        return bool(self.prefill or self.decode)


@dataclass
class SchedulerStats:
    num_encode_batches: int = 0
    num_model_steps: int = 0
    num_mixed_steps: int = 0
    num_preemptions: int = 0
    num_encoded_requests: int = 0
    num_prefilled_tokens: int = 0
    num_decoded_tokens: int = 0
    num_model_ooms: int = 0
    num_audio_ooms: int = 0


class Scheduler:
    def __init__(self, config: EngineConfig, block_manager: BlockManager):
        self.config = config
        self.block_manager = block_manager
        self.waiting_encode: deque[AsrRequest] = deque()
        self.waiting_prefill: deque[AsrRequest] = deque()
        self.running: deque[AsrRequest] = deque()
        self.aborted: list[AsrRequest] = []
        # Every in-flight request by id, so cancellation is a lookup rather than a
        # scan of three queues -- and so a request already inside the batch being
        # executed can still be found.
        self.tracked: dict[int, AsrRequest] = {}
        self.stats = SchedulerStats()
        # Adaptive caps, tightened when a forward pass runs out of memory and
        # relaxed again once one succeeds. Sizing the KV cache from a profiling run
        # gets the steady state right, but it cannot foresee allocator
        # fragmentation or another tenant growing on the same card.
        self.model_batch_limit = config.max_num_seqs
        self.audio_frame_limit = config.max_audio_batch_frames

    def add(self, request: AsrRequest) -> None:
        self.waiting_encode.append(request)
        self.tracked[request.request_id] = request

    def admit_prefill(self, requests: list[AsrRequest]) -> None:
        """Hand encoded requests to the prefill queue."""
        self.waiting_prefill.extend(requests)

    def drain_aborted(self) -> list[AsrRequest]:
        aborted, self.aborted = self.aborted, []
        return aborted

    def cancel(self, request_id: int) -> bool:
        """Drop a request wherever it currently sits. Returns whether it was found.

        A request already inside the batch being executed cannot be pulled out of a
        running kernel, so it is flagged instead and dropped when that step lands.
        Either way the caller gets an output carrying ``finish_reason="cancelled"``,
        which keeps delivery uniform with every other terminal state.
        """
        request = self.tracked.get(request_id)
        if request is None:
            return False
        for queue in (self.waiting_encode, self.waiting_prefill, self.running):
            try:
                queue.remove(request)
            except ValueError:
                continue
            self._give_up(request, "cancelled")
            return True
        request.cancel_requested = True
        return True

    @property
    def has_work(self) -> bool:
        return bool(self.waiting_encode or self.waiting_prefill or self.running)

    @property
    def num_in_flight(self) -> int:
        return len(self.waiting_encode) + len(self.waiting_prefill) + len(self.running)

    def schedule_audio(self) -> list[AsrRequest]:
        """Pick requests for one packed encoder call.

        The encoded backlog is capped: audio embeddings sit in GPU memory until
        their request is prefilled, so running far ahead of the decoder would just
        trade throughput for memory.
        """
        backlog_capacity = self.config.max_num_seqs - len(self.waiting_prefill)
        if backlog_capacity <= 0 or not self.waiting_encode:
            return []

        selected: list[AsrRequest] = []
        frames = 0
        while self.waiting_encode and len(selected) < min(
            backlog_capacity, self.config.max_audio_batch_size
        ):
            request = self.waiting_encode[0]
            request_frames = request.features.mel_frames
            if selected and frames + request_frames > self.audio_frame_limit:
                break
            self.waiting_encode.popleft()
            request.timings.encode_start = time.perf_counter()
            selected.append(request)
            frames += request_frames

        if selected:
            self.stats.num_encode_batches += 1
            self.stats.num_encoded_requests += len(selected)
        return selected

    def _preempt(self, request: AsrRequest) -> None:
        """Return a decoding request to the prefill queue to free its blocks.

        Its audio embeddings survive, so preemption never costs a re-encode --
        only the text prefill is repeated.
        """
        self.block_manager.deallocate(request)
        request.num_computed_tokens = 0
        request.stage = RequestStage.WAITING_PREFILL
        self.waiting_prefill.appendleft(request)
        self.stats.num_preemptions += 1

    def schedule_model(self) -> ModelBatch:
        batch = ModelBatch()

        # In-flight decodes go first; each needs at most one new block.
        while self.running:
            request = self.running[0]
            if self.block_manager.can_append(request):
                self.running.popleft()
                batch.decode.append(request)
                continue
            if len(self.running) > 1:
                self._preempt(self.running.pop())
                continue
            # A single request that cannot grow means the cache is exhausted.
            self.running.popleft()
            request.finish("aborted:out_of_kv_blocks")
            self.block_manager.deallocate(request)
            request.audio_embeds = None
            self.aborted.append(request)
            break

        token_budget = self.config.max_num_batched_tokens - len(batch.decode)
        seq_budget = self.model_batch_limit - len(batch.decode)

        while self.waiting_prefill and seq_budget > 0:
            request = self.waiting_prefill[0]
            # len(request) is an upper bound on the query length: a prefix-cache
            # hit can only shrink it. Checking the bound first avoids allocating
            # blocks only to hand them straight back.
            if len(request) > token_budget:
                break
            if not self.block_manager.can_ever_allocate(request):
                # Waiting cannot help: the request needs more blocks than the cache
                # holds. Leaving it queued would keep has_work true forever while
                # every batch came back empty, which is a hang rather than an error.
                self.waiting_prefill.popleft()
                request.finish("aborted:kv_cache_too_small")
                request.audio_embeds = None
                self.aborted.append(request)
                continue
            if not self.block_manager.can_allocate(request):
                break
            self.waiting_prefill.popleft()
            self.block_manager.allocate(request)
            batch.prefill.append(request)
            token_budget -= request.num_tokens_to_compute
            seq_budget -= 1

        if batch:
            self.stats.num_model_steps += 1
            if batch.prefill and batch.decode:
                self.stats.num_mixed_steps += 1
            self.stats.num_prefilled_tokens += sum(r.num_tokens_to_compute for r in batch.prefill)
            self.stats.num_decoded_tokens += len(batch.decode)
        return batch

    # ------------------------------------------------------------ oom recovery

    def _give_up(self, request: AsrRequest, reason: str) -> None:
        self.block_manager.deallocate(request)
        request.num_computed_tokens = 0
        request.audio_embeds = None
        request.finish(reason)
        self.tracked.pop(request.request_id, None)
        self.aborted.append(request)

    def on_model_oom(self, batch: ModelBatch) -> None:
        """Undo a decoder batch whose forward pass ran out of memory.

        Nothing in the batch has been mutated yet -- tokens are appended and blocks
        extended in ``postprocess``, which never ran -- so the batch can be put back
        exactly as it was and retried under a tighter cap. A batch of one that still
        fails has nowhere left to shrink, so that request is given up on.
        """
        self.stats.num_model_ooms += 1
        attempted = len(batch.requests)

        for request in reversed(batch.prefill):
            self.block_manager.deallocate(request)
            request.num_computed_tokens = 0
            self.waiting_prefill.appendleft(request)
        for request in reversed(batch.decode):
            self.running.appendleft(request)

        if attempted <= 1:
            self.model_batch_limit = 1
            if self.running:
                self._give_up(self.running.pop(), "aborted:out_of_memory")
            elif self.waiting_prefill:
                self._give_up(self.waiting_prefill.popleft(), "aborted:out_of_memory")
            return
        self.model_batch_limit = max(1, attempted // 2)

    def on_audio_oom(self, requests: list[AsrRequest]) -> None:
        """Undo an encoder batch that ran out of memory, tightening the frame cap."""
        self.stats.num_audio_ooms += 1
        frames = sum(request.features.mel_frames for request in requests)

        for request in reversed(requests):
            request.timings.encode_start = None
            request.audio_embeds = None
            self.waiting_encode.appendleft(request)

        if len(requests) <= 1:
            # One recording alone does not fit; no cap can rescue it.
            self._give_up(self.waiting_encode.popleft(), "aborted:out_of_memory")
            return
        self.audio_frame_limit = max(1, frames // 2)

    def relax_limits(self) -> None:
        """Walk the caps back toward configured values after a successful step."""
        if self.model_batch_limit < self.config.max_num_seqs:
            self.model_batch_limit = min(self.config.max_num_seqs, self.model_batch_limit * 2)
        if self.audio_frame_limit < self.config.max_audio_batch_frames:
            self.audio_frame_limit = min(
                self.config.max_audio_batch_frames, self.audio_frame_limit * 2
            )

    def postprocess(self, batch: ModelBatch, token_ids: list[int]) -> list[AsrRequest]:
        """Record sampled tokens and advance stages; return finished requests."""
        requests = batch.requests
        if len(requests) != len(token_ids):
            raise ValueError(
                f"sampler returned {len(token_ids)} tokens for {len(requests)} requests"
            )

        finished: list[AsrRequest] = []
        for request, token_id in zip(requests, token_ids):
            request.num_computed_tokens = len(request)
            request.append_token(token_id)
            if request.stage is RequestStage.WAITING_PREFILL:
                request.mark_prefilled()

            if request.cancel_requested:
                request.finish("cancelled")
            elif not request.check_finished():
                self.block_manager.may_append(request)
                self.running.append(request)
                continue

            self.block_manager.deallocate(request)
            request.audio_embeds = None
            self.tracked.pop(request.request_id, None)
            finished.append(request)
        return finished
