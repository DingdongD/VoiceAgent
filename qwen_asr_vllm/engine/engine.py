"""Engine main loop.

A step drains at most one audio-encode batch and one decoder batch. Because the
prefill backlog is capped, encoding stops running once the decoder is saturated,
so the two stages self-balance without an explicit pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch

from qwen_asr_vllm.audio.frontend import AudioFrontend
from qwen_asr_vllm.config import EngineConfig
from qwen_asr_vllm.engine.audio_runner import AudioRunner
from qwen_asr_vllm.engine.block_manager import BlockManager
from qwen_asr_vllm.engine.dual_stream import DualStreamExecutor
from qwen_asr_vllm.engine.model_runner import ModelRunner
from qwen_asr_vllm.engine.request import AsrRequest, RequestTimings, SamplingParams
from qwen_asr_vllm.engine.scheduler import ModelBatch, Scheduler
from qwen_asr_vllm.loader import load_weights
from qwen_asr_vllm.metrics import Metrics
from qwen_asr_vllm.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from qwen_asr_vllm.postprocess import parse_asr_output
from qwen_asr_vllm.profiling import StageTimer
from qwen_asr_vllm.prompt import PromptBuilder, normalize_language

logger = logging.getLogger(__name__)

# torch.cuda.OutOfMemoryError became torch.OutOfMemoryError in 2.5; the older name
# is kept as an alias, but not in every build we care about.
OutOfMemoryError = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)


def _set_current_cuda_device(device: str) -> None:
    """Align torch's process-local default with the configured engine device."""

    target = torch.device(device)
    if target.type == "cuda" and target.index is not None:
        torch.cuda.set_device(target)


@dataclass
class AsrOutput:
    request_id: int
    text: str
    language: str
    raw_text: str
    audio_seconds: float
    num_prompt_tokens: int
    num_audio_tokens: int
    num_output_tokens: int
    finish_reason: str | None
    timings: RequestTimings = field(default_factory=RequestTimings)
    # Token ids of the transcript body (stop tokens already stripped). Kept so a
    # speculative draft can be the previous request's output without re-tokenising
    # text, which would not round-trip through the chat template cleanly.
    output_token_ids: list[int] = field(default_factory=list)
    # Populated by ``transcribe_with_draft``: how much of the draft was accepted.
    draft_tokens: int = 0
    draft_accepted: int = 0


class AsrEngine:
    def __init__(self, model: str | None = None, config: EngineConfig | None = None, **kwargs):
        if (model is None) == (config is None):
            raise ValueError("pass exactly one of model or config")
        self.config = config or EngineConfig(model=model, **kwargs)
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Check that the active environment's torch build "
                "matches the installed driver."
            )
        _set_current_cuda_device(self.config.device)

        self._limit_frontend_threads()
        self.frontend = AudioFrontend(self.config.model)
        self.prompt_builder = PromptBuilder(self.config.model, self.config.model_config)
        self.model = self._build_model()

        self.model_runner = ModelRunner(self.model, self.config)
        num_blocks = self.model_runner.determine_num_blocks()
        self.model_runner.allocate_kv_cache(num_blocks)
        self.model_runner.capture_graphs()
        self.num_kvcache_blocks = num_blocks

        self.audio_runner = AudioRunner(self.model, self.config)
        self.block_manager = BlockManager(
            num_blocks, self.config.kvcache_block_size, self.config.enable_prefix_cache
        )
        self.scheduler = Scheduler(self.config, self.block_manager)
        # Wall-clock spent in each stage, so a benchmark can attribute cost without
        # reconstructing it from per-request timestamps.
        self.stage_seconds = {"frontend": 0.0, "audio_encode": 0.0, "model_step": 0.0}
        # Frontend work may run on worker threads, so its accumulator needs a lock.
        self._stage_lock = threading.Lock()
        self.metrics = Metrics()
        self.dual_stream: DualStreamExecutor | None = None
        if self.config.enable_dual_stream:
            self.dual_stream = DualStreamExecutor(device=self.config.device)
            logger.info(
                "dual-stream encode∥decode enabled (eager decode; overlap-eligible schedule)"
            )

        logger.info(
            "engine ready: model=%s kv_blocks=%d (%.1f GiB) max_num_seqs=%d max_model_len=%d",
            self.config.model,
            num_blocks,
            num_blocks * self.model_runner.kv_cache_bytes_per_block() / 2**30,
            self.config.max_num_seqs,
            self.config.max_model_len,
        )

    @property
    def frontend_seconds(self) -> float:
        return self.stage_seconds["frontend"]

    def _build_model(self) -> Qwen3ASRForConditionalGeneration:
        torch.set_default_dtype(self.config.dtype)
        try:
            with torch.device(self.config.device):
                model = Qwen3ASRForConditionalGeneration(
                    self.config.model_config, self.config.max_model_len
                )
        finally:
            torch.set_default_dtype(torch.float32)
        load_weights(model, self.config.model)
        return model.eval()

    # ---------------------------------------------------------------- requests

    def prepare_request(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        context: str = "",
        language: str | None = None,
        sampling: SamplingParams | None = None,
    ) -> AsrRequest:
        """Build a request without admitting it: mel features and prompt layout.

        Touches no engine state, so it is safe to run on a worker thread and thereby
        overlap the CPU frontend with GPU work. ``AsyncAsrEngine`` relies on that;
        the synchronous path just calls it inline.
        """
        started = time.perf_counter()
        features = self.frontend(waveform, sample_rate)
        canonical_language = normalize_language(language)
        layout = self.prompt_builder.build(
            features.num_audio_tokens, context=context, language=canonical_language
        )
        elapsed = time.perf_counter() - started
        with self._stage_lock:
            self.stage_seconds["frontend"] += elapsed

        if len(layout) > self.config.max_model_len:
            raise ValueError(
                f"prompt needs {len(layout)} tokens but max_model_len is "
                f"{self.config.max_model_len}; {features.audio_seconds:.1f}s of audio is too long"
            )

        return AsrRequest(
            features=features,
            layout=layout,
            sampling=sampling or SamplingParams(),
            block_size=self.config.kvcache_block_size,
            stop_token_ids=self.prompt_builder.stop_token_ids,
            language=canonical_language,
        )

    def admit(self, request: AsrRequest) -> AsrRequest:
        """Queue an already-prepared request. Must run on the stepping thread."""
        self.scheduler.add(request)
        self.metrics.record_received(request.features.audio_seconds)
        return request

    def add_request(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        context: str = "",
        language: str | None = None,
        sampling: SamplingParams | None = None,
    ) -> AsrRequest:
        return self.admit(
            self.prepare_request(waveform, sample_rate, context, language, sampling)
        )

    def _limit_frontend_threads(self) -> None:
        """Narrow torch's intra-op pool, which is oversized for the mel frontend.

        See ``EngineConfig.frontend_threads``. Only ever narrows: a caller who has
        already asked for fewer threads than we would has a reason, and the model
        itself runs on the GPU, so nothing here benefits from a wider pool.
        """
        wanted = self.config.frontend_threads
        current = torch.get_num_threads()
        if wanted <= 0 or current <= wanted:
            return
        torch.set_num_threads(wanted)
        logger.info(
            "narrowed torch intra-op threads from %d to %d for the mel frontend",
            current,
            wanted,
        )

    def enable_stage_profiling(self) -> StageTimer:
        """Time GPU phases with CUDA events. Returns the shared timer.

        Off by default: the phases sit in the per-batch path, and reading an event
        synchronises, which would destroy the CPU/GPU overlap being measured.
        """
        timer = StageTimer(enabled=True)
        self.model.audio_tower.timer = timer
        self.model_runner.timer = timer
        return timer

    def kv_bytes_for(self, num_tokens: int) -> int:
        """KV cache footprint of a context, in bytes."""
        text = self.config.model_config.text
        head_dim = text.head_dim or text.hidden_size // text.num_attention_heads
        element_size = torch.tensor([], dtype=self.config.dtype).element_size()
        return (
            2 * text.num_hidden_layers * num_tokens * text.num_key_value_heads * head_dim
        ) * element_size

    def cancel(self, request_id: int) -> bool:
        """Drop a request. Returns whether it was still in flight to be dropped.

        Counting happens in the layer the cancellation came from, not here, so a
        cancellation is never tallied twice.
        """
        return self.scheduler.cancel(request_id)

    def _finalize(self, request: AsrRequest) -> AsrOutput:
        output_ids = request.output_token_ids
        if output_ids and output_ids[-1] in request.stop_token_ids:
            output_ids = output_ids[:-1]
        raw_text = self.prompt_builder.decode(output_ids)
        language, text = parse_asr_output(raw_text, user_language=request.language)
        return AsrOutput(
            request_id=request.request_id,
            text=text,
            language=language,
            raw_text=raw_text,
            audio_seconds=request.features.audio_seconds,
            num_prompt_tokens=request.num_prompt_tokens,
            num_audio_tokens=request.layout.audio_length,
            num_output_tokens=len(output_ids),
            finish_reason=request.finish_reason,
            timings=request.timings,
            output_token_ids=list(output_ids),
        )

    def verify_draft(
        self,
        waveform: np.ndarray,
        draft_token_ids: list[int],
        sample_rate: int = 16000,
        context: str = "",
        language: str | None = None,
    ) -> tuple[int, list[int]]:
        """Run one speculative verify pass; return (accepted, predictions).

        Encodes the audio, prefills the prompt together with the draft in a single
        forward, and applies :func:`accepted_draft_length`. Does not admit the request
        into the scheduler and does not emit any further tokens — it is the measurement
        and integration primitive, not a full transcription.
        """
        from qwen_asr_vllm.engine.scheduler import ModelBatch
        from qwen_asr_vllm.engine.speculate import accepted_draft_length

        if not draft_token_ids:
            return 0, []

        request = self.prepare_request(
            waveform,
            sample_rate=sample_rate,
            context=context,
            language=language,
            # Cap of 1: this path never decodes; the sampling object is only required
            # so prepare_request has somewhere to hang the request's knobs.
            sampling=SamplingParams(max_new_tokens=1),
        )
        self.audio_runner.encode([request])
        for token_id in draft_token_ids:
            request.append_token(token_id)

        manager = self.scheduler.block_manager
        if not manager.can_allocate(request):
            raise RuntimeError(
                f"draft verify needs {request.num_blocks} KV blocks but only "
                f"{manager.num_free_blocks} are free"
            )
        manager.allocate(request)
        # Prefix-cache hits would shrink the query to the draft alone, dropping the
        # last prompt position that predicts the first draft token. Force a full
        # recompute so verify sees the n+1 positions its contract assumes.
        request.num_computed_tokens = 0
        try:
            predictions = self.model_runner.verify(
                ModelBatch(prefill=[request]), [len(draft_token_ids)]
            )[0]
        finally:
            manager.deallocate(request)
        return accepted_draft_length(draft_token_ids, predictions), predictions

    def transcribe_with_draft(
        self,
        waveform: np.ndarray,
        draft_token_ids: list[int] | None = None,
        sample_rate: int = 16000,
        context: str = "",
        language: str | None = None,
        sampling: SamplingParams | None = None,
    ) -> AsrOutput:
        """Transcribe using a greedy draft (previous transcript tokens).

        Verifies the draft in one forward, keeps the accepted prefix's KV, appends the
        bonus next token, then decodes only the residual. With an empty draft this is
        ordinary ``transcribe`` of one clip. Must run on the engine thread (the async
        wrapper serialises it onto the loop).
        """
        from qwen_asr_vllm.engine.scheduler import ModelBatch
        from qwen_asr_vllm.engine.speculate import accepted_draft_length

        draft = list(draft_token_ids or [])
        if sampling is None:
            sampling = SamplingParams.for_audio(len(waveform) / sample_rate)

        if not draft:
            return self.transcribe(
                [waveform],
                sample_rate=sample_rate,
                context=context,
                language=language,
                sampling=sampling,
            )[0]

        request = self.prepare_request(
            waveform,
            sample_rate=sample_rate,
            context=context,
            language=language,
            sampling=sampling,
        )
        self.audio_runner.encode([request])
        for token_id in draft:
            request.append_token(token_id)

        manager = self.block_manager
        if not manager.can_allocate(request):
            raise RuntimeError(
                f"draft transcribe needs {request.num_blocks} KV blocks but only "
                f"{manager.num_free_blocks} are free"
            )
        manager.allocate(request)
        request.num_computed_tokens = 0
        accepted = 0
        try:
            predictions = self.model_runner.verify(
                ModelBatch(prefill=[request]), [len(draft)]
            )[0]
            accepted = accepted_draft_length(draft, predictions)
            del request.token_ids[request.num_prompt_tokens + accepted :]
            request.num_computed_tokens = len(request)

            bonus = predictions[accepted]
            request.append_token(bonus)
            request.mark_prefilled()
            if not request.check_finished():
                manager.may_append(request)
                while not request.is_finished:
                    if not manager.can_append(request):
                        request.finish("aborted:out_of_kv_blocks")
                        break
                    token_id = self.model_runner.run(ModelBatch(decode=[request]))[0]
                    request.num_computed_tokens = len(request)
                    request.append_token(token_id)
                    if request.check_finished():
                        break
                    manager.may_append(request)

            if request.finish_reason is None:
                request.finish("stop")
            output = self._finalize(request)
            output.draft_tokens = len(draft)
            output.draft_accepted = accepted
            return output
        finally:
            manager.deallocate(request)
            request.audio_embeds = None

    # -------------------------------------------------------------------- loop

    def step(self) -> list[AsrOutput]:
        if self.dual_stream is not None:
            return self._step_dual_stream()
        return self._step_serial()

    def _emit_outputs(self, finished: list[AsrRequest]) -> list[AsrOutput]:
        outputs = [
            self._finalize(request) for request in self.scheduler.drain_aborted() + finished
        ]
        for output in outputs:
            self.metrics.record_finished(output)
            if output.finish_reason and output.finish_reason.startswith("aborted"):
                logger.warning(
                    "request %d aborted (%s) after %.2fs and %d tokens",
                    output.request_id,
                    output.finish_reason,
                    output.timings.total_seconds,
                    output.num_output_tokens,
                )
            elif output.finish_reason == "length":
                # Truncation loses transcript with no other signal. The default cap of
                # 440 tokens suits the ~30s clips it was chosen for and cuts a
                # 20-minute transcript to under a tenth, so this needs to be loud.
                logger.warning(
                    "request %d hit max_new_tokens=%d after %.1fs of audio; the "
                    "transcript is truncated. Raise max_new_tokens (roughly 8 per "
                    "audio second leaves headroom).",
                    output.request_id,
                    output.num_output_tokens,
                    output.audio_seconds,
                )
        return outputs

    def _step_serial(self) -> list[AsrOutput]:
        encode_batch = self.scheduler.schedule_audio()
        if encode_batch:
            started = time.perf_counter()
            try:
                self.audio_runner.encode(encode_batch)
                torch.cuda.synchronize()
            except OutOfMemoryError:
                logger.warning(
                    "audio encode ran out of memory on %d requests (%d mel frames); "
                    "rolling back and halving the frame cap",
                    len(encode_batch),
                    sum(request.features.mel_frames for request in encode_batch),
                )
                self._recover_from_oom()
                self.scheduler.on_audio_oom(encode_batch)
            else:
                self.scheduler.admit_prefill(encode_batch)
            self.stage_seconds["audio_encode"] += time.perf_counter() - started

        finished: list[AsrRequest] = []
        batch = self.scheduler.schedule_model()
        if batch:
            started = time.perf_counter()
            try:
                token_ids = self.model_runner.run(batch)
            except OutOfMemoryError:
                logger.warning(
                    "decoder step ran out of memory on %d requests (%d tokens); "
                    "rolling back and halving the batch cap",
                    len(batch.requests),
                    batch.num_tokens,
                )
                self._recover_from_oom()
                self.scheduler.on_model_oom(batch)
            else:
                finished = self.scheduler.postprocess(batch, token_ids)
                self.scheduler.relax_limits()
            self.stage_seconds["model_step"] += time.perf_counter() - started

        return self._emit_outputs(finished)

    def _step_dual_stream(self) -> list[AsrOutput]:
        """Overlap encode∥model when both batches are non-empty.

        Schedule order matches the Phase 1 bench: schedule audio and model first,
        execute (possibly overlapped), then admit newly encoded requests. That
        defers their prefill by one step versus ``_step_serial``.
        """
        assert self.dual_stream is not None
        encode_batch = self.scheduler.schedule_audio()
        batch: ModelBatch = self.scheduler.schedule_model()
        token_ids: list[int] | None = None
        finished: list[AsrRequest] = []

        def encode_fn() -> None:
            if encode_batch:
                self.audio_runner.encode(encode_batch)

        def decode_fn() -> None:
            nonlocal token_ids
            if batch:
                token_ids = self.model_runner.run(batch)

        started = time.perf_counter()
        try:
            if encode_batch and batch:
                self.dual_stream.run_overlap(encode_fn, decode_fn)
            else:
                self.dual_stream.run_serial(encode_fn, decode_fn)
        except OutOfMemoryError:
            logger.warning(
                "dual-stream step ran out of memory (encode=%d model=%d); rolling back",
                len(encode_batch),
                len(batch.requests) if batch else 0,
            )
            self._recover_from_oom()
            if encode_batch:
                self.scheduler.on_audio_oom(encode_batch)
            if batch:
                self.scheduler.on_model_oom(batch)
            return self._emit_outputs([])

        elapsed = time.perf_counter() - started
        if encode_batch:
            self.scheduler.admit_prefill(encode_batch)
            self.stage_seconds["audio_encode"] += elapsed
        if batch and token_ids is not None:
            finished = self.scheduler.postprocess(batch, token_ids)
            self.scheduler.relax_limits()
            self.stage_seconds["model_step"] += elapsed

        return self._emit_outputs(finished)

    @staticmethod
    def _recover_from_oom() -> None:
        """Return the failed pass's activations to the driver before retrying.

        Without this the allocator keeps the freed-but-cached blocks, so a retry at
        half the batch size can hit the same wall.
        """
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def transcribe(
        self,
        audios: Sequence[np.ndarray] | Iterable[np.ndarray],
        sample_rate: int = 16000,
        context: str = "",
        language: str | None = None,
        sampling: SamplingParams | None = None,
        show_progress: bool = False,
    ) -> list[AsrOutput]:
        """Transcribe a batch, preserving input order.

        Requests are admitted a few batches ahead of the scheduler instead of all
        at once: mel features live in host memory until their request is encoded,
        and a whole dataset's worth is a lot of memory to hold for no benefit.
        """
        source = iter(audios)
        admission_limit = max(self.config.max_num_seqs * 4, self.config.max_audio_batch_size)
        order: dict[int, int] = {}
        results: dict[int, AsrOutput] = {}
        pending: set[int] = set()
        exhausted = False
        next_index = 0

        progress = None
        if show_progress:
            from tqdm import tqdm

            total = len(audios) if hasattr(audios, "__len__") else None
            progress = tqdm(total=total, unit="clip", desc="transcribe")

        try:
            while pending or not exhausted:
                while not exhausted and self.scheduler.num_in_flight < admission_limit:
                    try:
                        waveform = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    request = self.add_request(
                        waveform,
                        sample_rate=sample_rate,
                        context=context,
                        language=language,
                        sampling=sampling,
                    )
                    order[request.request_id] = next_index
                    pending.add(request.request_id)
                    next_index += 1

                if not self.scheduler.has_work:
                    if pending:
                        raise RuntimeError(
                            f"{len(pending)} requests are unaccounted for but the scheduler has "
                            "no work left"
                        )
                    continue

                # A step that neither ran a batch nor returned anything means the
                # scheduler has work it will never make progress on. Hanging on that
                # is worse than failing on it. Encode-only steps (dual-stream admits
                # after scheduling model) advance num_encode_batches without model work.
                before_model = self.scheduler.stats.num_model_steps
                before_encode = self.scheduler.stats.num_encode_batches
                outputs = self.step()
                if (
                    not outputs
                    and self.scheduler.stats.num_model_steps == before_model
                    and self.scheduler.stats.num_encode_batches == before_encode
                ):
                    raise RuntimeError(
                        f"scheduler is not making progress with {self.scheduler.num_in_flight} "
                        "requests in flight; this is an engine bug"
                    )

                for output in outputs:
                    # Outputs can belong to an earlier call that raised partway
                    # through admission; those are not ours to return.
                    index = order.get(output.request_id)
                    if index is None:
                        continue
                    results[index] = output
                    pending.discard(output.request_id)
                    if progress is not None:
                        progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        return [results[index] for index in range(next_index)]
