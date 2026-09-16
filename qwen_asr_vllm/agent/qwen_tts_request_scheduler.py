"""Cooperative outer/inner codec scheduling owned by one service thread."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import queue
import threading
import time
import os
from contextlib import contextmanager

from transformers.cache_utils import Cache, CacheLayerMixin

from .qwen_tts_outer_static_engine import (
    PreparedOuterDecode, OuterStepOutput, _run_outer_decode_body,
    build_decode_position_ids, select_text_condition,
)
from .qwen_tts_streaming import _CodecFrameBuffer, _decode_codec_chunk


class _PrepareComplete(BaseException):
    def __init__(self, kwargs):
        self.kwargs = kwargs


@contextmanager
def _inner_compute_policy(precision, *, sdp_backend="default", deterministic=False):
    """Apply explicit inner GEMM/SDPA policy only around predictor work."""
    import torch

    if sdp_backend not in {"default", "math"}:
        raise ValueError("inner SDPA backend must be default or math")

    old_precision = torch.get_float32_matmul_precision()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_sdp = (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )
    torch.set_float32_matmul_precision(precision)
    # TF32 changes the reduction path for BF16/FP16 projections and can alter
    # greedy argmax at a codec step. Keep the setting local to this scheduler.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sdp_backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    if deterministic:
        torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.backends.cuda.enable_flash_sdp(old_sdp[0])
        torch.backends.cuda.enable_mem_efficient_sdp(old_sdp[1])
        torch.backends.cuda.enable_math_sdp(old_sdp[2])
        torch.use_deterministic_algorithms(old_deterministic)


class _JoinedLayer(CacheLayerMixin):
    is_compileable = False
    is_sliding = False

    def __init__(self, layers):
        super().__init__()
        self.sources = layers
        if not layers or not all(getattr(layer, "is_initialized", False) for layer in layers):
            raise ValueError("joined cache requires initialized source layers")
        first = layers[0].keys
        import torch
        shape = (len(layers), *first.shape[1:])
        self.joined_keys = torch.empty(shape, dtype=first.dtype, device=first.device)
        self.joined_values = torch.empty_like(self.joined_keys)
        # A joined cohort is stable across decode ticks. Populate each row once
        # on first use, then copy only the KV positions appended by that tick.
        self._copied_lengths = [0] * len(layers)
        self._aliased = [False] * len(layers)
        self._saved_backing = [None] * len(layers)
        self.copy_bytes = 0

    def lazy_initialization(self, key_states):
        raise RuntimeError("joined cache requires initialized request caches")

    def get_seq_length(self):
        return self.sources[0].get_seq_length()

    def get_mask_sizes(self, cache_position):
        return self.get_seq_length() + cache_position.numel(), 0

    def get_max_cache_shape(self):
        return self.sources[0].get_max_cache_shape()

    def _detach_row(self, index):
        if not self._aliased[index]:
            return
        source = self.sources[index]
        saved = self._saved_backing[index]
        if getattr(source, "_released", False):
            source._qav_joined_owner = None
        else:
            if saved is None:
                raise RuntimeError("aliased cache row has no saved backing")
            saved_keys, saved_values = saved
            length = source.get_seq_length()
            if length:
                saved_keys[:, :, :length].copy_(
                    self.joined_keys[index:index + 1, :, :length]
                )
                saved_values[:, :, :length].copy_(
                    self.joined_values[index:index + 1, :, :length]
                )
                self.copy_bytes += int(
                    2 * saved_keys[:, :, :length].numel()
                    * saved_keys.element_size()
                )
            source.keys = saved_keys
            source.values = saved_values
            source.is_initialized = True
            source._qav_joined_owner = None
        self._aliased[index] = False
        self._saved_backing[index] = None

    def detach_source(self, source):
        """Release a source row before another pool aliases the same cache."""
        for index, current in enumerate(self.sources):
            if current is source:
                self._detach_row(index)

    def _alias_row(self, index, source):
        owner = getattr(source, "_qav_joined_owner", None)
        if owner is not None and owner is not self:
            owner.detach_source(source)
        if not getattr(source, "is_initialized", False):
            raise RuntimeError("slot-backed cache requires initialized source layer")
        if source.keys is None or source.values is None:
            raise RuntimeError("slot-backed cache source has no backing storage")
        saved = (source.keys, source.values)
        length = source.get_seq_length()
        if length:
            self.joined_keys[index:index + 1, :, :length].copy_(
                source.keys[:, :, :length]
            )
            self.joined_values[index:index + 1, :, :length].copy_(
                source.values[:, :, :length]
            )
            self.copy_bytes += int(
                2 * source.keys[:, :, :length].numel()
                * source.keys.element_size()
            )
        source.keys = self.joined_keys[index:index + 1]
        source.values = self.joined_values[index:index + 1]
        source.is_initialized = True
        source._qav_joined_owner = self
        self._saved_backing[index] = saved
        self._aliased[index] = True
        self._copied_lengths[index] = length

    def bind(self, layers, *, reset_copied_lengths=False, alias_sources=False):
        if len(layers) != len(self.sources):
            raise ValueError("joined cache row count cannot change")
        if reset_copied_lengths:
            self._copied_lengths = [0] * len(self.sources)
        for index, layer in enumerate(layers):
            if self.sources[index] is layer and (
                not alias_sources or self._aliased[index]
            ):
                continue
            if self.sources[index] is not layer:
                self._detach_row(index)
            self.sources[index] = layer
            self._copied_lengths[index] = 0
            if alias_sources:
                self._alias_row(index, layer)

    def update(self, keys, values, cache_kwargs=None):
        for i, layer in enumerate(self.sources):
            layer.update(keys[i:i + 1], values[i:i + 1], cache_kwargs)
            length = layer.get_seq_length()
            start = self._copied_lengths[i]
            if start > length:
                raise RuntimeError("joined cache source length regressed")
            if self._aliased[i]:
                self._copied_lengths[i] = length
                continue
            if start < length:
                self.joined_keys[i:i + 1, :, start:length].copy_(
                    layer.keys[:, :, start:length]
                )
                self.joined_values[i:i + 1, :, start:length].copy_(
                    layer.values[:, :, start:length]
                )
                self.copy_bytes += int(
                    (self.joined_keys[i:i + 1, :, start:length].numel()
                     + self.joined_values[i:i + 1, :, start:length].numel())
                    * self.joined_keys.element_size()
                )
                self._copied_lengths[i] = length
        length = self.sources[0].get_seq_length()
        return self.joined_keys[:, :, :length], self.joined_values[:, :, :length]


def joined_cache(cohorts):
    lengths = {c.cache_position for c in cohorts}
    if len(lengths) != 1:
        raise ValueError("outer batch requires matching KV lengths")
    return Cache(layers=[
        _JoinedLayer([c.cache.layers[i] for c in cohorts])
        for i in range(len(cohorts[0].cache.layers))
    ])


class _FixedSlotJoinedCachePool:
    """Keep a fixed-size joined cache with stable request-id physical rows."""

    def __init__(self, entries):
        if not entries:
            raise ValueError("fixed slot pool requires at least one cohort")
        self.capacity = len(entries)
        self.request_to_slot = {}
        self.slot_to_request = [None] * self.capacity
        self.last_reused_rows = 0
        self.last_new_rows = 0
        self.cache = joined_cache([cohort for _request_id, cohort in entries])
        self.bind(entries)

    def bind(self, entries):
        if len(entries) != self.capacity:
            raise ValueError("fixed slot pool requires a stable batch size")
        request_ids = [int(request_id) for request_id, _cohort in entries]
        cohorts = [cohort for _request_id, cohort in entries]
        if len({c.cache_position for c in cohorts}) != 1:
            raise ValueError("fixed slot pool requires matching KV lengths")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("fixed slot pool requires unique request ids")
        incoming = dict(zip(request_ids, cohorts))
        previous_slots = list(self.slot_to_request)
        previous_request_to_slot = {
            request_id: slot
            for slot, request_id in enumerate(previous_slots)
            if request_id is not None
        }
        # Retain the physical row of every request that remains in the cohort.
        # Newly admitted requests are assigned only to rows released by a
        # completed request. This is the ownership boundary that makes the
        # joined tensor a real slot-backed cache instead of an allocation pool.
        next_slots = [None] * self.capacity
        retained = set(request_ids) & set(previous_request_to_slot)
        for request_id in retained:
            next_slots[previous_request_to_slot[request_id]] = request_id
        free_slots = [slot for slot, request_id in enumerate(next_slots)
                      if request_id is None]
        for request_id, slot in zip(
            (request_id for request_id in request_ids if request_id not in retained),
            free_slots,
        ):
            next_slots[slot] = request_id
        if any(request_id is None for request_id in next_slots):
            raise RuntimeError("fixed slot pool failed to assign every request")
        self.slot_to_request = next_slots
        self.request_to_slot = {
            request_id: slot for slot, request_id in enumerate(next_slots)
        }
        self.last_reused_rows = len(retained)
        self.last_new_rows = self.capacity - self.last_reused_rows
        ordered = [(request_id, incoming[request_id])
                   for request_id in self.slot_to_request]
        for layer_index, joined_layer in enumerate(self.cache.layers):
            joined_layer.bind(
                [cohort.cache.layers[layer_index] for _request_id, cohort in ordered],
                reset_copied_lengths=False,
                alias_sources=True,
            )
        self.ordered_entries = ordered
        return ordered


@dataclass
class _Request:
    request_id: int
    iterator: object
    pending: dict
    frames: _CodecFrameBuffer = field(default_factory=_CodecFrameBuffer)
    decoded: int = 0
    chunks: int = 0


class RequestIdCodecScheduler:
    """Batch inner steps across requests and outer steps by active KV length.

    KV joining uses fixed-size request-to-slot pools and copies only newly
    appended positions for requests that retain their physical row. It is
    still an explicit adapter, not paged attention or a claimed zero-copy
    fast path.
    """

    def __init__(self, backend, emit, *, is_cancelled, max_batch_size,
                 on_metrics=None):
        if (not backend._streaming or not backend._outer_active_prefix_talker_engine
                or not backend._cuda_graph_code_predictor):
            raise ValueError("request scheduler requires codec-step, active-prefix and inner CUDA Graph")
        parameters = backend._generation_parameters()
        if parameters.get("do_sample", True) or parameters.get("subtalker_dosample", True):
            raise ValueError("request scheduler requires greedy outer and inner decoding")
        self.backend = backend
        self.talker = backend._model.model.talker
        self.engine = self.talker.generate._qav_outer_engine
        self.predictor_engine = self.talker.code_predictor.generate._qav_engine
        self.strict_inner_parity = os.environ.get(
            "VOICE_TTS_STRICT_INNER_PARITY", "1"
        ) == "1"
        self.verify_inner_batch = os.environ.get(
            "VOICE_TTS_VERIFY_INNER_BATCH_PARITY", "0"
        ) == "1"
        self.inner_matmul_precision = os.environ.get(
            "VOICE_TTS_INNER_MATMUL_PRECISION", "high"
        ).lower()
        if self.inner_matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError(
                "VOICE_TTS_INNER_MATMUL_PRECISION must be highest, high, or medium"
            )
        self.inner_sdp_backend = os.environ.get(
            "VOICE_TTS_INNER_SDP_BACKEND", "default"
        ).lower()
        if self.inner_sdp_backend not in {"default", "math"}:
            raise ValueError("VOICE_TTS_INNER_SDP_BACKEND must be default or math")
        self.inner_deterministic = os.environ.get(
            "VOICE_TTS_INNER_DETERMINISTIC", "0"
        ) == "1"
        self.emit = emit
        self.cancelled = is_cancelled
        self.capacity = int(max_batch_size)
        if self.capacity <= 0:
            raise ValueError("max_batch_size must be positive")
        self.on_metrics = on_metrics
        self.queue = queue.Queue(maxsize=256)
        self.active = {}
        self._slot_pools = {}
        self._cohort_cursor = 0
        self.closed = False
        self.stats = dict(slot_batches=[], inner_scalar_steps=0,
                          inner_parity_checks=0, inner_parity_failures=0,
                          inner_first_logit_diff_step=None,
                          inner_first_logit_max_abs_diff=None,
                          inner_cache_match=None,
                          inner_first_hidden_diff_step=None,
                          inner_first_hidden_max_abs_diff=None,
                          outer_slot_batches=[], queue_wait_ms=[], prefill_ms=[],
                          joined_cache_allocations=0, joined_kv_copy_bytes=0,
                          slot_pool_binds=0, slot_reused_rows=0, slot_new_rows=0,
                          inner_matmul_precision=self.inner_matmul_precision,
                          inner_sdp_backend=self.inner_sdp_backend,
                          inner_deterministic=self.inner_deterministic,
                          completed_codec_sha256={}, errors=[], cancellations=0)
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name="tts-request-codec-scheduler")
        self.thread.start()

    def submit(self, request_id, text):
        if self.closed:
            raise RuntimeError("request scheduler is closed")
        self.queue.put_nowait((request_id, text, time.perf_counter()))

    def close(self):
        self.closed = True
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("request scheduler did not stop")

    def _prepare(self, text):
        original = self.talker.generate

        def capture(**kwargs):
            raise _PrepareComplete(kwargs)

        self.talker.generate = capture
        try:
            self.backend._run_custom_voice_generate_only(text)
        except _PrepareComplete as prepared:
            return self.engine.iterate(**prepared.kwargs)
        finally:
            self.talker.generate = original
        raise RuntimeError("Qwen wrapper did not provide outer generation inputs")

    def _admit(self, item):
        request_id, text, submitted = item
        iterator = None
        try:
            if self.cancelled(request_id):
                self.emit("done", request_id, None)
                return
            admitted = time.perf_counter()
            iterator = self._prepare(text)
            pending = next(iterator)
            self.active[request_id] = _Request(request_id, iterator, pending)
            self.stats["queue_wait_ms"].append((admitted - submitted) * 1000)
            self.stats["prefill_ms"].append((time.perf_counter() - admitted) * 1000)
        except StopIteration:
            self.emit("done", request_id, None)
        except BaseException as exc:
            if iterator is not None:
                iterator.close()
            self.emit("error", request_id, str(exc))

    def _run(self):
        import torch
        with torch.cuda.device(self.backend._model.model.device), torch.inference_mode():
            try:
                while not self.closed:
                    if not self.active:
                        try:
                            self._admit(self.queue.get(timeout=0.05))
                        except queue.Empty:
                            continue
                    while len(self.active) < self.capacity:
                        try:
                            self._admit(self.queue.get_nowait())
                        except queue.Empty:
                            break
                    for request in list(self.active.values()):
                        if self.cancelled(request.request_id):
                            self.stats["cancellations"] += 1
                            self._finish(request, cancelled=True)
                    if self.active:
                        self._step()
                    if self.on_metrics:
                        self.on_metrics(deepcopy(self.stats))
            except BaseException as exc:
                self.stats["errors"].append(str(exc))
                for request in list(self.active.values()):
                    self.emit("error", request.request_id, str(exc))
                self.closed = True
            finally:
                for request in self.active.values():
                    request.iterator.close()
                self.active.clear()
                self._slot_pools.clear()
                while True:
                    try:
                        request_id, _, _ = self.queue.get_nowait()
                    except queue.Empty:
                        break
                    self.emit("error", request_id, "request scheduler stopped")
                if self.on_metrics:
                    self.on_metrics(deepcopy(self.stats))

    def _select_step_cohort(self):
        """Select one fair cohort with a common outer cache position.

        Advancing every active request in one tick makes prompt-length skew
        permanent: requests then keep different KV lengths and outer batching
        collapses to batch-1. Advancing one cohort at a time lets the shorter
        request catch up while rotating equal-sized cohorts to avoid starvation.
        """

        grouped = {}
        for request in self.active.values():
            pending = request.pending
            cohort = pending["cohort"]
            key = (int(cohort.cache_position), bool(pending["output_hidden_states"]))
            grouped.setdefault(key, []).append(request)
        if not grouped:
            return []

        groups = list(grouped.values())
        # A position-skewed pair must advance the shortest prefix first. If
        # the longer prefix wins, both requests keep different KV lengths and
        # the outer model can never form a stable cohort. Once the minimum
        # position is selected, prefer the largest group and rotate ties.
        minimum_position = min(
            int(group[0].pending["cohort"].cache_position) for group in groups
        )
        minimum_groups = [
            group for group in groups
            if int(group[0].pending["cohort"].cache_position) == minimum_position
        ]
        largest = max(len(group) for group in minimum_groups)
        start = self._cohort_cursor % len(minimum_groups)
        selected_index = None
        for offset in range(len(minimum_groups)):
            index = (start + offset) % len(minimum_groups)
            if len(minimum_groups[index]) == largest:
                selected_index = index
                break
        if selected_index is None:
            raise RuntimeError("request scheduler could not select a cache cohort")
        self._cohort_cursor = (selected_index + 1) % len(minimum_groups)
        return minimum_groups[selected_index]

    def _bind_slot_cohort(self, requests):
        output_hidden_states = bool(requests[0].pending["output_hidden_states"])
        pool_key = (len(requests), output_hidden_states)
        pool = self._slot_pools.get(pool_key)
        entries = [
            (request.request_id, request.pending["cohort"])
            for request in requests
        ]
        if pool is None:
            pool = _FixedSlotJoinedCachePool(entries)
            self._slot_pools[pool_key] = pool
            self.stats["joined_cache_allocations"] += 1
            ordered_entries = pool.ordered_entries
        else:
            ordered_entries = pool.bind(entries)
        self._drain_joined_copy_bytes()
        self.stats["slot_pool_binds"] += 1
        self.stats["slot_reused_rows"] += pool.last_reused_rows
        self.stats["slot_new_rows"] += pool.last_new_rows
        requests_by_id = {request.request_id: request for request in requests}
        return pool, [
            requests_by_id[request_id] for request_id, _cohort in ordered_entries
        ]

    def _drain_joined_copy_bytes(self):
        copied = 0
        for pool in self._slot_pools.values():
            for layer in pool.cache.layers:
                copied += layer.copy_bytes
                layer.copy_bytes = 0
        self.stats["joined_kv_copy_bytes"] += copied

    def _step(self):
        import torch
        requests = self._select_step_cohort()
        pool, requests = self._bind_slot_cohort(requests)
        first_ids = torch.cat([r.pending["first_ids"] for r in requests])
        past = torch.cat([r.pending["cohort"].requests[0].past_hidden for r in requests])
        # This owner already batches requests; bypass the independent inner queue.
        p = requests[0].pending
        if self.strict_inner_parity:
            remaining = []
            for request in requests:
                row = request.pending
                row_past = row["cohort"].requests[0].past_hidden
                row_first = row["first_ids"]
                result = self.predictor_engine.generate(
                    inputs_embeds=torch.cat(
                        (row_past, self.talker.get_input_embeddings()(row_first)), dim=1
                    ),
                    max_new_tokens=self.talker.config.num_code_groups - 1,
                    do_sample=False, top_p=row["subtalker_top_p"],
                    top_k=row["subtalker_top_k"],
                    temperature=row["subtalker_temperature"],
                    output_hidden_states=True, return_dict_in_generate=True,
                )
                remaining.append(result.sequences)
            self.stats["inner_scalar_steps"] += len(requests)
            codes = torch.cat(
                [torch.cat((request.pending["first_ids"], tail), dim=-1)
                 for request, tail in zip(requests, remaining)], dim=0
            )
        else:
            with _inner_compute_policy(
                self.inner_matmul_precision,
                sdp_backend=self.inner_sdp_backend,
                deterministic=self.inner_deterministic,
            ):
                result = self.predictor_engine.generate(
                    inputs_embeds=torch.cat((past, self.talker.get_input_embeddings()(first_ids)), dim=1),
                    max_new_tokens=self.talker.config.num_code_groups - 1,
                    do_sample=False, top_p=p["subtalker_top_p"], top_k=p["subtalker_top_k"],
                    temperature=p["subtalker_temperature"], output_hidden_states=True,
                    return_dict_in_generate=True,
                )
            if self.verify_inner_batch and len(requests) > 1:
                self.stats["inner_parity_checks"] += 1
                scalar_rows = []
                for request in requests:
                    row = request.pending
                    with _inner_compute_policy(
                        self.inner_matmul_precision,
                        sdp_backend=self.inner_sdp_backend,
                        deterministic=self.inner_deterministic,
                    ):
                        scalar = self.predictor_engine.generate(
                            inputs_embeds=torch.cat(
                                (row["cohort"].requests[0].past_hidden,
                                 self.talker.get_input_embeddings()(row["first_ids"])),
                                dim=1,
                            ),
                            max_new_tokens=self.talker.config.num_code_groups - 1,
                            do_sample=False, top_p=row["subtalker_top_p"],
                            top_k=row["subtalker_top_k"],
                            temperature=row["subtalker_temperature"],
                            output_hidden_states=True, return_dict_in_generate=True,
                        )
                    scalar_rows.append(scalar.sequences)
                scalar = torch.cat(scalar_rows, dim=0)
                if os.environ.get("VOICE_TTS_TRACE_INNER_LOGITS", "0") == "1":
                    traces = self.predictor_engine.metrics_snapshot().get("trace_logits", [])
                    if len(traces) >= len(requests) + 1:
                        batch_trace = traces[-(len(requests) + 1)]
                        scalar_traces = traces[-len(requests):]
                        for step, batch_point in enumerate(batch_trace):
                            for row, scalar_trace in enumerate(scalar_traces):
                                scalar_point = scalar_trace[step]
                                diff = abs(batch_point["max"] - scalar_point["max"])
                                diff = max(diff, abs(batch_point["min"] - scalar_point["min"]))
                                diff = max(diff, abs(batch_point["mean"] - scalar_point["mean"]))
                                diff = max(diff, abs(batch_point["l2"] - scalar_point["l2"]))
                                if diff != 0.0:
                                    self.stats["inner_first_logit_diff_step"] = step
                                    self.stats["inner_first_logit_max_abs_diff"] = diff
                                    break
                            if self.stats["inner_first_logit_diff_step"] is not None:
                                break
                    cache_traces = self.predictor_engine.metrics_snapshot().get("trace_cache", [])
                    if len(cache_traces) >= len(requests) + 1:
                        batch_cache = cache_traces[-(len(requests) + 1)]
                        scalar_cache = cache_traces[-len(requests):]
                        self.stats["inner_cache_match"] = all(
                            batch_cache == item for item in scalar_cache
                        )
                    hidden_traces = self.predictor_engine.metrics_snapshot().get("trace_hidden", [])
                    if len(hidden_traces) >= len(requests) + 1:
                        batch_hidden = hidden_traces[-(len(requests) + 1)]
                        scalar_hidden = hidden_traces[-len(requests):]
                        for step, batch_point in enumerate(batch_hidden):
                            for scalar_trace in scalar_hidden:
                                if batch_point is None or scalar_trace[step] is None:
                                    continue
                                scalar_point = scalar_trace[step]
                                diff = max(
                                    abs(batch_point[key] - scalar_point[key])
                                    for key in ("max", "min", "mean", "l2")
                                )
                                if diff != 0.0:
                                    self.stats["inner_first_hidden_diff_step"] = step
                                    self.stats["inner_first_hidden_max_abs_diff"] = diff
                                    break
                            if self.stats["inner_first_hidden_diff_step"] is not None:
                                break
                if not torch.equal(result.sequences, scalar):
                    self.stats["inner_parity_failures"] += 1
                    raise RuntimeError(
                        "inner batch parity mismatch at scheduler step "
                        f"{self.stats['inner_parity_checks']}"
                    )
            codes = torch.cat((first_ids, result.sequences), dim=-1)
        self.stats["slot_batches"].append(len(requests))
        groups = defaultdict(list)
        for i, request in enumerate(requests):
            c = request.pending["cohort"]
            groups[(c.cache_position, request.pending["output_hidden_states"])].append(i)
        for (_, output_hidden_states), indices in groups.items():
            cohorts = [requests[i].pending["cohort"] for i in indices]
            states = [c.requests[0] for c in cohorts]
            masks = [self.engine.step_runtime.build_attention_mask(
                c.prompt_attention_mask, c.cache_position, self.engine.max_cache_len,
                c.requests[0].past_hidden.dtype) for c in cohorts]
            prepared = PreparedOuterDecode(
                codec_ids=codes[indices], condition=select_text_condition(states),
                attention_mask=None if all(m is None for m in masks) else torch.cat(masks),
                position_ids=build_decode_position_ids(cohorts[0].cache_position, states),
                cache_position=torch.tensor([cohorts[0].cache_position], device=codes.device),
                output_hidden_states=output_hidden_states,
            )
            if len(indices) != pool.capacity:
                raise RuntimeError("slot cohort changed batch size during outer decode")
            output = _run_outer_decode_body(self.talker, pool.cache, prepared)
            self._drain_joined_copy_bytes()
            self.stats["outer_slot_batches"].append(len(indices))
            for row, index in enumerate(indices):
                request = requests[index]
                frame = codes[index:index + 1]
                step_output = OuterStepOutput(
                    logits=output.logits[row:row + 1], past_hidden=output.past_hidden[row:row + 1],
                    hidden_states=tuple(h[row:row + 1] for h in output.hidden_states)
                    if output.hidden_states is not None else None,
                )
                eos = self.talker.config.codec_eos_token_id
                if int(frame[0, 0]) != eos:
                    request.frames.append(frame[0].detach().cpu().numpy())
                    self._decode(request)
                try:
                    request.pending = request.iterator.send((frame, step_output))
                except StopIteration:
                    self._finish(request)

    def _decode(self, request, final=False):
        chunk_size = (self.backend._stream_first_chunk_size or self.backend._stream_chunk_size
                      if request.chunks == 0 else self.backend._stream_chunk_size)
        end = len(request.frames)
        if end <= request.decoded or (not final and end - request.decoded < chunk_size):
            return
        chunk = _decode_codec_chunk(
            self.backend._model.model.speech_tokenizer, request.frames.prefix(end),
            chunk_index=request.chunks, code_start=request.decoded, code_end=end,
            left_context_size=self.backend._stream_left_context_size,
        )
        request.decoded = end
        request.chunks += 1
        self.emit("chunk", request.request_id, chunk.wav_bytes)

    def _finish(self, request, cancelled=False):
        if not cancelled:
            self._decode(request, final=True)
            self.stats["completed_codec_sha256"][str(request.request_id)] = hashlib.sha256(
                request.frames.prefix(len(request.frames)).tobytes()).hexdigest()
        request.iterator.close()
        self.active.pop(request.request_id)
        self.emit("done", request.request_id, None)
