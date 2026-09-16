"""Request-level metadata and scheduling primitives for Qwen-TTS varlen decode."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any, Callable, Iterable

from qwen_asr_vllm.agent.qwen_tts_outer_paged_cache import PageTable, PagedOuterCache


class VarlenMetadataError(RuntimeError):
    """Raised when request metadata cannot form a compatible packed step."""


@dataclass
class OuterVarlenRequest:
    request_id: str
    page_table: PageTable
    seq_len: int
    max_seq_len: int
    rope_delta: Any
    past_hidden: Any
    trailing_text_hidden: Any
    tts_pad_embed: Any
    active: bool = True
    ready: bool = True
    generation_step: int = 0
    first_codebook_history: list[Any] = field(default_factory=list)
    result: Any = None
    error: BaseException | None = None


@dataclass(frozen=True)
class PackedOuterStep:
    request_ids: tuple[str, ...]
    row_to_request_id: tuple[str, ...]
    requests: tuple[OuterVarlenRequest, ...]
    seq_lens: Any
    decode_positions: Any
    cu_seqlens: Any
    page_table: Any
    rope_deltas: Any
    past_hidden: Any


@dataclass(frozen=True)
class PackedOuterOutput:
    request_ids: tuple[str, ...]
    logits: Any
    past_hidden: Any
    hidden_states: Any = None
    codec_ids: Any = None
    done_request_ids: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)


def build_packed_step(requests: Iterable[OuterVarlenRequest]) -> PackedOuterStep:
    import torch

    selected = tuple(request for request in requests if request.active and request.ready)
    if not selected:
        raise VarlenMetadataError("no active ready requests")
    ids = tuple(str(request.request_id) for request in selected)
    if len(set(ids)) != len(ids):
        raise VarlenMetadataError("request IDs must be unique")
    reference = selected[0]
    if reference.seq_len <= 0 or reference.seq_len > reference.max_seq_len:
        raise VarlenMetadataError("request sequence length is invalid")
    for request in selected:
        if request.page_table.request_id != request.request_id:
            raise VarlenMetadataError("request page table does not match request")
        if request.seq_len <= 0 or request.seq_len > request.max_seq_len:
            raise VarlenMetadataError("request sequence length is invalid")
        if request.past_hidden.device != reference.past_hidden.device:
            raise VarlenMetadataError("requests must use the same device")
        if request.past_hidden.dtype != reference.past_hidden.dtype:
            raise VarlenMetadataError("requests must use the same dtype")
        if tuple(request.past_hidden.shape[1:]) != tuple(reference.past_hidden.shape[1:]):
            raise VarlenMetadataError("requests must use the same hidden geometry")
    device = reference.past_hidden.device
    seq_lens = torch.tensor([request.seq_len for request in selected], dtype=torch.long, device=device)
    decode_positions = seq_lens.clone()
    cu_seqlens = torch.cat(
        (torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(seq_lens, dim=0))
    )
    max_pages = max(len(request.page_table.page_ids) for request in selected)
    page_table = torch.full(
        (len(selected), max_pages), -1, dtype=torch.long, device=device
    )
    for row, request in enumerate(selected):
        ids_for_row = torch.tensor(request.page_table.page_ids, dtype=torch.long, device=device)
        page_table[row, : ids_for_row.numel()] = ids_for_row
    return PackedOuterStep(
        request_ids=ids,
        row_to_request_id=ids,
        requests=selected,
        seq_lens=seq_lens,
        decode_positions=decode_positions,
        cu_seqlens=cu_seqlens,
        page_table=page_table,
        rope_deltas=torch.cat([request.rope_delta for request in selected], dim=0),
        past_hidden=torch.cat([request.past_hidden for request in selected], dim=0),
    )


class OuterVarlenScheduler:
    """Tick-boundary request scheduler for packed outer-talker decode."""

    def __init__(
        self,
        *,
        cache: PagedOuterCache,
        prefill_fn: Callable[[OuterVarlenRequest], Any],
        decode_packed_fn: Callable[[PackedOuterStep], PackedOuterOutput],
        emit_fn: Callable[[str, Any], Any] | None = None,
    ):
        self.cache = cache
        self.prefill_fn = prefill_fn
        self.decode_packed_fn = decode_packed_fn
        self.emit_fn = emit_fn
        self._requests: dict[str, OuterVarlenRequest] = {}
        self._prefilled: set[str] = set()
        self._closed = False
        self.ticks = 0
        self.packed_requests = 0
        self.failures = 0

    def add_request(self, request: OuterVarlenRequest) -> str:
        if self._closed:
            raise VarlenMetadataError("scheduler is closed")
        request_id = str(request.request_id)
        if request_id in self._requests:
            raise VarlenMetadataError(f"request {request_id} is already scheduled")
        if request.page_table.request_id != request_id:
            raise VarlenMetadataError("request page table does not match request")
        self._requests[request_id] = request
        return request_id

    def request_ids(self) -> tuple[str, ...]:
        return tuple(self._requests)

    def cancel(self, request_id: str) -> bool:
        request = self._requests.pop(str(request_id), None)
        if request is None:
            return False
        request.active = False
        request.error = RuntimeError(f"request {request_id} cancelled")
        self._prefilled.discard(str(request_id))
        self.cache.release(str(request_id))
        return True

    def _fail_and_release(self, request: OuterVarlenRequest, error: BaseException) -> None:
        request.active = False
        request.error = error
        self.failures += 1
        self._requests.pop(request.request_id, None)
        self._prefilled.discard(request.request_id)
        self.cache.release(request.request_id)

    def step(self) -> PackedOuterOutput | None:
        if self._closed:
            raise VarlenMetadataError("scheduler is closed")
        if not self._requests:
            return None
        for request in tuple(self._requests.values()):
            if request.active and request.request_id not in self._prefilled:
                try:
                    self.prefill_fn(request)
                    self._prefilled.add(request.request_id)
                except Exception as error:
                    self._fail_and_release(request, error)
        candidates = tuple(
            request for request in self._requests.values() if request.active and request.ready
        )
        if not candidates:
            return None
        try:
            step = build_packed_step(candidates)
            output = self.decode_packed_fn(step)
            if tuple(output.request_ids) != step.request_ids:
                raise VarlenMetadataError("packed output request IDs do not match input rows")
            if output.past_hidden is not None:
                if output.past_hidden.shape[0] != len(step.requests):
                    raise VarlenMetadataError("packed output hidden row count mismatch")
            done = set(output.done_request_ids)
            for row, request in enumerate(step.requests):
                if output.past_hidden is not None:
                    request.past_hidden = output.past_hidden[row : row + 1].clone()
                request.seq_len += 1
                request.generation_step += 1
                if self.emit_fn is not None:
                    self.emit_fn(request.request_id, output)
                if request.request_id in done:
                    request.active = False
                    request.result = output
                    self._requests.pop(request.request_id, None)
                    self._prefilled.discard(request.request_id)
                    self.cache.release(request.request_id)
            self.ticks += 1
            self.packed_requests += len(step.requests)
            return output
        except Exception as error:
            for request in candidates:
                if request.request_id in self._requests:
                    self._fail_and_release(request, error)
            return None

    def run_until_idle(self, *, max_steps: int = 1024) -> int:
        steps = 0
        while self._requests and steps < int(max_steps):
            before = len(self._requests)
            self.step()
            steps += 1
            if len(self._requests) == before and not any(
                request.active and request.ready for request in self._requests.values()
            ):
                break
        if self._requests:
            raise VarlenMetadataError("scheduler did not become idle before max_steps")
        return steps

    def close(self) -> None:
        if self._closed:
            return
        for request in tuple(self._requests.values()):
            self._fail_and_release(request, RuntimeError("scheduler closed"))
        self._closed = True
        if self.cache.snapshot()["live_pages"]:
            raise VarlenMetadataError("scheduler close left live pages")

    def metrics(self) -> dict[str, int]:
        return {
            "ticks": self.ticks,
            "packed_requests": self.packed_requests,
            "failures": self.failures,
            "live_requests": len(self._requests),
        }


class ReferencePagedAttentionAdapter:
    """Correctness-first paged attention using explicit per-request gathers."""

    def __init__(self, cache: PagedOuterCache, *, scale: float | None = None):
        self.cache = cache
        self.scale = scale
        self.attention_calls = 0
        self.gather_bytes = 0

    def decode_packed(
        self,
        step: PackedOuterStep,
        queries: Any,
        *,
        layer_index: int = 0,
    ) -> PackedOuterOutput:
        import torch

        if not torch.is_tensor(queries) or queries.ndim != 4:
            raise VarlenMetadataError("queries must have shape [batch, heads, query, dim]")
        if queries.shape[0] != len(step.requests) or queries.shape[2] != 1:
            raise VarlenMetadataError("query rows must match packed request rows")
        rows = []
        for row, request in enumerate(step.requests):
            keys, values = self.cache.read(
                request.request_id,
                layer_index=layer_index,
                logical_length=int(step.seq_lens[row].item()),
            )
            query = queries[row : row + 1]
            if query.device != keys.device or query.dtype != keys.dtype:
                raise VarlenMetadataError("query and paged KV device/dtype must match")
            self.gather_bytes += (keys.numel() + values.numel()) * keys.element_size()
            rows.append(
                torch.nn.functional.scaled_dot_product_attention(
                    query,
                    keys,
                    values,
                    scale=self.scale,
                    is_causal=False,
                )
            )
            self.attention_calls += 1
        return PackedOuterOutput(
            request_ids=step.request_ids,
            logits=torch.cat(rows, dim=0),
            past_hidden=step.past_hidden,
            metrics={
                "attention_calls": self.attention_calls,
                "gather_bytes": self.gather_bytes,
            },
        )


try:
    from transformers.cache_utils import Cache as _TransformersCache
except ImportError:  # pragma: no cover - Qwen integration is optional at import time.
    _TransformersCache = object


class QwenPagedCache(_TransformersCache):
    """Transformers ``Cache`` bridge backed by request-owned fixed pages.

    The bridge intentionally exposes dense, padded KV tensors to the model's
    existing attention implementation.  The page pool remains the ownership
    and storage authority; padding is only a per-layer view for kernels that
    still expect a rectangular ``[batch, heads, kv, dim]`` tensor.
    """

    def __init__(
        self,
        cache: PagedOuterCache,
        *,
        num_layers: int,
        use_paged_attention_kernel: bool = False,
        paged_attention_block_n: int = 128,
        paged_attention_num_warps: int = 4,
    ):
        if _TransformersCache is object:
            raise VarlenCapabilityError("transformers is required for QwenPagedCache")
        super().__init__(layers=[])
        import torch

        self.page_cache = cache
        self.num_layers = int(num_layers)
        self.use_paged_attention_kernel = bool(use_paged_attention_kernel)
        self.paged_attention_block_n = int(paged_attention_block_n)
        self.paged_attention_num_warps = int(paged_attention_num_warps)
        if self.paged_attention_block_n <= 0 or self.paged_attention_block_n & (self.paged_attention_block_n - 1):
            raise VarlenMetadataError("paged attention block_n must be a positive power of two")
        if self.paged_attention_num_warps <= 0:
            raise VarlenMetadataError("paged attention num_warps must be positive")
        if self.num_layers <= 0 or self.num_layers != cache.num_layers:
            raise VarlenMetadataError("cache layer count does not match Qwen layer count")
        self._request_ids: tuple[str, ...] = ()
        self._query_positions: torch.Tensor | None = None
        self._device = None
        self._key_valid_masks: dict[str, torch.Tensor] = {}
        self._kernel_page_table: torch.Tensor | None = None
        self._kernel_valid_mask: torch.Tensor | None = None
        self._kernel_seq_len: int | None = None
        self._kernel_page_count: int | None = None

    def _bound_page_table(self, *, device: Any) -> Any:
        import torch

        if self._kernel_page_table is not None and self._kernel_page_table.device == device:
            return self._kernel_page_table

        tables = [self.page_cache.page_table(request_id) for request_id in self._request_ids]
        width = max(len(table.page_ids) for table in tables)
        result = torch.full(
            (len(tables), width),
            -1,
            dtype=torch.long,
            device=device,
        )
        for row, table in enumerate(tables):
            result[row, : len(table.page_ids)] = torch.tensor(
                table.page_ids,
                dtype=torch.long,
                device=device,
            )
        self._kernel_page_table = result
        return result

    def _bound_key_valid_mask(self, *, max_length: int, device: Any) -> Any:
        import torch

        if (
            self._kernel_valid_mask is not None
            and self._kernel_valid_mask.shape[1] == max_length
            and self._kernel_valid_mask.device == device
        ):
            return self._kernel_valid_mask

        rows = []
        for request_id in self._request_ids:
            valid = self._key_valid_masks.get(request_id)
            if valid is None:
                valid = torch.ones(0, dtype=torch.bool, device=device)
            valid = valid.to(device=device, dtype=torch.bool)
            if valid.shape[0] < max_length:
                valid = torch.cat(
                    (valid, torch.ones(max_length - valid.shape[0], dtype=torch.bool, device=device))
                )
            rows.append(valid[:max_length])
        result = torch.stack(rows, dim=0)
        self._kernel_valid_mask = result
        return result

    def paged_attention_decode(
        self,
        *,
        layer_idx: int,
        query_states: Any,
        key_states: Any,
        value_states: Any,
    ) -> Any:
        """Run the Triton kernel over the bound request rows and page table."""
        import torch

        if not self.use_paged_attention_kernel:
            raise VarlenCapabilityError("paged attention kernel is not enabled")
        if not torch.is_tensor(query_states) or query_states.ndim != 4:
            raise VarlenMetadataError("query states must have shape [batch, heads, 1, dim]")
        _, query_positions = self._require_bound()
        if query_positions.shape[1] != 1:
            raise VarlenMetadataError("paged attention kernel supports one decode token")
        if (
            not torch.is_tensor(key_states)
            or not torch.is_tensor(value_states)
            or key_states.ndim != 4
            or value_states.shape != key_states.shape
            or key_states.shape[0] != query_states.shape[0]
            or key_states.shape[2] != 1
        ):
            raise VarlenMetadataError(
                "decode KV must have shape [batch, kv_heads, 1, dim] and match query rows"
            )
        if self._kernel_seq_len is None or self._kernel_page_count is None:
            raise VarlenMetadataError("paged attention metadata was not staged for this bind")
        valid = self._bound_key_valid_mask(
            max_length=self._kernel_seq_len,
            device=query_states.device,
        )
        page_table = self._bound_page_table(device=query_states.device)
        key_pool = self.page_cache.layer_pool(int(layer_idx))
        from qwen_asr_vllm.agent.qwen_tts_paged_attention import paged_attention_decode

        return paged_attention_decode(
            query_states,
            key_pool,
            self.page_cache.layer_pool(int(layer_idx), values=True),
            page_table,
            query_positions[:, 0] + 1,
            valid,
            new_key=key_states,
            new_value=value_states,
            kernel_seq_len=self._kernel_seq_len,
            kernel_page_count=self._kernel_page_count,
            block_n=self.paged_attention_block_n,
            num_warps=self.paged_attention_num_warps,
            scale=1.0 / (float(query_states.shape[-1]) ** 0.5),
            use_triton=True,
            validate=False,
        )

    def commit_decode(self) -> None:
        """Commit the token written by every layer's fused decode kernel."""
        request_ids, query_positions = self._require_bound()
        if query_positions.shape[1] != 1:
            raise VarlenMetadataError("paged attention kernel supports one decode token")
        self.page_cache.commit_decode(request_ids, query_positions[:, 0])

    def bind_rows(
        self,
        request_ids: Iterable[str],
        query_positions: Any,
        key_valid_mask: Any = None,
    ) -> None:
        """Bind batch rows to requests and their contiguous logical positions."""
        import torch

        ids = tuple(str(request_id) for request_id in request_ids)
        if not ids or len(set(ids)) != len(ids):
            raise VarlenMetadataError("request IDs must be non-empty and unique")
        if not torch.is_tensor(query_positions) or query_positions.dtype != torch.long:
            raise VarlenMetadataError("query positions must be a torch.long tensor")
        if query_positions.ndim != 2 or query_positions.shape[0] != len(ids):
            raise VarlenMetadataError("query positions must have shape [batch rows, query tokens]")
        if query_positions.shape[1] <= 0:
            raise VarlenMetadataError("query positions must contain at least one token")
        if key_valid_mask is not None and (
            not torch.is_tensor(key_valid_mask)
            or key_valid_mask.ndim != 2
            or key_valid_mask.shape[0] != query_positions.shape[0]
        ):
            raise VarlenMetadataError("key_valid_mask must have one row per request")
        for row, request_id in enumerate(ids):
            record = self.page_cache.page_table(request_id)
            expected = torch.arange(
                record.seq_len,
                record.seq_len + query_positions.shape[1],
                dtype=torch.long,
                device=query_positions.device,
            )
            if not torch.equal(query_positions[row], expected):
                raise VarlenMetadataError("query positions must continue each request sequence")
            if key_valid_mask is not None:
                expected_key_length = record.seq_len + query_positions.shape[1]
                if key_valid_mask.shape[1] != expected_key_length:
                    raise VarlenMetadataError(
                        "key_valid_mask length must equal cached plus query tokens"
                    )
                self._key_valid_masks[request_id] = key_valid_mask[row].to(dtype=torch.bool).clone()
            elif request_id not in self._key_valid_masks:
                self._key_valid_masks[request_id] = torch.ones(
                    record.seq_len,
                    dtype=torch.bool,
                    device=query_positions.device,
                )
        self._request_ids = ids
        self._query_positions = query_positions
        self._device = query_positions.device
        self._kernel_page_table = None
        self._kernel_valid_mask = None
        if self.use_paged_attention_kernel:
            max_length = int(query_positions.max().item()) + 1
            self._kernel_seq_len = int(
                ceil(max_length / self.paged_attention_block_n) * self.paged_attention_block_n
            )
            self._kernel_page_count = int(ceil(self._kernel_seq_len / self.page_cache.page_size))
            self._bound_page_table(device=query_positions.device)
            self._bound_key_valid_mask(
                max_length=self._kernel_seq_len,
                device=query_positions.device,
            )

    def _require_bound(self) -> tuple[tuple[str, ...], Any]:
        if not self._request_ids or self._query_positions is None:
            raise VarlenMetadataError("bind_rows must be called before cache update")
        return self._request_ids, self._query_positions

    def update(
        self,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        import torch

        request_ids, query_positions = self._require_bound()
        if not torch.is_tensor(key_states) or key_states.ndim != 4:
            raise VarlenMetadataError("Qwen layer KV must have shape [batch, heads, query, dim]")
        if value_states.shape != key_states.shape:
            raise VarlenMetadataError("Qwen key/value layer shapes must match")
        if key_states.shape[0] != len(request_ids) or key_states.shape[2] != query_positions.shape[1]:
            raise VarlenMetadataError("Qwen KV batch rows do not match bound requests")
        if key_states.device != query_positions.device:
            raise VarlenMetadataError("Qwen KV and query positions must share a device")
        layer_idx = int(layer_idx)
        if not 0 <= layer_idx < self.num_layers:
            raise VarlenMetadataError("Qwen cache layer index is out of range")
        commit = layer_idx == self.num_layers - 1
        outputs = []
        value_outputs = []
        max_kv_len = int(query_positions.max().item()) + 1
        for row, request_id in enumerate(request_ids):
            positions = query_positions[row]
            self.page_cache.append_layer(
                request_id,
                layer_index=layer_idx,
                keys=key_states[row : row + 1],
                values=value_states[row : row + 1],
                logical_positions=positions,
                commit_seq_len=commit,
            )
            keys, values = self.page_cache.read_at(
                request_id,
                layer_index=layer_idx,
                logical_length=max_kv_len,
            )
            outputs.append(keys)
            value_outputs.append(values)
        return torch.cat(outputs, dim=0), torch.cat(value_outputs, dim=0)

    def attention_mask(
        self,
        *,
        dtype: Any,
        fill_value: float | None = None,
        key_valid_mask: Any = None,
    ) -> Any:
        """Build a causal mask for independent variable-length batch rows."""
        import torch

        _, query_positions = self._require_bound()
        max_kv_len = int(query_positions.max().item()) + 1
        if fill_value is None:
            fill_value = float(torch.finfo(dtype).min)
        mask = torch.full(
            (query_positions.shape[0], 1, query_positions.shape[1], max_kv_len),
            fill_value,
            dtype=dtype,
            device=query_positions.device,
        )
        key_positions = torch.arange(max_kv_len, device=query_positions.device)
        allowed = key_positions.unsqueeze(0).unsqueeze(0) <= query_positions.unsqueeze(-1)
        if key_valid_mask is None:
            valid_rows = []
            for request_id in self._request_ids:
                valid = self._key_valid_masks.get(request_id)
                if valid is None:
                    valid = torch.ones(0, dtype=torch.bool, device=query_positions.device)
                if valid.shape[0] < max_kv_len:
                    valid = torch.cat(
                        (valid, torch.ones(max_kv_len - valid.shape[0], dtype=torch.bool, device=valid.device))
                    )
                valid_rows.append(valid[:max_kv_len])
            key_valid_mask = torch.stack(valid_rows, dim=0)
        if key_valid_mask is not None:
            if (
                not torch.is_tensor(key_valid_mask)
                or key_valid_mask.ndim != 2
                or key_valid_mask.shape[0] != query_positions.shape[0]
                or key_valid_mask.shape[1] != max_kv_len
            ):
                raise VarlenMetadataError("key_valid_mask must have shape [batch rows, max kv length]")
            allowed &= key_valid_mask.to(dtype=torch.bool).unsqueeze(1)
        mask[:, 0].masked_fill_(allowed, 0)
        return mask

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self._request_ids:
            return 0
        return max(
            self.page_cache.page_table(request_id).seq_len for request_id in self._request_ids
        )

    def get_mask_sizes(self, cache_position: Any, layer_idx: int) -> tuple[int, int]:
        _, query_positions = self._require_bound()
        return int(query_positions.max().item()) + 1, 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return max(
            self.page_cache.page_table(request_id).max_seq_len for request_id in self._request_ids
        ) if self._request_ids else -1

    def reset(self) -> None:
        self._request_ids = ()
        self._query_positions = None
        self._key_valid_masks.clear()
        self._kernel_page_table = None
        self._kernel_valid_mask = None
        self._kernel_seq_len = None
        self._kernel_page_count = None


class VarlenCapabilityError(RuntimeError):
    """Raised when the Qwen model cannot use the reference varlen contract."""


class QwenOuterVarlenRunner:
    """Model-facing runner for request-id packed outer-talker steps.

    Codebook sampling/predictor generation stays outside this runner so it can
    be scheduled by the existing predictor service.  This class owns the
    shared Qwen outer model call and its request-addressed paged KV cache.
    """

    def __init__(self, talker: Any, adapter: "QwenOuterVarlenAdapter"):
        self.talker = talker
        self.adapter = adapter
        self.cache = adapter.create_qwen_cache()
        self._paged_attention_patch = None
        if adapter.use_paged_attention_kernel:
            from qwen_asr_vllm.agent.qwen_tts_paged_attention import (
                install_qwen_paged_decode_attention,
            )

            self._paged_attention_patch = install_qwen_paged_decode_attention(
                talker,
                fuse_qkv=adapter.use_fused_qkv,
            )

    def allocate_request(self, request_id: str, *, max_seq_len: int) -> PageTable:
        return self.adapter.cache.allocate(request_id, max_seq_len=max_seq_len, batch_size=1)

    def prefill(
        self,
        request_id: str,
        *,
        inputs_embeds: Any,
        position_ids: Any = None,
        attention_mask: Any = None,
    ) -> Any:
        import torch

        if not torch.is_tensor(inputs_embeds) or inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
            raise VarlenCapabilityError("Qwen varlen prefill expects inputs_embeds [1, tokens, hidden]")
        table = self.adapter.cache.page_table(request_id)
        if table.seq_len != 0:
            raise VarlenCapabilityError("Qwen varlen prefill can run only once per request")
        positions = torch.arange(inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device)
        self.cache.bind_rows(
            (request_id,),
            positions.unsqueeze(0),
            key_valid_mask=(
                attention_mask.to(dtype=torch.bool)
                if torch.is_tensor(attention_mask) and attention_mask.ndim == 2
                else None
            ),
        )
        if position_ids is None:
            position_ids = positions.view(1, 1, -1).expand(3, 1, -1)
        model_mask = self.cache.attention_mask(
            dtype=inputs_embeds.dtype,
            key_valid_mask=attention_mask.to(dtype=torch.bool) if torch.is_tensor(attention_mask) and attention_mask.ndim == 2 else None,
        )
        outputs = self.talker.model(
            inputs_embeds=inputs_embeds,
            attention_mask=model_mask,
            position_ids=position_ids,
            past_key_values=self.cache,
            cache_position=positions,
            use_cache=True,
            output_hidden_states=False,
        )
        return outputs.last_hidden_state

    def decode_packed(
        self,
        step: PackedOuterStep,
        *,
        codec_ids: Any,
        condition: Any,
        output_hidden_states: bool = False,
    ) -> PackedOuterOutput:
        import torch

        if not torch.is_tensor(codec_ids) or codec_ids.ndim != 2 or codec_ids.shape[0] != len(step.requests):
            raise VarlenMetadataError("codec_ids must have one row per packed request")
        if codec_ids.shape[1] != int(self.talker.config.num_code_groups):
            raise VarlenMetadataError("codec_ids does not match Qwen code group count")
        if not torch.is_tensor(condition) or condition.shape != (len(step.requests), 1, self.talker.config.hidden_size):
            raise VarlenMetadataError("condition must have shape [packed rows, 1, hidden]")
        codec_hiddens = [self.talker.get_input_embeddings()(codec_ids[:, :1])]
        predictor_embeddings = self.talker.code_predictor.get_input_embeddings()
        codec_hiddens.extend(
            predictor_embeddings[index](codec_ids[:, index + 1 : index + 2])
            for index in range(self.talker.config.num_code_groups - 1)
        )
        inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True) + condition
        self.cache.bind_rows(step.request_ids, step.decode_positions.view(-1, 1))
        position_ids = step.rope_deltas.reshape(1, len(step.requests), -1) + step.decode_positions.view(1, -1, 1)
        position_ids = position_ids.expand(3, -1, -1)
        outputs = self.talker.model(
            inputs_embeds=inputs_embeds,
            attention_mask=self.cache.attention_mask(dtype=inputs_embeds.dtype),
            position_ids=position_ids,
            past_key_values=self.cache,
            # The Qwen model API accepts one query-position vector, while the
            # cache bridge carries the per-row positions and supplies the 4D
            # mask.  The cache itself ignores this argument after binding rows;
            # use the first real position instead of a sentinel zero so model
            # implementations that inspect it see a valid decode position.
            cache_position=step.decode_positions[:1],
            use_cache=True,
            output_hidden_states=output_hidden_states,
        )
        if self.cache.use_paged_attention_kernel:
            self.cache.commit_decode()
        hidden_states = outputs.last_hidden_state
        return PackedOuterOutput(
            request_ids=step.request_ids,
            logits=self.talker.codec_head(hidden_states),
            past_hidden=hidden_states[:, -1:, :],
            hidden_states=getattr(outputs, "hidden_states", None),
        )

    def close(self) -> None:
        if self._paged_attention_patch is not None:
            self._paged_attention_patch.close()
            self._paged_attention_patch = None


class QwenOuterVarlenAdapter:
    """Opt-in Qwen integration boundary around the paged reference runtime.

    This object owns only request/cache infrastructure. The model-specific
    projection callback is injected by the future Qwen decode bridge, keeping
    installation side-effect free until a scheduler is explicitly created.
    """

    def __init__(
        self,
        talker: Any,
        *,
        max_cache_len: int,
        page_size: int = 16,
        max_pages: int = 256,
        use_paged_attention_kernel: bool = False,
        use_fused_qkv: bool = False,
        paged_attention_block_n: int = 128,
        paged_attention_num_warps: int = 4,
    ):
        config = getattr(talker, "config", None)
        layers = int(getattr(config, "num_hidden_layers", 0))
        heads = int(getattr(config, "num_attention_heads", 0))
        head_dim = int(getattr(config, "head_dim", 0) or 0)
        if layers <= 0 or heads <= 0 or head_dim <= 0:
            raise VarlenCapabilityError("model is missing outer attention geometry")
        if heads % 2:
            raise VarlenCapabilityError("unsupported outer attention head geometry")
        if int(max_cache_len) <= 0 or int(page_size) <= 0 or int(max_pages) <= 0:
            raise VarlenCapabilityError("max_cache_len, page_size, and max_pages must be positive")
        self.talker = talker
        self.max_cache_len = int(max_cache_len)
        self.page_size = int(page_size)
        self.max_pages = int(max_pages)
        self.use_paged_attention_kernel = bool(use_paged_attention_kernel)
        self.use_fused_qkv = bool(use_fused_qkv)
        self.paged_attention_block_n = int(paged_attention_block_n)
        self.paged_attention_num_warps = int(paged_attention_num_warps)
        self.num_layers = layers
        self.num_heads = heads
        self.head_dim = head_dim
        self.cache = PagedOuterCache(
            num_layers=layers,
            num_pages=self.max_pages,
            page_size=self.page_size,
        )
        self._scheduler: OuterVarlenScheduler | None = None

    def create_qwen_cache(self) -> QwenPagedCache:
        """Create a model-facing cache sharing this adapter's page pool."""
        return QwenPagedCache(
            self.cache,
            num_layers=self.num_layers,
            use_paged_attention_kernel=self.use_paged_attention_kernel,
            paged_attention_block_n=self.paged_attention_block_n,
            paged_attention_num_warps=self.paged_attention_num_warps,
        )

    def create_runner(self) -> QwenOuterVarlenRunner:
        return QwenOuterVarlenRunner(self.talker, self)

    def create_scheduler(
        self,
        *,
        prefill_fn: Callable[[OuterVarlenRequest], Any],
        decode_packed_fn: Callable[[PackedOuterStep], PackedOuterOutput],
        emit_fn: Callable[[str, Any], Any] | None = None,
    ) -> OuterVarlenScheduler:
        if self._scheduler is not None:
            raise VarlenCapabilityError("varlen scheduler is already created")
        self._scheduler = OuterVarlenScheduler(
            cache=self.cache,
            prefill_fn=prefill_fn,
            decode_packed_fn=decode_packed_fn,
            emit_fn=emit_fn,
        )
        return self._scheduler

    def metrics(self) -> dict[str, Any]:
        result = {"cache": self.cache.snapshot()}
        if self._scheduler is not None:
            result["scheduler"] = self._scheduler.metrics()
        return result

    def metrics_snapshot(self) -> dict[str, Any]:
        return self.metrics()

    def close(self) -> None:
        if self._scheduler is not None:
            self._scheduler.close()
            self._scheduler = None
        if self.cache.snapshot()["live_pages"]:
            raise VarlenCapabilityError("varlen adapter close left live pages")
