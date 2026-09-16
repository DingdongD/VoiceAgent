"""Fixed-page KV storage for the opt-in Qwen-TTS varlen reference path."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from threading import RLock
from typing import Any


class PagedCacheError(RuntimeError):
    """Raised when a paged outer cache contract is violated."""


@dataclass(frozen=True)
class PageTable:
    request_id: str
    page_ids: tuple[int, ...]
    batch_size: int
    max_seq_len: int
    seq_len: int = 0

    @property
    def num_pages(self) -> int:
        return len(self.page_ids)


@dataclass
class _RequestRecord:
    table: PageTable
    released: bool = False


class PagedOuterCache:
    """A request-owned fixed-page KV pool with explicit lifecycle checks.

    The pool stores all layers in one strided tensor and accepts a complete
    layer stack per append: ``[layers, batch, heads, tokens, head_dim]``.
    This keeps logical positions identical across layers while permitting
    different requests to have different sequence lengths.
    """

    def __init__(self, *, num_layers: int, num_pages: int, page_size: int):
        if int(num_layers) <= 0 or int(num_pages) <= 0 or int(page_size) <= 0:
            raise PagedCacheError("num_layers, num_pages, and page_size must be positive")
        self.num_layers = int(num_layers)
        self.num_pages = int(num_pages)
        self.page_size = int(page_size)
        self._lock = RLock()
        self._free_pages = list(range(self.num_pages))
        self._records: dict[str, _RequestRecord] = {}
        self._key_pool: Any = None
        self._value_pool: Any = None
        self._batch_size: int | None = None
        self._num_heads: int | None = None
        self._head_dim: int | None = None
        self._dtype: Any = None
        self._device: Any = None
        self._allocations = 0
        self._releases = 0
        self._overflows = 0

    def allocate(self, request_id: str, *, max_seq_len: int, batch_size: int = 1) -> PageTable:
        request_id = str(request_id)
        max_seq_len = int(max_seq_len)
        batch_size = int(batch_size)
        if not request_id:
            raise PagedCacheError("request_id must be non-empty")
        if max_seq_len <= 0 or batch_size <= 0:
            raise PagedCacheError("max_seq_len and batch_size must be positive")
        pages_needed = ceil(max_seq_len / self.page_size) * batch_size
        with self._lock:
            if request_id in self._records and not self._records[request_id].released:
                raise PagedCacheError(f"request {request_id} is already allocated")
            if pages_needed > len(self._free_pages):
                self._overflows += 1
                raise PagedCacheError("insufficient free pages")
            if self._batch_size is None:
                self._batch_size = batch_size
            elif self._batch_size != batch_size:
                raise PagedCacheError("all requests must use the same batch_size")
            page_ids = tuple(self._free_pages[:pages_needed])
            del self._free_pages[:pages_needed]
            table = PageTable(request_id, page_ids, batch_size, max_seq_len)
            self._records[request_id] = _RequestRecord(table)
            self._allocations += 1
            return table

    def _record(self, request_id: str) -> _RequestRecord:
        record = self._records.get(str(request_id))
        if record is None or record.released:
            raise PagedCacheError(f"unknown request {request_id}")
        return record

    def page_table(self, request_id: str) -> PageTable:
        """Return a snapshot of a live request's page table."""
        with self._lock:
            return self._record(request_id).table

    def layer_pool(self, layer_index: int, *, values: bool = False) -> Any:
        """Expose one layer's physical KV pages without making a copy."""
        with self._lock:
            layer_index = int(layer_index)
            if not 0 <= layer_index < self.num_layers:
                raise PagedCacheError("layer_index out of range")
            pool = self._value_pool if values else self._key_pool
            if pool is None:
                raise PagedCacheError("KV pool has not been initialized")
            return pool[layer_index]

    def _ensure_pool(self, keys: Any, values: Any) -> None:
        import torch

        if self._key_pool is not None:
            return
        _, _, heads, _, head_dim = keys.shape
        shape = (self.num_layers, self.num_pages, heads, self.page_size, head_dim)
        self._key_pool = torch.zeros(shape, dtype=keys.dtype, device=keys.device)
        self._value_pool = torch.zeros_like(self._key_pool)
        self._num_heads = int(heads)
        self._head_dim = int(head_dim)
        self._dtype = keys.dtype
        self._device = keys.device

    def append(self, request_id: str, keys: Any, values: Any, logical_positions: Any) -> None:
        import torch

        with self._lock:
            record = self._record(request_id)
            if not torch.is_tensor(keys) or not torch.is_tensor(values):
                raise PagedCacheError("key/value states must be tensors")
            if keys.ndim != 5 or values.shape != keys.shape:
                raise PagedCacheError("key/value states must have shape [layers,batch,heads,tokens,dim]")
            if keys.shape[0] != self.num_layers:
                raise PagedCacheError("key/value layer count mismatch")
            if keys.dtype != values.dtype or keys.device != values.device:
                raise PagedCacheError("key/value dtype and device must match")
            if not torch.is_tensor(logical_positions) or logical_positions.ndim != 1:
                raise PagedCacheError("logical_positions must be a one-dimensional tensor")
            if logical_positions.dtype != torch.long or logical_positions.device != keys.device:
                raise PagedCacheError("logical_positions must be torch.long on the KV device")
            _, batch, _, tokens, _ = keys.shape
            if batch != record.table.batch_size or logical_positions.numel() != tokens:
                raise PagedCacheError("append batch or position length mismatch")
            expected = torch.arange(
                record.table.seq_len,
                record.table.seq_len + tokens,
                dtype=torch.long,
                device=keys.device,
            )
            if not torch.equal(logical_positions, expected):
                self._overflows += 1
                raise PagedCacheError("logical positions must be contiguous")
            if record.table.seq_len + tokens > record.table.max_seq_len:
                self._overflows += 1
                raise PagedCacheError("logical sequence exceeds request capacity")
            self._ensure_pool(keys, values)
            if (
                keys.dtype != self._dtype
                or keys.device != self._device
                or int(keys.shape[2]) != self._num_heads
                or int(keys.shape[4]) != self._head_dim
            ):
                raise PagedCacheError("KV geometry, dtype, or device changed")
            pages_per_batch = ceil(record.table.max_seq_len / self.page_size)
            for layer in range(self.num_layers):
                for batch_index in range(batch):
                    row_pages = record.table.page_ids[
                        batch_index * pages_per_batch : (batch_index + 1) * pages_per_batch
                    ]
                    for token_index in range(tokens):
                        logical = record.table.seq_len + token_index
                        page_index, offset = divmod(logical, self.page_size)
                        physical = row_pages[page_index]
                        self._key_pool[layer, physical, :, offset, :] = keys[
                            layer, batch_index, :, token_index, :
                        ]
                        self._value_pool[layer, physical, :, offset, :] = values[
                            layer, batch_index, :, token_index, :
                        ]
            record.table = PageTable(
                record.table.request_id,
                record.table.page_ids,
                record.table.batch_size,
                record.table.max_seq_len,
                record.table.seq_len + tokens,
            )

    def append_layer(
        self,
        request_id: str,
        *,
        layer_index: int,
        keys: Any,
        values: Any,
        logical_positions: Any,
        commit_seq_len: bool = False,
    ) -> None:
        """Write one layer's KV without duplicating a full layer stack.

        ``logical_positions`` may be ``[tokens]`` for a single batch row or
        ``[batch, tokens]`` for an explicitly addressed batch.  The page
        table length is committed only when ``commit_seq_len`` is true.  This
        mirrors Transformers' per-layer ``Cache.update`` contract, where all
        layers observe the same logical query positions during one forward.
        """
        import torch

        with self._lock:
            record = self._record(request_id)
            layer_index = int(layer_index)
            if not 0 <= layer_index < self.num_layers:
                raise PagedCacheError("layer_index out of range")
            if not torch.is_tensor(keys) or not torch.is_tensor(values):
                raise PagedCacheError("key/value states must be tensors")
            if keys.ndim != 4 or values.shape != keys.shape:
                raise PagedCacheError("layer key/value states must have shape [batch,heads,tokens,dim]")
            if keys.dtype != values.dtype or keys.device != values.device:
                raise PagedCacheError("key/value dtype and device must match")
            if not torch.is_tensor(logical_positions) or logical_positions.dtype != torch.long:
                raise PagedCacheError("logical_positions must be torch.long")
            if logical_positions.device != keys.device:
                raise PagedCacheError("logical_positions must be on the KV device")
            batch, heads, tokens, head_dim = keys.shape
            if batch != record.table.batch_size:
                raise PagedCacheError("append layer batch size mismatch")
            if logical_positions.ndim == 1:
                if batch != 1 or logical_positions.numel() != tokens:
                    raise PagedCacheError("logical_positions shape does not match layer KV")
                logical_positions = logical_positions.unsqueeze(0)
            if logical_positions.shape != (batch, tokens):
                raise PagedCacheError("logical_positions shape does not match layer KV")
            expected = torch.arange(
                record.table.seq_len,
                record.table.seq_len + tokens,
                dtype=torch.long,
                device=keys.device,
            )
            if not torch.equal(logical_positions, expected.unsqueeze(0).expand(batch, -1)):
                self._overflows += 1
                raise PagedCacheError("logical positions must be contiguous")
            next_seq_len = record.table.seq_len + tokens
            if next_seq_len > record.table.max_seq_len:
                self._overflows += 1
                raise PagedCacheError("logical sequence exceeds request capacity")
            self._ensure_pool(keys.unsqueeze(0), values.unsqueeze(0))
            if (
                keys.dtype != self._dtype
                or keys.device != self._device
                or int(heads) != self._num_heads
                or int(head_dim) != self._head_dim
            ):
                raise PagedCacheError("KV geometry, dtype, or device changed")
            pages_per_batch = ceil(record.table.max_seq_len / self.page_size)
            for batch_index in range(batch):
                row_pages = record.table.page_ids[
                    batch_index * pages_per_batch : (batch_index + 1) * pages_per_batch
                ]
                page_ids = torch.tensor(row_pages, dtype=torch.long, device=keys.device)
                page_indices = torch.div(
                    logical_positions[batch_index], self.page_size, rounding_mode="floor"
                )
                offsets = logical_positions[batch_index] % self.page_size
                physical = page_ids.index_select(0, page_indices)
                self._key_pool[layer_index, physical, :, offsets, :] = keys[batch_index].transpose(0, 1)
                self._value_pool[layer_index, physical, :, offsets, :] = values[batch_index].transpose(0, 1)
            if commit_seq_len:
                record.table = PageTable(
                    record.table.request_id,
                    record.table.page_ids,
                    record.table.batch_size,
                    record.table.max_seq_len,
                    next_seq_len,
                )

    def commit_decode(self, request_ids: tuple[str, ...], positions: Any) -> None:
        """Commit one already-written decode position for every request row."""
        import torch

        if not torch.is_tensor(positions) or positions.ndim != 1:
            raise PagedCacheError("decode positions must be a one-dimensional tensor")
        if positions.dtype != torch.long:
            raise PagedCacheError("decode positions must be torch.long")
        with self._lock:
            if len(request_ids) != positions.shape[0]:
                raise PagedCacheError("decode positions must match request rows")
            if positions.device != self._device and self._device is not None:
                raise PagedCacheError("decode positions must be on the KV device")
            records = [self._record(request_id) for request_id in request_ids]
            next_lengths = []
            for record, position in zip(records, positions.tolist()):
                if int(position) != record.table.seq_len:
                    raise PagedCacheError("decode position does not continue request sequence")
                next_length = record.table.seq_len + 1
                if next_length > record.table.max_seq_len:
                    self._overflows += 1
                    raise PagedCacheError("logical sequence exceeds request capacity")
                next_lengths.append(next_length)
            for record, next_length in zip(records, next_lengths):
                table = record.table
                record.table = PageTable(
                    table.request_id,
                    table.page_ids,
                    table.batch_size,
                    table.max_seq_len,
                    next_length,
                )

    def read_at(self, request_id: str, *, layer_index: int, logical_length: int) -> tuple[Any, Any]:
        """Read a dense prefix including KV written by an uncommitted layer."""
        import torch

        with self._lock:
            record = self._record(request_id)
            layer_index = int(layer_index)
            logical_length = int(logical_length)
            if not 0 <= layer_index < self.num_layers:
                raise PagedCacheError("layer_index out of range")
            if logical_length < 0 or logical_length > record.table.max_seq_len:
                raise PagedCacheError("logical_length exceeds request capacity")
            if self._key_pool is None:
                raise PagedCacheError("request has no appended KV")
            pages_per_batch = ceil(record.table.max_seq_len / self.page_size)
            rows = []
            value_rows = []
            for batch_index in range(record.table.batch_size):
                row_pages = record.table.page_ids[
                    batch_index * pages_per_batch : (batch_index + 1) * pages_per_batch
                ]
                if logical_length:
                    logical = torch.arange(logical_length, dtype=torch.long, device=self._device)
                    page_ids = torch.tensor(row_pages, dtype=torch.long, device=self._device)
                    page_indices = torch.div(logical, self.page_size, rounding_mode="floor")
                    offsets = logical % self.page_size
                    physical = page_ids.index_select(0, page_indices)
                    rows.append(self._key_pool[layer_index, physical, :, offsets, :].transpose(0, 1))
                    value_rows.append(self._value_pool[layer_index, physical, :, offsets, :].transpose(0, 1))
                else:
                    rows.append(self._key_pool[layer_index, row_pages[0], :, :0, :])
                    value_rows.append(self._value_pool[layer_index, row_pages[0], :, :0, :])
            return torch.stack(rows, dim=0), torch.stack(value_rows, dim=0)

    def read(self, request_id: str, *, layer_index: int, logical_length: int) -> tuple[Any, Any]:
        with self._lock:
            record = self._record(request_id)
            logical_length = int(logical_length)
            if logical_length < 0 or logical_length > record.table.seq_len:
                raise PagedCacheError("logical_length exceeds request sequence length")
        return self.read_at(request_id, layer_index=layer_index, logical_length=logical_length)

    def read_page(self, request_id: str, *, page_id: int, layer_index: int) -> tuple[Any, Any]:
        """Read one physical page only after verifying request ownership."""

        with self._lock:
            record = self._record(request_id)
            page_id = int(page_id)
            layer_index = int(layer_index)
            if page_id not in record.table.page_ids:
                raise PagedCacheError(
                    f"page {page_id} does not belong to request {request_id}"
                )
            if self._key_pool is None or not 0 <= layer_index < self.num_layers:
                raise PagedCacheError("layer_index out of range or request has no appended KV")
            return (
                self._key_pool[layer_index, page_id].clone(),
                self._value_pool[layer_index, page_id].clone(),
            )

    def release(self, request_id: str) -> None:
        with self._lock:
            record = self._records.get(str(request_id))
            if record is None:
                raise PagedCacheError(f"unknown request {request_id}")
            if record.released:
                return
            self._free_pages.extend(record.table.page_ids)
            self._free_pages.sort()
            record.released = True
            self._releases += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            live = [r for r in self._records.values() if not r.released]
            return {
                "live_requests": len(live),
                "live_pages": sum(len(r.table.page_ids) for r in live),
                "free_pages": len(self._free_pages),
                "allocations": self._allocations,
                "releases": self._releases,
                "overflows": self._overflows,
                "page_size": self.page_size,
                "num_layers": self.num_layers,
            }
