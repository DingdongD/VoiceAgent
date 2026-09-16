from __future__ import annotations

from dataclasses import dataclass
from threading import Lock, RLock
from typing import Any

from qwen_asr_vllm.agent.qwen_tts_outer_static_engine import (
    OuterRuntimeLease,
    OuterTalkerStaticEngine,
    _run_outer_decode_body,
)


class ActivePrefixCacheError(RuntimeError):
    """Raised when active-prefix cache storage cannot satisfy its contract."""


try:
    import transformers
    from transformers.cache_utils import Cache, CacheLayerMixin
except Exception as exc:  # pragma: no cover - exercised only by broken installations.
    version = getattr(locals().get("transformers"), "__version__", "unavailable")
    raise ActivePrefixCacheError(
        "active-prefix cache requires public Transformers Cache and "
        f"CacheLayerMixin APIs (detected transformers {version})"
    ) from exc


class ActivePrefixLayer(CacheLayerMixin):
    """Fixed-capacity K/V storage which exposes only its active sequence prefix."""

    is_compileable = False

    def __init__(self, max_cache_len: int):
        super().__init__()
        if int(max_cache_len) <= 0:
            raise ActivePrefixCacheError("max_cache_len must be positive")
        self.max_cache_len = int(max_cache_len)
        self.active_length = 0
        self._released = False

    def lazy_initialization(self, key_states) -> None:
        batch, heads, _, head_dim = key_states.shape
        shape = (batch, heads, self.max_cache_len, head_dim)
        self.keys = key_states.new_zeros(shape)
        self.values = key_states.new_zeros(shape)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if self._released:
            raise ActivePrefixCacheError("active-prefix layer is released")

        torch = _torch()
        if not torch.is_tensor(key_states) or not torch.is_tensor(value_states):
            raise ActivePrefixCacheError("key/value states must be tensors")
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ActivePrefixCacheError(
                "key/value states must share a four-dimensional shape"
            )
        if key_states.shape != value_states.shape:
            raise ActivePrefixCacheError("key/value states must share an identical shape")
        if key_states.dtype != value_states.dtype:
            raise ActivePrefixCacheError("key/value states must share an identical dtype")
        if key_states.device != value_states.device:
            raise ActivePrefixCacheError("key/value states must share an identical device")
        if key_states.layout != torch.strided or value_states.layout != torch.strided:
            raise ActivePrefixCacheError("key/value states must use strided layout")

        if cache_kwargs is not None and not hasattr(cache_kwargs, "get"):
            raise ActivePrefixCacheError("cache_kwargs must be a mapping")
        cache_position = (cache_kwargs or {}).get("cache_position")
        if not torch.is_tensor(cache_position):
            raise ActivePrefixCacheError("cache_position must be a tensor")
        if cache_position.ndim != 1:
            raise ActivePrefixCacheError("one-dimensional cache_position is required")
        if cache_position.dtype != torch.long:
            raise ActivePrefixCacheError("cache_position must use torch.long dtype")
        if cache_position.device != key_states.device:
            raise ActivePrefixCacheError(
                "cache_position must be on the key/value device"
            )
        if cache_position.numel() != key_states.shape[-2]:
            raise ActivePrefixCacheError(
                "cache_position length must match query length"
            )

        query_length = int(key_states.shape[-2])
        next_length = self.active_length + query_length
        if next_length > self.max_cache_len:
            raise ActivePrefixCacheError("active-prefix cache capacity exceeded")

        if self.is_initialized:
            expected_shape = (
                self.keys.shape[0],
                self.keys.shape[1],
                self.keys.shape[3],
            )
            actual_shape = (
                key_states.shape[0],
                key_states.shape[1],
                key_states.shape[3],
            )
            if actual_shape != expected_shape:
                raise ActivePrefixCacheError(
                    "key/value layout changed after allocation"
                )
            if (
                key_states.dtype != self.keys.dtype
                or value_states.dtype != self.values.dtype
                or key_states.device != self.keys.device
                or value_states.device != self.values.device
                or self.keys.layout != torch.strided
                or self.values.layout != torch.strided
            ):
                raise ActivePrefixCacheError(
                    "key/value layout changed after allocation"
                )
            if any(
                _shares_storage_with_backing(tensor, self.keys, self.values)
                for tensor in (key_states, value_states, cache_position)
            ):
                raise ActivePrefixCacheError(
                    "cache update inputs must not alias key/value backing storage"
                )

        initialized_here = not self.is_initialized
        if initialized_here:
            self.lazy_initialization(key_states)
        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except Exception as exc:
            if initialized_here:
                self.keys = None
                self.values = None
                self.is_initialized = False
            raise ActivePrefixCacheError("active-prefix cache index update failed") from exc
        self.active_length = next_length
        return self.keys[:, :, :next_length], self.values[:, :, :next_length]

    def get_seq_length(self) -> int:
        return self.active_length

    def get_mask_sizes(self, cache_position) -> tuple[int, int]:
        return self.active_length + int(cache_position.shape[0]), 0

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def reset(self) -> None:
        if self._released:
            raise ActivePrefixCacheError("active-prefix layer is released")
        with _torch().inference_mode():
            if self.is_initialized and self.active_length:
                self.keys[:, :, : self.active_length].zero_()
                self.values[:, :, : self.active_length].zero_()
            self.active_length = 0

    def release(self) -> None:
        """Terminally discard backing storage; pooled lease reuse is a runtime concern."""

        if self._released:
            return
        self.keys = None
        self.values = None
        self.is_initialized = False
        self.active_length = 0
        self._released = True


class ActivePrefixCache(Cache):
    """Transformers cache composed of a fixed number of active-prefix layers."""

    def __init__(self, num_hidden_layers: int, max_cache_len: int):
        if int(num_hidden_layers) <= 0:
            raise ActivePrefixCacheError("num_hidden_layers must be positive")
        try:
            layers = [
                ActivePrefixLayer(max_cache_len)
                for _ in range(int(num_hidden_layers))
            ]
            super().__init__(layers=layers)
        except ActivePrefixCacheError:
            raise
        except Exception as exc:
            raise ActivePrefixCacheError(
                "active-prefix cache could not initialize public Transformers Cache "
                f"(detected transformers {transformers.__version__})"
            ) from exc

    def release(self) -> None:
        """Terminally release every layer's backing storage."""

        for layer in self.layers:
            layer.release()


@dataclass(frozen=True)
class _CacheRecord:
    key: tuple[int, str, str, int]
    batch_size: int
    max_cache_len: int


class _ActivePrefixRuntimeLease(OuterRuntimeLease):
    def __init__(self, runtime, talker, cache, record: _CacheRecord):
        super().__init__(cache)
        self._runtime = runtime
        self._talker = talker
        self._record = record
        self._lock = Lock()
        self._returned = False
        self._clean = cache.get_seq_length() == 0

    def reset(self) -> None:
        with self._lock:
            if self._returned:
                raise ActivePrefixCacheError("active-prefix lease is released")
            self._runtime._reset_cache(self.cache)
            self._clean = True

    def run(self, prepared_decode):
        with self._lock:
            if self._returned:
                raise ActivePrefixCacheError("active-prefix lease is released")
            self._clean = False
            try:
                output = _run_outer_decode_body(
                    self._talker, self.cache, prepared_decode
                )
            except ActivePrefixCacheError as exc:
                if "capacity" in str(exc).lower():
                    self._runtime._record_overflow()
                raise
            self._runtime._record_decode_step(self.cache, self._record)
            return output

    def release(self) -> None:
        with self._lock:
            if self._returned:
                return
            try:
                if not self._clean or self.cache.get_seq_length() != 0:
                    self._runtime._reset_cache(self.cache)
                    self._clean = True
            except BaseException:
                self._returned = True
                self._runtime._discard_cache(self.cache)
                raise
            self._returned = True
        self._runtime._return_cache(self.cache, self._record)


class ActivePrefixOuterStepRuntime:
    """Pool request-owned active-prefix caches behind outer runtime leases."""

    def __init__(self, talker: Any, *, max_cached_leases: int = 2):
        if int(max_cached_leases) < 0:
            raise ActivePrefixCacheError("max_cached_leases must be non-negative")
        self._talker = talker
        self._max_cached_leases = int(max_cached_leases)
        self._lock = RLock()
        self._idle: dict[
            tuple[int, str, str, int], list[ActivePrefixCache]
        ] = {}
        self._records: dict[int, _CacheRecord] = {}
        self._caches: dict[int, ActivePrefixCache] = {}
        self._closed = False
        self._cache_allocations = 0
        self._cache_resets = 0
        self._cache_overflows = 0
        self._active_kv_tokens_per_step: list[int] = []
        self._recorded_active_tokens = 0
        self._recorded_capacity_tokens = 0

    def acquire(self, batch_size, device, dtype, max_cache_len):
        batch_size = int(batch_size)
        max_cache_len = int(max_cache_len)
        if batch_size <= 0 or max_cache_len <= 0:
            raise ActivePrefixCacheError(
                "batch_size and max_cache_len must be positive"
            )
        key = (batch_size, str(device), str(dtype), max_cache_len)
        with self._lock:
            if self._closed:
                raise ActivePrefixCacheError("active-prefix runtime is closed")
            idle = self._idle.get(key)
            if idle:
                cache = idle.pop()
                if not idle:
                    self._idle.pop(key)
                reused = True
            else:
                cache = ActivePrefixCache(
                    num_hidden_layers=self._talker.config.num_hidden_layers,
                    max_cache_len=max_cache_len,
                )
                record = _CacheRecord(key, batch_size, max_cache_len)
                self._records[id(cache)] = record
                self._caches[id(cache)] = cache
                self._cache_allocations += 1
                reused = False
            record = self._records[id(cache)]
        if reused:
            try:
                self._reset_cache(cache)
            except BaseException:
                self._discard_cache(cache)
                raise
        return _ActivePrefixRuntimeLease(self, self._talker, cache, record)

    def build_attention_mask(
        self, prompt_attention_mask, cache_position, max_cache_len, dtype
    ):
        del dtype
        torch = _torch()
        if not torch.is_tensor(prompt_attention_mask) or prompt_attention_mask.ndim != 2:
            raise ActivePrefixCacheError(
                "prompt_attention_mask must be a two-dimensional tensor"
            )
        prompt_length = prompt_attention_mask.shape[1]
        live_length = int(cache_position) + 1
        if live_length < prompt_length or live_length > int(max_cache_len):
            raise ActivePrefixCacheError("active-prefix attention mask exceeds capacity")
        generated_length = live_length - prompt_length
        generated_mask = torch.ones(
            (prompt_attention_mask.shape[0], generated_length),
            dtype=prompt_attention_mask.dtype,
            device=prompt_attention_mask.device,
        )
        return torch.cat((prompt_attention_mask, generated_mask), dim=1)

    def metrics_snapshot(self) -> dict[str, Any]:
        with self._lock:
            allocated_kv_bytes = 0
            backing_capacity_tokens = 0
            for cache_id, record in self._records.items():
                cache = self._caches[cache_id]
                initialized_layers = [
                    layer for layer in cache.layers if layer.is_initialized
                ]
                if not initialized_layers:
                    continue
                backing_capacity_tokens += record.batch_size * record.max_cache_len
                allocated_kv_bytes += sum(
                    layer.keys.numel() * layer.keys.element_size()
                    + layer.values.numel() * layer.values.element_size()
                    for layer in initialized_layers
                )
            return {
                "cache_allocations": self._cache_allocations,
                "allocated_kv_bytes": allocated_kv_bytes,
                "active_kv_tokens_per_step": list(
                    self._active_kv_tokens_per_step
                ),
                "backing_capacity_tokens": backing_capacity_tokens,
                "active_capacity_ratio": (
                    self._recorded_active_tokens / self._recorded_capacity_tokens
                    if self._recorded_capacity_tokens
                    else 0.0
                ),
                "cache_resets": self._cache_resets,
                "cache_overflows": self._cache_overflows,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            idle = [cache for caches in self._idle.values() for cache in caches]
            self._idle.clear()
            for cache in idle:
                self._records.pop(id(cache), None)
                self._caches.pop(id(cache), None)
        for cache in idle:
            self._terminal_release(cache)

    def _reset_cache(self, cache: ActivePrefixCache) -> None:
        cache.reset()
        with self._lock:
            self._cache_resets += 1

    def _record_decode_step(
        self, cache: ActivePrefixCache, record: _CacheRecord
    ) -> None:
        active_tokens = record.batch_size * cache.get_seq_length()
        capacity_tokens = record.batch_size * record.max_cache_len
        with self._lock:
            self._active_kv_tokens_per_step.append(active_tokens)
            self._recorded_active_tokens += active_tokens
            self._recorded_capacity_tokens += capacity_tokens

    def _record_overflow(self) -> None:
        with self._lock:
            self._cache_overflows += 1

    def _return_cache(self, cache: ActivePrefixCache, record: _CacheRecord) -> None:
        terminal = False
        with self._lock:
            if self._closed or self._max_cached_leases == 0:
                self._records.pop(id(cache), None)
                self._caches.pop(id(cache), None)
                terminal = True
            else:
                idle = self._idle.get(record.key)
                if idle is not None and len(idle) >= self._max_cached_leases:
                    self._records.pop(id(cache), None)
                    self._caches.pop(id(cache), None)
                    terminal = True
                else:
                    self._idle.setdefault(record.key, []).append(cache)
        if terminal:
            self._terminal_release(cache)

    def _discard_cache(self, cache: ActivePrefixCache) -> None:
        with self._lock:
            self._records.pop(id(cache), None)
            self._caches.pop(id(cache), None)
        self._terminal_release(cache)

    @staticmethod
    def _terminal_release(cache: ActivePrefixCache) -> None:
        with _torch().inference_mode():
            cache.release()


def install_active_prefix_outer_talker(
    talker: Any, *, max_cache_len: int = 1024
) -> bool:
    """Install the active-prefix outer engine exactly once."""

    current_generate = talker.generate
    if getattr(current_generate, "_qav_active_prefix_outer_talker", False):
        return False
    if any(
        getattr(current_generate, name, False)
        for name in (
            "_qav_static_outer_talker",
            "_qav_explicit_talker_step_engine",
            "_qav_outer_engine",
        )
    ):
        raise ActivePrefixCacheError("another outer talker engine is already installed")

    runtime = ActivePrefixOuterStepRuntime(talker)
    engine = OuterTalkerStaticEngine(
        talker,
        max_cache_len=max_cache_len,
        step_runtime=runtime,
    )

    def active_prefix_generate(**kwargs):
        return engine.generate(**kwargs)

    active_prefix_generate._qav_active_prefix_outer_talker = True  # type: ignore[attr-defined]
    active_prefix_generate._qav_static_outer_talker = True  # type: ignore[attr-defined]
    active_prefix_generate._qav_original_generate = current_generate  # type: ignore[attr-defined]
    active_prefix_generate._qav_outer_engine = engine  # type: ignore[attr-defined]
    talker.generate = active_prefix_generate
    return True


def _torch():
    import torch

    return torch


def _shares_storage_with_backing(tensor, *backings) -> bool:
    storage_pointer = tensor.untyped_storage().data_ptr()
    if storage_pointer == 0:
        return False
    return any(
        tensor.device == backing.device
        and storage_pointer == backing.untyped_storage().data_ptr()
        for backing in backings
    )
