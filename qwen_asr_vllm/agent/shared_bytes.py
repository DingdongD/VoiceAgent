from __future__ import annotations

import threading
from dataclasses import dataclass
from inspect import signature
from multiprocessing import resource_tracker, shared_memory
from typing import Any


@dataclass(frozen=True)
class SharedBytesDescriptor:
    name: str
    size: int


_SUPPORTS_TRACK = "track" in signature(shared_memory.SharedMemory).parameters


def _create_segment(size: int):
    if _SUPPORTS_TRACK:
        return shared_memory.SharedMemory(create=True, size=size, track=False)
    segment = shared_memory.SharedMemory(create=True, size=size)
    resource_tracker.unregister(segment._name, "shared_memory")
    return segment


def _open_segment(name: str):
    if _SUPPORTS_TRACK:
        return shared_memory.SharedMemory(name=name, track=False)
    return shared_memory.SharedMemory(name=name)


class SharedBytesTransport:
    """Thresholded shared-memory transport for immutable byte payloads."""

    def __init__(self, *, threshold: int = 64 * 1024):
        self._threshold = max(1, int(threshold))
        self._lock = threading.Lock()
        self._owned_names: set[str] = set()
        self._closed = False
        self._created_segments = 0
        self._created_bytes = 0
        self._received_segments = 0
        self._received_bytes = 0
        self._unlinked_segments = 0
        self._cleanup_segments = 0

    def encode(self, payload: Any) -> Any:
        if not isinstance(payload, bytes) or len(payload) < self._threshold:
            return payload
        with self._lock:
            if self._closed:
                raise RuntimeError("shared bytes transport is closed")
        segment = _create_segment(len(payload))
        try:
            segment.buf[: len(payload)] = payload
            descriptor = SharedBytesDescriptor(segment.name, len(payload))
            with self._lock:
                self._owned_names.add(segment.name)
                self._created_segments += 1
                self._created_bytes += len(payload)
            return descriptor
        except BaseException:
            if not _SUPPORTS_TRACK:
                resource_tracker.register(segment._name, "shared_memory")
            segment.unlink()
            raise
        finally:
            segment.close()

    def decode(self, payload: Any) -> Any:
        if not isinstance(payload, SharedBytesDescriptor):
            return payload
        if payload.size < 0:
            raise ValueError("shared byte payload size must be non-negative")
        segment = _open_segment(payload.name)
        unlinked = False
        try:
            if payload.size > segment.size:
                raise ValueError("shared byte payload exceeds its segment")
            return bytes(segment.buf[: payload.size])
        finally:
            segment.close()
            try:
                segment.unlink()
                unlinked = True
            except FileNotFoundError:
                pass
            with self._lock:
                self._received_segments += 1
                self._received_bytes += payload.size
                self._unlinked_segments += int(unlinked)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            names = tuple(self._owned_names)
            self._owned_names.clear()
        cleaned = 0
        for name in names:
            try:
                segment = _open_segment(name)
            except FileNotFoundError:
                continue
            try:
                segment.unlink()
                cleaned += 1
            finally:
                segment.close()
        with self._lock:
            self._cleanup_segments += cleaned

    def metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "created_segments": self._created_segments,
                "created_bytes": self._created_bytes,
                "received_segments": self._received_segments,
                "received_bytes": self._received_bytes,
                "unlinked_segments": self._unlinked_segments,
                "cleanup_segments": self._cleanup_segments,
                "outstanding_segments": len(self._owned_names),
                "threshold": self._threshold,
            }
