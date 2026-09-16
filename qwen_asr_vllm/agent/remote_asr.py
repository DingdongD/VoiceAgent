"""TCP ASR client/server so the cloud coordinator can drive an edge GPU.

The coordinator's contract is `open_stream` / `feed` / `close`, the same as the
process-isolated runner. This module puts that contract on a length-prefixed JSON
socket. PCM travels as little-endian float32 so a 400 ms chunk is ~25 KiB; events
travel as dicts and are rebuilt into `StreamEvent` on the cloud side.

The protocol is one TCP connection per engine, multiplexed by session id. A
lock serializes RPCs because the cloud coordinator already feeds one session at
a time.
"""

from __future__ import annotations

import base64
import json
import socket
import struct
import threading
from typing import Any

import numpy as np

from qwen_asr_vllm.engine.streaming import StreamEvent

_HEADER = struct.Struct(">I")


def _pcm_to_b64(pcm: np.ndarray) -> str:
    audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
    return base64.b64encode(audio.tobytes()).decode("ascii")


def _b64_to_pcm(payload: str) -> np.ndarray:
    raw = base64.b64decode(payload.encode("ascii"))
    return np.frombuffer(raw, dtype=np.float32).copy()


def event_to_dict(event: StreamEvent) -> dict[str, Any]:
    return {
        "kind": event.kind,
        "text": event.text,
        "committed_text": event.committed_text,
        "audio_seconds": float(event.audio_seconds),
        "output_token_ids": list(event.output_token_ids or []),
        "chunk_index": int(event.chunk_index),
        "commit_violation": bool(event.commit_violation),
    }


def event_from_dict(payload: dict[str, Any]) -> StreamEvent:
    return StreamEvent(
        kind=str(payload.get("kind") or ""),
        text=str(payload.get("text") or ""),
        committed_text=str(payload.get("committed_text") or ""),
        audio_seconds=float(payload.get("audio_seconds") or 0.0),
        output_token_ids=list(payload.get("output_token_ids") or []),
        chunk_index=int(payload.get("chunk_index") or 0),
        commit_violation=bool(payload.get("commit_violation")),
    )


def send_message(sock: socket.socket, message: dict[str, Any]) -> None:
    body = json.dumps(message, ensure_ascii=False).encode("utf-8")
    sock.sendall(_HEADER.pack(len(body)) + body)


def recv_message(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exact(sock, _HEADER.size)
    (length,) = _HEADER.unpack(header)
    if length > 32 * 1024 * 1024:
        raise RuntimeError(f"ASR remote frame too large: {length} bytes")
    body = _recv_exact(sock, length)
    return json.loads(body.decode("utf-8"))


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        piece = sock.recv(size - len(chunks))
        if not piece:
            raise ConnectionError("ASR remote connection closed")
        chunks.extend(piece)
    return bytes(chunks)


class RemoteAsrSession:
    def __init__(self, engine: "RemoteAsrEngine", session_id: int):
        self._engine = engine
        self._session_id = session_id

    def feed(self, pcm: np.ndarray) -> list[StreamEvent]:
        reply = self._engine.rpc(
            {
                "op": "feed",
                "session_id": self._session_id,
                "pcm_b64": _pcm_to_b64(pcm),
            }
        )
        return [event_from_dict(item) for item in reply.get("events") or []]

    def close(self, reuse_last: bool = True) -> list[StreamEvent]:
        reply = self._engine.rpc(
            {
                "op": "close_session",
                "session_id": self._session_id,
                "reuse_last": bool(reuse_last),
            }
        )
        return [event_from_dict(item) for item in reply.get("events") or []]


class RemoteAsrEngine:
    """Cloud-side ASR engine that forwards feeds to an edge process."""

    def __init__(self, host: str, port: int, *, timeout: float = 300.0):
        self.host = host
        self.port = int(port)
        self._timeout = float(timeout)
        self._lock = threading.Lock()
        self._sock = socket.create_connection((self.host, self.port), timeout=self._timeout)
        self._sock.settimeout(self._timeout)
        hello = self.rpc({"op": "hello"})
        self._startup_metrics = dict(hello.get("metrics") or {})
        self._startup_metrics["remote"] = f"{self.host}:{self.port}"

    def rpc(self, message: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            send_message(self._sock, message)
            reply = recv_message(self._sock)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error") or "ASR remote RPC failed")
        return reply

    def open_stream(self, **kwargs) -> RemoteAsrSession:
        reply = self.rpc({"op": "open_stream", "kwargs": dict(kwargs)})
        return RemoteAsrSession(self, int(reply["session_id"]))

    def close(self) -> None:
        with self._lock:
            if self._sock is None:
                return
            try:
                send_message(self._sock, {"op": "shutdown"})
            except OSError:
                pass
            try:
                self._sock.close()
            finally:
                self._sock = None  # type: ignore[assignment]

    def runtime_metrics(self) -> dict[str, Any]:
        return dict(self._startup_metrics)


def handle_connection(sock: socket.socket, engine: Any) -> None:
    """Serve one client against a local `open_stream` ASR engine."""
    sessions: dict[int, Any] = {}
    next_id = 1
    metrics = {}
    runtime_metrics = getattr(engine, "runtime_metrics", None)
    if callable(runtime_metrics):
        try:
            metrics = dict(runtime_metrics())
        except Exception:  # noqa: BLE001 - metrics must not take the server down
            metrics = {"runtime_metrics_error": True}
    try:
        while True:
            request = recv_message(sock)
            op = request.get("op")
            if op == "hello":
                send_message(sock, {"ok": True, "metrics": metrics})
                continue
            if op == "shutdown":
                send_message(sock, {"ok": True})
                return
            if op == "open_stream":
                session_id = next_id
                next_id += 1
                sessions[session_id] = engine.open_stream(**(request.get("kwargs") or {}))
                send_message(sock, {"ok": True, "session_id": session_id})
                continue
            if op == "feed":
                session = sessions.get(int(request["session_id"]))
                if session is None:
                    send_message(sock, {"ok": False, "error": "unknown session"})
                    continue
                pcm = _b64_to_pcm(str(request.get("pcm_b64") or ""))
                events = [event_to_dict(event) for event in session.feed(pcm)]
                send_message(sock, {"ok": True, "events": events})
                continue
            if op == "close_session":
                session_id = int(request["session_id"])
                session = sessions.pop(session_id, None)
                if session is None:
                    send_message(sock, {"ok": True, "events": []})
                    continue
                events = [
                    event_to_dict(event)
                    for event in session.close(reuse_last=bool(request.get("reuse_last", True)))
                ]
                send_message(sock, {"ok": True, "events": events})
                continue
            send_message(sock, {"ok": False, "error": f"unknown op: {op}"})
    except ConnectionError:
        return
    finally:
        for session in sessions.values():
            close = getattr(session, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - best-effort teardown
                    pass


def serve_asr_engine(engine: Any, host: str, port: int) -> socket.socket:
    """Bind and accept in a background thread. Returns the listening socket."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(8)
    listener.settimeout(1.0)

    def loop() -> None:
        while True:
            try:
                client, _addr = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            client.settimeout(300.0)
            threading.Thread(
                target=handle_connection,
                args=(client, engine),
                daemon=True,
            ).start()

    threading.Thread(target=loop, daemon=True).start()
    return listener
