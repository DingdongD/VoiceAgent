"""TCP TTS client/server so the cloud LLM can drive edge synthesis and playback.

The coordinator already speaks in text fragments and consumes WAV chunks. This
module puts that generator on a length-prefixed JSON socket: one `synthesize_stream`
RPC yields zero or more `audio` frames then a `done`. Playback therefore happens
where the chunks are received; the cloud can still record them for timing.
"""

from __future__ import annotations

import base64
import json
import socket
import struct
import threading
from typing import Any, Iterable

from qwen_asr_vllm.agent.remote_asr import recv_message, send_message


class RemoteTtsBackend:
    """Cloud-side TTS backend that forwards text to an edge synthesizer."""

    supports_streaming_tts = True

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
            raise RuntimeError(reply.get("error") or "TTS remote RPC failed")
        return reply

    def synthesize(self, text: str) -> bytes | None:
        chunks = list(self.synthesize_stream(text))
        audio = b"".join(chunk for chunk in chunks if chunk)
        return audio or None

    def synthesize_stream(self, text: str) -> Iterable[bytes]:
        with self._lock:
            send_message(self._sock, {"op": "synthesize_stream", "text": text})
            while True:
                reply = recv_message(self._sock)
                if not reply.get("ok"):
                    raise RuntimeError(reply.get("error") or "TTS remote stream failed")
                if reply.get("done"):
                    return
                payload = reply.get("audio_b64") or ""
                if payload:
                    yield base64.b64decode(payload.encode("ascii"))

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


def handle_tts_connection(sock: socket.socket, backend: Any) -> None:
    metrics = {}
    runtime_metrics = getattr(backend, "runtime_metrics", None)
    if callable(runtime_metrics):
        try:
            metrics = dict(runtime_metrics())
        except Exception:  # noqa: BLE001
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
            if op == "synthesize_stream":
                text = str(request.get("text") or "")
                try:
                    stream = backend.synthesize_stream(text)
                    for chunk in stream:
                        if not chunk:
                            continue
                        send_message(
                            sock,
                            {
                                "ok": True,
                                "audio_b64": base64.b64encode(chunk).decode("ascii"),
                            },
                        )
                    send_message(sock, {"ok": True, "done": True})
                except Exception as exc:  # noqa: BLE001
                    send_message(sock, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                continue
            send_message(sock, {"ok": False, "error": f"unknown op: {op}"})
    except (ConnectionError, json.JSONDecodeError, struct.error):
        return


def serve_tts_backend(backend: Any, host: str, port: int) -> socket.socket:
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
                target=handle_tts_connection,
                args=(client, backend),
                daemon=True,
            ).start()

    threading.Thread(target=loop, daemon=True).start()
    return listener
