import numpy as np

from qwen_asr_vllm.agent.remote_tts import RemoteTtsBackend, serve_tts_backend


class FakeRemoteTts:
    supports_streaming_tts = True

    def __init__(self):
        self.texts = []

    def synthesize_stream(self, text):
        self.texts.append(text)
        yield b"chunk-a:" + text.encode()
        yield b"chunk-b:" + text.encode()

    def runtime_metrics(self):
        return {"backend": "fake-edge-tts"}


def test_remote_tts_streams_chunks_over_localhost():
    backend = FakeRemoteTts()
    listener = serve_tts_backend(backend, "127.0.0.1", 0)
    host, port = listener.getsockname()
    client = RemoteTtsBackend(host, port, timeout=5.0)
    try:
        assert client.runtime_metrics()["backend"] == "fake-edge-tts"
        chunks = list(client.synthesize_stream("hello"))
        assert chunks == [b"chunk-a:hello", b"chunk-b:hello"]
        assert backend.texts == ["hello"]
        whole = client.synthesize("hello")
        assert whole == b"chunk-a:hellochunk-b:hello"
    finally:
        client.close()
        listener.close()
