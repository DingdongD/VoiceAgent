from types import SimpleNamespace

import numpy as np

from qwen_asr_vllm.agent.local_tts import (
    QwenTtsBackend,
    _group_stream_batch_texts,
)


class FakeSpeechTokenizer:
    def __init__(self):
        self.requests = []

    def get_decode_upsample_rate(self):
        return 2

    def decode(self, encoded):
        codes = np.asarray(encoded[0]["audio_codes"])
        self.requests.append(codes.copy())
        audio = np.repeat(codes[:, 0].astype(np.float32), 2)
        return [audio], 16000


class FakeTalker:
    def __init__(self):
        self._next = 0
        self.batch_size = 1

    def forward(self, **_kwargs):
        if self.batch_size == 1:
            frame = np.array([[self._next, self._next + 100]], dtype=np.int64)
        else:
            frame = np.array(
                [
                    [self._next + offset * 10, self._next + offset * 10 + 100]
                    for offset in range(self.batch_size)
                ],
                dtype=np.int64,
            )
        self._next += 1
        return SimpleNamespace(hidden_states=(None, frame))


class FakeQwenModel:
    def __init__(self):
        self.talker = FakeTalker()
        self.speech_tokenizer = FakeSpeechTokenizer()
        self.config = SimpleNamespace(
            talker_config=SimpleNamespace(codec_eos_token_id=-1)
        )
        self.calls = []

    def generate_custom_voice(self, **kwargs):
        self.calls.append(kwargs)
        texts = kwargs["text"] if isinstance(kwargs["text"], list) else [kwargs["text"]]
        self.talker.batch_size = len(texts)
        for _ in range(6):
            self.talker.forward()
        self.talker.batch_size = 1
        return [np.arange(12, dtype=np.float32)], 16000


def make_backend(*, streaming=True):
    backend = QwenTtsBackend.__new__(QwenTtsBackend)
    backend._language = "chinese"
    backend._speaker = "speaker"
    backend._model = FakeQwenModel()
    backend._streaming = streaming
    backend._stream_chunk_size = 4
    backend._stream_first_chunk_size = 2
    backend._stream_left_context_size = 1
    backend._stream_batch_exact_parity = False
    backend._stream_batch_parity_gate = True
    backend._stream_batch_gate_stats = {
        "native_batches": 0,
        "scalar_fallback_batches": 0,
        "scalar_fallback_requests": 0,
    }
    backend._generation_kwargs = {}
    backend.supports_streaming_tts = streaming
    return backend


def test_stream_batch_groups_texts_by_tokenized_prompt_length():
    class FakeWrapper:
        def _build_assistant_text(self, text):
            return text

        def _tokenize_texts(self, texts):
            lengths = {"short": 4, "long-a": 7, "long-b": 7}
            return [SimpleNamespace(shape=(1, lengths[text])) for text in texts]

    groups = _group_stream_batch_texts(FakeWrapper(), ["short", "long-a", "long-b"])

    assert groups == [([0], ["short"]), ([1, 2], ["long-a", "long-b"])]


def test_stream_batch_compatibility_key_separates_prompt_lengths_and_fallbacks():
    backend = make_backend(streaming=True)
    backend._model._build_assistant_text = lambda text: text
    lengths = {"a": 4, "b": 4, "long": 7}
    backend._model._tokenize_texts = lambda texts: [
        SimpleNamespace(shape=(1, lengths[text])) for text in texts
    ]

    assert backend.stream_batch_compatibility_key("a") == (
        "native",
        4,
    )
    assert backend.stream_batch_compatibility_key("a") == (
        backend.stream_batch_compatibility_key("b")
    )
    assert backend.stream_batch_compatibility_key("a") != (
        backend.stream_batch_compatibility_key("long")
    )

    backend._generation_kwargs = {"do_sample": True}
    assert backend.stream_batch_compatibility_key("a") != (
        backend.stream_batch_compatibility_key("b")
    )


def test_qwen_tts_backend_streams_codec_step_chunks_from_local_model():
    backend = make_backend(streaming=True)

    chunks = list(backend.synthesize_stream("你好"))

    assert len(chunks) == 2
    assert all(chunk.startswith(b"RIFF") for chunk in chunks)
    assert backend._model.calls == [
        {
            "text": "你好",
            "speaker": "speaker",
            "language": "chinese",
            "non_streaming_mode": True,
        }
    ]
    assert [request[:, 0].tolist() for request in backend._model.speech_tokenizer.requests] == [
        [0, 1],
        [1, 2, 3, 4, 5],
    ]


def test_qwen_tts_backend_stream_falls_back_to_full_synthesis_when_disabled():
    backend = make_backend(streaming=False)

    chunks = list(backend.synthesize_stream("你好"))

    assert len(chunks) == 1
    assert chunks[0].startswith(b"RIFF")
    assert backend._model.speech_tokenizer.requests == []


def test_qwen_tts_backend_streams_batched_codec_steps_from_local_model():
    backend = make_backend(streaming=True)

    chunks = list(backend.synthesize_stream_batch(["你好", "世界"]))

    assert {request_id for request_id, _chunk in chunks} == {0, 1}
    assert all(chunk.startswith(b"RIFF") for _request_id, chunk in chunks)
    assert backend._model.calls == [
        {
            "text": ["你好", "世界"],
            "speaker": ["speaker", "speaker"],
            "language": ["chinese", "chinese"],
            "non_streaming_mode": True,
        }
    ]
    assert backend._stream_batch_gate_stats == {
        "native_batches": 1,
        "scalar_fallback_batches": 0,
        "scalar_fallback_requests": 0,
    }


def test_qwen_tts_backend_parity_gate_falls_back_for_sampling_requests():
    backend = make_backend(streaming=True)
    backend._generation_kwargs = {"do_sample": True}

    chunks = list(backend.synthesize_stream_batch(["你好", "世界"]))

    assert {request_id for request_id, _chunk in chunks} == {0, 1}
    assert [call["text"] for call in backend._model.calls] == ["你好", "世界"]
    assert backend._stream_batch_gate_stats["native_batches"] == 0
    assert backend._stream_batch_gate_stats["scalar_fallback_requests"] == 2


def test_qwen_tts_backend_exact_stream_batch_mode_keeps_requests_independent():
    backend = make_backend(streaming=True)
    backend._stream_batch_exact_parity = True

    chunks = list(backend.synthesize_stream_batch(["你好", "世界"]))

    assert {request_id for request_id, _chunk in chunks} == {0, 1}
    assert backend._model.calls == [
        {
            "text": "你好",
            "speaker": "speaker",
            "language": "chinese",
            "non_streaming_mode": True,
        },
        {
            "text": "世界",
            "speaker": "speaker",
            "language": "chinese",
            "non_streaming_mode": True,
        },
    ]


def test_qwen_tts_backend_exact_stream_batch_mode_reuses_scalar_stream_decoder():
    backend = make_backend(streaming=True)
    backend._stream_batch_exact_parity = True
    calls = []

    def scalar_stream(text):
        calls.append(text)
        yield f"scalar:{text}".encode()

    backend.synthesize_stream = scalar_stream

    chunks = list(backend.synthesize_stream_batch(["你好", "世界"]))

    assert calls == ["你好", "世界"]
    assert chunks == [(0, b"scalar:\xe4\xbd\xa0\xe5\xa5\xbd"), (1, b"scalar:\xe4\xb8\x96\xe7\x95\x8c")]
