import time
from types import SimpleNamespace

import numpy as np

from qwen_asr_vllm.agent.qwen_tts_streaming import (
    _CodecFrameBuffer,
    chunked_decode_codec_codes,
    concatenate_audio_chunks,
    run_with_codec_frame_batch_hook,
    run_with_codec_frame_hook,
    stream_decode_codec_frame_batches,
    stream_decode_codec_frames,
)


def test_codec_frame_buffer_exposes_only_the_generated_prefix():
    buffer = _CodecFrameBuffer(initial_capacity=2)
    buffer.append(np.array([1, 2], dtype=np.int64))
    buffer.append(np.array([3, 4], dtype=np.int64))
    buffer.append(np.array([5, 6], dtype=np.int64))

    assert len(buffer) == 3
    np.testing.assert_array_equal(
        buffer.prefix(2),
        np.array([[1, 2], [3, 4]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        buffer.prefix(3),
        np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64),
    )


class FakeSpeechTokenizer:
    def __init__(self, *, upsample_rate=4, sample_rate=16000):
        self.upsample_rate = upsample_rate
        self.sample_rate = sample_rate
        self.requests = []

    def get_decode_upsample_rate(self):
        return self.upsample_rate

    def decode(self, encoded):
        codes = np.asarray(encoded[0]["audio_codes"])
        self.requests.append(codes.copy())
        values = np.repeat(codes[:, 0].astype(np.float32), self.upsample_rate)
        return [values], self.sample_rate


def test_chunked_decode_codec_codes_trims_left_context_and_preserves_audio_order():
    tokenizer = FakeSpeechTokenizer(upsample_rate=4)
    codes = np.arange(10, dtype=np.int64).reshape(10, 1)

    chunks = list(
        chunked_decode_codec_codes(
            tokenizer,
            codes,
            chunk_size=4,
            left_context_size=2,
        )
    )

    assert [(chunk.code_start, chunk.code_end) for chunk in chunks] == [
        (0, 4),
        (4, 8),
        (8, 10),
    ]
    assert [request[:, 0].tolist() for request in tokenizer.requests] == [
        [0, 1, 2, 3],
        [2, 3, 4, 5, 6, 7],
        [6, 7, 8, 9],
    ]
    decoded = concatenate_audio_chunks(chunks)
    np.testing.assert_array_equal(decoded, np.repeat(np.arange(10, dtype=np.float32), 4))


def test_chunked_decode_codec_codes_emits_wav_bytes_for_each_chunk():
    tokenizer = FakeSpeechTokenizer(upsample_rate=2)
    codes = np.arange(5, dtype=np.int64).reshape(5, 1)

    chunks = list(chunked_decode_codec_codes(tokenizer, codes, chunk_size=3))

    assert len(chunks) == 2
    assert all(chunk.wav_bytes[:4] == b"RIFF" for chunk in chunks)
    assert all(chunk.sample_rate == 16000 for chunk in chunks)


def test_stream_decode_codec_frames_decodes_before_generation_finishes():
    generation_done = False
    decode_done_states = []

    class ObservingTokenizer(FakeSpeechTokenizer):
        def decode(self, encoded):
            decode_done_states.append(generation_done)
            return super().decode(encoded)

    tokenizer = ObservingTokenizer(upsample_rate=2)

    def run_generate(on_codec_frame):
        nonlocal generation_done
        for value in range(5):
            on_codec_frame(np.array([value], dtype=np.int64))
            time.sleep(0.01)
        generation_done = True

    chunks = list(
        stream_decode_codec_frames(
            run_generate,
            tokenizer,
            chunk_size=2,
            left_context_size=1,
        )
    )

    assert [(chunk.code_start, chunk.code_end) for chunk in chunks] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    assert decode_done_states[0] is False
    decoded = concatenate_audio_chunks(chunks)
    np.testing.assert_array_equal(decoded, np.repeat(np.arange(5, dtype=np.float32), 2))


def test_stream_decode_codec_frames_uses_adaptive_first_chunk_size():
    tokenizer = FakeSpeechTokenizer(upsample_rate=2)

    def run_generate(on_codec_frame):
        for value in range(10):
            on_codec_frame(np.array([value], dtype=np.int64))

    chunks = list(
        stream_decode_codec_frames(
            run_generate,
            tokenizer,
            chunk_size=4,
            first_chunk_size=2,
            left_context_size=1,
        )
    )

    assert [(chunk.code_start, chunk.code_end) for chunk in chunks] == [
        (0, 2),
        (2, 6),
        (6, 10),
    ]
    assert [request[:, 0].tolist() for request in tokenizer.requests] == [
        [0, 1],
        [1, 2, 3, 4, 5],
        [5, 6, 7, 8, 9],
    ]


def test_stream_decode_codec_frame_batches_demuxes_request_chunks():
    tokenizer = FakeSpeechTokenizer(upsample_rate=2)

    def run_generate(on_codec_frame_batch):
        for value in range(5):
            on_codec_frame_batch(
                np.array([[value], [value + 10]], dtype=np.int64)
            )

    chunks = list(
        stream_decode_codec_frame_batches(
            run_generate,
            tokenizer,
            request_ids=["a", "b"],
            chunk_size=2,
            first_chunk_size=1,
            left_context_size=1,
        )
    )

    assert [(request_id, chunk.code_start, chunk.code_end) for request_id, chunk in chunks] == [
        ("a", 0, 1),
        ("b", 0, 1),
        ("a", 1, 3),
        ("b", 1, 3),
        ("a", 3, 5),
        ("b", 3, 5),
    ]
    decoded = {
        request_id: np.concatenate([chunk.audio for rid, chunk in chunks if rid == request_id])
        for request_id in ("a", "b")
    }
    np.testing.assert_array_equal(decoded["a"], np.repeat(np.arange(5, dtype=np.float32), 2))
    np.testing.assert_array_equal(decoded["b"], np.repeat(np.arange(10, 15, dtype=np.float32), 2))


def test_run_with_codec_frame_hook_extracts_frames_and_restores_forward():
    class FakeTalker:
        def __init__(self):
            self.calls = 0

        def forward(self):
            self.calls += 1
            frame = np.array([[self.calls, self.calls + 10]], dtype=np.int64)
            return SimpleNamespace(hidden_states=(None, frame))

    talker = FakeTalker()
    original_forward = talker.forward
    frames = []

    def run_generate():
        talker.forward()
        talker.forward()
        return "ok"

    result = run_with_codec_frame_hook(talker, run_generate, frames.append)

    assert result == "ok"
    assert [frame.tolist() for frame in frames] == [[1, 11], [2, 12]]
    assert talker.forward == original_forward


def test_run_with_codec_frame_hook_uses_outer_engine_callback():
    from contextlib import contextmanager

    class FakeOuterEngine:
        def __init__(self):
            self.callback = None

        @contextmanager
        def codec_frame_callback(self, callback):
            previous = self.callback
            self.callback = callback
            try:
                yield
            finally:
                self.callback = previous

        def emit(self, frame):
            self.callback(frame)

    class FakeGenerate:
        _qav_outer_engine = FakeOuterEngine()

    class FakeTalker:
        generate = FakeGenerate()

        def forward(self):
            raise AssertionError("outer engine path must not patch talker.forward")

    frames = []
    talker = FakeTalker()

    def run_generate():
        talker.generate._qav_outer_engine.emit(np.array([[3, 4]], dtype=np.int64))
        return "ok"

    assert run_with_codec_frame_hook(talker, run_generate, frames.append) == "ok"
    assert [frame.tolist() for frame in frames] == [[[3, 4]]]
    assert talker.generate._qav_outer_engine.callback is None


def test_run_with_codec_frame_batch_hook_extracts_batch_and_restores_forward():
    class FakeTalker:
        def __init__(self):
            self.calls = 0

        def forward(self):
            self.calls += 1
            frame = np.array(
                [[self.calls, self.calls + 10], [self.calls + 20, self.calls + 30]],
                dtype=np.int64,
            )
            return SimpleNamespace(hidden_states=(None, frame))

    talker = FakeTalker()
    original_forward = talker.forward
    batches = []

    def run_generate():
        talker.forward()
        return "ok"

    result = run_with_codec_frame_batch_hook(talker, run_generate, batches.append)

    assert result == "ok"
    assert [batch.tolist() for batch in batches] == [[[1, 11], [21, 31]]]
    assert talker.forward == original_forward
