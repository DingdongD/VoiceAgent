from __future__ import annotations

import io
import inspect
import queue
import threading
import time
import wave
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class CodecAudioChunk:
    index: int
    code_start: int
    code_end: int
    context_start: int
    context_frames: int
    audio: np.ndarray
    sample_rate: int
    decode_ms: float
    wav_bytes: bytes


class _CodecFrameBuffer:
    """Small growable row-major buffer for frames arriving from the talker.

    The Qwen-TTS package does not expose a stateful waveform decoder. Keeping
    the generated codec prefix in one contiguous host buffer still removes a
    repeated ``np.stack(all_frames)`` on every decode boundary and gives a
    future native decoder a stable prefix interface.
    """

    def __init__(self, *, initial_capacity: int = 8):
        self._capacity = max(1, int(initial_capacity))
        self._length = 0
        self._width: int | None = None
        self._buffer: np.ndarray | None = None

    def __len__(self) -> int:
        return self._length

    def append(self, frame: Any) -> None:
        array = np.asarray(frame, dtype=np.int64).reshape(-1)
        if self._width is None:
            self._width = int(array.size)
            if self._width <= 0:
                raise ValueError("codec frame must contain at least one code")
            self._buffer = np.empty(
                (self._capacity, self._width),
                dtype=np.int64,
            )
        elif int(array.size) != self._width:
            raise ValueError(
                f"codec frame width {array.size} does not match {self._width}"
            )
        if self._length >= self._capacity:
            assert self._buffer is not None
            self._capacity = max(self._capacity * 2, self._length + 1)
            grown = np.empty((self._capacity, self._width), dtype=np.int64)
            grown[: self._length] = self._buffer[: self._length]
            self._buffer = grown
        assert self._buffer is not None
        self._buffer[self._length] = array
        self._length += 1

    def prefix(self, end: int) -> np.ndarray:
        end = min(max(0, int(end)), self._length)
        if self._buffer is None:
            width = self._width or 0
            return np.empty((0, width), dtype=np.int64)
        return self._buffer[:end]


def chunked_decode_codec_codes(
    speech_tokenizer: Any,
    audio_codes: Any,
    *,
    chunk_size: int,
    left_context_size: int = 0,
    decode_upsample_rate: int | None = None,
) -> Iterable[CodecAudioChunk]:
    """Decode pre-generated Qwen-TTS codec codes in chunks.

    This is a stage-A probe helper: codec generation is still full-sequence, but
    codec-to-waveform decode is split with left context to test chunk boundaries.
    """

    chunk_size = max(1, int(chunk_size))
    left_context_size = max(0, int(left_context_size))
    upsample_rate = _decode_upsample_rate(speech_tokenizer, decode_upsample_rate)
    code_length = int(audio_codes.shape[0])

    index = 0
    code_start = 0
    while code_start < code_length:
        code_end = min(code_start + chunk_size, code_length)
        context_start = max(0, code_start - left_context_size)
        context_frames = code_start - context_start
        codes_chunk = audio_codes[context_start:code_end]

        started = time.perf_counter()
        wavs, sample_rate = speech_tokenizer.decode([{"audio_codes": codes_chunk}])
        decode_ms = (time.perf_counter() - started) * 1000
        if not wavs:
            audio = np.zeros(0, dtype=np.float32)
        else:
            audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        trim = min(audio.size, context_frames * upsample_rate)
        if trim:
            audio = audio[trim:]

        yield CodecAudioChunk(
            index=index,
            code_start=code_start,
            code_end=code_end,
            context_start=context_start,
            context_frames=context_frames,
            audio=audio,
            sample_rate=int(sample_rate),
            decode_ms=decode_ms,
            wav_bytes=numpy_to_wav(audio, int(sample_rate)),
        )

        index += 1
        code_start = code_end


def stream_decode_codec_frames(
    run_generate,
    speech_tokenizer: Any,
    *,
    chunk_size: int,
    first_chunk_size: int | None = None,
    left_context_size: int = 0,
    eos_token_id: int | None = None,
) -> Iterable[CodecAudioChunk]:
    """Decode codec frames while a generator is still producing them.

    `run_generate` is called in a background thread with one callback argument.
    The callback must be invoked with one codec frame shaped `(num_code_groups,)`
    as soon as it is available.
    """

    chunk_size = max(1, int(chunk_size))
    first_chunk_size = (
        max(1, int(first_chunk_size)) if first_chunk_size is not None else chunk_size
    )
    left_context_size = max(0, int(left_context_size))
    frames: queue.Queue[Any] = queue.Queue()
    sentinel = object()

    def on_codec_frame(frame: Any) -> None:
        array = _to_numpy_frame(frame)
        if eos_token_id is not None and array.size and int(array[0]) == int(eos_token_id):
            return
        frames.put(array)

    def worker() -> None:
        try:
            run_generate(on_codec_frame)
        except BaseException as exc:  # noqa: BLE001 - surfaced to consumer
            frames.put(exc)
        finally:
            frames.put(sentinel)

    thread = threading.Thread(target=worker, name="qwen-tts-codec-stream", daemon=True)
    thread.start()

    codec_frames = _CodecFrameBuffer()
    next_decode_start = 0
    chunk_index = 0
    finished = False
    while not finished:
        item = frames.get()
        if item is sentinel:
            finished = True
        elif isinstance(item, BaseException):
            thread.join(timeout=1)
            raise item
        else:
            codec_frames.append(item)

        current_chunk_size = first_chunk_size if chunk_index == 0 else chunk_size
        while len(codec_frames) - next_decode_start >= current_chunk_size or (
            finished and next_decode_start < len(codec_frames)
        ):
            current_chunk_size = first_chunk_size if chunk_index == 0 else chunk_size
            code_end = (
                next_decode_start + current_chunk_size
                if len(codec_frames) - next_decode_start >= current_chunk_size
                else len(codec_frames)
            )
            codes = codec_frames.prefix(code_end)
            chunk = _decode_codec_chunk(
                speech_tokenizer,
                codes,
                chunk_index=chunk_index,
                code_start=next_decode_start,
                code_end=code_end,
                left_context_size=left_context_size,
            )
            yield chunk
            chunk_index += 1
            next_decode_start = code_end

    thread.join(timeout=1)


def stream_decode_codec_frame_batches(
    run_generate,
    speech_tokenizer: Any,
    *,
    request_ids: Iterable[Any],
    chunk_size: int,
    first_chunk_size: int | None = None,
    left_context_size: int = 0,
    eos_token_id: int | None = None,
) -> Iterable[tuple[Any, CodecAudioChunk]]:
    """Decode batched codec frames while one batched generator is running."""

    request_ids = list(request_ids)
    if not request_ids:
        return
    chunk_size = max(1, int(chunk_size))
    first_chunk_size = (
        max(1, int(first_chunk_size)) if first_chunk_size is not None else chunk_size
    )
    left_context_size = max(0, int(left_context_size))
    frame_batches: queue.Queue[Any] = queue.Queue()
    sentinel = object()

    def on_codec_frame_batch(frames: Any) -> None:
        frame_batches.put(_to_numpy_frame_batch(frames))

    def worker() -> None:
        try:
            run_generate(on_codec_frame_batch)
        except BaseException as exc:  # noqa: BLE001 - surfaced to consumer
            frame_batches.put(exc)
        finally:
            frame_batches.put(sentinel)

    thread = threading.Thread(
        target=worker, name="qwen-tts-codec-batch-stream", daemon=True
    )
    thread.start()

    codec_frames = [_CodecFrameBuffer() for _ in request_ids]
    next_decode_starts = [0 for _ in request_ids]
    chunk_indexes = [0 for _ in request_ids]
    finished_requests = [False for _ in request_ids]
    finished = False
    while not finished:
        item = frame_batches.get()
        if item is sentinel:
            finished = True
        elif isinstance(item, BaseException):
            thread.join(timeout=1)
            raise item
        else:
            batch = np.asarray(item, dtype=np.int64)
            if batch.shape[0] != len(request_ids):
                raise ValueError(
                    f"codec frame batch size {batch.shape[0]} does not match "
                    f"{len(request_ids)} request ids"
                )
            for index, frame in enumerate(batch):
                if finished_requests[index]:
                    continue
                if (
                    eos_token_id is not None
                    and frame.size
                    and int(frame.reshape(-1)[0]) == int(eos_token_id)
                ):
                    finished_requests[index] = True
                    continue
                codec_frames[index].append(frame)

        for request_id, index, chunk in _iter_ready_batch_chunks(
            request_ids,
            codec_frames,
            next_decode_starts,
            chunk_indexes,
            speech_tokenizer,
            chunk_size=chunk_size,
            first_chunk_size=first_chunk_size,
            left_context_size=left_context_size,
            flush=finished,
        ):
            yield request_id, chunk

    thread.join(timeout=1)


def _iter_ready_batch_chunks(
    request_ids: list[Any],
    codec_frames: list[_CodecFrameBuffer],
    next_decode_starts: list[int],
    chunk_indexes: list[int],
    speech_tokenizer: Any,
    *,
    chunk_size: int,
    first_chunk_size: int,
    left_context_size: int,
    flush: bool,
) -> Iterable[tuple[Any, int, CodecAudioChunk]]:
    while True:
        emitted = False
        for index, request_id in enumerate(request_ids):
            frames = codec_frames[index]
            next_decode_start = next_decode_starts[index]
            current_chunk_size = (
                first_chunk_size if chunk_indexes[index] == 0 else chunk_size
            )
            while len(frames) - next_decode_start >= current_chunk_size or (
                flush and next_decode_start < len(frames)
            ):
                current_chunk_size = (
                    first_chunk_size if chunk_indexes[index] == 0 else chunk_size
                )
                code_end = (
                    next_decode_start + current_chunk_size
                    if len(frames) - next_decode_start >= current_chunk_size
                    else len(frames)
                )
                codes = frames.prefix(code_end)
                chunk = _decode_codec_chunk(
                    speech_tokenizer,
                    codes,
                    chunk_index=chunk_indexes[index],
                    code_start=next_decode_start,
                    code_end=code_end,
                    left_context_size=left_context_size,
                )
                yield request_id, index, chunk
                next_decode_start = code_end
                next_decode_starts[index] = next_decode_start
                chunk_indexes[index] += 1
                emitted = True
        if not emitted:
            return


def run_with_codec_frame_hook(talker: Any, run_generate, on_codec_frame):
    """Run generation while observing per-step codec frames from talker.forward."""

    outer_engine = getattr(
        getattr(talker, "generate", None), "_qav_outer_engine", None
    )
    outer_callback = getattr(outer_engine, "codec_frame_callback", None)
    if callable(outer_callback):
        with outer_callback(on_codec_frame):
            return run_generate()

    original_forward = talker.forward

    def hooked_forward(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        frame = _extract_codec_frame(outputs)
        if frame is not None:
            on_codec_frame(frame)
        return outputs

    hooked_forward.__signature__ = inspect.signature(original_forward)  # type: ignore[attr-defined]
    talker.forward = hooked_forward
    try:
        return run_generate()
    finally:
        talker.forward = original_forward


def run_with_codec_frame_batch_hook(talker: Any, run_generate, on_codec_frame_batch):
    """Run generation while observing batched per-step codec frames."""

    outer_engine = getattr(
        getattr(talker, "generate", None), "_qav_outer_engine", None
    )
    outer_callback = getattr(outer_engine, "codec_frame_callback", None)
    if callable(outer_callback):
        with outer_callback(on_codec_frame_batch):
            return run_generate()

    original_forward = talker.forward

    def hooked_forward(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        frames = _extract_codec_frame_batch(outputs)
        if frames is not None:
            on_codec_frame_batch(frames)
        return outputs

    hooked_forward.__signature__ = inspect.signature(original_forward)  # type: ignore[attr-defined]
    talker.forward = hooked_forward
    try:
        return run_generate()
    finally:
        talker.forward = original_forward


def concatenate_audio_chunks(chunks: Iterable[CodecAudioChunk]) -> np.ndarray:
    audios = [np.asarray(chunk.audio, dtype=np.float32).reshape(-1) for chunk in chunks]
    if not audios:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(audios)


def numpy_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def _decode_codec_chunk(
    speech_tokenizer: Any,
    audio_codes: Any,
    *,
    chunk_index: int,
    code_start: int,
    code_end: int,
    left_context_size: int,
    decode_upsample_rate: int | None = None,
) -> CodecAudioChunk:
    upsample_rate = _decode_upsample_rate(speech_tokenizer, decode_upsample_rate)
    context_start = max(0, code_start - left_context_size)
    context_frames = code_start - context_start
    codes_chunk = audio_codes[context_start:code_end]
    started = time.perf_counter()
    wavs, sample_rate = speech_tokenizer.decode([{"audio_codes": codes_chunk}])
    decode_ms = (time.perf_counter() - started) * 1000
    if not wavs:
        audio = np.zeros(0, dtype=np.float32)
    else:
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    trim = min(audio.size, context_frames * upsample_rate)
    if trim:
        audio = audio[trim:]
    return CodecAudioChunk(
        index=chunk_index,
        code_start=code_start,
        code_end=code_end,
        context_start=context_start,
        context_frames=context_frames,
        audio=audio,
        sample_rate=int(sample_rate),
        decode_ms=decode_ms,
        wav_bytes=numpy_to_wav(audio, int(sample_rate)),
    )


def _to_numpy_frame(frame: Any) -> np.ndarray:
    detach = getattr(frame, "detach", None)
    if callable(detach):
        frame = detach()
    cpu = getattr(frame, "cpu", None)
    if callable(cpu):
        frame = cpu()
    numpy = getattr(frame, "numpy", None)
    if callable(numpy):
        frame = numpy()
    return np.asarray(frame, dtype=np.int64).reshape(-1)


def _to_numpy_frame_batch(frames: Any) -> np.ndarray:
    detach = getattr(frames, "detach", None)
    if callable(detach):
        frames = detach()
    cpu = getattr(frames, "cpu", None)
    if callable(cpu):
        frames = cpu()
    numpy = getattr(frames, "numpy", None)
    if callable(numpy):
        frames = numpy()
    array = np.asarray(frames, dtype=np.int64)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array.reshape(array.shape[0], -1)


def _extract_codec_frame(outputs: Any) -> np.ndarray | None:
    batch = _extract_codec_frame_batch(outputs)
    if batch is None:
        return None
    return np.asarray(batch[0], dtype=np.int64).reshape(-1)


def _extract_codec_frame_batch(outputs: Any) -> np.ndarray | None:
    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states or len(hidden_states) < 2:
        return None
    codec_ids = hidden_states[1]
    if codec_ids is None:
        return None
    array = _to_numpy_frame(codec_ids)
    if array.size == 0:
        return None
    return _to_numpy_frame_batch(codec_ids)


def _decode_upsample_rate(speech_tokenizer: Any, override: int | None) -> int:
    if override is not None:
        return max(1, int(override))
    getter = getattr(speech_tokenizer, "get_decode_upsample_rate", None)
    if callable(getter):
        return max(1, int(getter()))
    return 1
