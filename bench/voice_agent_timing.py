"""Timing harness for sync vs async voice-agent scheduling.

Fake mode is deterministic and requires no models:

    python bench/voice_agent_timing.py --mode fake

Real mode tries to reuse `/home/voice_assistant_app` backends and needs local ASR,
LLM and TTS checkpoints plus an input WAV:

    python bench/voice_agent_timing.py --mode real --audio /path/to/input.wav
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import time
import wave
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_asr_vllm.agent.async_coordinator import (
    BARGE_IN_POLICIES,
    AsyncVoiceAgentCoordinator,
)
from qwen_asr_vllm.agent.coordinator import VoiceAgentCoordinator
from qwen_asr_vllm.agent.local_tts import QwenTtsBackend
from qwen_asr_vllm.agent.nano_llm import NanoVllmStepBatchingBackend
from qwen_asr_vllm.agent.runtime_profiles import (
    RUNTIME_PROFILES,
    activate_runtime_profile_environment,
    default_runtime_profile,
    parse_profiled_args,
)
from qwen_asr_vllm.agent.service_runners import (
    ProcessAsrEngine,
    ProcessConcurrentTtsBackend,
    ProcessNanoLlmBackend,
    ProcessTtsBackend,
)
from qwen_asr_vllm.engine.streaming import StreamEvent


@dataclass
class TimingResult:
    name: str
    first_audio_ms: float | None
    total_ms: float
    events: list[str]
    errors: list[str] | None = None
    event_offsets_ms: dict[str, float] | None = None
    details: dict | None = None

    def as_dict(self) -> dict:
        event_offsets = self.event_offsets_ms or {}
        details = dict(self.details or {})
        derived = _derive_stage_metrics(event_offsets, details)
        if derived:
            details["derived_ms"] = derived
        return {
            "name": self.name,
            "first_audio_ms": round(self.first_audio_ms, 1)
            if self.first_audio_ms is not None
            else None,
            "total_ms": round(self.total_ms, 1),
            "events": self.events,
            "errors": self.errors or [],
            "event_offsets_ms": {
                key: round(value, 1)
                for key, value in (self.event_offsets_ms or {}).items()
            },
            "details": details,
        }


def _derive_stage_metrics(event_offsets: dict[str, float], details: dict) -> dict[str, float]:
    derived: dict[str, float] = {}

    def add(name: str, end: str, start: str) -> None:
        if end in event_offsets and start in event_offsets:
            derived[name] = round(event_offsets[end] - event_offsets[start], 1)

    add("llm_ttft", "llm_first_chunk", "llm_start")
    add("llm_to_first_tts_text", "llm_sentence_ready", "llm_first_chunk")
    add("tts_wait", "tts_first_chunk", "llm_sentence_ready")
    add("tts_compute_wait", "tts_audio_ready", "llm_sentence_ready")
    add("tts_release_delay", "tts_first_chunk", "tts_audio_ready")
    add("committed_lead_vs_final", "asr_final", "asr_committed")
    if "input_wall_ms" in details and "tts_first_chunk" in event_offsets:
        derived["first_audio_after_input_end"] = round(
            event_offsets["tts_first_chunk"] - float(details["input_wall_ms"]), 1
        )
    return derived


class FakeAsrSession:
    def __init__(self, trigger: str):
        self.trigger = trigger

    def feed(self, pcm):
        if self.trigger != "committed":
            return []
        return [
            StreamEvent(
                kind="committed",
                text="status",
                committed_text="status",
                audio_seconds=0.4,
            )
        ]

    def close(self, reuse_last=True):
        return [
            StreamEvent(
                kind="final",
                text="status",
                committed_text="status" if self.trigger == "committed" else "",
                audio_seconds=0.8,
            )
        ]


class FakeAsrEngine:
    def __init__(self, trigger: str):
        self.trigger = trigger

    def open_stream(self, **kwargs):
        return FakeAsrSession(self.trigger)


class SlowLlm:
    def __init__(self, chunks: Iterable[str], delay: float):
        self.chunks = list(chunks)
        self.delay = float(delay)

    def chat_stream(self, text):
        for chunk in self.chunks:
            time.sleep(self.delay)
            yield chunk


class SlowTts:
    def __init__(self, delay: float):
        self.delay = float(delay)

    def synthesize(self, text):
        time.sleep(self.delay)
        return ("wav:" + text).encode()


def _first_audio_from_sync(events, started: float, ended: float) -> float | None:
    for event in events:
        if event.type == "tts_chunk":
            # Sync coordinator returns only after the blocking call completes; this is
            # the first instant a caller could send audio.
            return (ended - started) * 1000
    return None


def run_fake_sync(args) -> TimingResult:
    engine = FakeAsrEngine(trigger="final")
    llm = SlowLlm(["First.", " Second."], args.llm_delay)
    tts = SlowTts(args.tts_delay)
    coordinator = VoiceAgentCoordinator(engine, llm, tts, llm_trigger="final")

    started = time.perf_counter()
    time.sleep(args.asr_final_delay)
    events = coordinator.close()
    ended = time.perf_counter()
    return TimingResult(
        name="sync_final_blocking",
        first_audio_ms=_first_audio_from_sync(events, started, ended),
        total_ms=(ended - started) * 1000,
        events=[event.type for event in events],
    )


async def run_fake_async(args) -> TimingResult:
    engine = FakeAsrEngine(trigger=args.llm_trigger)
    llm = SlowLlm(["First.", " Second."], args.llm_delay)
    tts = SlowTts(args.tts_delay)
    coordinator = AsyncVoiceAgentCoordinator(
        engine,
        llm,
        tts,
        llm_trigger=args.llm_trigger,
        tts_concurrency=args.tts_concurrency,
        tts_flush_chars=args.tts_flush_chars,
        tts_flush_after_ms=args.tts_flush_after_ms,
        tts_flush_min_chars=args.tts_flush_min_chars,
        tts_coalesce_chars=args.tts_coalesce_chars,
        tts_coalesce_wait_ms=args.tts_coalesce_wait_ms,
        tts_first_sentence_immediate=args.tts_first_sentence_immediate,
        tts_defer_short_segments_chars=args.tts_defer_short_segments_chars,
        tts_defer_short_segments_ms=args.tts_defer_short_segments_ms,
        tts_stream_first_segment_only=args.tts_stream_first_segment_only,
        min_committed_words=args.min_committed_words,
        min_committed_audio_seconds=args.min_committed_audio_seconds,
        max_committed_audio_seconds=args.max_committed_audio_seconds,
        defer_tts_audio_until_asr_final=args.defer_tts_audio_until_asr_final,
        tts_playback_preroll_ms=args.tts_playback_preroll_ms,
        barge_in_policy=args.barge_in_policy,
        barge_in_rms_threshold=args.barge_in_rms_threshold,
    )

    started = time.perf_counter()
    first_audio = None
    names: list[str] = []
    errors: list[str] = []
    offsets: dict[str, float] = {}

    async def collect():
        nonlocal first_audio
        while True:
            event = await coordinator.next_event(timeout=5)
            names.append(event.type)
            offsets.setdefault(event.type, (time.perf_counter() - started) * 1000)
            if event.type == "error":
                errors.append(str(event.as_dict()))
            if event.type == "tts_chunk" and first_audio is None:
                first_audio = (time.perf_counter() - started) * 1000
            if event.type == "done":
                return

    collector = asyncio.create_task(collect())
    if args.llm_trigger == "committed":
        time.sleep(args.asr_commit_delay)
        await coordinator.feed(np.zeros(1600, dtype=np.float32))
        await asyncio.sleep(max(0.0, args.asr_final_delay - args.asr_commit_delay))
        await coordinator.close()
    else:
        await asyncio.sleep(args.asr_final_delay)
        await coordinator.close()
    await collector
    ended = time.perf_counter()
    return TimingResult(
        name=f"async_{args.llm_trigger}",
        first_audio_ms=first_audio,
        total_ms=(ended - started) * 1000,
        events=names,
        errors=errors,
        event_offsets_ms=offsets,
    )


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError("real mode expects 16-bit PCM WAV")
        raw = handle.readframes(handle.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        channels = handle.getnchannels()
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio


def ensure_voice_app_path() -> None:
    app_root = Path("/home/voice_assistant_app")
    if not app_root.is_dir():
        raise RuntimeError("/home/voice_assistant_app is not present")
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))


def create_real_asr_backend():
    ensure_voice_app_path()
    from src.asr_backend import create_asr_backend

    return create_asr_backend()


def create_real_llm_agent(max_num_seqs: int | None = None):
    ensure_voice_app_path()
    from src import config

    configured_max_num_seqs = (
        int(max_num_seqs) if max_num_seqs is not None else int(config.LLM_MAX_NUM_SEQS)
    )
    return NanoVllmStepBatchingBackend(
        model_path=config.LLM_MODEL_PATH,
        nanovllm_root=config.NANOVLLM_ROOT,
        device=config.LLM_DEVICE,
        max_model_len=int(config.LLM_MAX_MODEL_LEN),
        max_num_batched_tokens=max(int(config.LLM_MAX_MODEL_LEN), 2048),
        max_num_seqs=configured_max_num_seqs,
        gpu_memory_utilization=float(config.LLM_GPU_MEMORY_UTILIZATION),
        num_kvcache_blocks=int(getattr(config, "LLM_NUM_KVCACHE_BLOCKS", -1)),
        warmup_batch_sizes=tuple(getattr(config, "LLM_WARMUP_BATCH_SIZES", (1,))),
        enforce_eager=bool(config.LLM_ENFORCE_EAGER),
        system_prompt=config.LLM_SYSTEM_PROMPT,
        temperature=float(config.LLM_TEMPERATURE),
        max_new_tokens=int(config.LLM_MAX_NEW_TOKENS),
    )


def create_real_tts_agent(
    *,
    streaming: bool = False,
    stream_chunk_size: int = 8,
    stream_first_chunk_size: int | None = None,
    stream_left_context_size: int = 4,
    stream_batch_exact_parity: bool = True,
    fast_code_predictor: bool = False,
    static_code_predictor: bool = False,
    cuda_graph_code_predictor: bool = False,
    cuda_graph_fixed_slots: int = 1,
    cuda_graph_batch_window_ms: float = 0.0,
    fast_code_predictor_batch_window_ms: float = 0.0,
    fast_code_predictor_max_batch_size: int = 8,
    explicit_talker_step_engine: bool = False,
    outer_active_prefix_talker_engine: bool = False,
    outer_cuda_graph_talker_engine: bool = False,
    outer_graph_fixed_slots: int = 2,
    outer_graph_max_cache_len: int = 16384,
    compile_step_engine: bool = False,
    compile_step_engine_mode: str = "reduce-overhead",
    do_sample: bool | None = None,
    subtalker_dosample: bool | None = None,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
    eos_token_id: int | None = None,
):
    ensure_voice_app_path()
    from src import config

    backend_kwargs = dict(
        model_path=config.TTS_MODEL_PATH,
        device=config.TTS_DEVICE,
        language=config.TTS_LANGUAGE,
        speaker=config.TTS_SPEAKER,
        streaming=streaming,
        stream_chunk_size=stream_chunk_size,
        stream_first_chunk_size=stream_first_chunk_size,
        stream_left_context_size=stream_left_context_size,
        stream_batch_exact_parity=stream_batch_exact_parity,
        fast_code_predictor=fast_code_predictor,
        static_code_predictor=static_code_predictor,
        cuda_graph_code_predictor=cuda_graph_code_predictor,
        cuda_graph_fixed_slots=cuda_graph_fixed_slots,
        cuda_graph_batch_window_ms=cuda_graph_batch_window_ms,
        fast_code_predictor_batch_window_ms=fast_code_predictor_batch_window_ms,
        fast_code_predictor_max_batch_size=fast_code_predictor_max_batch_size,
        explicit_talker_step_engine=explicit_talker_step_engine,
        outer_active_prefix_talker_engine=outer_active_prefix_talker_engine,
        outer_cuda_graph_talker_engine=outer_cuda_graph_talker_engine,
        outer_graph_fixed_slots=outer_graph_fixed_slots,
        outer_graph_max_cache_len=outer_graph_max_cache_len,
        compile_step_engine=compile_step_engine,
        compile_step_engine_mode=compile_step_engine_mode,
    )
    generation_kwargs = {
        key: value
        for key, value in {
            "do_sample": do_sample,
            "subtalker_dosample": subtalker_dosample,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": eos_token_id,
        }.items()
        if value is not None
    }
    backend_kwargs.update(generation_kwargs)
    return QwenTtsBackend(**backend_kwargs)


def resolve_tts_process_batching(
    *,
    stream_workers: int,
    batch_window_ms: float,
    max_batch_size: int,
    cuda_graph_code_predictor: bool,
    cuda_graph_fixed_slots: int,
    exact_parity: bool = False,
) -> tuple[int, float, int]:
    if exact_parity:
        if os.environ.get("VOICE_TTS_REQUEST_STEP_SCHEDULER") == "1":
            raise ValueError(
                "strict request-id codec scheduler is incompatible with exact parity"
            )
        return 1, 0.0, 1
    workers = max(1, int(stream_workers))
    window_ms = max(0.0, float(batch_window_ms))
    max_batch = max(1, int(max_batch_size))
    slots = max(1, int(cuda_graph_fixed_slots))
    if cuda_graph_code_predictor and slots > 1:
        workers = max(workers, slots)
        window_ms = window_ms or 10.0
        max_batch = min(max_batch, slots)
    return workers, window_ms, max_batch


def _validate_strict_tts_architecture(
    *,
    process_isolated: bool,
    tts_streaming: bool,
    tts_stream_batch_exact_parity: bool,
    tts_cuda_graph_code_predictor: bool,
    tts_outer_active_prefix_talker_engine: bool,
    tts_do_sample: bool | None,
    tts_subtalker_do_sample: bool | None,
    tts_fast_code_predictor: bool,
    tts_static_code_predictor: bool,
    tts_explicit_talker_step_engine: bool,
    tts_outer_cuda_graph_talker_engine: bool,
    tts_compile_step_engine: bool,
) -> None:
    if os.environ.get("VOICE_TTS_REQUEST_STEP_SCHEDULER") != "1":
        return
    errors = []
    if not process_isolated:
        errors.append("process-isolated TTS service")
    if not tts_streaming:
        errors.append("codec-step streaming")
    if tts_stream_batch_exact_parity:
        errors.append("legacy exact parity")
    if not tts_cuda_graph_code_predictor:
        errors.append("CUDA Graph code predictor")
    if not tts_outer_active_prefix_talker_engine:
        errors.append("active-prefix outer talker")
    if tts_do_sample is not False:
        errors.append("greedy outer decoding")
    if tts_subtalker_do_sample is not False:
        errors.append("greedy codec decoding")
    if tts_fast_code_predictor or tts_static_code_predictor:
        errors.append("single inner predictor engine")
    if tts_explicit_talker_step_engine or tts_outer_cuda_graph_talker_engine:
        errors.append("single active-prefix outer engine")
    if tts_compile_step_engine:
        errors.append("resident CUDA Graph predictor without compile")
    if errors:
        raise ValueError(
            "strict request-id codec scheduler requires " + ", ".join(errors)
        )


def load_real_backends(
    timeout: float = 300.0,
    *,
    process_isolated: bool = False,
    llm_max_num_seqs: int | None = None,
    tts_streaming: bool = False,
    tts_stream_chunk_size: int = 8,
    tts_stream_first_chunk_size: int | None = None,
    tts_stream_left_context_size: int = 4,
    tts_stream_batch_exact_parity: bool = True,
    tts_fast_code_predictor: bool = False,
    tts_static_code_predictor: bool = False,
    tts_cuda_graph_code_predictor: bool = False,
    tts_cuda_graph_fixed_slots: int = 1,
    tts_cuda_graph_batch_window_ms: float = 0.0,
    tts_fast_code_predictor_batch_window_ms: float = 0.0,
    tts_fast_code_predictor_max_batch_size: int = 8,
    tts_explicit_talker_step_engine: bool = False,
    tts_outer_active_prefix_talker_engine: bool = False,
    tts_outer_cuda_graph_talker_engine: bool = False,
    tts_outer_graph_fixed_slots: int = 2,
    tts_outer_graph_max_cache_len: int = 16384,
    tts_compile_step_engine: bool = False,
    tts_compile_step_engine_mode: str = "reduce-overhead",
    tts_do_sample: bool | None = None,
    tts_subtalker_do_sample: bool | None = None,
    tts_temperature: float | None = None,
    tts_max_new_tokens: int | None = None,
    tts_eos_token_id: int | None = None,
    tts_process_stream_workers: int = 1,
    tts_process_batch_window_ms: float = 0.0,
    tts_process_max_batch_size: int = 8,
    tts_shared_memory_threshold_bytes: int = 64 * 1024,
    asr_remote: str | None = None,
    tts_remote: str | None = None,
):
    _validate_strict_tts_architecture(
        process_isolated=process_isolated,
        tts_streaming=tts_streaming,
        tts_stream_batch_exact_parity=tts_stream_batch_exact_parity,
        tts_cuda_graph_code_predictor=tts_cuda_graph_code_predictor,
        tts_outer_active_prefix_talker_engine=tts_outer_active_prefix_talker_engine,
        tts_do_sample=tts_do_sample,
        tts_subtalker_do_sample=tts_subtalker_do_sample,
        tts_fast_code_predictor=tts_fast_code_predictor,
        tts_static_code_predictor=tts_static_code_predictor,
        tts_explicit_talker_step_engine=tts_explicit_talker_step_engine,
        tts_outer_cuda_graph_talker_engine=tts_outer_cuda_graph_talker_engine,
        tts_compile_step_engine=tts_compile_step_engine,
    )
    tts_factory = partial(
        create_real_tts_agent,
        streaming=tts_streaming,
        stream_chunk_size=tts_stream_chunk_size,
        stream_first_chunk_size=tts_stream_first_chunk_size,
        stream_left_context_size=tts_stream_left_context_size,
        stream_batch_exact_parity=tts_stream_batch_exact_parity,
        fast_code_predictor=tts_fast_code_predictor,
        static_code_predictor=tts_static_code_predictor,
        cuda_graph_code_predictor=tts_cuda_graph_code_predictor,
        cuda_graph_fixed_slots=tts_cuda_graph_fixed_slots,
        cuda_graph_batch_window_ms=tts_cuda_graph_batch_window_ms,
        fast_code_predictor_batch_window_ms=tts_fast_code_predictor_batch_window_ms,
        fast_code_predictor_max_batch_size=tts_fast_code_predictor_max_batch_size,
        explicit_talker_step_engine=tts_explicit_talker_step_engine,
        outer_active_prefix_talker_engine=tts_outer_active_prefix_talker_engine,
        outer_cuda_graph_talker_engine=tts_outer_cuda_graph_talker_engine,
        outer_graph_fixed_slots=tts_outer_graph_fixed_slots,
        outer_graph_max_cache_len=tts_outer_graph_max_cache_len,
        compile_step_engine=tts_compile_step_engine,
        compile_step_engine_mode=tts_compile_step_engine_mode,
        do_sample=tts_do_sample,
        subtalker_dosample=tts_subtalker_do_sample,
        temperature=tts_temperature,
        max_new_tokens=tts_max_new_tokens,
        eos_token_id=tts_eos_token_id,
    )
    if process_isolated:
        if asr_remote:
            from qwen_asr_vllm.agent.remote_asr import RemoteAsrEngine

            host, port_text = asr_remote.rsplit(":", 1)
            asr_backend = RemoteAsrEngine(host, int(port_text), timeout=timeout)
        else:
            asr_backend = ProcessAsrEngine(create_real_asr_backend, timeout=timeout)
        (
            tts_process_stream_workers,
            tts_process_batch_window_ms,
            tts_process_max_batch_size,
        ) = resolve_tts_process_batching(
            stream_workers=tts_process_stream_workers,
            batch_window_ms=tts_process_batch_window_ms,
            max_batch_size=tts_process_max_batch_size,
            cuda_graph_code_predictor=tts_cuda_graph_code_predictor,
            cuda_graph_fixed_slots=tts_cuda_graph_fixed_slots,
            exact_parity=tts_stream_batch_exact_parity,
        )
        tts_cls = (
            ProcessConcurrentTtsBackend
            if int(tts_process_stream_workers) > 1
            or float(tts_process_batch_window_ms) > 0.0
            else ProcessTtsBackend
        )
        tts_kwargs = {"shared_memory_threshold": int(tts_shared_memory_threshold_bytes)}
        if tts_cls is ProcessConcurrentTtsBackend:
            tts_kwargs.update(
                max_workers=int(tts_process_stream_workers),
                batch_window_ms=float(tts_process_batch_window_ms),
                max_batch_size=int(tts_process_max_batch_size),
            )
        if tts_remote:
            from qwen_asr_vllm.agent.remote_tts import RemoteTtsBackend

            host, port_text = tts_remote.rsplit(":", 1)
            tts_backend = RemoteTtsBackend(host, int(port_text), timeout=timeout)
        else:
            tts_backend = tts_cls(tts_factory, timeout=timeout, **tts_kwargs)
        return (
            asr_backend,
            ProcessNanoLlmBackend(
                partial(create_real_llm_agent, max_num_seqs=llm_max_num_seqs),
                timeout=timeout,
            ),
            tts_backend,
        )
    return (
        create_real_asr_backend(),
        create_real_llm_agent(max_num_seqs=llm_max_num_seqs),
        tts_factory(),
    )


def run_real_sync(pcm: np.ndarray, args) -> TimingResult:
    ensure_voice_app_path()
    from src.asr_stream import transcribe_pcm_chunked

    asr, llm, tts = load_real_backends(
        args.timeout,
        process_isolated=True,
        llm_max_num_seqs=args.llm_max_num_seqs,
        tts_streaming=args.tts_streaming_engine == "codec-step",
        tts_stream_chunk_size=args.tts_stream_chunk_size,
        tts_stream_first_chunk_size=args.tts_stream_first_chunk_size,
        tts_stream_left_context_size=args.tts_stream_left_context_size,
        tts_stream_batch_exact_parity=args.tts_stream_batch_exact_parity,
        tts_fast_code_predictor=args.tts_fast_code_predictor,
        tts_static_code_predictor=args.tts_static_code_predictor,
        tts_cuda_graph_code_predictor=args.tts_cuda_graph_code_predictor,
        tts_cuda_graph_fixed_slots=args.tts_cuda_graph_fixed_slots,
        tts_cuda_graph_batch_window_ms=args.tts_cuda_graph_batch_window_ms,
        tts_fast_code_predictor_batch_window_ms=args.tts_fast_code_predictor_batch_window_ms,
        tts_fast_code_predictor_max_batch_size=args.tts_fast_code_predictor_max_batch_size,
        tts_explicit_talker_step_engine=args.tts_explicit_talker_step_engine,
        tts_outer_active_prefix_talker_engine=args.tts_outer_active_prefix_talker_engine,
        tts_outer_cuda_graph_talker_engine=args.tts_outer_cuda_graph_talker_engine,
        tts_outer_graph_fixed_slots=args.tts_outer_graph_fixed_slots,
        tts_outer_graph_max_cache_len=args.tts_outer_graph_max_cache_len,
        tts_compile_step_engine=args.tts_compile_step_engine,
        tts_compile_step_engine_mode=args.tts_compile_step_engine_mode,
        tts_do_sample=args.tts_do_sample,
        tts_subtalker_do_sample=args.tts_subtalker_do_sample,
        tts_temperature=args.tts_temperature,
        tts_max_new_tokens=args.tts_max_new_tokens,
        tts_eos_token_id=args.tts_eos_token_id,
        tts_process_stream_workers=args.tts_process_stream_workers,
        tts_process_batch_window_ms=args.tts_process_batch_window_ms,
        tts_process_max_batch_size=args.tts_process_max_batch_size,
        tts_shared_memory_threshold_bytes=args.tts_shared_memory_threshold_bytes,
        asr_remote=args.asr_remote,
        tts_remote=args.tts_remote,
    )
    started = time.perf_counter()
    transcript = transcribe_pcm_chunked(asr, pcm)
    names = ["asr_final"]
    sentence = ""
    first_audio = None
    for chunk in llm.chat_stream(transcript.text or ""):
        names.append("llm_chunk")
        sentence += chunk
        if any(mark in sentence for mark in ".!?。！？\n"):
            audio = tts.synthesize(sentence.strip())
            names.append("tts_chunk")
            if audio and first_audio is None:
                first_audio = (time.perf_counter() - started) * 1000
            sentence = ""
    if sentence.strip():
        audio = tts.synthesize(sentence.strip())
        names.append("tts_chunk")
        if audio and first_audio is None:
            first_audio = (time.perf_counter() - started) * 1000
    ended = time.perf_counter()
    asr.close()
    llm.close()
    close_tts = getattr(tts, "close", None)
    if callable(close_tts):
        close_tts()
    return TimingResult("real_sync_final", first_audio, (ended - started) * 1000, names)


async def run_real_async(pcm: np.ndarray, args) -> TimingResult:
    asr, llm, tts = load_real_backends(
        args.timeout,
        process_isolated=True,
        llm_max_num_seqs=args.llm_max_num_seqs,
        tts_streaming=args.tts_streaming_engine == "codec-step",
        tts_stream_chunk_size=args.tts_stream_chunk_size,
        tts_stream_first_chunk_size=args.tts_stream_first_chunk_size,
        tts_stream_left_context_size=args.tts_stream_left_context_size,
        tts_stream_batch_exact_parity=args.tts_stream_batch_exact_parity,
        tts_fast_code_predictor=args.tts_fast_code_predictor,
        tts_static_code_predictor=args.tts_static_code_predictor,
        tts_cuda_graph_code_predictor=args.tts_cuda_graph_code_predictor,
        tts_cuda_graph_fixed_slots=args.tts_cuda_graph_fixed_slots,
        tts_cuda_graph_batch_window_ms=args.tts_cuda_graph_batch_window_ms,
        tts_fast_code_predictor_batch_window_ms=args.tts_fast_code_predictor_batch_window_ms,
        tts_fast_code_predictor_max_batch_size=args.tts_fast_code_predictor_max_batch_size,
        tts_explicit_talker_step_engine=args.tts_explicit_talker_step_engine,
        tts_outer_active_prefix_talker_engine=args.tts_outer_active_prefix_talker_engine,
        tts_outer_cuda_graph_talker_engine=args.tts_outer_cuda_graph_talker_engine,
        tts_outer_graph_fixed_slots=args.tts_outer_graph_fixed_slots,
        tts_outer_graph_max_cache_len=args.tts_outer_graph_max_cache_len,
        tts_compile_step_engine=args.tts_compile_step_engine,
        tts_compile_step_engine_mode=args.tts_compile_step_engine_mode,
        tts_do_sample=args.tts_do_sample,
        tts_subtalker_do_sample=args.tts_subtalker_do_sample,
        tts_temperature=args.tts_temperature,
        tts_max_new_tokens=args.tts_max_new_tokens,
        tts_eos_token_id=args.tts_eos_token_id,
        tts_process_stream_workers=args.tts_process_stream_workers,
        tts_process_batch_window_ms=args.tts_process_batch_window_ms,
        tts_process_max_batch_size=args.tts_process_max_batch_size,
        tts_shared_memory_threshold_bytes=args.tts_shared_memory_threshold_bytes,
        asr_remote=args.asr_remote,
        tts_remote=args.tts_remote,
    )
    coordinator = AsyncVoiceAgentCoordinator(
        asr,
        llm,
        tts,
        llm_trigger=args.llm_trigger,
        asr_kwargs={
            "language": args.language,
            "chunk_policy": args.asr_policy,
            "commit_lag_words": args.commit_lag_words,
        },
        tts_concurrency=args.tts_concurrency,
        tts_flush_chars=args.tts_flush_chars,
        tts_flush_after_ms=args.tts_flush_after_ms,
        tts_flush_min_chars=args.tts_flush_min_chars,
        tts_coalesce_chars=args.tts_coalesce_chars,
        tts_coalesce_wait_ms=args.tts_coalesce_wait_ms,
        tts_first_sentence_immediate=args.tts_first_sentence_immediate,
        tts_defer_short_segments_chars=args.tts_defer_short_segments_chars,
        tts_defer_short_segments_ms=args.tts_defer_short_segments_ms,
        tts_stream_first_segment_only=args.tts_stream_first_segment_only,
        min_committed_words=args.min_committed_words,
        min_committed_audio_seconds=args.min_committed_audio_seconds,
        max_committed_audio_seconds=args.max_committed_audio_seconds,
        defer_tts_audio_until_asr_final=args.defer_tts_audio_until_asr_final,
        tts_playback_preroll_ms=args.tts_playback_preroll_ms,
        barge_in_policy=args.barge_in_policy,
        barge_in_rms_threshold=args.barge_in_rms_threshold,
    )
    started = time.perf_counter()
    first_audio = None
    names: list[str] = []
    errors: list[str] = []
    offsets: dict[str, float] = {}
    last_offsets: dict[str, float] = {}
    details: dict = {
        "tts_chunk_timeline": [],
        "asr_timeline": [],
        "asr_hypothesis": "",
        # With llm_trigger=committed the LLM is prompted with a *prefix* of the
        # utterance, not the final transcript. Recording both is the only way to
        # see whether a latency win silently changed what the agent answered.
        "llm_prompt": "",
        "asr_final_text": "",
        "llm_output": "",
        "tts_inputs": [],
        "tts_rendered_inputs": [],
        "tts_streaming": None,
        "first_audio_bytes": None,
        "tts_chunk_count": 0,
        "tts_audio_bytes": 0,
        "tts_audio_duration_ms": 0.0,
    }
    input_complete = asyncio.Event()

    async def collect():
        nonlocal first_audio
        def record(event) -> None:
            nonlocal first_audio
            payload = event.as_dict()
            names.append(event.type)
            now_ms = (time.perf_counter() - started) * 1000
            offsets.setdefault(event.type, now_ms)
            # First-occurrence offsets cannot show whether a stage kept up with
            # real time: they say when audio started, never when it stopped.
            # Last offsets plus the per-chunk timeline give the emission rate.
            last_offsets[event.type] = round(now_ms, 1)
            if event.type == "error":
                errors.append(str(payload))
            if event.type.startswith("asr_"):
                text = payload.get("text") or payload.get("committed_text") or ""
                committed = payload.get("committed_text") or ""
                details["asr_hypothesis"] = (text or committed).strip()
                # How early each word is available, and whether it later changed,
                # is what decides if a downstream stage can start on a prefix.
                # Counts alone cannot answer that.
                details["asr_timeline"].append(
                    {
                        "at_ms": round(now_ms, 1),
                        "kind": event.type.removeprefix("asr_"),
                        "text": (text or committed).strip(),
                        "committed": committed.strip(),
                    }
                )
                if event.type == "asr_final":
                    details["asr_final_text"] = (text or committed).strip()
            if event.type == "llm_start" and not details["llm_prompt"]:
                details["llm_prompt"] = str(payload.get("text") or "").strip()
            if event.type == "llm_chunk":
                details["llm_output"] += str(payload.get("text") or "")
            if event.type == "llm_sentence_ready":
                details["tts_inputs"].append(str(payload.get("text") or ""))
            if event.type == "tts_audio_ready":
                text = str(payload.get("text") or "")
                if text not in details["tts_rendered_inputs"]:
                    details["tts_rendered_inputs"].append(text)
            if event.type == "tts_chunk" and first_audio is None:
                first_audio = (time.perf_counter() - started) * 1000
                audio = payload.get("audio")
                details["tts_streaming"] = bool(payload.get("tts_streaming"))
                details["first_audio_bytes"] = len(audio) if audio is not None else 0
            if event.type == "tts_chunk":
                audio = payload.get("audio")
                if audio:
                    details["tts_chunk_count"] += 1
                    details["tts_audio_bytes"] += len(audio)
                    chunk_ms = None
                    try:
                        with wave.open(io.BytesIO(audio), "rb") as wav:
                            chunk_ms = wav.getnframes() / wav.getframerate() * 1000.0
                            if details["tts_audio_duration_ms"] is not None:
                                details["tts_audio_duration_ms"] += chunk_ms
                    except (EOFError, wave.Error):
                        details["tts_audio_duration_ms"] = None
                    details["tts_chunk_timeline"].append(
                        {
                            "at_ms": round(now_ms, 1),
                            "audio_ms": None if chunk_ms is None else round(chunk_ms, 1),
                        }
                    )
                text = str(payload.get("text") or "")
                if text and text not in details["tts_rendered_inputs"]:
                    details["tts_rendered_inputs"].append(text)

        await collect_until_input_complete(
            lambda: coordinator.next_event(timeout=args.timeout),
            record,
            input_complete,
        )

    collector = asyncio.create_task(collect())
    chunk = int(args.chunk_ms * 16000 / 1000)
    input_started = time.perf_counter()
    for start in range(0, len(pcm), chunk):
        block = pcm[start : start + chunk]
        await coordinator.feed(block)
        if args.realtime_input:
            # A microphone delivers audio on its own clock, so pace against an
            # absolute schedule. Sleeping for the chunk duration *after* feeding
            # stacked ASR compute on top of every chunk, which made "realtime"
            # input run 1.34x slower than real time (11027 ms of wall for 8250 ms
            # of audio) and serialized ASR against speech by construction.
            # With absolute pacing, `input_wall_ms` exceeding `input_audio_ms` now
            # means ASR genuinely failed to keep up.
            delay = input_started + (start + len(block)) / 16000 - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
    details["input_wall_ms"] = round((time.perf_counter() - input_started) * 1000, 1)
    # ASR real-time factor is input_wall / input_audio, so the audio duration has
    # to travel with the report rather than be assumed by whoever reads it.
    details["input_audio_ms"] = round(len(pcm) / 16000 * 1000, 1)
    await coordinator.close()
    input_complete.set()
    await collector
    ended = time.perf_counter()
    details["gpu_state"] = _gpu_state_snapshot()
    details["event_last_offsets_ms"] = dict(last_offsets)
    details["resident_services"] = {
        "asr": _runtime_metrics_snapshot(asr),
        "llm": _runtime_metrics_snapshot(llm),
        "tts": _runtime_metrics_snapshot(tts),
    }
    asr.close()
    llm.close()
    close_tts = getattr(tts, "close", None)
    if callable(close_tts):
        close_tts()
    return TimingResult(
        f"real_async_{args.llm_trigger}_processes",
        first_audio,
        (ended - started) * 1000,
        names,
        errors=errors,
        event_offsets_ms=offsets,
        details=details,
    )


def speedup(base: TimingResult, other: TimingResult) -> dict:
    return {
        "total": round(base.total_ms / other.total_ms, 2) if other.total_ms else None,
        "first_audio": round(base.first_audio_ms / other.first_audio_ms, 2)
        if base.first_audio_ms and other.first_audio_ms
        else None,
    }


async def collect_until_input_complete(
    next_event,
    handle_event,
    input_complete: asyncio.Event,
) -> None:
    """Collect through ASR close even when an agent turn finishes early."""

    turn_done = False
    while True:
        if turn_done and input_complete.is_set():
            return
        if not turn_done:
            event = await next_event()
        else:
            event_task = asyncio.create_task(next_event())
            input_task = asyncio.create_task(input_complete.wait())
            completed, pending = await asyncio.wait(
                (event_task, input_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if event_task in completed:
                event = event_task.result()
            else:
                event_task.cancel()
                with suppress(asyncio.CancelledError):
                    await event_task
                return
            for task in pending:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        handle_event(event)
        event_type = event.get("type") if isinstance(event, dict) else event.type
        if event_type == "done":
            turn_done = True


def compare_timing_outputs(base: TimingResult, candidate: TimingResult) -> dict:
    """Gate speedup reporting on matching semantic and PCM outputs."""

    base_details = base.details or {}
    candidate_details = candidate.details or {}
    fields = (
        "asr_hypothesis",
        # A configuration that changes which prefix triggers the LLM changes the
        # prompt, and therefore the answer. That is a semantic difference even
        # when the transcript and the audio byte counts happen to match.
        "llm_prompt",
        "llm_output",
        "tts_rendered_inputs",
        "tts_audio_bytes",
        "tts_audio_duration_ms",
    )
    mismatches = [
        field
        for field in fields
        if base_details.get(field) != candidate_details.get(field)
    ]
    eligible = not mismatches and not base.errors and not candidate.errors
    return {
        "eligible": eligible,
        "mismatches": mismatches,
        "speedup": (
            {
                "first_audio": round(base.first_audio_ms / candidate.first_audio_ms, 2)
                if base.first_audio_ms and candidate.first_audio_ms
                else None,
                "total": round(base.total_ms / candidate.total_ms, 2)
                if candidate.total_ms
                else None,
            }
            if eligible
            else None
        ),
    }


def _gpu_state_snapshot() -> list[dict] | None:
    """Record SM clock and throttle reasons so a timing number can be interpreted.

    This node holds all four cards in `sw_thermal_slowdown` at roughly half their
    max clock, and throughput moves ~1.9x with it. Without this, an arm's absolute
    latency cannot be compared against an arm from another session.
    """
    try:
        from bench.gpu_contention_probe import query_smi

        return query_smi()
    except Exception:  # noqa: BLE001 - never fail a timing run over telemetry
        return None


def _runtime_metrics_snapshot(service) -> dict:
    runtime_metrics = getattr(service, "runtime_metrics", None)
    if not callable(runtime_metrics):
        return {}
    try:
        return dict(runtime_metrics())
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-profile",
        choices=RUNTIME_PROFILES,
        default=default_runtime_profile(),
    )
    parser.add_argument("--mode", choices=["fake", "real"], default="fake")
    parser.add_argument("--real-target", choices=["sync", "async", "both"], default="both")
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument(
        "--llm-trigger",
        choices=["final", "committed"],
        default="final",
        help=(
            "final answers the whole utterance. committed triggers on an ASR "
            "prefix and is a research flag only: with the default gate it "
            "prompted the LLM with one word of a 22-word utterance"
        ),
    )
    parser.add_argument(
        "--asr-remote",
        default=None,
        help="host:port of an edge ASR server; PCM is streamed there instead of a local GPU",
    )
    parser.add_argument(
        "--tts-remote",
        default=None,
        help="host:port of an edge TTS server; text fragments are synthesized there",
    )
    parser.add_argument("--asr-policy", default="speculate")
    parser.add_argument("--commit-lag-words", type=int, default=1)
    parser.add_argument("--language", default=None)
    parser.add_argument("--chunk-ms", type=int, default=400)
    parser.add_argument("--tts-concurrency", type=int, default=2)
    parser.add_argument("--realtime-input", action="store_true")
    parser.add_argument(
        "--llm-max-num-seqs",
        type=int,
        default=None,
        help="real mode: override nano-vLLM max_num_seqs for LLM batching",
    )
    parser.add_argument(
        "--tts-flush-chars",
        type=int,
        default=None,
        help="async mode: flush TTS fragments after this many buffered chars even without punctuation",
    )
    parser.add_argument(
        "--tts-flush-after-ms",
        type=int,
        default=None,
        help="async mode: flush pending LLM text to TTS after this many milliseconds without a sentence boundary",
    )
    parser.add_argument(
        "--tts-flush-min-chars",
        type=int,
        default=12,
        help="async mode: minimum pending characters for time-based TTS flush",
    )
    parser.add_argument(
        "--tts-coalesce-chars",
        type=int,
        default=None,
        help="async mode: coalesce queued TTS fragments up to this many chars while TTS is busy",
    )
    parser.add_argument(
        "--tts-coalesce-wait-ms",
        type=int,
        default=None,
        help="async mode: wait this long for nearby queued TTS fragments before starting synthesis",
    )
    parser.add_argument(
        "--tts-first-sentence-immediate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="async mode: submit the first TTS segment of each LLM generation without the coalesce wait",
    )
    parser.add_argument(
        "--tts-defer-short-segments-chars",
        type=int,
        default=None,
        help="async mode: hold shorter completed LLM segments so nearby text can merge before TTS",
    )
    parser.add_argument(
        "--tts-defer-short-segments-ms",
        type=int,
        default=None,
        help="async mode: max wait before flushing a deferred short TTS segment",
    )
    parser.add_argument(
        "--tts-stream-first-segment-only",
        action="store_true",
        help="async mode: use streaming TTS only for the first segment in each generation",
    )
    parser.add_argument(
        "--tts-process-stream-workers",
        type=int,
        default=1,
        help="real mode: use request-id concurrent TTS process backend with this worker count when >1",
    )
    parser.add_argument(
        "--tts-process-batch-window-ms",
        type=float,
        default=0.0,
        help="real mode: admission window for batching TTS stream request ids in one codec generation",
    )
    parser.add_argument(
        "--tts-process-max-batch-size",
        type=int,
        default=8,
        help="real mode: maximum TTS stream request ids per batched codec generation",
    )
    parser.add_argument(
        "--tts-shared-memory-threshold-bytes",
        type=int,
        default=64 * 1024,
        help="real mode: minimum TTS byte payload moved through shared memory",
    )
    parser.add_argument(
        "--tts-streaming-engine",
        choices=["off", "codec-step"],
        default="off",
        help="real mode: opt into Qwen-TTS codec-step audio streaming",
    )
    parser.add_argument(
        "--tts-stream-batch-exact-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="real mode: keep streamed TTS requests on scalar outer generation for legacy codec parity",
    )
    parser.add_argument(
        "--tts-do-sample",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="real mode: explicitly control TTS sampling for reproducible A/B runs",
    )
    parser.add_argument(
        "--tts-subtalker-do-sample",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="real mode: explicitly control Qwen-TTS codec predictor sampling",
    )
    parser.add_argument(
        "--tts-temperature",
        type=float,
        default=None,
        help="real mode: override TTS generation temperature",
    )
    parser.add_argument(
        "--tts-max-new-tokens",
        type=int,
        default=None,
        help="real mode: cap TTS generation length for matched-output A/B runs",
    )
    parser.add_argument(
        "--tts-eos-token-id",
        type=int,
        default=None,
        help="real mode: override TTS codec EOS id; use max-new-tokens to cap generation",
    )
    parser.add_argument(
        "--tts-stream-chunk-size",
        type=int,
        default=8,
        help="real mode: steady-state codec frames per streamed TTS audio chunk",
    )
    parser.add_argument(
        "--tts-stream-first-chunk-size",
        type=int,
        default=None,
        help="real mode: codec frames for the first streamed TTS audio chunk",
    )
    parser.add_argument(
        "--tts-stream-left-context-size",
        type=int,
        default=4,
        help="real mode: left-context codec frames used when decoding streamed TTS chunks",
    )
    parser.add_argument(
        "--tts-fast-code-predictor",
        action="store_true",
        help="real mode: replace Qwen-TTS inner code predictor HF generate calls with a fixed-step loop",
    )
    parser.add_argument(
        "--tts-static-code-predictor",
        action="store_true",
        help="real mode: use a pooled static KV cache for Qwen-TTS inner code predictor steps",
    )
    parser.add_argument(
        "--tts-cuda-graph-code-predictor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="real mode: replay fixed-shape Qwen-TTS predictor steps with CUDA Graphs",
    )
    parser.add_argument(
        "--tts-cuda-graph-fixed-slots",
        type=int,
        default=1,
        help="real mode: pre-captured CUDA Graph slots for concurrent predictor request ids",
    )
    parser.add_argument(
        "--tts-cuda-graph-batch-window-ms",
        type=float,
        default=0.0,
        help="real mode: admission window for compatible CUDA Graph predictor requests",
    )
    parser.add_argument(
        "--tts-fast-code-predictor-batch-window-ms",
        type=float,
        default=0.0,
        help="real mode: wait this long to batch concurrent Qwen-TTS code predictor steps",
    )
    parser.add_argument(
        "--tts-fast-code-predictor-max-batch-size",
        type=int,
        default=8,
        help="real mode: maximum concurrent request ids per code predictor step batch",
    )
    parser.add_argument(
        "--tts-explicit-talker-step-engine",
        action="store_true",
        help="real mode: replace Qwen-TTS outer HF talker.generate with an explicit codec step loop",
    )
    parser.add_argument(
        "--tts-outer-active-prefix-talker-engine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="real mode: use the exact-parity active-prefix outer talker step engine",
    )
    parser.add_argument(
        "--tts-outer-cuda-graph-talker-engine",
        action="store_true",
        help="real mode: replay fixed-shape outer Qwen-TTS talker decode steps with CUDA Graphs",
    )
    parser.add_argument(
        "--tts-outer-graph-fixed-slots",
        type=int,
        default=2,
        help="real mode: number of fixed outer talker CUDA Graph request slots",
    )
    parser.add_argument(
        "--tts-outer-graph-max-cache-len",
        type=int,
        default=16384,
        help="real mode: maximum cached outer talker decode length",
    )
    parser.add_argument(
        "--tts-compile-step-engine",
        action="store_true",
        help="real mode: compile Qwen-TTS fast predictor and explicit talker step callables with torch.compile",
    )
    parser.add_argument(
        "--tts-compile-step-engine-mode",
        default="reduce-overhead",
        help="real mode: torch.compile mode for Qwen-TTS step callables",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--min-committed-words", type=int, default=1)
    parser.add_argument("--min-committed-audio-seconds", type=float, default=0.0)
    parser.add_argument("--max-committed-audio-seconds", type=float, default=None)
    parser.add_argument(
        "--defer-tts-audio-until-asr-final",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--tts-playback-preroll-ms",
        type=float,
        default=0.0,
        help=(
            "hold the opening TTS chunks until this much audio is buffered. "
            "Playback that starts on the first chunk starves whenever TTS runs "
            "slower than real time; 0 disables the gate"
        ),
    )
    parser.add_argument(
        "--barge-in-policy",
        choices=sorted(BARGE_IN_POLICIES),
        default="auto",
        help=(
            "when incoming user audio may cancel a reply already being spoken; "
            "'after-asr-final' stops the tail of the triggering utterance from "
            "self-interrupting once first audio is fast"
        ),
    )
    parser.add_argument("--barge-in-rms-threshold", type=float, default=1e-4)
    parser.add_argument("--asr-commit-delay", type=float, default=0.4)
    parser.add_argument("--asr-final-delay", type=float, default=0.8)
    parser.add_argument("--llm-delay", type=float, default=0.1)
    parser.add_argument("--tts-delay", type=float, default=0.3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parse_profiled_args(build_parser(), argv)
    activate_runtime_profile_environment(args)
    if args.mode == "fake":
        sync = run_fake_sync(args)
        async_result = asyncio.run(run_fake_async(args))
    else:
        if args.audio is None:
            raise SystemExit("--audio is required for --mode real")
        pcm = read_wav(args.audio)
        if args.real_target == "sync":
            sync = run_real_sync(pcm, args)
            print(json.dumps({"baseline": sync.as_dict()}, ensure_ascii=False, indent=2))
            return 0
        if args.real_target == "async":
            async_result = asyncio.run(run_real_async(pcm, args))
            print(json.dumps({"async": async_result.as_dict()}, ensure_ascii=False, indent=2))
            return 0
        sync = run_real_sync(pcm, args)
        async_result = asyncio.run(run_real_async(pcm, args))
    print(
        json.dumps(
            {
                "baseline": sync.as_dict(),
                "async": async_result.as_dict(),
                "speedup": speedup(sync, async_result),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
