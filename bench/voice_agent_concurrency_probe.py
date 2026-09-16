"""Shared-service 1/2/4/8 concurrency probe for the local voice agent."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench import voice_agent_timing
from qwen_asr_vllm.agent.async_coordinator import AsyncVoiceAgentCoordinator
from qwen_asr_vllm.agent.runtime_profiles import (
    activate_runtime_profile_environment,
    parse_profiled_args,
)


def normalize_session_counts(counts) -> tuple[int, ...]:
    values = tuple(int(value) for value in counts)
    if not values or any(value <= 0 for value in values):
        raise ValueError("session counts must be positive")
    if len(set(values)) != len(values):
        raise ValueError("session counts must be unique")
    return values


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction, 1)


def _coordinator_kwargs(args) -> dict[str, Any]:
    return {
        "llm_trigger": args.llm_trigger,
        "asr_kwargs": {
            "language": args.language,
            "chunk_policy": args.asr_policy,
            "commit_lag_words": args.commit_lag_words,
        },
        "tts_concurrency": args.tts_concurrency,
        "tts_flush_chars": args.tts_flush_chars,
        "tts_flush_after_ms": args.tts_flush_after_ms,
        "tts_flush_min_chars": args.tts_flush_min_chars,
        "tts_coalesce_chars": args.tts_coalesce_chars,
        "tts_coalesce_wait_ms": args.tts_coalesce_wait_ms,
        "tts_first_sentence_immediate": args.tts_first_sentence_immediate,
        "tts_defer_short_segments_chars": args.tts_defer_short_segments_chars,
        "tts_defer_short_segments_ms": args.tts_defer_short_segments_ms,
        "tts_stream_first_segment_only": args.tts_stream_first_segment_only,
        "min_committed_words": args.min_committed_words,
        "min_committed_audio_seconds": args.min_committed_audio_seconds,
        "max_committed_audio_seconds": args.max_committed_audio_seconds,
        "defer_tts_audio_until_asr_final": args.defer_tts_audio_until_asr_final,
        "reset_llm_on_start": False,
    }


def _real_backend_kwargs(args) -> dict[str, Any]:
    return {
        "process_isolated": True,
        "llm_max_num_seqs": args.llm_max_num_seqs,
        "tts_streaming": args.tts_streaming_engine == "codec-step",
        "tts_stream_chunk_size": args.tts_stream_chunk_size,
        "tts_stream_first_chunk_size": args.tts_stream_first_chunk_size,
        "tts_stream_left_context_size": args.tts_stream_left_context_size,
        "tts_stream_batch_exact_parity": args.tts_stream_batch_exact_parity,
        "tts_fast_code_predictor": args.tts_fast_code_predictor,
        "tts_static_code_predictor": args.tts_static_code_predictor,
        "tts_cuda_graph_code_predictor": args.tts_cuda_graph_code_predictor,
        "tts_cuda_graph_fixed_slots": args.tts_cuda_graph_fixed_slots,
        "tts_cuda_graph_batch_window_ms": args.tts_cuda_graph_batch_window_ms,
        "tts_fast_code_predictor_batch_window_ms": args.tts_fast_code_predictor_batch_window_ms,
        "tts_fast_code_predictor_max_batch_size": args.tts_fast_code_predictor_max_batch_size,
        "tts_explicit_talker_step_engine": args.tts_explicit_talker_step_engine,
        "tts_outer_active_prefix_talker_engine": args.tts_outer_active_prefix_talker_engine,
        "tts_outer_cuda_graph_talker_engine": args.tts_outer_cuda_graph_talker_engine,
        "tts_outer_graph_fixed_slots": args.tts_outer_graph_fixed_slots,
        "tts_outer_graph_max_cache_len": args.tts_outer_graph_max_cache_len,
        "tts_compile_step_engine": args.tts_compile_step_engine,
        "tts_compile_step_engine_mode": args.tts_compile_step_engine_mode,
        "tts_do_sample": args.tts_do_sample,
        "tts_subtalker_do_sample": args.tts_subtalker_do_sample,
        "tts_temperature": args.tts_temperature,
        "tts_max_new_tokens": args.tts_max_new_tokens,
        "tts_eos_token_id": args.tts_eos_token_id,
        "tts_process_stream_workers": args.tts_process_stream_workers,
        "tts_process_batch_window_ms": args.tts_process_batch_window_ms,
        "tts_process_max_batch_size": args.tts_process_max_batch_size,
        "tts_shared_memory_threshold_bytes": args.tts_shared_memory_threshold_bytes,
    }


def _load_services(args):
    if args.mode == "fake":
        return (
            voice_agent_timing.FakeAsrEngine(trigger=args.llm_trigger),
            voice_agent_timing.SlowLlm(["First.", " Second."], args.llm_delay),
            voice_agent_timing.SlowTts(args.tts_delay),
        )
    return voice_agent_timing.load_real_backends(args.timeout, **_real_backend_kwargs(args))


async def _run_session(session_id: int, coordinator, pcm: np.ndarray, args) -> dict:
    started = time.perf_counter()
    first_audio_ms = None
    asr_hypothesis = ""
    llm_output = ""
    tts_inputs: list[str] = []
    audio_by_text: dict[str, bytearray] = {}
    audio_bytes = 0
    errors: list[str] = []
    event_types: list[str] = []
    done_seen = False
    asr_final_seen = False

    async def collect() -> None:
        nonlocal first_audio_ms, asr_hypothesis, llm_output
        nonlocal audio_bytes, done_seen, asr_final_seen
        while not (done_seen and asr_final_seen):
            event = await coordinator.next_event(timeout=args.timeout)
            payload = event.as_dict()
            event_types.append(event.type)
            if event.type == "asr_final":
                asr_final_seen = True
                asr_hypothesis = str(
                    payload.get("text") or payload.get("committed_text") or ""
                ).strip()
            elif event.type == "llm_chunk":
                llm_output += str(payload.get("text") or "")
            elif event.type == "llm_sentence_ready":
                tts_inputs.append(str(payload.get("text") or ""))
            elif event.type == "tts_chunk":
                audio = payload.get("audio") or b""
                if first_audio_ms is None:
                    first_audio_ms = (time.perf_counter() - started) * 1000.0
                text = str(payload.get("text") or "")
                audio_by_text.setdefault(text, bytearray()).extend(audio)
                audio_bytes += len(audio)
            elif event.type == "error":
                errors.append(str(payload))
            elif event.type == "done":
                done_seen = True

    collector = asyncio.create_task(collect())
    chunk_samples = max(1, int(args.chunk_ms * 16000 / 1000))
    try:
        for offset in range(0, len(pcm), chunk_samples):
            part = pcm[offset : offset + chunk_samples]
            await coordinator.feed(part)
            if args.realtime_input:
                await asyncio.sleep(len(part) / 16000.0)
        await coordinator.close()
        await collector
    except BaseException:
        collector.cancel()
        await asyncio.gather(collector, return_exceptions=True)
        raise
    total_ms = (time.perf_counter() - started) * 1000.0
    audio_hash = hashlib.sha256()
    hashed_texts: set[str] = set()
    for text in tts_inputs:
        if text in hashed_texts:
            continue
        hashed_texts.add(text)
        audio_hash.update(text.encode())
        audio_hash.update(audio_by_text.get(text, b""))
    for text in sorted(audio_by_text.keys() - hashed_texts):
        audio_hash.update(text.encode())
        audio_hash.update(audio_by_text[text])
    return {
        "session_id": session_id,
        "first_audio_ms": round(first_audio_ms, 1) if first_audio_ms is not None else None,
        "total_ms": round(total_ms, 1),
        "asr_hypothesis": asr_hypothesis,
        "llm_output": llm_output,
        "tts_inputs": tts_inputs,
        "tts_audio_bytes": audio_bytes,
        "tts_audio_sha256": audio_hash.hexdigest(),
        "events": event_types,
        "errors": errors,
    }


def _signature(session: dict[str, Any]) -> str:
    payload = {
        key: session[key]
        for key in (
            "asr_hypothesis",
            "llm_output",
            "tts_inputs",
            "tts_audio_bytes",
            "tts_audio_sha256",
        )
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


async def run_sweep(args) -> dict[str, Any]:
    counts = normalize_session_counts(args.sessions)
    pcm = (
        np.full(1600, 0.01, dtype=np.float32)
        if args.mode == "fake"
        else voice_agent_timing.read_wav(args.audio)
    )
    asr, llm, tts = _load_services(args)
    levels = []
    baseline_signature = None
    try:
        for count in counts:
            reset = getattr(llm, "reset", None)
            if callable(reset):
                reset()
            coordinators = [
                AsyncVoiceAgentCoordinator(asr, llm, tts, **_coordinator_kwargs(args))
                for _ in range(count)
            ]
            level_started = time.perf_counter()
            sessions = await asyncio.gather(
                *(
                    _run_session(index, coordinator, pcm, args)
                    for index, coordinator in enumerate(coordinators)
                )
            )
            wall_ms = (time.perf_counter() - level_started) * 1000.0
            signatures = [_signature(session) for session in sessions]
            if baseline_signature is None:
                baseline_signature = signatures[0]
            errors = [error for session in sessions for error in session["errors"]]
            first_audio = [
                session["first_audio_ms"]
                for session in sessions
                if session["first_audio_ms"] is not None
            ]
            totals = [session["total_ms"] for session in sessions]
            eligible = not errors and len(first_audio) == count
            levels.append(
                {
                    "session_count": count,
                    "wall_ms": round(wall_ms, 1),
                    "latency_ms": {
                        "first_audio_p50": _percentile(first_audio, 0.50),
                        "first_audio_p95": _percentile(first_audio, 0.95),
                        "total_p50": _percentile(totals, 0.50),
                        "total_p95": _percentile(totals, 0.95),
                    },
                    "throughput": {
                        "sessions_per_s": round(count / max(wall_ms / 1000.0, 1e-9), 3),
                        "audio_bytes_per_s": round(
                            sum(session["tts_audio_bytes"] for session in sessions)
                            / max(wall_ms / 1000.0, 1e-9),
                            1,
                        ),
                    },
                    "parity": {
                        "eligible": eligible,
                        "matched": eligible
                        and all(signature == baseline_signature for signature in signatures),
                    },
                    "errors": errors,
                    "resident_services": {
                        "asr": voice_agent_timing._runtime_metrics_snapshot(asr),
                        "llm": voice_agent_timing._runtime_metrics_snapshot(llm),
                        "tts": voice_agent_timing._runtime_metrics_snapshot(tts),
                    },
                    "sessions": sessions,
                }
            )
    finally:
        for service in (asr, llm, tts):
            close = getattr(service, "close", None)
            if callable(close):
                close()
    return {
        "mode": args.mode,
        "runtime_profile": args.runtime_profile,
        "session_counts": list(counts),
        "baseline_signature": baseline_signature,
        "levels": levels,
    }


def _assert_idle_gpu(index: int, max_memory_mib: int, max_utilization: int) -> None:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    used_text, utilization_text = (part.strip() for part in output.split(",", 1))
    used = int(used_text)
    utilization = int(utilization_text)
    if used > max_memory_mib or utilization > max_utilization:
        raise RuntimeError(
            f"CUDA {index} is not idle: {used} MiB, {utilization}% utilization"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = voice_agent_timing.build_parser()
    parser.description = __doc__
    parser.set_defaults(real_target="async")
    parser.add_argument("--sessions", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--max-preflight-memory-mib", type=int, default=512)
    parser.add_argument("--max-preflight-utilization", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parse_profiled_args(build_parser(), argv)
    normalize_session_counts(args.sessions)
    activate_runtime_profile_environment(args)
    if args.mode == "real":
        if args.audio is None:
            raise SystemExit("--audio is required for --mode real")
        _assert_idle_gpu(
            args.physical_gpu,
            args.max_preflight_memory_mib,
            args.max_preflight_utilization,
        )
    report = asyncio.run(run_sweep(args))
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
