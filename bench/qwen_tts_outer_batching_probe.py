"""Measure serial versus unified cross-session Qwen-TTS stream batching."""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
import wave
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.qwen_tts_streaming_engine_probe import default_config  # noqa: E402
from qwen_asr_vllm.agent.local_tts import QwenTtsBackend  # noqa: E402
from qwen_asr_vllm.agent.service_runners import (  # noqa: E402
    ProcessConcurrentTtsBackend,
)


def _backend_factory(args):
    return partial(
        QwenTtsBackend,
        model_path=args.model_path,
        device=args.device,
        language=args.language,
        speaker=args.speaker,
        streaming=True,
        stream_batch_exact_parity=args.stream_batch_exact_parity,
        stream_chunk_size=args.chunk_size,
        stream_first_chunk_size=args.first_chunk_size,
        stream_left_context_size=args.left_context_size,
        do_sample=False,
        subtalker_dosample=False,
        max_new_tokens=args.max_new_tokens,
        cuda_graph_code_predictor=args.inner_cuda_graph,
        cuda_graph_fixed_slots=args.inner_graph_slots,
        cuda_graph_batch_window_ms=args.inner_graph_batch_window_ms,
    )


def _collect_stream(backend, text: str) -> dict:
    started = time.perf_counter()
    chunks = list(backend.synthesize_stream(text))
    stats = _audio_stats(chunks)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "elapsed_ms": elapsed_ms,
        "chunks": len(chunks),
        "audio_bytes": sum(len(chunk) for chunk in chunks),
        **stats,
    }


def _run_serial(backend, texts: list[str]) -> dict:
    started = time.perf_counter()
    requests = [_collect_stream(backend, text) for text in texts]
    return {
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "requests": requests,
    }


def _run_native_batch(backend, texts: list[str]) -> dict:
    started = time.perf_counter()
    chunks_by_request = [[] for _ in texts]
    for index, chunk in backend.synthesize_stream_batch(texts):
        chunks_by_request[int(index)].append(chunk)
    return {
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "requests": [_stream_result(chunks) for chunks in chunks_by_request],
    }


def _stream_result(chunks: list[bytes]) -> dict:
    stats = _audio_stats(chunks)
    return {
        "chunks": len(chunks),
        "audio_bytes": sum(len(chunk) for chunk in chunks),
        **stats,
    }


def _audio_stats(chunks: list[bytes]) -> dict:
    """Estimate codec work from emitted PCM, excluding per-chunk WAV headers."""

    payload_bytes = 0
    sample_rate = 16000
    for chunk in chunks:
        try:
            with wave.open(io.BytesIO(chunk), "rb") as handle:
                sample_rate = int(handle.getframerate())
                payload_bytes += int(handle.getnframes()) * handle.getsampwidth()
        except (EOFError, wave.Error):
            continue
    audio_duration_ms = payload_bytes / max(1, sample_rate * 2) * 1000.0
    codec_frames = audio_duration_ms / 1000.0 * 12.0
    return {
        "audio_pcm_bytes": payload_bytes,
        "audio_duration_ms": audio_duration_ms,
        "codec_frames_estimate": codec_frames,
        "codec_tokens_estimate": codec_frames * 16.0,
        "codec_sample_rate_hz": 12.0,
        "codec_codebooks": 16,
    }


def _throughput(result: dict) -> dict:
    wall_seconds = max(1.0e-9, float(result["wall_ms"]) / 1000.0)
    requests = result.get("requests", [])
    frames = sum(float(item.get("codec_frames_estimate", 0.0)) for item in requests)
    tokens = sum(float(item.get("codec_tokens_estimate", 0.0)) for item in requests)
    return {
        "requests_per_s": len(requests) / wall_seconds,
        "codec_frames_per_s": frames / wall_seconds,
        "codec_tokens_per_s": tokens / wall_seconds,
        "audio_real_time_factor": sum(
            float(item.get("audio_duration_ms", 0.0)) for item in requests
        )
        / max(1.0, float(result["wall_ms"])),
    }


def _run_unified_service(factory, texts: list[str], args) -> dict:
    service = ProcessConcurrentTtsBackend(
        factory,
        context="spawn",
        timeout=args.timeout,
        max_workers=1,
        batch_window_ms=args.batch_window_ms,
        max_batch_size=len(texts),
    )
    results = [{} for _ in texts]
    barrier = threading.Barrier(len(texts) + 1)

    def worker(index: int) -> None:
        barrier.wait()
        results[index] = _collect_stream(service, texts[index])

    threads = [threading.Thread(target=worker, args=(index,), daemon=True) for index in range(len(texts))]
    for thread in threads:
        thread.start()
    started = time.perf_counter()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=args.timeout)
    wall_ms = (time.perf_counter() - started) * 1000.0
    service.close()
    result = {"wall_ms": wall_ms, "requests": results}
    result["throughput"] = _throughput(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    defaults = default_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=defaults.get("model_path"))
    parser.add_argument("--device", default=defaults.get("device", "cuda:3"))
    parser.add_argument("--language", default=defaults.get("language", "english"))
    parser.add_argument("--speaker", default=defaults.get("speaker", ""))
    parser.add_argument(
        "--texts",
        nargs="+",
        default=["I'm here to help!", "What can I do for you?"],
    )
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--first-chunk-size", type=int, default=3)
    parser.add_argument("--left-context-size", type=int, default=4)
    parser.add_argument(
        "--stream-batch-exact-parity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use the scalar stream decoder for each request instead of native outer batching",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--inner-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--inner-graph-slots", type=int, default=2)
    parser.add_argument("--inner-graph-batch-window-ms", type=float, default=10.0)
    parser.add_argument("--batch-window-ms", type=float, default=10.0)
    parser.add_argument(
        "--sweep-concurrency",
        nargs="+",
        type=int,
        default=[],
        help="also run independent unified-service load points at these request counts",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model_path:
        raise SystemExit("--model-path is required")

    factory = _backend_factory(args)
    backend = factory()
    try:
        serial = _run_serial(backend, list(args.texts))
        native_batch = _run_native_batch(backend, list(args.texts))
        runtime_metrics = backend.runtime_metrics()
    finally:
        backend.close()

    unified = _run_unified_service(factory, list(args.texts), args)
    for result in (serial, native_batch):
        result["throughput"] = _throughput(result)
    concurrency_sweep = {}
    for level in sorted({max(1, int(value)) for value in args.sweep_concurrency}):
        sweep_texts = [args.texts[index % len(args.texts)] for index in range(level)]
        concurrency_sweep[str(level)] = _run_unified_service(
            factory,
            sweep_texts,
            args,
        )
    report = {
        "stage": "qwen_tts_outer_unified_batching",
        "device": args.device,
        "texts": list(args.texts),
        "batch_window_ms": args.batch_window_ms,
        "max_new_tokens": args.max_new_tokens,
        "inner_cuda_graph": bool(args.inner_cuda_graph),
        "inner_graph_slots": int(args.inner_graph_slots),
        "inner_graph_batch_window_ms": args.inner_graph_batch_window_ms,
        "stream_batch_exact_parity": bool(args.stream_batch_exact_parity),
        "serial": serial,
        "native_batch": native_batch,
        "unified_service": unified,
        "concurrency_sweep": concurrency_sweep,
        "resident_runtime_metrics": runtime_metrics,
        "speedup_vs_serial": {
            "native_batch": serial["wall_ms"] / native_batch["wall_ms"],
            "unified_service": serial["wall_ms"] / unified["wall_ms"],
        },
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
