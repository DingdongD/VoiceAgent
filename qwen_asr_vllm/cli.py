"""Command line entry points: transcribe files, or serve HTTP.

``transcribe`` submits every file at once rather than looping one at a time, which
is the whole point of the engine: a directory of short clips is exactly the workload
continuous batching is for, and feeding them in sequentially would leave most of the
batch empty.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".webm"}


def _add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="path to the Qwen3-ASR checkpoint")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="fraction of *free* GPU memory the KV cache may take, not of total",
    )
    parser.add_argument("--max-audio-batch-frames", type=int, default=12000)
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="cap on transcript tokens; 0 scales it with audio length, which the "
        "440-token default does not (a 20-minute transcript needs about 4900)",
    )
    parser.add_argument(
        "--frontend-threads",
        type=int,
        default=8,
        help="torch intra-op threads for the CPU mel frontend, or 0 for torch's default",
    )
    parser.add_argument(
        "--dual-stream",
        action="store_true",
        help="overlap audio encode and text decode on separate CUDA streams "
        "(forces eager decode)",
    )
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )


def _engine_kwargs(args: argparse.Namespace) -> dict:
    return {
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_audio_batch_frames": args.max_audio_batch_frames,
        "enable_prefix_cache": not args.no_prefix_cache,
        "frontend_threads": args.frontend_threads,
        "enable_dual_stream": args.dual_stream,
    }


def collect_audio_paths(inputs: list[str]) -> list[Path]:
    """Expand files and directories into a sorted list of audio paths."""
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(
                child
                for child in sorted(path.rglob("*"))
                if child.suffix.lower() in AUDIO_SUFFIXES
            )
        elif path.exists():
            paths.append(path)
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")
    if not paths:
        raise ValueError(f"no audio files found under {' '.join(inputs)}")
    return paths


def transcribe_command(args: argparse.Namespace) -> int:
    from qwen_asr_vllm.audio.decode import decode_audio
    from qwen_asr_vllm.engine.async_engine import AsyncAsrEngine
    from qwen_asr_vllm.engine.request import SamplingParams

    paths = collect_audio_paths(args.inputs)
    engine = AsyncAsrEngine(model=args.model, **_engine_kwargs(args))
    started = time.perf_counter()
    try:
        # Every file is in flight at once; the scheduler decides the batching.
        handles = []
        for path in paths:
            audio = decode_audio(path.read_bytes())
            handles.append(
                (
                    path,
                    engine.submit(
                        audio,
                        context=args.context,
                        language=args.language,
                        sampling=SamplingParams.for_audio(
                            len(audio) / 16000, args.max_new_tokens
                        ),
                        timeout=args.timeout,
                    ),
                )
            )

        results = []
        for path, handle in handles:
            try:
                output = handle.result()
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
                print(f"{path}: FAILED ({exc})", file=sys.stderr)
                results.append({"path": str(path), "error": str(exc)})
                continue
            results.append(
                {
                    "path": str(path),
                    "text": output.text,
                    "language": output.language,
                    "audio_seconds": output.audio_seconds,
                }
            )
            if args.output_format == "text":
                print(f"{path}\t{output.text}")
    finally:
        engine.close()

    elapsed = time.perf_counter() - started
    if args.output_format == "json":
        json.dump(results, sys.stdout, ensure_ascii=False, indent=2)
        print()
    if args.output_file:
        Path(args.output_file).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    audio_seconds = sum(r.get("audio_seconds", 0.0) for r in results)
    failures = sum("error" in r for r in results)
    print(
        f"{len(results) - failures}/{len(results)} transcribed, "
        f"{audio_seconds:.1f}s of audio in {elapsed:.1f}s "
        f"({audio_seconds / elapsed if elapsed else 0:.1f}x realtime)",
        file=sys.stderr,
    )
    return 1 if failures else 0


def serve_command(args: argparse.Namespace) -> int:
    from qwen_asr_vllm.server import serve

    serve(
        model=args.model,
        host=args.host,
        port=args.port,
        frontend_workers=args.frontend_workers,
        log_level=args.log_level,
        **_engine_kwargs(args),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-asr-vllm", description="Qwen3-ASR inference with continuous batching"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe = subparsers.add_parser("transcribe", help="transcribe files or directories")
    _add_engine_arguments(transcribe)
    transcribe.add_argument("inputs", nargs="+", help="audio files or directories")
    transcribe.add_argument("--language", default=None, help="force a language, e.g. en")
    transcribe.add_argument("--context", default="", help="hotword or domain context")
    transcribe.add_argument("--timeout", type=float, default=None, help="per-request seconds")
    transcribe.add_argument("--output-format", default="text", choices=["text", "json"])
    transcribe.add_argument("--output-file", default=None, help="also write JSON here")
    transcribe.set_defaults(handler=transcribe_command)

    server = subparsers.add_parser("serve", help="run the HTTP server")
    _add_engine_arguments(server)
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--frontend-workers", type=int, default=4)
    server.set_defaults(handler=serve_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
