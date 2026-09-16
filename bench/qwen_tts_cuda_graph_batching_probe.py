"""Measure concurrent Qwen-TTS sessions through fixed-slot CUDA Graph batching."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.qwen_tts_streaming_engine_probe import default_config  # noqa: E402
from qwen_asr_vllm.agent.local_tts import QwenTtsBackend  # noqa: E402


def _cuda_synchronize(device: str) -> None:
    import torch

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _predictor_runtime(backend: QwenTtsBackend):
    generate = backend._model.model.talker.code_predictor.generate
    return (
        getattr(generate, "_qav_engine", None),
        getattr(generate, "_qav_scheduler", None),
    )


def run_concurrent_round(
    backend: QwenTtsBackend,
    *,
    text: str,
    requests: int,
    device: str,
) -> dict:
    barrier = threading.Barrier(requests + 1)
    durations = [0.0] * requests
    audio_bytes = [0] * requests
    errors: list[str | None] = [None] * requests

    def worker(index: int) -> None:
        barrier.wait()
        started = time.perf_counter()
        try:
            audio = backend.synthesize(text)
            audio_bytes[index] = len(audio or b"")
        except BaseException as exc:  # noqa: BLE001 - report worker failures
            errors[index] = repr(exc)
        finally:
            durations[index] = (time.perf_counter() - started) * 1000.0

    threads = [
        threading.Thread(target=worker, args=(index,), daemon=True)
        for index in range(requests)
    ]
    for thread in threads:
        thread.start()
    _cuda_synchronize(device)
    started = time.perf_counter()
    barrier.wait()
    for thread in threads:
        thread.join()
    _cuda_synchronize(device)
    wall_ms = (time.perf_counter() - started) * 1000.0
    return {
        "wall_ms": wall_ms,
        "request_ms": durations,
        "audio_bytes": audio_bytes,
        "errors": [error for error in errors if error is not None],
    }


def _representative_predictor_inputs(engine):
    for key, bundle in engine._bundles.items():
        shape = key[0]
        if int(shape[0]) == 1 and hasattr(bundle, "_inputs_embeds"):
            return bundle._inputs_embeds.detach().clone(), int(key[-1])
    raise RuntimeError("batch-1 CUDA Graph bundle was not captured during warmup")


def run_predictor_throughput(
    backend: QwenTtsBackend,
    *,
    requests: int,
    iterations: int,
    device: str,
) -> dict:
    engine, scheduler = _predictor_runtime(backend)
    inputs, max_new_tokens = _representative_predictor_inputs(engine)
    generate = backend._model.model.talker.code_predictor.generate
    if scheduler is not None:
        scheduler.request_batches.clear()
        scheduler.slot_batches.clear()
        scheduler.padded_slots = 0
    replay_start = int(engine.graph_replays)
    barrier = threading.Barrier(requests + 1)
    errors: list[str] = []

    def worker() -> None:
        barrier.wait()
        try:
            for _ in range(iterations):
                generate(
                    inputs_embeds=inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
        except BaseException as exc:  # noqa: BLE001 - report worker failures
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(requests)]
    for thread in threads:
        thread.start()
    _cuda_synchronize(device)
    started = time.perf_counter()
    barrier.wait()
    for thread in threads:
        thread.join()
    _cuda_synchronize(device)
    wall_ms = (time.perf_counter() - started) * 1000.0
    total_requests = requests * iterations
    return {
        "wall_ms": wall_ms,
        "predictor_requests": total_requests,
        "requests_per_second": total_requests / (wall_ms / 1000.0),
        "graph_replays": int(engine.graph_replays) - replay_start,
        "request_batches": list(getattr(scheduler, "request_batches", [])),
        "slot_batches": list(getattr(scheduler, "slot_batches", [])),
        "padded_slots": getattr(scheduler, "padded_slots", 0),
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = default_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=defaults.get("model_path"))
    parser.add_argument("--device", default=defaults.get("device", "cuda:1"))
    parser.add_argument("--language", default=defaults.get("language", "chinese"))
    parser.add_argument("--speaker", default=defaults.get("speaker", ""))
    parser.add_argument("--text", default="你好，请确认实时语音系统已经准备好。")
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--predictor-iterations", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--greedy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fixed-slots", type=int, default=2)
    parser.add_argument("--batch-window-ms", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model_path:
        raise SystemExit("--model-path is required")
    if args.requests <= 0 or args.rounds <= 0:
        raise SystemExit("--requests and --rounds must be positive")

    backend = QwenTtsBackend(
        model_path=args.model_path,
        device=args.device,
        language=args.language,
        speaker=args.speaker,
        cuda_graph_code_predictor=True,
        cuda_graph_fixed_slots=args.fixed_slots,
        cuda_graph_batch_window_ms=args.batch_window_ms,
    )
    if args.greedy:
        backend._model.generate_defaults.update(
            do_sample=False,
            subtalker_dosample=False,
            max_new_tokens=args.max_new_tokens,
        )
    engine, scheduler = _predictor_runtime(backend)
    if scheduler is not None:
        scheduler.request_batches.clear()
        scheduler.slot_batches.clear()
        scheduler.padded_slots = 0

    rounds = [
        run_concurrent_round(
            backend,
            text=args.text,
            requests=args.requests,
            device=args.device,
        )
        for _ in range(args.rounds)
    ]
    tts_request_batches = list(getattr(scheduler, "request_batches", []))
    tts_slot_batches = list(getattr(scheduler, "slot_batches", []))
    tts_padded_slots = getattr(scheduler, "padded_slots", 0)
    predictor_throughput = run_predictor_throughput(
        backend,
        requests=args.requests,
        iterations=args.predictor_iterations,
        device=args.device,
    )
    report = {
        "stage": "qwen_tts_cuda_graph_request_batching",
        "device": args.device,
        "requests": args.requests,
        "fixed_slots": args.fixed_slots,
        "batch_window_ms": args.batch_window_ms,
        "greedy": bool(args.greedy),
        "max_new_tokens": int(args.max_new_tokens),
        "rounds": rounds,
        "mean_wall_ms": sum(item["wall_ms"] for item in rounds) / len(rounds),
        "graph_captures": getattr(engine, "graph_captures", None),
        "graph_replays": getattr(engine, "graph_replays", None),
        "request_batches": tts_request_batches,
        "slot_batches": tts_slot_batches,
        "padded_slots": tts_padded_slots,
        "predictor_throughput": predictor_throughput,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    close = getattr(backend, "close", None)
    if callable(close):
        close()
    failed = any(item["errors"] for item in rounds) or predictor_throughput["errors"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
