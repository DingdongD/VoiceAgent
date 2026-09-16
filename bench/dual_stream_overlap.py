"""Compare serial vs dual-stream encode∥decode (Phase 1 prototype).

Both timings use *overlap-eligible* schedule order: ``schedule_audio`` and
``schedule_model`` first, execute, then ``admit_prefill`` / ``postprocess``.
That defers newly encoded prefills by one step versus production ``step``, but
keeps serial vs overlap an apples-to-apples compare of launch overlap only.

Usage:
  /opt/conda/envs/nano-vllm/bin/python bench/dual_stream_overlap.py \\
      --model /mnt/llm_data/voice_ckpt/Qwen3-ASR-0.6B \\
      --mode both --num-samples 32 --max-num-seqs 8
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.data import load_librispeech
from qwen_asr_vllm.engine.dual_stream import DualStreamExecutor, DualStreamReport, compute_gate
from qwen_asr_vllm.engine.engine import AsrEngine, OutOfMemoryError
from qwen_asr_vllm.engine.scheduler import ModelBatch

DEFAULT_MODEL = "/mnt/llm_data/voice_ckpt/Qwen3-ASR-0.6B"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def step_instrumented(
    engine: AsrEngine,
    executor: DualStreamExecutor,
    mode: str,
) -> tuple[list, float, bool]:
    """One engine step with serial or overlapped encode∥model execution.

    Returns ``(outputs, elapsed_seconds, was_mixed)``.
    """
    if mode not in ("serial", "overlap"):
        raise ValueError(f"unknown mode {mode!r}")

    encode_batch = engine.scheduler.schedule_audio()
    batch: ModelBatch = engine.scheduler.schedule_model()
    was_mixed = bool(encode_batch) and bool(batch)
    token_ids: list[int] | None = None

    def encode_fn() -> None:
        if encode_batch:
            engine.audio_runner.encode(encode_batch)

    def decode_fn() -> None:
        nonlocal token_ids
        if batch:
            token_ids = engine.model_runner.run(batch)

    try:
        if mode == "serial":
            elapsed = executor.run_serial(encode_fn, decode_fn)
        else:
            elapsed = executor.run_overlap(encode_fn, decode_fn)
    except OutOfMemoryError:
        engine._recover_from_oom()
        if encode_batch:
            engine.scheduler.on_audio_oom(encode_batch)
        if batch:
            engine.scheduler.on_model_oom(batch)
        return [], 0.0, False

    finished = []
    if encode_batch:
        engine.scheduler.admit_prefill(encode_batch)
    if batch and token_ids is not None:
        finished = engine.scheduler.postprocess(batch, token_ids)
        engine.scheduler.relax_limits()

    outputs = [engine._finalize(r) for r in engine.scheduler.drain_aborted() + finished]
    for output in outputs:
        engine.metrics.record_finished(output)
    return outputs, elapsed, was_mixed


def transcribe_instrumented(
    engine: AsrEngine,
    waveforms: list[np.ndarray],
    executor: DualStreamExecutor,
    mode: str,
    *,
    language: str | None = "en",
) -> tuple[list[str], float, float, int]:
    """Transcribe with instrumented steps.

    Returns ``(texts_in_order, wall_s, mixed_step_s, n_mixed_steps)``.
    """
    admission_limit = max(engine.config.max_num_seqs * 4, engine.config.max_audio_batch_size)
    order: dict[int, int] = {}
    results: dict[int, str] = {}
    pending: set[int] = set()
    source = iter(waveforms)
    exhausted = False
    next_index = 0
    mixed_s = 0.0
    n_mixed = 0

    wall_start = time.perf_counter()
    while pending or not exhausted:
        while not exhausted and engine.scheduler.num_in_flight < admission_limit:
            try:
                waveform = next(source)
            except StopIteration:
                exhausted = True
                break
            request = engine.add_request(waveform, language=language)
            order[request.request_id] = next_index
            pending.add(request.request_id)
            next_index += 1

        if not engine.scheduler.has_work:
            break

        outputs, elapsed, was_mixed = step_instrumented(engine, executor, mode)
        if was_mixed:
            mixed_s += elapsed
            n_mixed += 1
        for output in outputs:
            pending.discard(output.request_id)
            results[order[output.request_id]] = output.text

    wall_s = time.perf_counter() - wall_start
    texts = [results[i] for i in range(len(waveforms))]
    return texts, wall_s, mixed_s, n_mixed


def _build_engine(model: str, max_num_seqs: int, gpu_memory_utilization: float) -> AsrEngine:
    return AsrEngine(
        model,
        max_num_seqs=max_num_seqs,
        max_model_len=2048,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
    )


def run_comparison(
    *,
    mode_label: str,
    model: str,
    waveforms: list[np.ndarray],
    max_num_seqs: int,
    gpu_memory_utilization: float,
) -> DualStreamReport:
    """Run serial then overlap on fresh engines; compare texts and mixed-step time."""
    executor = DualStreamExecutor(device="cuda")

    engine_s = _build_engine(model, max_num_seqs, gpu_memory_utilization)
    try:
        texts_s, wall_s, mixed_s, n_mixed_s = transcribe_instrumented(
            engine_s, waveforms, executor, "serial"
        )
    finally:
        del engine_s
        _free_gpu()

    engine_o = _build_engine(model, max_num_seqs, gpu_memory_utilization)
    try:
        texts_o, wall_o, mixed_o, n_mixed_o = transcribe_instrumented(
            engine_o, waveforms, executor, "overlap"
        )
    finally:
        del engine_o
        _free_gpu()

    correctness_ok = texts_s == texts_o
    n_mixed = min(n_mixed_s, n_mixed_o)
    if mixed_o > 0 and n_mixed > 0:
        speedup_mixed = mixed_s / mixed_o
    else:
        speedup_mixed = 0.0
    speedup_wall = wall_s / wall_o if wall_o > 0 else 0.0
    notes = ""
    if n_mixed == 0:
        notes = "no mixed encode+model steps observed; increase samples or lower max_num_seqs"
    elif n_mixed_s != n_mixed_o:
        notes = f"mixed step counts differ serial={n_mixed_s} overlap={n_mixed_o}"
    if not correctness_ok:
        mismatches = sum(a != b for a, b in zip(texts_s, texts_o))
        notes = (notes + "; " if notes else "") + f"{mismatches}/{len(texts_s)} transcripts differ"

    gate = compute_gate(correctness_ok=correctness_ok, speedup_mixed_steps=speedup_mixed)
    return DualStreamReport(
        mode=mode_label,
        model=model,
        speedup_wall=speedup_wall,
        speedup_mixed_steps=speedup_mixed,
        correctness_ok=correctness_ok,
        gate_1_15=gate,
        n_mixed_steps=n_mixed,
        notes=notes,
        serial_mixed_seconds=mixed_s,
        overlap_mixed_seconds=mixed_o,
    )


def _synthetic_waveforms(num: int, seconds: float = 2.0, sr: int = 16000) -> list[np.ndarray]:
    """Deterministic tones so synthetic mode does not require dataset I/O."""
    out: list[np.ndarray] = []
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    for i in range(num):
        freq = 220.0 + 15.0 * (i % 11)
        out.append(0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=("synthetic", "real", "both"), default="both")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports: list[DualStreamReport] = []

    if args.mode in ("synthetic", "both"):
        waves = _synthetic_waveforms(args.num_samples)
        print(f"[synthetic] {len(waves)} tones, max_num_seqs={args.max_num_seqs}")
        report = run_comparison(
            mode_label="synthetic",
            model=args.model,
            waveforms=waves,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        reports.append(report)
        print(
            f"  mixed_steps={report.n_mixed_steps} "
            f"speedup_mixed={report.speedup_mixed_steps:.3f} "
            f"speedup_wall={report.speedup_wall:.3f} "
            f"correct={report.correctness_ok} gate={report.gate_1_15}"
        )
        if report.notes:
            print(f"  notes: {report.notes}")

    if args.mode in ("real", "both"):
        samples = load_librispeech("test-clean", num_samples=args.num_samples, seed=0)
        waves = [s.audio for s in samples]
        print(f"[real] {len(waves)} librispeech clips, max_num_seqs={args.max_num_seqs}")
        report = run_comparison(
            mode_label="real",
            model=args.model,
            waveforms=waves,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        reports.append(report)
        print(
            f"  mixed_steps={report.n_mixed_steps} "
            f"speedup_mixed={report.speedup_mixed_steps:.3f} "
            f"speedup_wall={report.speedup_wall:.3f} "
            f"correct={report.correctness_ok} gate={report.gate_1_15}"
        )
        if report.notes:
            print(f"  notes: {report.notes}")

    payload = {
        "created": stamp,
        "args": {
            "model": args.model,
            "mode": args.mode,
            "num_samples": args.num_samples,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "reports": [r.to_dict() for r in reports],
        "any_gate_pass": any(r.gate_1_15 for r in reports),
    }
    out_path = args.results_dir / f"dual_stream_{args.mode}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}")
    print("GATE PASS" if payload["any_gate_pass"] else "GATE FAIL")


if __name__ == "__main__":
    main()
