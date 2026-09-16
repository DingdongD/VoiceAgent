"""Is the decoder step launch-bound? Measured on a real workload.

CUDA graphs only pay off when the CPU cannot issue kernels as fast as the GPU
retires them. Rather than construct a synthetic batch and time it -- which is easy
to get wrong, because ``ModelRunner.run`` ends in a device-to-host copy that hides
the CPU/GPU split -- this runs the actual engine and uses the counters it already
keeps: total seconds spent in decoder steps, and how many steps that was.

Batch assembly is timed separately by wrapping ``_prepare``, since that is the CPU
work a captured graph would eliminate. If assembly is a small fraction of the step,
there is nothing for a graph to recover and the unified varlen attention path is
better left alone.

    /opt/conda/envs/nano-vllm/bin/python -m bench.decode_launch_overhead \
        --model /mnt/llm_data/voice_ckpt/Qwen3-ASR-0.6B
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_asr_vllm.engine.engine import AsrEngine


def run_workload(engine: AsrEngine, samples, concurrency: int) -> dict:
    """Transcribe every sample, accounting for CPU batch assembly separately."""
    runner = engine.model_runner
    original_prepare = runner._prepare
    prepare_seconds = 0.0
    prepare_calls = 0

    def timed_prepare(batch):
        nonlocal prepare_seconds, prepare_calls
        started = time.perf_counter()
        result = original_prepare(batch)
        # _prepare ends with host-to-device copies; sync so the cost lands here and
        # not in whatever touches the tensors next.
        torch.cuda.synchronize()
        prepare_seconds += time.perf_counter() - started
        prepare_calls += 1
        return result

    runner._prepare = timed_prepare
    try:
        pending = list(samples)
        in_flight = 0
        outputs = []
        started = time.perf_counter()
        while pending or in_flight:
            while pending and in_flight < concurrency:
                engine.add_request(pending.pop(0).audio, language="en")
                in_flight += 1
            finished = engine.step()
            in_flight -= len(finished)
            outputs.extend(finished)
        elapsed = time.perf_counter() - started
    finally:
        runner._prepare = original_prepare

    stats = engine.scheduler.stats
    steps = stats.num_model_steps
    return {
        "elapsed": elapsed,
        "steps": steps,
        "step_ms": engine.stage_seconds["model_step"] / steps * 1000,
        "prepare_ms": prepare_seconds / prepare_calls * 1000 if prepare_calls else 0.0,
        "decoded_tokens": stats.num_decoded_tokens,
        "audio_seconds": sum(output.audio_seconds for output in outputs),
        "model_step_share": engine.stage_seconds["model_step"] / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = parser.parse_args()

    from bench.data import load_librispeech

    samples = load_librispeech(split="test-clean", num_samples=args.num_samples)

    print(
        f"{'conc':>5} {'steps':>7} {'step':>9} {'assembly':>9} {'assembly%':>10} "
        f"{'realtime':>9} {'step share':>11}"
    )
    for concurrency in args.concurrency:
        engine = AsrEngine(
            args.model,
            max_num_seqs=max(concurrency, 8),
            max_model_len=2048,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        try:
            result = run_workload(engine, samples, concurrency)
        finally:
            del engine
            torch.cuda.empty_cache()

        share = result["prepare_ms"] / result["step_ms"] * 100 if result["step_ms"] else 0
        print(
            f"{concurrency:>5} {result['steps']:>7} {result['step_ms']:>8.2f}ms "
            f"{result['prepare_ms']:>8.2f}ms {share:>9.1f}% "
            f"{result['audio_seconds'] / result['elapsed']:>8.1f}x "
            f"{result['model_step_share'] * 100:>10.1f}%"
        )


if __name__ == "__main__":
    main()
