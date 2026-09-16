"""How the encoder/prefill versus decode balance shifts with concurrency.

Single-stream profiling (``bench/profile_stages.py``) finds decode dominating almost
everything past 30s of audio, because decode spends one sequential step per output
token while prefill consumes the whole audio prompt in one pass.

That is a statement about batch size one. Decode steps batch across requests -- each
step reads the weights once no matter how many sequences ride along -- whereas
encoder and prefill work is per-request and already compute-bound. So the balance
should tilt back towards the front of the pipeline as concurrency rises, and the
question this script answers is where.

Stage shares are reported against device time. Mixed batches, where the scheduler
packs a prefill alongside running decodes, are reported separately rather than being
split: those tokens share one kernel launch, so any division between the two would
be invented.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.data import load_librispeech
from bench.profile_stages import MODELS, concatenated_speech
from qwen_asr_vllm.engine.engine import AsrEngine
from qwen_asr_vllm.engine.request import SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="0.6B", choices=list(MODELS))
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--pool-samples", type=int, default=200)
    args = parser.parse_args()

    samples = load_librispeech(split="test-clean", num_samples=args.pool_samples)
    pool = np.concatenate([s.audio for s in samples])

    peak = max(args.concurrency)
    context = int(args.seconds * 13 * 1.05) + args.max_new_tokens + 256
    engine = AsrEngine(
        MODELS[args.model],
        max_num_seqs=peak,
        max_model_len=context,
        max_num_batched_tokens=max(context, 8192),
        gpu_memory_utilization=0.9,
        enable_prefix_cache=False,
    )
    timer = engine.enable_stage_profiling()
    sampling = SamplingParams(max_new_tokens=args.max_new_tokens)
    clip = concatenated_speech(pool, args.seconds)

    engine.transcribe([clip], language="en", sampling=sampling)

    header = (
        f"{'conc':>5} {'wall':>7} {'audio/s':>8} {'RTF':>8} {'tok/s':>8} "
        f"{'encoder':>8} {'prefill':>8} {'mixed':>8} {'decode':>8} {'front%':>7}"
    )
    print(f"\n{args.model}, {args.seconds:.0f}s clips on {torch.cuda.get_device_name()}")
    print(header)
    print("-" * len(header))

    for concurrency in args.concurrency:
        timer.reset()
        start = time.perf_counter()
        outputs = engine.transcribe([clip] * concurrency, language="en", sampling=sampling)
        wall = time.perf_counter() - start

        totals = timer.totals
        encoder = totals.get("conv", 0.0) + totals.get("audio_transformer", 0.0) + totals.get("projector", 0.0)
        prefill = totals.get("llm_prefill", 0.0)
        mixed = totals.get("llm_mixed", 0.0)
        decode = totals.get("llm_decode", 0.0)
        device = encoder + prefill + mixed + decode or 1.0
        audio_seconds = concurrency * args.seconds
        tokens = sum(o.num_output_tokens for o in outputs)
        print(
            f"{concurrency:>5} {wall:>6.2f}s {audio_seconds:>7.0f}s {wall / audio_seconds:>8.5f} "
            f"{tokens / wall:>8.1f} {encoder / device:>7.1%} {prefill / device:>7.1%} "
            f"{mixed / device:>7.1%} {decode / device:>7.1%} "
            f"{(encoder + prefill) / device:>6.1%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
