"""Concurrency sweep: default (CUDA graph) vs eager serial vs dual-stream.

Answers whether ``enable_dual_stream`` helps end-to-end ASR throughput relative to
the production default, and separates that from the cost of forcing eager decode.

Usage:
  /opt/conda/envs/nano-vllm/bin/python bench/concurrency_dual_stream_sweep.py \\
      --num-samples 64 --concurrency 1,4,8,16,32
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.data import AudioSample, load_librispeech
from bench.metrics import compute_cer, compute_wer

DEFAULT_MODEL = "/mnt/llm_data/voice_ckpt/Qwen3-ASR-0.6B"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


@dataclass
class SweepRow:
    name: str
    concurrency: int
    wall_seconds: float
    audio_seconds: float
    output_tokens: int
    wer: float
    cer: float
    num_model_steps: int = 0
    num_mixed_steps: int = 0
    num_encode_batches: int = 0
    stage_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        return self.wall_seconds / self.audio_seconds if self.audio_seconds else 0.0

    @property
    def audio_per_s(self) -> float:
        return self.audio_seconds / self.wall_seconds if self.wall_seconds else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rtf"] = self.rtf
        d["audio_seconds_per_second"] = self.audio_per_s
        return d


def _free_gpu() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def run_one(
    *,
    name: str,
    model: str,
    samples: list[AudioSample],
    concurrency: int,
    max_model_len: int,
    language: str | None,
    enable_dual_stream: bool,
    enforce_eager: bool,
    gpu_memory_utilization: float,
) -> SweepRow:
    from qwen_asr_vllm import AsrEngine

    engine = AsrEngine(
        model,
        max_num_seqs=concurrency,
        max_model_len=max_model_len,
        max_num_batched_tokens=max(8192, max_model_len),
        max_audio_batch_size=concurrency,
        enable_dual_stream=enable_dual_stream,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    try:
        engine.transcribe([samples[0].audio], language=language)
        engine.stage_seconds = dict.fromkeys(engine.stage_seconds, 0.0)

        started = time.perf_counter()
        outputs = engine.transcribe([s.audio for s in samples], language=language)
        wall = time.perf_counter() - started

        refs = [s.text for s in samples]
        hyps = [o.text for o in outputs]
        stats = engine.scheduler.stats
        return SweepRow(
            name=name,
            concurrency=concurrency,
            wall_seconds=wall,
            audio_seconds=sum(s.duration for s in samples),
            output_tokens=sum(o.num_output_tokens for o in outputs),
            wer=compute_wer(refs, hyps),
            cer=compute_cer(refs, hyps),
            num_model_steps=stats.num_model_steps,
            num_mixed_steps=stats.num_mixed_steps,
            num_encode_batches=stats.num_encode_batches,
            stage_seconds=dict(engine.stage_seconds),
        )
    finally:
        del engine
        _free_gpu()


def print_table(rows: list[SweepRow], baseline_by_conc: dict[int, SweepRow]) -> None:
    header = (
        f"{'config':<22}{'conc':>5}{'wall':>8}{'RTF':>9}{'audio/s':>10}"
        f"{'vs_def':>8}{'mixed':>7}{'WER':>8}"
    )
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        base = baseline_by_conc.get(row.concurrency)
        vs = (base.wall_seconds / row.wall_seconds) if base and row.wall_seconds > 0 else float("nan")
        print(
            f"{row.name:<22}{row.concurrency:>5}{row.wall_seconds:>8.2f}{row.rtf:>9.4f}"
            f"{row.audio_per_s:>10.1f}{vs:>8.3f}{row.num_mixed_steps:>7}{row.wer:>8.4f}"
        )
    print()
    print("vs_def = default_wall / this_wall (>1 means faster than default)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", default="1,4,8,16,32")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--language", default="en")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    samples = load_librispeech("test-clean", num_samples=args.num_samples, seed=args.seed)
    audio_s = sum(s.duration for s in samples)
    concurrencies = [int(x) for x in args.concurrency.split(",") if x.strip()]
    print(
        f"{len(samples)} clips, {audio_s:.1f}s audio (mean {audio_s / len(samples):.2f}s); "
        f"concurrency={concurrencies}"
    )

    configs = [
        ("default(graph)", False, False),
        ("eager_serial", False, True),
        ("dual_stream", True, True),
    ]

    rows: list[SweepRow] = []
    for conc in concurrencies:
        for name, dual, eager in configs:
            # dual_stream forces eager in config; pass eager=True explicitly too
            print(f"  {name} @ conc={conc} ...", flush=True)
            rows.append(
                run_one(
                    name=name,
                    model=args.model,
                    samples=samples,
                    concurrency=conc,
                    max_model_len=args.max_model_len,
                    language=args.language,
                    enable_dual_stream=dual,
                    enforce_eager=eager,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                )
            )

    baseline = {r.concurrency: r for r in rows if r.name == "default(graph)"}
    print_table(rows, baseline)

    # Pairwise: dual vs eager_serial at each conc (isolates overlap gain)
    print("dual_stream vs eager_serial (overlap-only, both eager):")
    for conc in concurrencies:
        eager_row = next(r for r in rows if r.name == "eager_serial" and r.concurrency == conc)
        dual_row = next(r for r in rows if r.name == "dual_stream" and r.concurrency == conc)
        speedup = eager_row.wall_seconds / dual_row.wall_seconds if dual_row.wall_seconds else 0
        print(
            f"  conc={conc:>2}: eager={eager_row.wall_seconds:.2f}s  "
            f"dual={dual_row.wall_seconds:.2f}s  speedup={speedup:.3f}x  "
            f"mixed_steps={dual_row.num_mixed_steps}"
        )

    out = args.out or (
        DEFAULT_RESULTS_DIR / f"concurrency_dual_stream_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "num_samples": len(samples),
        "audio_seconds": audio_s,
        "concurrency": concurrencies,
        "rows": [r.to_dict() for r in rows],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
