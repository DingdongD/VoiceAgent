"""Throughput and quality benchmark.

Two things the existing nano-vLLM ASR benchmark could not show:

*Concurrency sweep.* Continuous batching only pays off above one in-flight
request, so a single-clip measurement cannot see it at all.

*Stage decomposition.* Wall time is split across the mel frontend, the audio
encoder and the decoder, which is what tells you whether a change helped the part
that was actually the bottleneck.

Usage:
    python bench/run_bench.py --num-samples 200 --concurrency 1,4,8,16,32
    python bench/run_bench.py --num-samples 64 --compare engine,reference
"""

from __future__ import annotations

import argparse
import json
import statistics
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
class RunReport:
    name: str
    concurrency: int
    num_samples: int
    audio_seconds: float
    wall_seconds: float
    output_tokens: int
    wer: float
    cer: float
    stage_seconds: dict[str, float] = field(default_factory=dict)
    request_stage_means: dict[str, float] = field(default_factory=dict)
    scheduler: dict[str, int] = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        return self.wall_seconds / self.audio_seconds

    @property
    def audio_seconds_per_second(self) -> float:
        return self.audio_seconds / self.wall_seconds

    @property
    def output_tokens_per_second(self) -> float:
        return self.output_tokens / self.wall_seconds

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.update(
            rtf=self.rtf,
            audio_seconds_per_second=self.audio_seconds_per_second,
            output_tokens_per_second=self.output_tokens_per_second,
        )
        return payload


def _free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def run_engine(
    model_path: str,
    samples: list[AudioSample],
    concurrency: int,
    max_model_len: int,
    language: str | None,
) -> RunReport:
    from qwen_asr_vllm import AsrEngine

    engine = AsrEngine(
        model_path,
        max_num_seqs=concurrency,
        max_model_len=max_model_len,
        max_num_batched_tokens=max(8192, max_model_len),
        max_audio_batch_size=concurrency,
    )
    try:
        # Warm up so kernel autotuning and lazy CUDA init are not billed to the run.
        engine.transcribe([samples[0].audio], language=language)
        engine.stage_seconds = dict.fromkeys(engine.stage_seconds, 0.0)

        started = time.perf_counter()
        outputs = engine.transcribe([sample.audio for sample in samples], language=language)
        wall_seconds = time.perf_counter() - started

        references = [sample.text for sample in samples]
        hypotheses = [output.text for output in outputs]
        return RunReport(
            name="qwen-asr-vllm",
            concurrency=concurrency,
            num_samples=len(samples),
            audio_seconds=sum(sample.duration for sample in samples),
            wall_seconds=wall_seconds,
            output_tokens=sum(output.num_output_tokens for output in outputs),
            wer=compute_wer(references, hypotheses),
            cer=compute_cer(references, hypotheses),
            stage_seconds=dict(engine.stage_seconds),
            request_stage_means={
                stage: statistics.fmean(
                    getattr(output.timings, f"{stage}_seconds") for output in outputs
                )
                for stage in ("queue", "encode", "prefill", "decode", "total")
            },
            scheduler=asdict(engine.scheduler.stats),
        )
    finally:
        del engine
        _free_gpu()


def run_reference(
    model_path: str, samples: list[AudioSample], language: str | None
) -> RunReport:
    from bench.baselines import ReferenceTranscriber

    transcriber = ReferenceTranscriber(model_path)
    try:
        transcriber.transcribe([samples[0].audio], language=language)
        result = transcriber.transcribe([sample.audio for sample in samples], language=language)
        references = [sample.text for sample in samples]
        return RunReport(
            name="reference (transformers)",
            concurrency=1,
            num_samples=len(samples),
            audio_seconds=sum(sample.duration for sample in samples),
            wall_seconds=result.wall_seconds,
            output_tokens=sum(result.num_output_tokens),
            wer=compute_wer(references, result.texts),
            cer=compute_cer(references, result.texts),
        )
    finally:
        del transcriber
        _free_gpu()


def run_nano_vllm(
    model_path: str, samples: list[AudioSample], language: str | None
) -> RunReport:
    from bench.baselines import NanoVllmAsrTranscriber

    transcriber = NanoVllmAsrTranscriber(model_path)
    try:
        transcriber.transcribe([samples[0].audio], language=language)
        result = transcriber.transcribe([sample.audio for sample in samples], language=language)
        references = [sample.text for sample in samples]
        return RunReport(
            name="nano-vllm asr",
            concurrency=1,
            num_samples=len(samples),
            audio_seconds=sum(sample.duration for sample in samples),
            wall_seconds=result.wall_seconds,
            output_tokens=sum(result.num_output_tokens),
            wer=compute_wer(references, result.texts),
            cer=compute_cer(references, result.texts),
        )
    finally:
        del transcriber
        _free_gpu()


def print_table(reports: list[RunReport]) -> None:
    header = (
        f"{'backend':<24}{'conc':>5}{'wall(s)':>10}{'RTF':>9}"
        f"{'audio_s/s':>11}{'out_tok/s':>11}{'WER':>8}{'CER':>8}"
    )
    print()
    print(header)
    print("-" * len(header))
    for report in reports:
        print(
            f"{report.name:<24}{report.concurrency:>5}{report.wall_seconds:>10.2f}"
            f"{report.rtf:>9.4f}{report.audio_seconds_per_second:>11.1f}"
            f"{report.output_tokens_per_second:>11.1f}{report.wer:>8.4f}{report.cer:>8.4f}"
        )

    staged = [report for report in reports if report.stage_seconds]
    if not staged:
        return
    print()
    print("stage decomposition (engine wall seconds)")
    stage_header = (
        f"{'conc':>5}{'frontend':>11}{'audio_enc':>11}{'model':>11}{'other':>11}{'steps':>8}{'mixed':>8}"
    )
    print(stage_header)
    print("-" * len(stage_header))
    for report in staged:
        stages = report.stage_seconds
        accounted = sum(stages.values())
        print(
            f"{report.concurrency:>5}{stages.get('frontend', 0):>11.2f}"
            f"{stages.get('audio_encode', 0):>11.2f}{stages.get('model_step', 0):>11.2f}"
            f"{report.wall_seconds - accounted:>11.2f}"
            f"{report.scheduler.get('num_model_steps', 0):>8}"
            f"{report.scheduler.get('num_mixed_steps', 0):>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", default="test-clean")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0, help="sampling seed; omit for head of split")
    parser.add_argument("--concurrency", default="1,4,8,16,32")
    parser.add_argument("--compare", default="engine", help="engine,reference,nano-vllm")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--language", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    samples = load_librispeech(args.split, num_samples=args.num_samples, seed=args.seed)
    audio_seconds = sum(sample.duration for sample in samples)
    print(
        f"{len(samples)} clips from librispeech {args.split}: {audio_seconds:.1f}s of audio "
        f"(mean {audio_seconds / len(samples):.2f}s)"
    )

    backends = [name.strip() for name in args.compare.split(",") if name.strip()]
    concurrencies = [int(value) for value in args.concurrency.split(",") if value.strip()]

    reports: list[RunReport] = []
    if "engine" in backends:
        for concurrency in concurrencies:
            print(f"  running qwen-asr-vllm at concurrency {concurrency} ...", flush=True)
            reports.append(
                run_engine(args.model, samples, concurrency, args.max_model_len, args.language)
            )
    if "reference" in backends:
        print("  running reference (transformers, sequential) ...", flush=True)
        reports.append(run_reference(args.model, samples, args.language))
    if "nano-vllm" in backends:
        print("  running nano-vllm asr path ...", flush=True)
        reports.append(run_nano_vllm(args.model, samples, args.language))

    print_table(reports)

    output_path = Path(args.out) if args.out else (
        DEFAULT_RESULTS_DIR / f"bench_{args.split}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": args.model,
                "split": args.split,
                "num_samples": len(samples),
                "audio_seconds": audio_seconds,
                "reports": [report.to_dict() for report in reports],
            },
            handle,
            indent=2,
        )
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
