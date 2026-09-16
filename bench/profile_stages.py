"""Per-stage cost of one transcription, swept over audio duration and model size.

Answers where the time actually goes for a single request: mel extraction, the
encoder's convolutional downsampler, the audio transformer, the projector, LLM
prefill, and decode -- plus TTFT, RTF, throughput, peak memory and KV footprint.

Batch size is one on purpose. Stage attribution is only honest when no two stages
share a kernel launch, and continuous batching deliberately overlaps prefill with
decode. Throughput numbers here are therefore *latency-bound* single-stream figures,
not the engine's peak; ``bench/throughput.py`` measures that.

``max_new_tokens`` is raised well above the 440-token default because that default
truncates anything longer than a few minutes -- a 1200s transcript needs ~4200
tokens, and measuring decode against a truncated transcript would understate it by
several times over.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.data import load_librispeech, load_tedlium_long_form
from qwen_asr_vllm.engine.engine import AsrEngine
from qwen_asr_vllm.engine.request import SamplingParams

DURATIONS = (2, 10, 30, 120, 600, 1200)
MODELS = {
    "0.6B": "/mnt/llm_data/voice_ckpt/Qwen3-ASR-0.6B",
    "1.7B": "/mnt/llm_data/voice_ckpt/Qwen3-ASR-1.7B",
}
GIB = 2**30


@dataclass
class StageRow:
    model: str
    audio_seconds: float
    audio_tokens: int
    prompt_tokens: int
    output_tokens: int
    finish_reason: str
    fbank: float
    conv: float
    audio_transformer: float
    projector: float
    llm_prefill: float
    llm_decode: float
    decode_steps: int
    ttft: float
    wall: float
    peak_gib: float
    activation_gib: float
    kv_mib: float
    text: str = field(repr=False, default="")

    @property
    def gpu_total(self) -> float:
        return self.conv + self.audio_transformer + self.projector + self.llm_prefill + self.llm_decode

    @property
    def rtf(self) -> float:
        return self.wall / self.audio_seconds

    @property
    def tokens_per_second(self) -> float:
        return self.output_tokens / self.wall if self.wall else 0.0

    @property
    def ms_per_decode_step(self) -> float:
        return 1000 * self.llm_decode / self.decode_steps if self.decode_steps else 0.0


def concatenated_speech(pool: np.ndarray, seconds: float) -> np.ndarray:
    """A clip of the requested length, built from real speech.

    Synthetic noise would be easier but sends decode off the distribution the model
    was trained on: it emits degenerate loops, the repetition guard collapses them,
    and the resulting output length -- which is what decode cost is proportional to
    -- stops meaning anything.
    """
    total = int(seconds * 16000)
    repeats = int(np.ceil(total / len(pool)))
    return np.tile(pool, repeats)[:total].astype(np.float32)


def profile_model(
    label: str,
    path: str,
    clips: list[tuple[float, np.ndarray]],
    max_new_tokens: int,
    repeats: int,
) -> list[StageRow]:
    longest = max(seconds for seconds, _ in clips)
    # 13 audio tokens per second plus the prompt frame, then room for the transcript.
    context = int(longest * 13 * 1.05) + max_new_tokens + 256
    engine = AsrEngine(
        path,
        max_num_seqs=4,
        max_model_len=context,
        max_num_batched_tokens=context,
        gpu_memory_utilization=0.85,
        enable_prefix_cache=False,
    )
    timer = engine.enable_stage_profiling()
    sampling = SamplingParams(max_new_tokens=max_new_tokens)

    # Warm up the allocator, the CUDA graphs and the tokenizer before measuring.
    engine.transcribe([clips[0][1][: 10 * 16000]], language="en", sampling=sampling)
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()

    def measure_once(clip: np.ndarray) -> StageRow:
        timer.reset()
        for key in engine.stage_seconds:
            engine.stage_seconds[key] = 0.0
        steps_before = engine.scheduler.stats.num_model_steps
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        output = engine.transcribe([clip], language="en", sampling=sampling)[0]
        wall = time.perf_counter() - start

        totals = timer.totals
        steps = engine.scheduler.stats.num_model_steps - steps_before
        peak = torch.cuda.max_memory_allocated()
        return StageRow(
            model=label,
            audio_seconds=len(clip) / 16000,
            audio_tokens=output.num_audio_tokens,
            prompt_tokens=output.num_prompt_tokens,
            output_tokens=output.num_output_tokens,
            finish_reason=output.finish_reason,
            fbank=engine.stage_seconds.get("frontend", 0.0),
            conv=totals.get("conv", 0.0),
            audio_transformer=totals.get("audio_transformer", 0.0),
            projector=totals.get("projector", 0.0),
            llm_prefill=totals.get("llm_prefill", 0.0),
            llm_decode=totals.get("llm_decode", 0.0),
            # One prefill step; the rest produced a token each.
            decode_steps=max(steps - 1, 1),
            ttft=output.timings.queue_seconds
            + output.timings.encode_seconds
            + output.timings.prefill_seconds,
            wall=wall,
            peak_gib=peak / GIB,
            activation_gib=max(peak - resident, 0) / GIB,
            kv_mib=engine.kv_bytes_for(output.num_prompt_tokens + output.num_output_tokens)
            / 2**20,
            text=output.text,
        )

    rows: list[StageRow] = []
    for seconds, clip in clips:
        # The first call at a new mel shape pays one-off autotuning inside the conv
        # stack, which lands entirely on T_conv/T_AuT and dwarfs the real cost at
        # short durations. Discard it, then take the median of the rest.
        measure_once(clip)
        trials = sorted((measure_once(clip) for _ in range(repeats)), key=lambda r: r.wall)
        row = trials[len(trials) // 2]
        rows.append(row)
        print(
            f"  {seconds:>5.0f}s  audio_tok={row.audio_tokens:>6}  out_tok={row.output_tokens:>5}"
            f"  wall={row.wall:>6.2f}s  rtf={row.rtf:.4f}  ttft={row.ttft:.3f}s"
            f"  finish={row.finish_reason}",
            flush=True,
        )
        if row.finish_reason == "length":
            print(
                f"      WARNING: hit the {max_new_tokens}-token cap, so decode is truncated",
                flush=True,
            )

    return rows


def print_stage_table(rows: list[StageRow]) -> None:
    print("\n== Stage decomposition (seconds of device time; T_Fbank is CPU) ==")
    header = (
        f"{'model':>6} {'audio':>7} {'T_Fbank':>8} {'T_conv':>8} {'T_AuT':>8} {'T_proj':>8} "
        f"{'T_prefill':>10} {'T_decode':>9} {'GPU tot':>8} {'wall':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.model:>6} {r.audio_seconds:>6.0f}s {r.fbank:>8.3f} {r.conv:>8.3f} "
            f"{r.audio_transformer:>8.3f} {r.projector:>8.3f} {r.llm_prefill:>10.3f} "
            f"{r.llm_decode:>9.3f} {r.gpu_total:>8.3f} {r.wall:>7.2f}"
        )

    print("\n== Share of measured GPU time ==")
    header = (
        f"{'model':>6} {'audio':>7} {'conv':>7} {'AuT':>7} {'proj':>7} {'prefill':>8} "
        f"{'decode':>8} {'encoder+prefill':>16}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        total = r.gpu_total or 1.0
        front = (r.conv + r.audio_transformer + r.projector + r.llm_prefill) / total
        print(
            f"{r.model:>6} {r.audio_seconds:>6.0f}s {r.conv / total:>6.1%} "
            f"{r.audio_transformer / total:>6.1%} {r.projector / total:>6.1%} "
            f"{r.llm_prefill / total:>7.1%} {r.llm_decode / total:>7.1%} {front:>15.1%}"
        )

    print("\n== Serving metrics ==")
    header = (
        f"{'model':>6} {'audio':>7} {'TTFT':>8} {'RTF':>8} {'tok/s':>8} {'steps':>7} "
        f"{'ms/step':>8} {'peak GiB':>9} {'act GiB':>8} {'KV MiB':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.model:>6} {r.audio_seconds:>6.0f}s {r.ttft:>7.3f}s {r.rtf:>8.4f} "
            f"{r.tokens_per_second:>8.1f} {r.decode_steps:>7} {r.ms_per_decode_step:>8.2f} "
            f"{r.peak_gib:>9.2f} {r.activation_gib:>8.2f} {r.kv_mib:>8.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument(
        "--dataset",
        default="tedlium",
        choices=["tedlium", "librispeech"],
        help="tedlium uses whole TED talks at their recorded length; librispeech "
        "stitches clips to hit --durations exactly",
    )
    parser.add_argument("--durations", nargs="+", type=int, default=list(DURATIONS))
    parser.add_argument("--max-new-tokens", type=int, default=16000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pool-samples", type=int, default=400)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.dataset == "tedlium":
        talks = load_tedlium_long_form()
        clips = [(talk.duration, talk.audio) for talk in talks]
        print(f"{len(talks)} whole TED talks, {clips[0][0]:.0f}s to {clips[-1][0]:.0f}s")
    else:
        samples = load_librispeech(split="test-clean", num_samples=args.pool_samples)
        pool = np.concatenate([s.audio for s in samples])
        print(f"speech pool: {len(pool) / 16000:.0f}s from {len(samples)} LibriSpeech clips")
        clips = [(float(sec), concatenated_speech(pool, sec)) for sec in args.durations]

    rows: list[StageRow] = []
    for label in args.models:
        print(f"\n=== {label} ===", flush=True)
        # The engine holds a KV cache sized to most of the card, so it has to be gone
        # before the next model loads.
        torch.cuda.empty_cache()
        rows.extend(
            profile_model(label, MODELS[label], clips, args.max_new_tokens, args.repeats)
        )

    print_stage_table(rows)
    if args.json:
        payload = [{k: v for k, v in vars(r).items() if k != "text"} for r in rows]
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
