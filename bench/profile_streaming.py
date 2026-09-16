"""Streaming ASR with a growing audio prefix: latency, recompute and revision.

Measures the re-transcribe baseline, which is what the engine can do today: on every
chunk boundary the whole utterance so far is sent as one request. That is the honest
starting point, and it is what any cache-reuse design has to be compared against.

Why the usual prefix/KV cache does not transfer
-----------------------------------------------
The prompt is ``[prefix][audio ...][suffix][transcript ...]``. A new chunk appends
audio tokens *in the middle* of that layout, ahead of the suffix and of everything
already generated. Three separate consequences, only the first of which a normal
prefix cache handles:

1. Audio tokens 0..A(k) keep their positions and attend only leftwards, so their KV
   is still valid. But the engine cannot reuse it through the prefix cache: blocks are
   keyed by token id, every audio position carries the same ``<|audio_pad|>`` id, and
   caching them would let any two equal-length audios collide. ``BlockManager``
   deliberately refuses to hash blocks containing audio.
2. The suffix and the transcript shift right by one chunk's worth of tokens. Their
   RoPE-encoded keys are computed from absolute positions, so their KV is *wrong*, not
   merely stale, and no amount of prefix matching saves it.
3. The transcript is not append-only in the first place. Later audio changes what the
   earlier audio should have been transcribed as, so tokens already emitted have to be
   allowed to change -- which is the revision rate measured here.

So the reported recompute ratio is prefill tokens actually computed over prefill
tokens in the full context, under two policies: this baseline (everything, ratio 1.0)
and the best an audio-KV-reuse design could do (new audio plus the shifted tail).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.data import load_contiguous_speech
from bench.profile_stages import MODELS
from qwen_asr_vllm.engine.engine import AsrEngine
from qwen_asr_vllm.engine.request import SamplingParams

SAMPLE_RATE = 16000


@dataclass
class ChunkResult:
    index: int
    audio_seconds: float
    latency: float
    prompt_tokens: int
    audio_tokens: int
    output_tokens: int
    words: list[str]
    revised_words: int
    """Words the previous transcript emitted that this one changed or dropped."""
    revision_depth: int
    """Words back from the end of the previous transcript to its earliest change.

    How much of an already-emitted transcript a strict commit policy would have had to
    take back. Distinct from ``revised_words``: changing one word far from the end is a
    single revision but a deep one.
    """


def word_revisions(previous: list[str], current: list[str]) -> tuple[int, int]:
    """Revised word count and revision depth between consecutive transcripts.

    Alignment rather than prefix comparison. Comparing prefixes charges every word
    after the first disagreement as revised, which reports near-total rewriting for
    transcripts that are in fact 99% identical -- one changed word early on poisons the
    entire count.

    Words appended past the end of ``previous`` are new audio being transcribed for the
    first time, so they are growth, not revision.
    """
    matcher = SequenceMatcher(None, previous, current, autojunk=False)
    stable = 0
    earliest: int | None = None
    for tag, begin, end, _, _ in matcher.get_opcodes():
        if tag == "equal":
            stable += end - begin
            continue
        if tag == "insert" and begin >= len(previous):
            continue
        if earliest is None:
            earliest = begin
    depth = 0 if earliest is None else len(previous) - earliest
    return len(previous) - stable, depth


def run_session(
    engine: AsrEngine,
    audio: np.ndarray,
    chunk_seconds: float,
    sampling: SamplingParams,
    language: str,
) -> list[ChunkResult]:
    total_chunks = int(np.ceil(len(audio) / (chunk_seconds * SAMPLE_RATE)))
    results: list[ChunkResult] = []
    previous: list[str] = []

    for index in range(total_chunks):
        end = min(int((index + 1) * chunk_seconds * SAMPLE_RATE), len(audio))
        visible = audio[:end]
        if visible.size < 160:
            continue

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        start.record()
        output = engine.transcribe([visible], language=language, sampling=sampling)[0]
        finish.record()
        torch.cuda.synchronize()

        words = output.text.split()
        revised, depth = word_revisions(previous, words)
        results.append(
            ChunkResult(
                index=index,
                audio_seconds=end / SAMPLE_RATE,
                latency=start.elapsed_time(finish) / 1000,
                prompt_tokens=output.num_prompt_tokens,
                audio_tokens=output.num_audio_tokens,
                output_tokens=output.num_output_tokens,
                words=words,
                revised_words=revised,
                revision_depth=depth,
            )
        )
        previous = words

    return results


def report(results: list[ChunkResult], chunk_seconds: float, label: str) -> None:
    latencies = sorted(r.latency for r in results)
    quantile = statistics.quantiles(latencies, n=100, method="inclusive")

    print(f"\n=== {label} ===")
    print(f"chunks: {len(results)}  chunk size: {chunk_seconds}s")
    print(
        f"chunk latency: p50={quantile[49] * 1000:.0f}ms  p95={quantile[94] * 1000:.0f}ms  "
        f"max={latencies[-1] * 1000:.0f}ms  mean={statistics.fmean(latencies) * 1000:.0f}ms"
    )
    over_budget = sum(1 for value in latencies if value > chunk_seconds)
    print(
        f"chunks slower than real time: {over_budget}/{len(results)} "
        f"({over_budget / len(results):.1%})"
    )

    # Recompute accounting. The baseline prefills the whole context every chunk; an
    # audio-reuse design would only prefill the new audio plus the tail that shifted.
    non_audio = [r.prompt_tokens - r.audio_tokens for r in results]
    per_chunk_audio = results[0].audio_tokens if results else 0
    baseline_tokens = sum(r.prompt_tokens for r in results)
    reuse_tokens = sum(per_chunk_audio + tail for tail in non_audio)
    full_context = results[-1].prompt_tokens if results else 1
    print(
        f"prefill tokens over the session: baseline={baseline_tokens}  "
        f"audio-reuse floor={reuse_tokens}  ({baseline_tokens / max(reuse_tokens, 1):.1f}x)"
    )
    print(
        f"recompute ratio vs final context ({full_context} tokens): "
        f"baseline={baseline_tokens / full_context:.1f}x  "
        f"audio-reuse={reuse_tokens / full_context:.2f}x"
    )
    decoded = sum(r.output_tokens for r in results)
    print(
        f"decode tokens over the session: {decoded} to emit a final "
        f"{results[-1].output_tokens}-token transcript "
        f"({decoded / max(results[-1].output_tokens, 1):.1f}x redundant)"
    )

    revised = sum(r.revised_words for r in results)
    emitted = sum(len(r.words) for r in results)
    changed_chunks = sum(1 for r in results if r.revised_words)
    print(
        f"revision rate: {changed_chunks}/{len(results)} chunks revised earlier words "
        f"({changed_chunks / len(results):.1%});  {revised} revised of {emitted} emitted "
        f"words ({revised / max(emitted, 1):.1%})"
    )
    if len(results) > 1:
        similarity = SequenceMatcher(
            None, " ".join(results[-2].words), " ".join(results[-1].words)
        ).ratio()
        print(f"last two transcripts similarity: {similarity:.3f}")

    # How far back a revision reaches decides whether a prefix can be committed and
    # kept out of the next chunk's decode. Few-but-deep revisions are the awkward case:
    # little text actually changes, but no small commit lag is safe.
    depths = sorted(r.revision_depth for r in results)
    if depths:
        deep = statistics.quantiles(depths, n=100, method="inclusive")
        print(
            f"revision depth (words back from the end): p50={depths[len(depths) // 2]}  "
            f"p95={deep[94]:.0f}  max={depths[-1]}"
        )
        print("  commit lag violated:", end="")
        for lag in (1, 2, 4, 8, 16, 32, 64):
            missed = sum(1 for d in depths if d > lag)
            print(f"  {lag}w {missed / len(depths):>5.1%}", end="")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="0.6B", choices=list(MODELS))
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--chunk-seconds", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--pool-samples", type=int, default=200)
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    audio, reference = load_contiguous_speech(args.seconds)
    print(f"audio: {len(audio) / SAMPLE_RATE:.0f}s contiguous, reference has {len(reference.split())} words")

    context = int(args.seconds * 13 * 1.05) + args.max_new_tokens + 256
    engine = AsrEngine(
        MODELS[args.model],
        max_num_seqs=4,
        max_model_len=context,
        max_num_batched_tokens=context,
        gpu_memory_utilization=0.85,
    )
    sampling = SamplingParams(max_new_tokens=args.max_new_tokens)
    engine.transcribe([audio[: 10 * SAMPLE_RATE]], language=args.language, sampling=sampling)

    timer = engine.enable_stage_profiling()
    results = run_session(engine, audio, args.chunk_seconds, sampling, args.language)
    report(results, args.chunk_seconds, f"{args.model}, {args.seconds:.0f}s utterance")

    stages = timer.totals
    encoder = sum(stages.get(name, 0.0) for name in ("conv", "audio_transformer", "projector"))
    prefill = stages.get("llm_prefill", 0.0)
    decode = stages.get("llm_decode", 0.0)
    device = encoder + prefill + decode or 1.0
    print(
        f"\nsession device time: encoder={encoder:.2f}s ({encoder / device:.1%})  "
        f"prefill={prefill:.2f}s ({prefill / device:.1%})  "
        f"decode={decode:.2f}s ({decode / device:.1%})"
    )
    print(
        "  eliminating all prefill recompute would save at most "
        f"{prefill / device:.1%} of device time"
    )

    print("\nper-chunk detail (every 5th chunk)")
    header = (
        f"{'chunk':>6} {'audio':>7} {'latency':>9} {'prompt':>7} {'audio_tok':>10} "
        f"{'out_tok':>8} {'words':>6} {'revised':>8} {'depth':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results[::5]:
        print(
            f"{r.index:>6} {r.audio_seconds:>6.0f}s {r.latency * 1000:>8.0f}ms "
            f"{r.prompt_tokens:>7} {r.audio_tokens:>10} {r.output_tokens:>8} "
            f"{len(r.words):>6} {r.revised_words:>8} {r.revision_depth:>6}"
        )


if __name__ == "__main__":
    main()
