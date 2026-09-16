"""Acceptance rate of the previous transcript as a speculative draft.

Streaming re-transcribe regenerates the whole transcript every chunk. Decode is
97% of that cost, and word-level revisions are about 1%, so the natural draft is
the previous chunk's tokens. This script measures how much of that draft the model
actually accepts when the audio has grown by one chunk.

Two numbers, because they answer different questions:

* ``verify`` acceptance is what a serving integration would see: one forward over
  prompt plus draft, then the leading agreed prefix. That is the number that
  decides whether wiring ``ModelRunner.verify`` into the loop is worth it.
* Token LCP against an independent greedy transcription of the new audio is the
  same quantity at temperature 0 (both compare the draft to the same argmax), and
  is reported alongside as a cross-check. A gap means the verify path is wrong.

Usage::

    python bench/profile_speculation.py --seconds 300 --chunk-seconds 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.data import load_tedlium_long_form
from qwen_asr_vllm.engine.engine import AsrEngine
from qwen_asr_vllm.engine.request import SamplingParams
from qwen_asr_vllm.engine.speculate import longest_common_prefix

SAMPLE_RATE = 16000
DEFAULT_MODEL = "/mnt/llm_data/voice_ckpt/Qwen3-ASR-0.6B"


@dataclass
class ChunkSpec:
    index: int
    audio_seconds: float
    draft_tokens: int
    accepted: int
    lcp: int
    new_tokens: int
    verify_agrees: bool

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.draft_tokens if self.draft_tokens else float("nan")


def pick_talk(seconds: float):
    talks = [t for t in load_tedlium_long_form() if t.duration >= seconds]
    if not talks:
        raise SystemExit(f"no TED talk of at least {seconds}s in tedlium_long_form")
    # Shortest talk that covers the window: less wall time for the same duration.
    return min(talks, key=lambda t: t.duration)


def run_session(
    engine: AsrEngine,
    audio: np.ndarray,
    chunk_seconds: float,
    language: str,
    check_verify: bool,
) -> list[ChunkSpec]:
    total_chunks = int(np.ceil(len(audio) / (chunk_seconds * SAMPLE_RATE)))
    previous_tokens: list[int] = []
    results: list[ChunkSpec] = []

    for index in range(total_chunks):
        end = min(int((index + 1) * chunk_seconds * SAMPLE_RATE), len(audio))
        visible = audio[:end]
        if visible.size < 160:
            continue

        sampling = SamplingParams.for_audio(len(visible) / SAMPLE_RATE)
        output = engine.transcribe([visible], language=language, sampling=sampling)[0]
        current = output.output_token_ids
        if not previous_tokens:
            previous_tokens = current
            continue

        lcp = longest_common_prefix(previous_tokens, current)
        if check_verify:
            accepted, _ = engine.verify_draft(
                visible, previous_tokens, language=language
            )
        else:
            # At temperature 0 this equals verify; skip the extra forward on long runs.
            accepted = lcp

        results.append(
            ChunkSpec(
                index=index,
                audio_seconds=len(visible) / SAMPLE_RATE,
                draft_tokens=len(previous_tokens),
                accepted=accepted,
                lcp=lcp,
                new_tokens=len(current),
                verify_agrees=accepted == lcp,
            )
        )
        previous_tokens = current
    return results


def summarise(results: list[ChunkSpec]) -> dict:
    if not results:
        return {}
    draft = sum(r.draft_tokens for r in results)
    accepted = sum(r.accepted for r in results)
    rates = [r.accept_rate for r in results]
    # Tokens the next chunk still has to decode after a full speculative accept.
    residual = [max(0, r.new_tokens - r.accepted) for r in results]
    return {
        "chunks": len(results),
        "draft_tokens": draft,
        "accepted_tokens": accepted,
        "corpus_accept_rate": accepted / draft if draft else float("nan"),
        "per_chunk_accept_rate_p50": statistics.median(rates),
        "per_chunk_accept_rate_p05": sorted(rates)[max(0, int(0.05 * len(rates)) - 1)],
        # Speculative decoding needs a *prefix*. One early punctuation flip zeros the
        # rest of the draft, so the per-chunk rate is bimodal rather than clustered
        # around the corpus mean — that is the shape the serving path has to live with.
        "chunks_accept_ge_0_9": sum(1 for r in rates if r >= 0.9),
        "chunks_accept_lt_0_1": sum(1 for r in rates if r < 0.1),
        "full_accept_chunks": sum(1 for r in results if r.accepted == r.draft_tokens),
        "mean_residual_tokens": statistics.mean(residual),
        "mean_draft_tokens": statistics.mean(r.draft_tokens for r in results),
        "verify_lcp_mismatches": sum(1 for r in results if not r.verify_agrees),
        # Decode steps saved if every accepted token replaces one sequential step.
        "decode_steps_saved_fraction": accepted
        / sum(r.new_tokens for r in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--chunk-seconds", type=float, default=2.0)
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--check-verify",
        action="store_true",
        help="run ModelRunner.verify on every chunk (slow); default uses LCP after a probe",
    )
    parser.add_argument("--max-model-len", type=int, default=24576)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    talk = pick_talk(args.seconds)
    audio = talk.audio[: int(args.seconds * SAMPLE_RATE)]
    print(
        f"talk={talk.sample_id} using first {args.seconds:.0f}s of "
        f"{talk.duration:.0f}s, chunk={args.chunk_seconds}s"
    )

    engine = AsrEngine(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
    )
    try:
        # Always check verify on a short probe so a broken path cannot hide behind LCP.
        probe = audio[: int(min(8.0, args.seconds) * SAMPLE_RATE)]
        first = engine.transcribe(
            [probe[: len(probe) // 2]],
            language=args.language,
            sampling=SamplingParams.for_audio(len(probe) / (2 * SAMPLE_RATE)),
        )[0]
        second = engine.transcribe(
            [probe],
            language=args.language,
            sampling=SamplingParams.for_audio(len(probe) / SAMPLE_RATE),
        )[0]
        accepted, _ = engine.verify_draft(
            probe, first.output_token_ids, language=args.language
        )
        lcp = longest_common_prefix(first.output_token_ids, second.output_token_ids)
        print(
            f"probe verify: accepted={accepted} lcp={lcp} "
            f"draft={len(first.output_token_ids)} new={len(second.output_token_ids)}"
        )
        if accepted != lcp:
            raise SystemExit(
                f"verify/LCP disagree on probe ({accepted} vs {lcp}); aborting"
            )

        results = run_session(
            engine,
            audio,
            args.chunk_seconds,
            args.language,
            check_verify=args.check_verify,
        )
    finally:
        del engine

    summary = summarise(results)
    print("\nacceptance (previous transcript as draft)")
    print(
        f"{'chunk':>6}{'audio_s':>9}{'draft':>8}{'accept':>8}"
        f"{'rate':>8}{'new':>8}{'residual':>10}"
    )
    print("-" * 60)
    sample_rows = list(results[:: max(1, len(results) // 12)])
    if results and results[-1] not in sample_rows:
        sample_rows.append(results[-1])
    for r in sample_rows:
        print(
            f"{r.index:>6}{r.audio_seconds:>9.1f}{r.draft_tokens:>8}{r.accepted:>8}"
            f"{r.accept_rate:>8.3f}{r.new_tokens:>8}"
            f"{max(0, r.new_tokens - r.accepted):>10}"
        )
    print("-" * 60)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(args.out or f"results/speculation_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "talk_id": talk.sample_id,
                "seconds": args.seconds,
                "chunk_seconds": args.chunk_seconds,
                "summary": summary,
                "chunks": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
