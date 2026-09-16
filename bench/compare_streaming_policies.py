"""Compare streaming policies on the same growing-prefix session.

Default comparison is WER-preserving ``retranscribe`` vs ``speculate`` (wall
clock, RTF, gt WER, policy-gap WER, draft accept). ``incremental`` can be added
via ``--policies`` but regresses WER.

Usage::

    python bench/compare_streaming_policies.py --policies retranscribe,speculate \\
        --sources libri,ted --seconds 120 --ted-mode full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.data import load_contiguous_speech, load_tedlium_long_form
from bench.metrics import load_english_normalizer, score_words
from qwen_asr_vllm.engine.async_engine import AsyncAsrEngine

SAMPLE_RATE = 16000
DEFAULT_MODEL = "/mnt/llm_data/voice_ckpt/Qwen3-ASR-0.6B"


@dataclass
class PolicyRun:
    policy: str
    text: str
    wer: float | None
    substitutions: int
    insertions: int
    deletions: int
    ref_words: int
    transcribed_seconds: float
    wall_seconds: float
    window_recomputes: int
    draft_tokens: int
    draft_accepted: int
    audio_seconds: float

    @property
    def rtf(self) -> float:
        return self.wall_seconds / max(self.audio_seconds, 1e-9)

    @property
    def draft_accept_rate(self) -> float | None:
        if self.draft_tokens <= 0:
            return None
        return self.draft_accepted / self.draft_tokens


@dataclass
class UtteranceReport:
    source: str
    sample_id: str
    audio_seconds: float
    policies: dict[str, PolicyRun]
    gaps_vs_baseline: dict[str, float]
    gt_aligned: bool
    baseline: str


def load_sources(
    sources: list[str], seconds: float, ted_mode: str
) -> list[tuple[str, str, np.ndarray, str | None]]:
    out: list[tuple[str, str, np.ndarray, str | None]] = []
    if "libri" in sources:
        audio, text = load_contiguous_speech(seconds)
        out.append(("libri", f"contiguous-{len(audio) / SAMPLE_RATE:.0f}s", audio, text))
    if "ted" in sources:
        talks = [t for t in load_tedlium_long_form() if t.duration >= seconds]
        if not talks:
            raise SystemExit(f"no TED talk >= {seconds}s")
        talk = min(talks, key=lambda t: t.duration)
        if ted_mode == "full":
            out.append(("ted", talk.sample_id, talk.audio, talk.text))
        elif ted_mode == "prefix":
            audio = talk.audio[: int(seconds * SAMPLE_RATE)]
            out.append(("ted", f"{talk.sample_id}-prefix{seconds:.0f}s", audio, None))
        else:
            raise SystemExit("ted_mode must be 'full' or 'prefix'")
    return out


def run_policy(
    engine: AsyncAsrEngine,
    audio: np.ndarray,
    policy: str,
    chunk_seconds: float,
    language: str,
    commit_lag_words: int,
    recompute_seconds: float,
    recompute_overlap_seconds: float,
) -> tuple[str, float, int, float, int, int]:
    kwargs = {
        "language": language,
        "chunk_policy": policy,
        "commit_lag_words": commit_lag_words if policy == "incremental" else 0,
    }
    if policy == "incremental":
        kwargs.update(
            recompute_seconds=recompute_seconds,
            recompute_overlap_seconds=recompute_overlap_seconds,
            tail_min_seconds=0.5,
        )
    session = engine.open_stream(**kwargs)
    chunk = int(chunk_seconds * SAMPLE_RATE)
    started = time.perf_counter()
    for start in range(0, len(audio), chunk):
        session.feed(audio[start : start + chunk])
    finals = session.close(reuse_last=True)
    wall = time.perf_counter() - started
    text = finals[0].text if finals else ""
    return (
        text,
        session.total_transcribed_seconds,
        session.window_recomputes,
        wall,
        session.total_draft_tokens,
        session.total_draft_accepted,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sources", default="libri,ted", help="libri,ted")
    parser.add_argument(
        "--policies",
        default="retranscribe,speculate",
        help="comma list; first policy is the baseline for policy-gap",
    )
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument(
        "--ted-mode",
        default="full",
        choices=["prefix", "full"],
        help="prefix: cut TED audio (no gt WER); full: whole talk >= seconds (gt WER)",
    )
    parser.add_argument("--chunk-seconds", type=float, default=2.0)
    parser.add_argument("--language", default="en")
    parser.add_argument("--commit-lag-words", type=int, default=16)
    parser.add_argument("--recompute-seconds", type=float, default=6.0)
    parser.add_argument("--recompute-overlap-seconds", type=float, default=2.0)
    parser.add_argument("--max-model-len", type=int, default=24576)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    if not policies:
        raise SystemExit("need at least one policy")
    baseline = policies[0]
    utterances = load_sources(sources, args.seconds, args.ted_mode)
    normalizer = load_english_normalizer()

    engine = AsyncAsrEngine(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
    )
    reports: list[UtteranceReport] = []
    try:
        for source, sample_id, audio, reference in utterances:
            gt_aligned = reference is not None
            print(
                f"\n=== {source} {sample_id}  "
                f"{len(audio) / SAMPLE_RATE:.0f}s  chunk={args.chunk_seconds}s  "
                f"gt={'aligned' if gt_aligned else 'n/a'} ==="
            )
            policy_runs: dict[str, PolicyRun] = {}
            for policy in policies:
                text, transcribed, recomputes, wall, draft_n, draft_ok = run_policy(
                    engine,
                    audio,
                    policy,
                    args.chunk_seconds,
                    args.language,
                    args.commit_lag_words,
                    args.recompute_seconds,
                    args.recompute_overlap_seconds,
                )
                if gt_aligned:
                    scored = score_words([reference], [text], normalizer)
                    wer: float | None = scored.wer
                    sub, ins, dele, ref_w = (
                        scored.substitutions,
                        scored.insertions,
                        scored.deletions,
                        scored.ref_words,
                    )
                else:
                    wer, sub, ins, dele, ref_w = None, 0, 0, 0, 0
                run = PolicyRun(
                    policy=policy,
                    text=text,
                    wer=wer,
                    substitutions=sub,
                    insertions=ins,
                    deletions=dele,
                    ref_words=ref_w,
                    transcribed_seconds=transcribed,
                    wall_seconds=wall,
                    window_recomputes=recomputes,
                    draft_tokens=draft_n,
                    draft_accepted=draft_ok,
                    audio_seconds=len(audio) / SAMPLE_RATE,
                )
                policy_runs[policy] = run
                wer_s = f"{run.wer:.4f}" if run.wer is not None else "n/a"
                accept_s = (
                    f"{run.draft_accept_rate:.3f}"
                    if run.draft_accept_rate is not None
                    else "n/a"
                )
                print(
                    f"{policy:<14} WER={wer_s}  "
                    f"wall={run.wall_seconds:.1f}s  RTF={run.rtf:.3f}  "
                    f"transcribed={run.transcribed_seconds:.1f}s  "
                    f"draft_accept={accept_s}  "
                    f"recomputes={run.window_recomputes}"
                )

            gaps: dict[str, float] = {}
            base_text = policy_runs[baseline].text
            for policy in policies[1:]:
                disagree = score_words([base_text], [policy_runs[policy].text], normalizer)
                gaps[policy] = disagree.wer
                speedup = (
                    policy_runs[baseline].wall_seconds
                    / max(policy_runs[policy].wall_seconds, 1e-9)
                )
                print(
                    f"gap vs {baseline:<6} {policy}: WER={disagree.wer:.4f}  "
                    f"({disagree.errors}/{disagree.ref_words} edits)  "
                    f"wall speedup={speedup:.2f}x"
                )

            reports.append(
                UtteranceReport(
                    source=source,
                    sample_id=sample_id,
                    audio_seconds=len(audio) / SAMPLE_RATE,
                    policies=policy_runs,
                    gaps_vs_baseline=gaps,
                    gt_aligned=gt_aligned,
                    baseline=baseline,
                )
            )
    finally:
        engine.close()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(args.out or f"results/streaming_policy_compare_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "seconds": args.seconds,
        "ted_mode": args.ted_mode,
        "chunk_seconds": args.chunk_seconds,
        "policies": policies,
        "baseline": baseline,
        "utterances": [
            {
                "source": r.source,
                "sample_id": r.sample_id,
                "audio_seconds": r.audio_seconds,
                "gt_aligned": r.gt_aligned,
                "baseline": r.baseline,
                "gaps_vs_baseline": r.gaps_vs_baseline,
                "policies": {k: asdict(v) for k, v in r.policies.items()},
            }
            for r in reports
        ],
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
