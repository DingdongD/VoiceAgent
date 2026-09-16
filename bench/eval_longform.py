"""Word error rate on whole recordings — the gate this project was missing.

Short-clip accuracy has been checked all along and passes at parity with the reference
implementation. Nothing checked accuracy on whole recordings, and that is the regime
where the interesting failures live: the model is running 40x past the ~30s clips it was
tuned on, and every optimisation queued up next (KV quantisation, KV eviction,
speculative decoding) would show up here first. This project has already shipped one
silent accuracy regression — SDPA ignoring ``cu_seqlens``, 78% relative WER — precisely
because a number like this was not being watched.

Two comparisons, because they answer different questions:

* engine vs reference on the same audio isolates *our* errors from the model's. A gap
  here is a bug we introduced.
* either against ground truth measures whether the model is usable at this length at
  all, which no amount of engine correctness can fix.

Usage::

    python bench/eval_longform.py --model /path/to/Qwen3-ASR-0.6B --compare engine,reference
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.data import load_tedlium_long_form
from bench.metrics import load_english_normalizer, normalize_text, score_words

DEFAULT_MODEL = "/mnt/llm_data/voice_ckpt/Qwen3-ASR-0.6B"
SAMPLE_RATE = 16000

logger = logging.getLogger("eval_longform")


@dataclass
class TalkResult:
    talk_id: str
    duration: float
    ref_words: int
    hyp_words: int
    wer: float
    substitutions: int
    insertions: int
    deletions: int
    wall_seconds: float
    finish_reason: str | None


def transcribe_with_engine(
    talks: list, model: str, max_model_len: int, concurrency: int
) -> tuple[list[str], list[float], list[str | None]]:
    """Whole talks, all in flight at once, with a duration-scaled token cap."""
    from qwen_asr_vllm.engine.async_engine import AsyncAsrEngine
    from qwen_asr_vllm.engine.request import SamplingParams

    engine = AsyncAsrEngine(
        model=model,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        max_num_seqs=concurrency,
    )
    try:
        handles = [
            engine.submit(
                talk.audio,
                language="en",
                sampling=SamplingParams.for_audio(talk.duration),
            )
            for talk in talks
        ]
        texts: list[str] = []
        walls: list[float] = []
        reasons: list[str | None] = []
        for handle in handles:
            output = handle.result()
            texts.append(output.text)
            walls.append(output.timings.total_seconds)
            reasons.append(output.finish_reason)
    finally:
        engine.close()
    return texts, walls, reasons


def transcribe_with_reference(
    talks: list, model: str
) -> tuple[list[str], list[float], list[str | None]]:
    """The transformers path, one talk at a time.

    Whether it survives a 20-minute input at all is part of what we are measuring, so a
    failure is recorded per talk instead of aborting the run.
    """
    from bench.baselines import ReferenceTranscriber

    transcriber = ReferenceTranscriber(model)
    texts: list[str] = []
    walls: list[float] = []
    reasons: list[str | None] = []
    for talk in talks:
        started = time.perf_counter()
        try:
            result = transcriber.transcribe(
                [talk.audio], language="en", max_new_tokens=int(talk.duration * 8) + 64
            )
            texts.append(result.texts[0])
            reasons.append("stop")
        except Exception as exc:  # noqa: BLE001 - a failure at length is a finding
            logger.warning("reference failed on %s (%.0fs): %s", talk.sample_id, talk.duration, exc)
            texts.append("")
            reasons.append(f"failed: {type(exc).__name__}")
        walls.append(time.perf_counter() - started)
    return texts, walls, reasons


def score(talks: list, texts: list[str], walls: list[float], reasons: list, normalizer):
    results = []
    for talk, text, wall, reason in zip(talks, texts, walls, reasons):
        one = score_words([talk.text], [text], normalizer)
        results.append(
            TalkResult(
                talk_id=talk.sample_id,
                duration=talk.duration,
                ref_words=one.ref_words,
                hyp_words=len(normalizer(text).split()),
                wer=one.wer,
                substitutions=one.substitutions,
                insertions=one.insertions,
                deletions=one.deletions,
                wall_seconds=wall,
                finish_reason=reason,
            )
        )
    return results


def print_table(name: str, results: list[TalkResult], corpus) -> None:
    print(f"\n{name}")
    print(
        f"{'talk':<22}{'dur(s)':>8}{'ref_w':>7}{'hyp_w':>7}"
        f"{'WER':>8}{'sub':>6}{'ins':>6}{'del':>6}{'RTF':>8}{'finish':>10}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r.talk_id[:22]:<22}{r.duration:>8.0f}{r.ref_words:>7}{r.hyp_words:>7}"
            f"{r.wer:>8.4f}{r.substitutions:>6}{r.insertions:>6}{r.deletions:>6}"
            f"{r.wall_seconds / r.duration:>8.4f}{r.finish_reason!s:>10}"
        )
    print("-" * 90)
    print(
        f"{'corpus':<22}{sum(r.duration for r in results):>8.0f}{corpus.ref_words:>7}"
        f"{'':>7}{corpus.wer:>8.4f}{corpus.substitutions:>6}{corpus.insertions:>6}"
        f"{corpus.deletions:>6}"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--compare", default="engine", help="engine,reference")
    parser.add_argument("--max-model-len", type=int, default=24576)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--min-duration", type=float, default=0.0)
    parser.add_argument(
        "--normalizer",
        default="english",
        choices=["english", "basic"],
        help="english = Whisper's, which reconciles spelled numbers; basic = ours",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    talks = [t for t in load_tedlium_long_form() if t.duration >= args.min_duration]
    talks.sort(key=lambda t: t.duration)
    total_audio = sum(t.duration for t in talks)
    print(
        f"{len(talks)} talks, {total_audio / 3600:.2f}h of audio, "
        f"{min(t.duration for t in talks):.0f}-{max(t.duration for t in talks):.0f}s each"
    )

    normalizer = load_english_normalizer() if args.normalizer == "english" else normalize_text

    payload: dict = {
        "model": args.model,
        "normalizer": args.normalizer,
        "num_talks": len(talks),
        "audio_hours": total_audio / 3600,
        "backends": {},
    }
    backends = [b.strip() for b in args.compare.split(",") if b.strip()]
    hypotheses: dict[str, list[str]] = {}

    for backend in backends:
        if backend == "engine":
            texts, walls, reasons = transcribe_with_engine(
                talks, args.model, args.max_model_len, args.concurrency
            )
        elif backend == "reference":
            texts, walls, reasons = transcribe_with_reference(talks, args.model)
        else:
            raise SystemExit(f"unknown backend {backend}")
        hypotheses[backend] = texts
        results = score(talks, texts, walls, reasons, normalizer)
        corpus = score_words([t.text for t in talks], texts, normalizer)
        print_table(backend, results, corpus)
        payload["backends"][backend] = {
            "corpus": asdict(corpus) | {"errors": corpus.errors},
            "talks": [asdict(r) for r in results],
        }

    if "engine" in hypotheses and "reference" in hypotheses:
        # Engine against reference rather than against truth: this is the number that
        # says whether we introduced anything, with the model's own errors cancelled.
        agreement = score_words(hypotheses["reference"], hypotheses["engine"], normalizer)
        print(
            f"\nengine vs reference: {agreement.wer:.4f} word disagreement "
            f"({agreement.errors} edits over {agreement.ref_words} words)"
        )
        payload["engine_vs_reference"] = asdict(agreement) | {"errors": agreement.errors}

    out = Path(args.out or f"results/longform_wer_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
