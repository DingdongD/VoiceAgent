"""The long-form accuracy gate.

This suite already checked that the engine matches the reference numerically on short
clips, and that was not enough: the checks operate on ~30s inputs, and the engine's
whole reason to exist is inputs 40x longer. An error that only appears once the audio
occupies thousands of KV slots -- a wrong ``cu_seqlens``, an attention kernel that
silently falls back, a token cap that truncates, a block reused across requests -- can
pass every existing test.

That is not hypothetical here. Attention degrading to SDPA, which ignores ``cu_seqlens``
and therefore lets sequences read each other's keys, once cost 78% relative WER and was
found by accident rather than by a test.

Measured with ``bench/eval_longform.py`` on 1.56h of whole TED talks, 187-1299s each,
Qwen3-ASR-0.6B: corpus WER 0.0270 for the engine against 0.0273 for the reference, and
0.0012 word disagreement between the two. The thresholds below sit just above those, so
a regression of even a few percent relative fails rather than being absorbed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Corpus WER measured at 0.0270; a real regression clears 0.030 easily while
# run-to-run scheduling nondeterminism does not.
MAX_CORPUS_WER = 0.030
# Worst single talk measured at 0.0538, on 381s of hard content rather than on the
# longest input. Guards against one recording collapsing while the corpus average hides
# it.
MAX_TALK_WER = 0.065

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.checkpoint,
    pytest.mark.dataset,
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def longform_transcripts(model_dir):
    """Transcribe every talk once; the assertions below all read this."""
    from bench.data import load_tedlium_long_form
    from bench.metrics import load_english_normalizer, score_words
    from qwen_asr_vllm.engine.async_engine import AsyncAsrEngine
    from qwen_asr_vllm.engine.request import SamplingParams

    talks = sorted(load_tedlium_long_form(), key=lambda t: t.duration)
    engine = AsyncAsrEngine(
        model=model_dir,
        max_model_len=24576,
        max_num_batched_tokens=24576,
        max_num_seqs=8,
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
        outputs = [handle.result() for handle in handles]
    finally:
        engine.close()

    normalizer = load_english_normalizer()
    return {
        "talks": talks,
        "outputs": outputs,
        "normalizer": normalizer,
        "corpus": score_words(
            [talk.text for talk in talks], [out.text for out in outputs], normalizer
        ),
    }


class TestLongFormAccuracy:
    def test_corpus_wer_does_not_regress(self, longform_transcripts):
        corpus = longform_transcripts["corpus"]
        assert corpus.wer <= MAX_CORPUS_WER, (
            f"long-form WER {corpus.wer:.4f} exceeds {MAX_CORPUS_WER} "
            f"({corpus.errors} edits over {corpus.ref_words} words)"
        )

    def test_no_talk_collapses(self, longform_transcripts):
        from bench.metrics import score_words

        normalizer = longform_transcripts["normalizer"]
        worst = []
        for talk, out in zip(longform_transcripts["talks"], longform_transcripts["outputs"]):
            one = score_words([talk.text], [out.text], normalizer)
            if one.wer > MAX_TALK_WER:
                worst.append(f"{talk.sample_id} ({talk.duration:.0f}s): {one.wer:.4f}")
        assert not worst, "talks above the per-recording ceiling: " + ", ".join(worst)

    def test_nothing_is_truncated(self, longform_transcripts):
        """A cap hit means transcript was dropped, which WER alone under-reports.

        Deleting the tail of a 20-minute transcript is a huge accuracy loss, but it
        reads as deletions spread over one recording and the corpus average softens it.
        The finish reason is unambiguous, so assert on that directly.
        """
        truncated = [
            f"{talk.sample_id} ({talk.duration:.0f}s, {out.num_output_tokens} tokens)"
            for talk, out in zip(longform_transcripts["talks"], longform_transcripts["outputs"])
            if out.finish_reason == "length"
        ]
        assert not truncated, "hit max_new_tokens: " + ", ".join(truncated)

    def test_length_does_not_degrade_accuracy(self, longform_transcripts):
        """The longest recordings must not be systematically worse than the shortest.

        Accuracy decaying with length is the failure mode a corpus average is worst at
        showing, because the long recordings that carry the most words would also be
        setting the average they are compared against.
        """
        from bench.metrics import score_words

        normalizer = longform_transcripts["normalizer"]
        talks = longform_transcripts["talks"]
        outputs = longform_transcripts["outputs"]
        half = len(talks) // 2
        short = score_words(
            [t.text for t in talks[:half]], [o.text for o in outputs[:half]], normalizer
        )
        long = score_words(
            [t.text for t in talks[half:]], [o.text for o in outputs[half:]], normalizer
        )
        # Measured 0.0308 short (187-381s) against 0.0259 long (908-1299s): if anything
        # the long half is better, so a 2x ratio means something genuinely broke.
        assert long.wer <= max(2 * short.wer, MAX_CORPUS_WER), (
            f"long half {long.wer:.4f} vs short half {short.wer:.4f}"
        )
