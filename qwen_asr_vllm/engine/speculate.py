"""Greedy speculative decoding helpers for ASR.

The interesting draft for streaming is the previous chunk's transcript: word-level
revision data says about 99% of emitted words stay put, so the open question is how
much of that survives as a *token* prefix the model will actually accept when the
audio has grown. That number is acceptance rate, and it is what decides whether
speculative decoding is worth wiring into the engine loop.

``accepted_draft_length`` is the pure rule. ``ModelRunner.verify`` is the GPU half:
one forward over the prompt plus the draft yields a prediction per draft position
(plus the free next token), and the accepted prefix is the leading run where the two
agree. At temperature 0 that length equals the longest common prefix of the draft and
an independent greedy transcription of the same audio — both are checked against the
same argmax — so the streaming measurement can use either path and the GPU path is
what a serving integration would call.
"""

from __future__ import annotations


def accepted_draft_length(draft: list[int], predictions: list[int]) -> int:
    """How many leading draft tokens the model would keep.

    ``predictions`` comes from :meth:`ModelRunner.verify` and has length
    ``len(draft) + 1``: one entry per draft token, then the token that follows the
    whole draft. Only the first ``len(draft)`` entries are compared; the bonus token
    is what decoding would emit next after a full accept.
    """
    limit = min(len(draft), len(predictions))
    for index in range(limit):
        if draft[index] != predictions[index]:
            return index
    return limit


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    """Token-level LCP. Equal to :func:`accepted_draft_length` under greedy decode."""
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit
