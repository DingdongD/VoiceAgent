"""Extract the trailing JSON report from `voice_agent_timing.py` logs and compare arms.

The timing harness prints one JSON document after its progress output. These logs
are the only artifact for arms driven from a shell ladder, so the parser has to
recover the report rather than depend on a `--out` flag.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Reports now carry `input_audio_ms`. This is the duration of
# `results/cuda0_librispeech_sample.wav`, the fixed input for every arm in the
# 2026-09 study, and is used only to keep those older logs readable.
LEGACY_INPUT_AUDIO_MS = 8250.0

SEGMENTS = (
    ("asr_to_committed", "asr_committed", None),
    ("llm_ttft", "llm_first_chunk", "llm_start"),
    ("llm_to_first_sentence", "llm_sentence_ready", "llm_first_chunk"),
    ("tts_first_audio", "tts_audio_ready", "llm_sentence_ready"),
)


def extract_report(text: str) -> dict:
    start = text.index("{")
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("no balanced JSON object found")


def load_arm(path: Path) -> dict:
    report = extract_report(path.read_text())
    target = "async" if "async" in report else next(iter(report))
    payload = report[target]
    offsets = payload.get("event_offsets_ms", {})
    details = payload.get("details", {})

    segments = {}
    for name, end_event, start_event in SEGMENTS:
        end = offsets.get(end_event)
        if end is None:
            segments[name] = None
            continue
        start = 0.0 if start_event is None else offsets.get(start_event)
        segments[name] = None if start is None else round(end - start, 1)

    arm = {
        "arm": path.stem.replace("arm_", ""),
        "first_audio_ms": payload.get("first_audio_ms"),
        "total_ms": payload.get("total_ms"),
        "input_wall_ms": details.get("input_wall_ms"),
        "input_audio_ms": details.get("input_audio_ms"),
        "asr_partials": payload.get("events", []).count("asr_partial"),
        "llm_chunks": payload.get("events", []).count("llm_chunk"),
        "tts_chunks": details.get("tts_chunk_count"),
        "tts_audio_bytes": details.get("tts_audio_bytes"),
        "tts_audio_duration_ms": details.get("tts_audio_duration_ms"),
        "tts_chunk_timeline": details.get("tts_chunk_timeline"),
        "last_tts_chunk_ms": (details.get("event_last_offsets_ms") or {}).get("tts_chunk"),
        "tts_streaming": details.get("tts_streaming"),
        "asr_hypothesis": details.get("asr_hypothesis"),
        "llm_prompt": details.get("llm_prompt"),
        "asr_final_text": details.get("asr_final_text"),
        "llm_output": details.get("llm_output"),
        "errors": payload.get("errors", []),
        "segments_ms": segments,
        "offsets_ms": offsets,
    }
    arm["cost"] = derive_cost(arm, offsets)
    return arm


def derive_cost(arm: dict, offsets: dict) -> dict:
    """Per-turn stage occupancy and real-time factors.

    A voice agent is only sustainable if each stage keeps up with speech. ASR has
    to consume audio faster than it arrives, and TTS has to emit audio faster
    than it is played, so both are reported as real-time factors where a value
    above 1.0 means the stage cannot keep up.
    """
    cost: dict[str, object] = {}

    input_wall = arm.get("input_wall_ms")
    input_audio_ms = arm.get("input_audio_ms") or LEGACY_INPUT_AUDIO_MS
    if input_wall:
        cost["asr_ingest_wall_ms"] = round(input_wall, 1)
        cost["asr_rtf"] = round(input_wall / input_audio_ms, 4)
        cost["asr_audio_sec_per_sec"] = round(input_audio_ms / input_wall, 1)

    llm_start = offsets.get("llm_start")
    llm_done = offsets.get("llm_done")
    if llm_start is not None and llm_done is not None:
        llm_wall = llm_done - llm_start
        cost["llm_wall_ms"] = round(llm_wall, 1)
        chunks = arm.get("llm_chunks") or 0
        if chunks and llm_wall > 0:
            cost["llm_ms_per_chunk"] = round(llm_wall / chunks, 1)
            cost["llm_chunks_per_s"] = round(chunks / (llm_wall / 1000), 1)

    # TTS work runs from the first text it receives to the last audio it emits.
    # `done` is not that boundary: it also covers turn teardown, so it overstates
    # the stage. Prefer the last `tts_chunk` and fall back only when no timeline
    # was recorded.
    tts_start = offsets.get("llm_sentence_ready")
    tts_end = arm.get("last_tts_chunk_ms") or offsets.get("done")
    audio_ms = arm.get("tts_audio_duration_ms")
    if tts_start is not None and tts_end is not None:
        tts_wall = tts_end - tts_start
        cost["tts_wall_ms"] = round(tts_wall, 1)
        if audio_ms:
            cost["tts_audio_ms"] = round(audio_ms, 1)
            cost["tts_rtf"] = round(tts_wall / audio_ms, 3)
            cost["tts_sustainable"] = bool(tts_wall < audio_ms)

    # A stage can average below real time and still starve playback if each
    # chunk costs more than the last. The slope separates a stage that keeps up
    # from one that is falling behind as the utterance grows.
    timeline = arm.get("tts_chunk_timeline") or []
    gaps = [
        round(timeline[i]["at_ms"] - timeline[i - 1]["at_ms"], 1)
        for i in range(1, len(timeline))
    ]
    if len(gaps) >= 4:
        cost["tts_chunk_gaps_ms"] = gaps
        # A playback preroll releases its held chunks back to back, and those
        # near-zero gaps are a delivery artifact rather than generation cost, so
        # they would otherwise swamp the ratio.
        trend = list(gaps)
        while len(trend) >= 4 and trend[0] < 5.0:
            trend.pop(0)
        if len(trend) >= 4:
            head = sum(trend[:2]) / 2
            # The final gap is a short tail chunk, so it is excluded.
            tail = sum(trend[-3:-1]) / 2
            cost["tts_gap_growth"] = round(tail / head, 2) if head else None

    # Whether the listener actually hears a gap. A player that starts on the
    # first chunk and consumes audio in real time holds `received - elapsed`
    # milliseconds; once that goes negative the reply audibly breaks up, which
    # RTF alone does not show.
    # A non-streaming arm delivers the whole reply in one chunk, so it cannot
    # starve by construction. Calling that "gapless" would read as a virtue when
    # it is just the absence of streaming, so the verdict needs two chunks.
    if len(timeline) >= 2:
        origin = timeline[0]["at_ms"]
        received = 0.0
        worst = None
        starves_at = None
        for chunk in timeline:
            received += chunk.get("audio_ms") or 0.0
            buffer_ms = received - (chunk["at_ms"] - origin)
            worst = buffer_ms if worst is None else min(worst, buffer_ms)
            if buffer_ms < 0 and starves_at is None:
                starves_at = round(chunk["at_ms"] - origin, 1)
        cost["playback_worst_buffer_ms"] = round(worst, 1)
        cost["playback_starves_at_ms"] = starves_at
        cost["playback_gapless"] = starves_at is None
    return cost


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    arms = []
    for path in args.logs:
        try:
            arms.append(load_arm(path))
        except Exception as exc:  # noqa: BLE001 - a failed arm must not hide the others
            arms.append({"arm": path.stem.replace("arm_", ""), "parse_error": str(exc)})

    header = (
        f"{'arm':<14}{'first_audio':>12}{'total':>10}{'input_wall':>11}"
        f"{'asr>commit':>11}{'llm_ttft':>9}{'llm>sent':>10}{'tts>audio':>10}"
        f"{'partials':>9}{'chunks':>7}"
    )
    print(header)
    print("-" * len(header))
    for arm in arms:
        if "parse_error" in arm:
            print(f"{arm['arm']:<14}  PARSE ERROR: {arm['parse_error']}")
            continue
        seg = arm["segments_ms"]

        def fmt(value: object, width: int) -> str:
            return f"{value:>{width}}" if value is not None else f"{'-':>{width}}"

        print(
            f"{arm['arm']:<14}{fmt(arm['first_audio_ms'], 12)}{fmt(arm['total_ms'], 10)}"
            f"{fmt(arm['input_wall_ms'], 11)}{fmt(seg['asr_to_committed'], 11)}"
            f"{fmt(seg['llm_ttft'], 9)}{fmt(seg['llm_to_first_sentence'], 10)}"
            f"{fmt(seg['tts_first_audio'], 10)}{fmt(arm['asr_partials'], 9)}"
            f"{fmt(arm['llm_chunks'], 7)}"
        )

    print()
    cost_header = (
        f"{'arm':<14}{'asr_wall':>10}{'asr_rtf':>9}{'llm_wall':>10}"
        f"{'ms/chunk':>10}{'tts_wall':>10}{'tts_audio':>10}{'tts_rtf':>9}"
        f"{'gap_grow':>10}{'buffer':>9}{'gapless':>9}{'ok':>4}"
    )
    print(cost_header)
    print("-" * len(cost_header))
    for arm in arms:
        if "parse_error" in arm:
            continue
        cost = arm.get("cost", {})

        def cfmt(key: str, width: int) -> str:
            value = cost.get(key)
            return f"{value:>{width}}" if value is not None else f"{'-':>{width}}"

        sustainable = cost.get("tts_sustainable")
        flag = "-" if sustainable is None else ("yes" if sustainable else "NO")
        heard = cost.get("playback_gapless")
        gapless = "-" if heard is None else ("yes" if heard else "STARVES")
        print(
            f"{arm['arm']:<14}{cfmt('asr_ingest_wall_ms', 10)}{cfmt('asr_rtf', 9)}"
            f"{cfmt('llm_wall_ms', 10)}{cfmt('llm_ms_per_chunk', 10)}"
            f"{cfmt('tts_wall_ms', 10)}{cfmt('tts_audio_ms', 10)}"
            f"{cfmt('tts_rtf', 9)}{cfmt('tts_gap_growth', 10)}"
            f"{cfmt('playback_worst_buffer_ms', 9)}{gapless:>9}{flag:>4}"
        )

    print()
    for arm in arms:
        if "parse_error" in arm:
            continue
        print(f"[{arm['arm']}] llm_prompt={arm['llm_prompt']!r}")
        print(f"[{arm['arm']}] llm_output={arm['llm_output']!r}")
        print(f"[{arm['arm']}] asr={arm['asr_hypothesis']!r}")
        print(
            f"[{arm['arm']}] tts_chunks={arm['tts_chunks']} "
            f"bytes={arm['tts_audio_bytes']} streaming={arm['tts_streaming']} "
            f"errors={arm['errors']}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(arms, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
