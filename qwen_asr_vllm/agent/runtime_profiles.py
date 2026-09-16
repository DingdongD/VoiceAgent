from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence


RUNTIME_PROFILES = ("compat", "cuda0-throughput", "low-latency")

# Measured configuration for a single interactive session, on real ASR +
# nano-vLLM + Qwen-TTS; see the 2026-09-15 entries in findings.md. Every setting
# here already existed as an opt-in flag.
_LOW_LATENCY_DEFAULTS = {
    "tts_streaming_engine": "codec-step",
    "tts_cuda_graph_code_predictor": True,
    "tts_cuda_graph_fixed_slots": 2,
    # The LLM must answer the whole utterance. `committed` was the largest
    # reported first-audio win (3194.2 -> 994.3 ms) and it was not a speedup: the
    # shipped gate prompted the LLM with the first committed word, `"Have"` out
    # of 22, so the agent answered a different question and never revised.
    "llm_trigger": "final",
    # Neither hand-written outer talker engine belongs in a measured profile.
    # Output-matched isolated profiling (identical codec SHA-256) measured TTS
    # total RTF 1.390 upstream, 3.141 for active-prefix, 8.659 for the CUDA-graph
    # talker; five contention-controlled end-to-end pairs put the active-prefix
    # regression at 2.39x (median RTF 3.137 vs 1.312) with per-chunk cost growing
    # 2.2-2.4x through the reply. It bought no first audio to pay for that.
    "tts_outer_active_prefix_talker_engine": False,
    "tts_outer_cuda_graph_talker_engine": False,
    "tts_stream_batch_exact_parity": False,
    "tts_do_sample": False,
    "tts_subtalker_do_sample": False,
    "tts_temperature": 0.0,
    # First audio cannot exist before this many codec frames are generated, so
    # it is the floor on perceived latency. 8 -> 2 measured 2.64x on the
    # TTS-to-first-audio segment.
    "tts_stream_first_chunk_size": 2,
    # Sentence-boundary segmentation only. Character-level flushing was measured
    # output-matched against it and lost: first audio 830.2 vs 869.4 ms (inside
    # run-to-run noise) while total synthesized audio grew 2080 -> 2480 ms and
    # TTS compute grew 3882 -> 4531 ms. Splitting a reply into sub-sentence
    # fragments makes each fragment carry its own lead-in, so it costs real
    # audio and real GPU time for no measurable latency gain.
    "tts_first_sentence_immediate": True,
    # Once first audio arrives before the user stops speaking, the tail of the
    # triggering utterance would otherwise cancel the turn and restart the LLM.
    "barge_in_policy": "after-asr-final",
    # Deferring audio until ASR final is the workaround this profile replaces.
    "defer_tts_audio_until_asr_final": False,
}

_PROFILE_DEFAULTS = {
    "compat": {},
    "low-latency": _LOW_LATENCY_DEFAULTS,
    "cuda0-throughput": {
        "tts_streaming_engine": "codec-step",
        # PrefixStaticCache keeps fixed storage while exposing the same
        # active-prefix mask and KV shape as HF DynamicCache.
        "tts_cuda_graph_code_predictor": True,
        "tts_cuda_graph_fixed_slots": 2,
        "tts_process_stream_workers": 2,
        "tts_process_batch_window_ms": 10.0,
        "tts_shared_memory_threshold_bytes": 16 * 1024,
        # This legacy flag stays false so the process wrapper allocates request
        # slots. Scheduler selection itself is controlled by the strict
        # profile environment below, not by this historical parity switch.
        "tts_stream_batch_exact_parity": False,
        # Rejected here for the same reason as in `low-latency`: output-matched
        # profiling put it at 2.26x upstream, and a tight-static-cache attempt to
        # rescue it measured 0.763 once and then 2.10x slower under identical
        # parameters, so it is bimodal on top of being slow.
        "tts_outer_active_prefix_talker_engine": False,
        "tts_do_sample": False,
        "tts_subtalker_do_sample": False,
        "tts_temperature": 0.0,
    },
}


def default_runtime_profile() -> str:
    profile = os.getenv("VOICE_RUNTIME_PROFILE", "compat")
    if profile not in RUNTIME_PROFILES:
        raise ValueError(f"unknown VOICE_RUNTIME_PROFILE: {profile}")
    return profile


def _explicit_option_dests(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
) -> set[str]:
    destinations = set()
    actions = parser._option_string_actions
    for token in argv:
        option = token.split("=", 1)[0]
        action = actions.get(option)
        if action is not None:
            destinations.add(action.dest)
    return destinations


def parse_profiled_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
):
    raw = list(sys.argv[1:] if argv is None else argv)
    explicit = _explicit_option_dests(parser, raw)
    args = parser.parse_args(raw)
    profile = getattr(args, "runtime_profile", "compat")
    for destination, value in _PROFILE_DEFAULTS[profile].items():
        if destination not in explicit and hasattr(args, destination):
            setattr(args, destination, value)
    return args


def activate_runtime_profile_environment(args) -> None:
    profile = getattr(args, "runtime_profile", "compat")
    os.environ["VOICE_RUNTIME_PROFILE"] = profile
    if profile == "cuda0-throughput":
        os.environ["VOICE_TTS_REQUEST_STEP_SCHEDULER"] = "1"
        os.environ["VOICE_TTS_LEGACY_STREAM_BATCH"] = "0"
        os.environ.setdefault("VOICE_TTS_STRICT_INNER_PARITY", "1")
    else:
        # Compatibility is an explicit legacy profile, never an implicit
        # runtime fallback from the production scheduler. `low-latency` shares
        # this scheduler on purpose: the strict request-id scheduler measured
        # 0.805x first audio for a single session, so it belongs to the
        # throughput profile only.
        os.environ["VOICE_TTS_REQUEST_STEP_SCHEDULER"] = "0"
        os.environ["VOICE_TTS_LEGACY_STREAM_BATCH"] = "1"
