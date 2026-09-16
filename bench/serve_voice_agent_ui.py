"""Serve the realtime voice-agent latency UI.

The browser keeps the microphone and speaker. This process is the resident
cloud coordinator: it can load ASR/TTS locally, or forward them to already
running edge servers through ``--asr-remote`` / ``--tts-remote``.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.voice_agent_timing import (
    create_real_asr_backend,
    create_real_llm_agent,
    create_real_tts_agent,
    resolve_tts_process_batching,
)
from qwen_asr_vllm.agent.async_coordinator import (
    BARGE_IN_POLICIES,
    AsyncVoiceAgentCoordinator,
)
from qwen_asr_vllm.agent.service_runners import (
    ProcessAsrEngine,
    ProcessConcurrentTtsBackend,
    ProcessNanoLlmBackend,
    ProcessTtsBackend,
)
from qwen_asr_vllm.agent.runtime_profiles import (
    RUNTIME_PROFILES,
    activate_runtime_profile_environment,
    default_runtime_profile,
    parse_profiled_args,
)
from qwen_asr_vllm.server import create_app


class AgentUiEngine:
    def __init__(self, topology: dict | None = None) -> None:
        self.config = SimpleNamespace(model="voice-agent-real")
        self.topology = dict(topology or {})

    def close(self) -> None:
        pass

    def health(self) -> dict:
        return {
            "status": "ok",
            "mode": "voice-agent-real",
            "topology": self.topology,
        }


def voice_topology_from_args(args) -> dict:
    return {
        "microphone": "browser-local",
        "speaker": "browser-local",
        "asr": args.asr_remote or "cloud-process",
        "llm": "cloud-process",
        "tts": args.tts_remote or "cloud-process",
        "resident": True,
    }


def build_real_voice_factory(args):
    if args.asr_remote:
        from qwen_asr_vllm.agent.remote_asr import RemoteAsrEngine

        host, port_text = args.asr_remote.rsplit(":", 1)
        asr = RemoteAsrEngine(host, int(port_text), timeout=args.timeout)
    else:
        asr = ProcessAsrEngine(create_real_asr_backend, timeout=args.timeout)
    llm = ProcessNanoLlmBackend(
        partial(create_real_llm_agent, max_num_seqs=args.llm_max_num_seqs),
        timeout=args.timeout,
    )
    if args.tts_remote:
        from qwen_asr_vllm.agent.remote_tts import RemoteTtsBackend

        host, port_text = args.tts_remote.rsplit(":", 1)
        tts = RemoteTtsBackend(host, int(port_text), timeout=args.timeout)
    else:
        tts_workers, tts_batch_window_ms, tts_max_batch_size = (
            resolve_tts_process_batching(
                stream_workers=args.tts_process_stream_workers,
                batch_window_ms=args.tts_process_batch_window_ms,
                max_batch_size=args.tts_process_max_batch_size,
                cuda_graph_code_predictor=args.tts_cuda_graph_code_predictor,
                cuda_graph_fixed_slots=args.tts_cuda_graph_fixed_slots,
                exact_parity=args.tts_stream_batch_exact_parity,
            )
        )
        tts_cls = (
            ProcessConcurrentTtsBackend
            if tts_workers > 1 or tts_batch_window_ms > 0.0
            else ProcessTtsBackend
        )
        tts_kwargs = {
            "shared_memory_threshold": args.tts_shared_memory_threshold_bytes,
        }
        if tts_cls is ProcessConcurrentTtsBackend:
            tts_kwargs.update(
                max_workers=tts_workers,
                batch_window_ms=tts_batch_window_ms,
                max_batch_size=tts_max_batch_size,
            )
        tts = tts_cls(
            partial(
                create_real_tts_agent,
                streaming=args.tts_streaming_engine == "codec-step",
                stream_chunk_size=args.tts_stream_chunk_size,
                stream_first_chunk_size=args.tts_stream_first_chunk_size,
                stream_left_context_size=args.tts_stream_left_context_size,
                stream_batch_exact_parity=args.tts_stream_batch_exact_parity,
                fast_code_predictor=args.tts_fast_code_predictor,
                static_code_predictor=args.tts_static_code_predictor,
                cuda_graph_code_predictor=args.tts_cuda_graph_code_predictor,
                cuda_graph_fixed_slots=args.tts_cuda_graph_fixed_slots,
                cuda_graph_batch_window_ms=args.tts_cuda_graph_batch_window_ms,
                fast_code_predictor_batch_window_ms=args.tts_fast_code_predictor_batch_window_ms,
                fast_code_predictor_max_batch_size=args.tts_fast_code_predictor_max_batch_size,
                explicit_talker_step_engine=args.tts_explicit_talker_step_engine,
                outer_active_prefix_talker_engine=args.tts_outer_active_prefix_talker_engine,
                compile_step_engine=args.tts_compile_step_engine,
                compile_step_engine_mode=args.tts_compile_step_engine_mode,
            ),
            timeout=args.timeout,
            **tts_kwargs,
        )

    def voice_factory(_engine):
        return AsyncVoiceAgentCoordinator(
            asr,
            llm,
            tts,
            llm_trigger=args.llm_trigger,
            asr_kwargs={
                "language": args.language,
                "chunk_policy": args.asr_policy,
                "commit_lag_words": args.commit_lag_words,
            },
            tts_concurrency=args.tts_concurrency,
            tts_flush_chars=args.tts_flush_chars,
            tts_flush_after_ms=args.tts_flush_after_ms,
            tts_flush_min_chars=args.tts_flush_min_chars,
            tts_coalesce_chars=args.tts_coalesce_chars,
            tts_coalesce_wait_ms=args.tts_coalesce_wait_ms,
            tts_first_sentence_immediate=args.tts_first_sentence_immediate,
            tts_defer_short_segments_chars=args.tts_defer_short_segments_chars,
            tts_defer_short_segments_ms=args.tts_defer_short_segments_ms,
            tts_stream_first_segment_only=args.tts_stream_first_segment_only,
            min_committed_words=args.min_committed_words,
            min_committed_audio_seconds=args.min_committed_audio_seconds,
            max_committed_audio_seconds=args.max_committed_audio_seconds,
            defer_tts_audio_until_asr_final=args.defer_tts_audio_until_asr_final,
            tts_playback_preroll_ms=args.tts_playback_preroll_ms,
            barge_in_policy=args.barge_in_policy,
            barge_in_rms_threshold=args.barge_in_rms_threshold,
        )

    def close_all() -> None:
        asr.close()
        llm.close()
        tts.close()

    voice_factory.close = close_all  # type: ignore[attr-defined]
    return voice_factory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-profile",
        choices=RUNTIME_PROFILES,
        default=default_runtime_profile(),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--llm-trigger",
        choices=["final", "committed"],
        default="final",
        help=(
            "final answers the whole utterance. committed triggers on an ASR "
            "prefix and is a research flag only: with the default gate it "
            "prompted the LLM with one word of a 22-word utterance"
        ),
    )
    parser.add_argument("--asr-policy", default="speculate")
    parser.add_argument("--commit-lag-words", type=int, default=1)
    parser.add_argument("--language", default=None)
    parser.add_argument("--llm-max-num-seqs", type=int, default=4)
    parser.add_argument("--tts-concurrency", type=int, default=2)
    parser.add_argument("--tts-flush-chars", type=int, default=12)
    parser.add_argument("--tts-flush-after-ms", type=int, default=250)
    parser.add_argument("--tts-flush-min-chars", type=int, default=5)
    parser.add_argument("--tts-coalesce-chars", type=int, default=80)
    parser.add_argument("--tts-coalesce-wait-ms", type=int, default=None)
    parser.add_argument(
        "--tts-first-sentence-immediate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--tts-defer-short-segments-chars", type=int, default=None)
    parser.add_argument("--tts-defer-short-segments-ms", type=int, default=None)
    parser.add_argument("--tts-stream-first-segment-only", action="store_true")
    parser.add_argument("--tts-process-stream-workers", type=int, default=1)
    parser.add_argument("--tts-process-batch-window-ms", type=float, default=0.0)
    parser.add_argument("--tts-process-max-batch-size", type=int, default=8)
    parser.add_argument(
        "--tts-shared-memory-threshold-bytes", type=int, default=64 * 1024
    )
    parser.add_argument("--tts-streaming-engine", choices=["off", "codec-step"], default="off")
    parser.add_argument(
        "--tts-stream-batch-exact-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--tts-stream-chunk-size", type=int, default=8)
    parser.add_argument("--tts-stream-first-chunk-size", type=int, default=None)
    parser.add_argument("--tts-stream-left-context-size", type=int, default=4)
    parser.add_argument("--tts-fast-code-predictor", action="store_true")
    parser.add_argument("--tts-static-code-predictor", action="store_true")
    parser.add_argument(
        "--tts-cuda-graph-code-predictor",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--tts-cuda-graph-fixed-slots", type=int, default=1)
    parser.add_argument("--tts-cuda-graph-batch-window-ms", type=float, default=0.0)
    parser.add_argument("--tts-fast-code-predictor-batch-window-ms", type=float, default=0.0)
    parser.add_argument("--tts-fast-code-predictor-max-batch-size", type=int, default=8)
    parser.add_argument("--tts-explicit-talker-step-engine", action="store_true")
    parser.add_argument(
        "--tts-outer-active-prefix-talker-engine",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--tts-compile-step-engine", action="store_true")
    parser.add_argument("--tts-compile-step-engine-mode", default="reduce-overhead")
    parser.add_argument("--min-committed-words", type=int, default=3)
    parser.add_argument("--min-committed-audio-seconds", type=float, default=0.8)
    parser.add_argument("--max-committed-audio-seconds", type=float, default=2.0)
    parser.add_argument(
        "--defer-tts-audio-until-asr-final",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--tts-playback-preroll-ms",
        type=float,
        default=0.0,
        help=(
            "hold the opening TTS chunks until this much audio is buffered, so "
            "playback does not starve when TTS runs slower than real time"
        ),
    )
    parser.add_argument(
        "--barge-in-policy",
        choices=sorted(BARGE_IN_POLICIES),
        default="auto",
        help=(
            "when incoming user audio may cancel a reply already being spoken; "
            "use 'after-asr-final' together with "
            "--no-defer-tts-audio-until-asr-final to stream audio without the "
            "tail of the triggering utterance self-interrupting the turn"
        ),
    )
    parser.add_argument("--barge-in-rms-threshold", type=float, default=1e-4)
    parser.add_argument(
        "--asr-remote",
        default=None,
        help="host:port of a resident edge ASR server; the browser mic still "
        "enters this process, which forwards PCM there",
    )
    parser.add_argument(
        "--tts-remote",
        default=None,
        help="host:port of a resident edge TTS server; WAV chunks come back "
        "here so the same page can play them",
    )
    parser.add_argument("--log-level", default="info")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parse_profiled_args(build_parser(), argv)
    activate_runtime_profile_environment(args)
    topology = voice_topology_from_args(args)
    print(
        "voice UI topology: "
        f"mic={topology['microphone']} asr={topology['asr']} "
        f"llm={topology['llm']} tts={topology['tts']} "
        f"speaker={topology['speaker']}",
        flush=True,
    )
    voice_factory = build_real_voice_factory(args)
    app = create_app(AgentUiEngine(topology), voice_factory=voice_factory)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
