import os
import sys
import asyncio
import inspect
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from bench import serve_voice_agent_ui, voice_agent_timing
from qwen_asr_vllm.agent.events import agent_event
from qwen_asr_vllm.agent.local_tts import QwenTtsBackend


def test_nonexecuting_varlen_adapter_is_not_exposed_as_production_option():
    assert "outer_varlen_paged_attention" not in inspect.signature(
        QwenTtsBackend
    ).parameters
    for parser in (
        voice_agent_timing.build_parser(),
        serve_voice_agent_ui.build_parser(),
    ):
        assert (
            "--tts-outer-varlen-paged-attention"
            not in parser._option_string_actions
        )


def test_create_real_llm_agent_passes_fixed_kv_blocks(monkeypatch):
    captured = {}

    class RecordingBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    config = SimpleNamespace(
        LLM_MODEL_PATH="/ckpt/llm",
        NANOVLLM_ROOT="/home/nano-vllm",
        LLM_DEVICE="cuda:0",
        LLM_MAX_MODEL_LEN=4096,
        LLM_MAX_NUM_SEQS=4,
        LLM_GPU_MEMORY_UTILIZATION=0.5,
        LLM_NUM_KVCACHE_BLOCKS=64,
        LLM_WARMUP_BATCH_SIZES=(1, 2, 4),
        LLM_ENFORCE_EAGER=False,
        LLM_SYSTEM_PROMPT="system",
        LLM_TEMPERATURE=0.0,
        LLM_MAX_NEW_TOKENS=64,
    )
    src = ModuleType("src")
    src.config = config
    monkeypatch.setitem(sys.modules, "src", src)
    monkeypatch.setattr(voice_agent_timing, "ensure_voice_app_path", lambda: None)
    monkeypatch.setattr(
        voice_agent_timing,
        "NanoVllmStepBatchingBackend",
        RecordingBackend,
    )

    voice_agent_timing.create_real_llm_agent()

    assert captured["num_kvcache_blocks"] == 64
    assert captured["warmup_batch_sizes"] == (1, 2, 4)


def test_cuda_graph_slots_enable_outer_request_batching_defaults(monkeypatch):
    monkeypatch.setenv("VOICE_TTS_REQUEST_STEP_SCHEDULER", "0")
    assert voice_agent_timing.resolve_tts_process_batching(
        stream_workers=1,
        batch_window_ms=0,
        max_batch_size=8,
        cuda_graph_code_predictor=True,
        cuda_graph_fixed_slots=2,
    ) == (2, 10.0, 2)
    assert voice_agent_timing.resolve_tts_process_batching(
        stream_workers=1,
        batch_window_ms=0,
        max_batch_size=8,
        cuda_graph_code_predictor=False,
        cuda_graph_fixed_slots=2,
    ) == (1, 0.0, 8)
    assert voice_agent_timing.resolve_tts_process_batching(
        stream_workers=1,
        batch_window_ms=0,
        max_batch_size=8,
        cuda_graph_code_predictor=True,
        cuda_graph_fixed_slots=2,
        exact_parity=True,
    ) == (1, 0.0, 1)


def test_cuda0_runtime_profile_enables_validated_throughput_path():
    args = voice_agent_timing.parse_profiled_args(
        voice_agent_timing.build_parser(),
        [
            "--runtime-profile",
            "cuda0-throughput",
            "--mode",
            "real",
            "--audio",
            "sample.wav",
        ],
    )

    assert args.tts_streaming_engine == "codec-step"
    assert args.tts_cuda_graph_code_predictor is True
    assert args.tts_cuda_graph_fixed_slots == 2
    assert args.tts_process_stream_workers == 2
    assert args.tts_process_batch_window_ms == 10.0
    assert args.tts_shared_memory_threshold_bytes == 16384
    assert args.tts_stream_batch_exact_parity is False
    # This profile used to enable the active-prefix outer talker engine. Later
    # output-matched profiling measured it at 2.26x upstream, and a tight-cache
    # rescue attempt measured 0.763 once before failing to replicate at 2.10x
    # slower, so the throughput profile no longer ships it either.
    assert args.tts_outer_active_prefix_talker_engine is False
    assert args.tts_do_sample is False
    assert args.tts_subtalker_do_sample is False
    assert args.tts_temperature == 0.0


def test_runtime_profile_selects_one_tts_scheduler_architecture(monkeypatch):
    monkeypatch.delenv("VOICE_TTS_REQUEST_STEP_SCHEDULER", raising=False)
    monkeypatch.delenv("VOICE_TTS_LEGACY_STREAM_BATCH", raising=False)

    throughput = voice_agent_timing.parse_profiled_args(
        voice_agent_timing.build_parser(),
        ["--runtime-profile", "cuda0-throughput"],
    )
    voice_agent_timing.activate_runtime_profile_environment(throughput)

    assert os.environ["VOICE_TTS_REQUEST_STEP_SCHEDULER"] == "1"
    assert os.environ["VOICE_TTS_LEGACY_STREAM_BATCH"] == "0"

    compat = voice_agent_timing.parse_profiled_args(
        voice_agent_timing.build_parser(),
        ["--runtime-profile", "compat"],
    )
    voice_agent_timing.activate_runtime_profile_environment(compat)

    assert os.environ["VOICE_TTS_REQUEST_STEP_SCHEDULER"] == "0"
    assert os.environ["VOICE_TTS_LEGACY_STREAM_BATCH"] == "1"


def test_strict_tts_scheduler_rejects_non_service_backend(monkeypatch):
    monkeypatch.setenv("VOICE_TTS_REQUEST_STEP_SCHEDULER", "1")

    with pytest.raises(ValueError, match="process-isolated"):
        voice_agent_timing.load_real_backends(process_isolated=False)


def test_strict_tts_scheduler_rejects_legacy_parity_downgrade(monkeypatch):
    monkeypatch.setenv("VOICE_TTS_REQUEST_STEP_SCHEDULER", "1")

    with pytest.raises(ValueError, match="exact parity"):
        voice_agent_timing.resolve_tts_process_batching(
            stream_workers=2,
            batch_window_ms=10,
            max_batch_size=2,
            cuda_graph_code_predictor=True,
            cuda_graph_fixed_slots=2,
            exact_parity=True,
        )


def test_runtime_profile_environment_selects_matching_cli_defaults(monkeypatch):
    monkeypatch.setenv("VOICE_RUNTIME_PROFILE", "cuda0-throughput")

    args = voice_agent_timing.parse_profiled_args(
        voice_agent_timing.build_parser(),
        [],
    )

    assert args.runtime_profile == "cuda0-throughput"
    assert args.tts_cuda_graph_code_predictor is True
    assert args.tts_streaming_engine == "codec-step"


def test_explicit_compat_profile_overrides_cuda0_environment(monkeypatch):
    monkeypatch.setenv("VOICE_RUNTIME_PROFILE", "cuda0-throughput")
    args = voice_agent_timing.parse_profiled_args(
        voice_agent_timing.build_parser(),
        ["--runtime-profile", "compat"],
    )

    voice_agent_timing.activate_runtime_profile_environment(args)

    assert os.environ["VOICE_RUNTIME_PROFILE"] == "compat"


@pytest.mark.parametrize(
    "argv",
    [
        [
            "--runtime-profile",
            "cuda0-throughput",
            "--tts-streaming-engine",
            "off",
            "--no-tts-cuda-graph-code-predictor",
            "--tts-stream-batch-exact-parity",
            "--no-tts-outer-active-prefix-talker-engine",
        ],
        [
            "--tts-streaming-engine",
            "off",
            "--no-tts-cuda-graph-code-predictor",
            "--tts-stream-batch-exact-parity",
            "--no-tts-outer-active-prefix-talker-engine",
            "--runtime-profile",
            "cuda0-throughput",
        ],
    ],
)
def test_explicit_tts_options_override_runtime_profile_independent_of_order(argv):
    args = voice_agent_timing.parse_profiled_args(
        voice_agent_timing.build_parser(),
        argv,
    )

    assert args.tts_streaming_engine == "off"
    assert args.tts_cuda_graph_code_predictor is False
    assert args.tts_stream_batch_exact_parity is True
    assert args.tts_outer_active_prefix_talker_engine is False


def test_voice_agent_timing_parser_accepts_tts_streaming_flags():
    args = voice_agent_timing.build_parser().parse_args(
        [
            "--mode",
            "real",
            "--audio",
            "sample.wav",
            "--tts-coalesce-wait-ms",
            "120",
            "--tts-defer-short-segments-chars",
            "48",
            "--tts-defer-short-segments-ms",
            "80",
            "--tts-stream-first-segment-only",
            "--tts-process-stream-workers",
            "2",
            "--tts-process-batch-window-ms",
            "60",
            "--tts-process-max-batch-size",
            "4",
            "--tts-shared-memory-threshold-bytes",
            "8192",
            "--tts-streaming-engine",
            "codec-step",
            "--tts-fast-code-predictor",
            "--tts-static-code-predictor",
            "--tts-cuda-graph-code-predictor",
            "--tts-cuda-graph-fixed-slots",
            "2",
            "--tts-cuda-graph-batch-window-ms",
            "3",
            "--tts-fast-code-predictor-batch-window-ms",
            "6",
            "--tts-fast-code-predictor-max-batch-size",
            "3",
            "--tts-explicit-talker-step-engine",
            "--tts-outer-active-prefix-talker-engine",
            "--tts-outer-cuda-graph-talker-engine",
            "--tts-outer-graph-fixed-slots",
            "3",
            "--tts-outer-graph-max-cache-len",
            "512",
            "--no-tts-do-sample",
            "--no-tts-subtalker-do-sample",
            "--tts-temperature",
            "0",
            "--tts-max-new-tokens",
            "64",
            "--tts-eos-token-id",
            "2150",
            "--tts-compile-step-engine",
            "--tts-compile-step-engine-mode",
            "max-autotune",
            "--tts-stream-chunk-size",
            "8",
            "--tts-stream-first-chunk-size",
            "4",
            "--tts-stream-left-context-size",
            "2",
            "--no-tts-stream-batch-exact-parity",
        ]
    )

    assert args.tts_streaming_engine == "codec-step"
    assert args.tts_coalesce_wait_ms == 120
    assert args.tts_defer_short_segments_chars == 48
    assert args.tts_defer_short_segments_ms == 80
    assert args.tts_stream_first_segment_only is True
    assert args.tts_process_stream_workers == 2
    assert args.tts_process_batch_window_ms == 60
    assert args.tts_process_max_batch_size == 4
    assert args.tts_shared_memory_threshold_bytes == 8192
    assert args.tts_stream_chunk_size == 8
    assert args.tts_fast_code_predictor is True
    assert args.tts_static_code_predictor is True
    assert args.tts_cuda_graph_code_predictor is True
    assert args.tts_cuda_graph_fixed_slots == 2
    assert args.tts_cuda_graph_batch_window_ms == 3
    assert args.tts_fast_code_predictor_batch_window_ms == 6
    assert args.tts_fast_code_predictor_max_batch_size == 3
    assert args.tts_explicit_talker_step_engine is True
    assert args.tts_outer_active_prefix_talker_engine is True
    assert args.tts_outer_cuda_graph_talker_engine is True
    assert args.tts_outer_graph_fixed_slots == 3
    assert args.tts_outer_graph_max_cache_len == 512
    assert args.tts_do_sample is False
    assert args.tts_subtalker_do_sample is False
    assert args.tts_temperature == 0.0
    assert args.tts_max_new_tokens == 64
    assert args.tts_eos_token_id == 2150
    assert args.tts_compile_step_engine is True
    assert args.tts_compile_step_engine_mode == "max-autotune"
    assert args.tts_stream_first_chunk_size == 4
    assert args.tts_stream_left_context_size == 2
    assert args.tts_stream_batch_exact_parity is False
    assert args.tts_first_sentence_immediate is False


def test_serve_voice_agent_ui_parser_accepts_tts_streaming_flags():
    args = serve_voice_agent_ui.build_parser().parse_args(
        [
            "--tts-coalesce-wait-ms",
            "150",
            "--tts-defer-short-segments-chars",
            "52",
            "--tts-defer-short-segments-ms",
            "90",
            "--tts-stream-first-segment-only",
            "--tts-process-stream-workers",
            "3",
            "--tts-process-batch-window-ms",
            "70",
            "--tts-process-max-batch-size",
            "5",
            "--tts-shared-memory-threshold-bytes",
            "4096",
            "--tts-streaming-engine",
            "codec-step",
            "--tts-fast-code-predictor",
            "--tts-static-code-predictor",
            "--tts-cuda-graph-code-predictor",
            "--tts-cuda-graph-fixed-slots",
            "2",
            "--tts-cuda-graph-batch-window-ms",
            "4",
            "--tts-fast-code-predictor-batch-window-ms",
            "7",
            "--tts-fast-code-predictor-max-batch-size",
            "4",
            "--tts-explicit-talker-step-engine",
            "--tts-outer-active-prefix-talker-engine",
            "--tts-compile-step-engine",
            "--tts-compile-step-engine-mode",
            "max-autotune",
            "--tts-stream-chunk-size",
            "12",
            "--tts-stream-first-chunk-size",
            "4",
            "--tts-stream-left-context-size",
            "3",
        ]
    )

    assert args.tts_streaming_engine == "codec-step"
    assert args.tts_coalesce_wait_ms == 150
    assert args.tts_defer_short_segments_chars == 52
    assert args.tts_defer_short_segments_ms == 90
    assert args.tts_stream_first_segment_only is True
    assert args.tts_first_sentence_immediate is False
    assert args.tts_process_stream_workers == 3
    assert args.tts_process_batch_window_ms == 70
    assert args.tts_process_max_batch_size == 5
    assert args.tts_shared_memory_threshold_bytes == 4096
    assert args.tts_stream_chunk_size == 12
    assert args.tts_fast_code_predictor is True
    assert args.tts_static_code_predictor is True
    assert args.tts_cuda_graph_code_predictor is True
    assert args.tts_cuda_graph_fixed_slots == 2
    assert args.tts_cuda_graph_batch_window_ms == 4
    assert args.tts_fast_code_predictor_batch_window_ms == 7
    assert args.tts_fast_code_predictor_max_batch_size == 4
    assert args.tts_explicit_talker_step_engine is True
    assert args.tts_outer_active_prefix_talker_engine is True
    assert args.tts_compile_step_engine is True
    assert args.tts_compile_step_engine_mode == "max-autotune"
    assert args.tts_stream_first_chunk_size == 4
    assert args.tts_stream_left_context_size == 3


def test_real_async_timing_passes_tts_latency_tuning_options(monkeypatch):
    captured = {}
    loaded = {}

    class Closeable:
        next_id = 0

        def __init__(self):
            self.metric_id = Closeable.next_id
            Closeable.next_id += 1

        def runtime_metrics(self):
            return {"service_id": self.metric_id}

        def close(self):
            return None

    class FakeCoordinator:
        def __init__(self, asr, llm, tts, **kwargs):
            captured.update(kwargs)
            self.events = asyncio.Queue()

        async def feed(self, pcm):
            return None

        async def close(self):
            await self.events.put(agent_event("done", text=""))

        async def next_event(self, timeout=None):
            return await self.events.get()

    def fake_load_real_backends(*args, **kwargs):
        loaded.update(kwargs)
        return Closeable(), Closeable(), Closeable()

    monkeypatch.setattr(voice_agent_timing, "load_real_backends", fake_load_real_backends)
    monkeypatch.setattr(
        voice_agent_timing,
        "AsyncVoiceAgentCoordinator",
        FakeCoordinator,
    )
    args = voice_agent_timing.build_parser().parse_args(
        [
            "--mode",
            "real",
            "--audio",
            "sample.wav",
            "--tts-coalesce-wait-ms",
            "120",
            "--tts-defer-short-segments-chars",
            "48",
            "--tts-defer-short-segments-ms",
            "80",
            "--tts-stream-first-segment-only",
            "--tts-process-stream-workers",
            "2",
            "--tts-process-batch-window-ms",
            "60",
            "--tts-process-max-batch-size",
            "4",
            "--tts-shared-memory-threshold-bytes",
            "8192",
            "--tts-fast-code-predictor",
            "--tts-static-code-predictor",
            "--tts-cuda-graph-code-predictor",
            "--tts-cuda-graph-fixed-slots",
            "2",
            "--tts-cuda-graph-batch-window-ms",
            "3",
            "--tts-fast-code-predictor-batch-window-ms",
            "6",
            "--tts-fast-code-predictor-max-batch-size",
            "3",
            "--tts-explicit-talker-step-engine",
            "--tts-outer-active-prefix-talker-engine",
            "--tts-outer-cuda-graph-talker-engine",
            "--tts-outer-graph-fixed-slots",
            "2",
            "--tts-outer-graph-max-cache-len",
            "512",
            "--no-tts-do-sample",
            "--tts-temperature",
            "0",
            "--tts-max-new-tokens",
            "64",
            "--tts-eos-token-id",
            "-1",
            "--tts-compile-step-engine",
            "--tts-compile-step-engine-mode",
            "max-autotune",
        ]
    )

    result = asyncio.run(
        voice_agent_timing.run_real_async(np.zeros(1600, dtype=np.float32), args)
    )

    assert captured["tts_coalesce_wait_ms"] == 120
    assert captured["tts_defer_short_segments_chars"] == 48
    assert captured["tts_defer_short_segments_ms"] == 80
    assert captured["tts_stream_first_segment_only"] is True
    assert loaded["tts_process_stream_workers"] == 2
    assert loaded["tts_process_batch_window_ms"] == 60
    assert loaded["tts_process_max_batch_size"] == 4
    assert loaded["tts_shared_memory_threshold_bytes"] == 8192
    assert loaded["tts_fast_code_predictor"] is True
    assert loaded["tts_static_code_predictor"] is True
    assert loaded["tts_cuda_graph_code_predictor"] is True
    assert loaded["tts_cuda_graph_fixed_slots"] == 2
    assert loaded["tts_cuda_graph_batch_window_ms"] == 3
    assert loaded["tts_fast_code_predictor_batch_window_ms"] == 6
    assert loaded["tts_fast_code_predictor_max_batch_size"] == 3
    assert loaded["tts_explicit_talker_step_engine"] is True
    assert loaded["tts_outer_active_prefix_talker_engine"] is True
    assert loaded["tts_outer_cuda_graph_talker_engine"] is True
    assert loaded["tts_outer_graph_fixed_slots"] == 2
    assert loaded["tts_outer_graph_max_cache_len"] == 512
    assert loaded["tts_do_sample"] is False
    assert loaded["tts_temperature"] == 0.0
    assert loaded["tts_max_new_tokens"] == 64
    assert loaded["tts_eos_token_id"] == -1
    assert loaded["tts_compile_step_engine"] is True
    assert loaded["tts_compile_step_engine_mode"] == "max-autotune"
    assert result.details["resident_services"] == {
        "asr": {"service_id": 0},
        "llm": {"service_id": 1},
        "tts": {"service_id": 2},
    }


def test_create_real_tts_agent_passes_streaming_config(monkeypatch):
    captured = {}

    class RecordingBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    config = SimpleNamespace(
        TTS_MODEL_PATH="/ckpt/tts",
        TTS_DEVICE="cuda:1",
        TTS_LANGUAGE="chinese",
        TTS_SPEAKER="speaker",
    )
    src = ModuleType("src")
    src.config = config
    monkeypatch.setitem(sys.modules, "src", src)
    monkeypatch.setattr(voice_agent_timing, "ensure_voice_app_path", lambda: None)
    monkeypatch.setattr(voice_agent_timing, "QwenTtsBackend", RecordingBackend)

    voice_agent_timing.create_real_tts_agent(
        streaming=True,
        stream_chunk_size=8,
        stream_first_chunk_size=4,
        stream_left_context_size=2,
        stream_batch_exact_parity=False,
        fast_code_predictor=True,
        static_code_predictor=True,
        cuda_graph_code_predictor=True,
        cuda_graph_fixed_slots=2,
        cuda_graph_batch_window_ms=3,
        fast_code_predictor_batch_window_ms=6,
        fast_code_predictor_max_batch_size=3,
        explicit_talker_step_engine=True,
        outer_active_prefix_talker_engine=True,
        outer_cuda_graph_talker_engine=True,
        outer_graph_fixed_slots=3,
        outer_graph_max_cache_len=512,
        do_sample=False,
        subtalker_dosample=False,
        temperature=0.0,
        max_new_tokens=64,
        eos_token_id=2150,
        compile_step_engine=True,
        compile_step_engine_mode="max-autotune",
    )

    assert captured == {
        "model_path": "/ckpt/tts",
        "device": "cuda:1",
        "language": "chinese",
        "speaker": "speaker",
        "streaming": True,
        "stream_chunk_size": 8,
        "stream_first_chunk_size": 4,
        "stream_left_context_size": 2,
        "stream_batch_exact_parity": False,
        "fast_code_predictor": True,
        "static_code_predictor": True,
        "cuda_graph_code_predictor": True,
        "cuda_graph_fixed_slots": 2,
        "cuda_graph_batch_window_ms": 3,
        "fast_code_predictor_batch_window_ms": 6,
        "fast_code_predictor_max_batch_size": 3,
        "explicit_talker_step_engine": True,
        "outer_active_prefix_talker_engine": True,
        "outer_cuda_graph_talker_engine": True,
        "outer_graph_fixed_slots": 3,
        "outer_graph_max_cache_len": 512,
        "do_sample": False,
        "subtalker_dosample": False,
        "temperature": 0.0,
        "max_new_tokens": 64,
        "eos_token_id": 2150,
        "compile_step_engine": True,
        "compile_step_engine_mode": "max-autotune",
    }


def test_low_latency_runtime_profile_enables_measured_first_audio_path():
    """The measured-best first-audio configuration must be selectable by name.

    Ablation on real ASR/LLM/TTS measured 7007.6 -> 960.8 ms first audio from
    these settings alone, so leaving them as scattered opt-in flags means the
    gain is unused in practice.
    """
    args = voice_agent_timing.parse_profiled_args(
        voice_agent_timing.build_parser(),
        [
            "--runtime-profile",
            "low-latency",
            "--mode",
            "real",
            "--audio",
            "sample.wav",
        ],
    )

    assert args.tts_streaming_engine == "codec-step"
    assert args.tts_cuda_graph_code_predictor is True
    assert args.tts_stream_first_chunk_size == 2
    assert args.tts_first_sentence_immediate is True
    # Sentence-boundary segmentation only; sub-sentence flushing measured worse
    # on total audio and TTS compute for no first-audio gain.
    assert args.tts_flush_chars is None
    assert args.tts_flush_after_ms is None
    assert args.barge_in_policy == "after-asr-final"
    assert args.defer_tts_audio_until_asr_final is False
    assert args.tts_do_sample is False
    assert args.tts_temperature == 0.0


def test_low_latency_profile_avoids_the_custom_outer_talker_engines():
    """Both hand-written outer talker engines are real-time-factor regressions.

    Output-matched isolated profiling (identical codec SHA-256) measured TTS
    total RTF 1.390 upstream, 3.141 for `active_prefix`, and 8.659 for the
    CUDA-graph talker. Five contention-controlled end-to-end pairs confirmed it:
    median TTS RTF 3.137 with `active_prefix` versus 1.312 without, a 2.39x
    regression, with per-chunk cost growing 2.2-2.4x across the reply instead of
    staying flat. It bought no first audio either (3321-3513 vs 3431-3451 ms).
    """
    args = voice_agent_timing.parse_profiled_args(
        voice_agent_timing.build_parser(),
        ["--runtime-profile", "low-latency"],
    )

    assert args.tts_outer_active_prefix_talker_engine is False
    assert args.tts_outer_cuda_graph_talker_engine is False


def test_low_latency_profile_answers_the_whole_utterance():
    """A latency profile must not buy latency by truncating the question.

    `--llm-trigger committed` with the shipped gate prompted the LLM with the
    single word `"Have"` out of a 22-word utterance and produced an unrelated
    reply, so the profile pins the trigger that answers the full transcript.
    """
    args = voice_agent_timing.parse_profiled_args(
        voice_agent_timing.build_parser(),
        ["--runtime-profile", "low-latency"],
    )

    assert args.llm_trigger == "final"


def test_committed_trigger_is_not_the_default():
    """The unsafe trigger must be opted into, never inherited from the CLI."""
    args = voice_agent_timing.build_parser().parse_args([])

    assert args.llm_trigger == "final"


def test_live_ui_server_answers_the_whole_utterance_by_default():
    """The WebSocket server is the production path and had the same unsafe default."""
    import importlib

    serve_voice_agent_ui = importlib.import_module("bench.serve_voice_agent_ui")
    args = serve_voice_agent_ui.build_parser().parse_args([])

    assert args.llm_trigger == "final"


def test_live_ui_parser_accepts_edge_remote_backends():
    """The latency app is how a microphone and speaker drive the cloud-edge split."""
    args = serve_voice_agent_ui.build_parser().parse_args(
        ["--asr-remote", "127.0.0.1:18765", "--tts-remote", "127.0.0.1:18766"]
    )

    assert args.asr_remote == "127.0.0.1:18765"
    assert args.tts_remote == "127.0.0.1:18766"


def test_live_ui_factory_uses_remote_asr_and_tts_when_configured(monkeypatch):
    created = {}

    class FakeRemote:
        def __init__(self, host, port, *, timeout=300.0):
            created[port] = (host, timeout)

        def close(self):
            pass

    class FakeLlm:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        "qwen_asr_vllm.agent.remote_asr.RemoteAsrEngine", FakeRemote
    )
    monkeypatch.setattr(
        "qwen_asr_vllm.agent.remote_tts.RemoteTtsBackend", FakeRemote
    )
    monkeypatch.setattr(
        "bench.serve_voice_agent_ui.ProcessNanoLlmBackend", FakeLlm
    )
    args = serve_voice_agent_ui.build_parser().parse_args(
        [
            "--runtime-profile",
            "low-latency",
            "--asr-remote",
            "127.0.0.1:18765",
            "--tts-remote",
            "127.0.0.1:18766",
            "--timeout",
            "12",
        ]
    )
    factory = serve_voice_agent_ui.build_real_voice_factory(args)
    try:
        assert created[18765] == ("127.0.0.1", 12.0)
        assert created[18766] == ("127.0.0.1", 12.0)
    finally:
        factory.close()


def test_live_ui_health_places_mic_speaker_on_the_browser():
    args = serve_voice_agent_ui.build_parser().parse_args(
        ["--asr-remote", "127.0.0.1:18765", "--tts-remote", "127.0.0.1:18766"]
    )
    topology = serve_voice_agent_ui.voice_topology_from_args(args)
    health = serve_voice_agent_ui.AgentUiEngine(topology).health()

    assert health["status"] == "ok"
    assert health["topology"]["microphone"] == "browser-local"
    assert health["topology"]["speaker"] == "browser-local"
    assert health["topology"]["asr"] == "127.0.0.1:18765"
    assert health["topology"]["tts"] == "127.0.0.1:18766"
    assert health["topology"]["llm"] == "cloud-process"
    assert health["topology"]["resident"] is True


def test_low_latency_profile_keeps_explicit_cli_overrides():
    args = voice_agent_timing.parse_profiled_args(
        voice_agent_timing.build_parser(),
        [
            "--runtime-profile",
            "low-latency",
            "--tts-stream-first-chunk-size",
            "6",
            "--barge-in-policy",
            "auto",
        ],
    )

    assert args.tts_stream_first_chunk_size == 6
    assert args.barge_in_policy == "auto"


def test_low_latency_profile_uses_the_measured_scheduler(monkeypatch):
    """Single-session latency measured better on the legacy scheduler.

    The strict request-id scheduler measured 0.805x first audio for one session,
    so a latency profile must not silently select it.
    """
    monkeypatch.delenv("VOICE_TTS_REQUEST_STEP_SCHEDULER", raising=False)
    monkeypatch.delenv("VOICE_TTS_LEGACY_STREAM_BATCH", raising=False)

    args = voice_agent_timing.parse_profiled_args(
        voice_agent_timing.build_parser(),
        ["--runtime-profile", "low-latency"],
    )
    voice_agent_timing.activate_runtime_profile_environment(args)

    assert os.environ["VOICE_RUNTIME_PROFILE"] == "low-latency"
    assert os.environ["VOICE_TTS_REQUEST_STEP_SCHEDULER"] == "0"


def test_no_runtime_profile_enables_the_rejected_outer_talker_engines():
    """Both hand-written outer talker engines are rejected, so no profile ships them.

    Output-matched isolated profiling (identical codec SHA-256) measured TTS total
    RTF 1.390 upstream against 3.141 for active-prefix and 8.659 for the CUDA-graph
    talker. A later attempt to rescue active-prefix with a tight static cache
    measured 0.763 and then failed to replicate at 2.10x slower under identical
    parameters, so it is also bimodal and no single measurement of it can justify
    shipping it.
    """
    from qwen_asr_vllm.agent.runtime_profiles import _PROFILE_DEFAULTS

    for name, defaults in _PROFILE_DEFAULTS.items():
        for flag in (
            "tts_outer_active_prefix_talker_engine",
            "tts_outer_cuda_graph_talker_engine",
        ):
            assert defaults.get(flag, False) is False, (
                f"profile {name!r} enables {flag}, a measured regression"
            )
