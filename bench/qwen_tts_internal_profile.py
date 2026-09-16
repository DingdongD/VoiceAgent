"""Internal Qwen-TTS profiler for codec generation and decode stages."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.qwen_tts_streaming_engine_probe import (  # noqa: E402
    _run_custom_voice_generate,
    default_config,
)
from qwen_asr_vllm.agent.local_tts import QwenTtsBackend  # noqa: E402
from qwen_asr_vllm.agent.qwen_tts_outer_static_engine import (  # noqa: E402
    OuterTalkerEngineError,
)
from qwen_asr_vllm.agent.qwen_tts_streaming import (  # noqa: E402
    _extract_codec_frame_batch,
    chunked_decode_codec_codes,
    concatenate_audio_chunks,
)


def _set_seed(seed: int) -> None:
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def profile_qwen_tts_batch(
    wrapper: Any,
    *,
    texts: list[str],
    chunk_size: int,
    left_context_size: int,
    non_streaming_mode: bool = True,
    max_new_tokens: int | None = None,
    sync_each_step: bool = False,
) -> dict[str, Any]:
    texts = [str(text) for text in texts if str(text).strip()]
    if not texts:
        raise ValueError("texts must contain at least one non-empty item")

    speaker = getattr(wrapper, "_speaker", "")
    language = getattr(wrapper, "_language", "auto")
    model = wrapper.model
    talker_events: list[dict[str, Any]] = []
    code_predictor_events: list[dict[str, Any]] = []
    talker_model_events: list[dict[str, Any]] = []

    def run_generate():
        return _run_custom_voice_generate(
            wrapper,
            text=texts,
            speaker=speaker,
            language=language,
            non_streaming_mode=non_streaming_mode,
            max_new_tokens=max_new_tokens,
        )

    generation_started = time.perf_counter()
    _cuda_synchronize()
    generation_result = _run_with_talker_profile(
        model.talker,
        run_generate,
        talker_events,
        code_predictor_events,
        talker_model_events,
        sync_each_step=sync_each_step,
    )
    _cuda_synchronize()
    generation_ms = (time.perf_counter() - generation_started) * 1000

    codes_list = _normalise_codes(generation_result)
    codec_ids = _codec_ids_report(codes_list)
    outer_engine, outer_engine_metrics = _outer_engine_snapshot(model.talker)
    full_decode_started = time.perf_counter()
    _cuda_synchronize()
    audios, sample_rate = model.speech_tokenizer.decode(
        [{"audio_codes": codes} for codes in codes_list]
    )
    _cuda_synchronize()
    full_decode_ms = (time.perf_counter() - full_decode_started) * 1000

    chunked_started = time.perf_counter()
    chunk_counts: list[int] = []
    chunk_decode_ms: list[list[float]] = []
    chunked_duration_s: list[float] = []
    for codes in codes_list:
        chunks = list(
            chunked_decode_codec_codes(
                model.speech_tokenizer,
                codes,
                chunk_size=chunk_size,
                left_context_size=left_context_size,
            )
        )
        chunk_counts.append(len(chunks))
        chunk_decode_ms.append([round(chunk.decode_ms, 3) for chunk in chunks])
        audio = concatenate_audio_chunks(chunks)
        chunk_sample_rate = chunks[0].sample_rate if chunks else int(sample_rate)
        chunked_duration_s.append(
            round(float(audio.size) / float(chunk_sample_rate), 3)
            if chunk_sample_rate
            else 0.0
        )
    chunked_decode_total_ms = (time.perf_counter() - chunked_started) * 1000

    audio_duration_s = [
        round(float(np.asarray(audio).reshape(-1).size) / float(sample_rate), 3)
        if sample_rate
        else 0.0
        for audio in audios
    ]
    audio_arrays = [np.asarray(audio).reshape(-1) for audio in audios]
    return {
        "stage": "qwen_tts_internal",
        "batch_size": len(texts),
        "text_chars": [len(text) for text in texts],
        "codec_frames": [_first_dim(codes) for codes in codes_list],
        "codec_ids_sha256": codec_ids["combined_sha256"],
        "codec_ids_item_sha256": codec_ids["item_sha256"],
        "codec_ids_shapes": codec_ids["shapes"],
        "codec_ids_dtypes": codec_ids["dtypes"],
        "codec_ids_sha256_scheme": (
            "SHA-256 of ordered concatenated CPU-contiguous tensor bytes; "
            "per-item hashes preserve tensor boundaries"
        ),
        "outer_engine": outer_engine,
        "outer_engine_metrics": outer_engine_metrics,
        "generation_ms": round(generation_ms, 1),
        "full_decode_ms": round(full_decode_ms, 1),
        "chunked_decode_total_ms": round(chunked_decode_total_ms, 1),
        "chunked_chunks": chunk_counts,
        "chunk_decode_ms": chunk_decode_ms,
        "audio_duration_s": audio_duration_s,
        "audio_nonempty": [bool(audio.size) for audio in audio_arrays],
        "audio_finite": [bool(np.isfinite(audio).all()) for audio in audio_arrays],
        "chunked_duration_s": chunked_duration_s,
        "decode_rtf": _rtf(full_decode_ms, audio_duration_s),
        "chunked_decode_rtf": _rtf(chunked_decode_total_ms, chunked_duration_s),
        # Codec generation dominates the TTS wall, so it, not decode, sets
        # whether the stage can sustain a conversation. `total_rtf` is the
        # streaming path end to end: generate codes, then decode them in chunks.
        "generation_rtf": _rtf(generation_ms, audio_duration_s),
        "total_rtf": _rtf(generation_ms + chunked_decode_total_ms, chunked_duration_s),
        "codec_frames_per_s": _frames_per_second(codes_list, generation_ms),
        "talker_forward": _summarise_timed_events(talker_events, generation_ms),
        "code_predictor_generate": _summarise_timed_events(
            code_predictor_events,
            generation_ms,
        ),
        "talker_model_forward": _summarise_timed_events(
            talker_model_events,
            generation_ms,
        ),
        "cuda_synchronized_generation": _torch_cuda_available(),
        "sync_each_step": bool(sync_each_step),
    }


def _run_with_talker_profile(
    talker: Any,
    run_generate,
    events: list[dict[str, Any]],
    code_predictor_events: list[dict[str, Any]],
    talker_model_events: list[dict[str, Any]],
    *,
    sync_each_step: bool,
):
    original_forward = talker.forward
    code_predictor = getattr(talker, "code_predictor", None)
    original_code_predictor_generate = (
        getattr(code_predictor, "generate", None) if code_predictor is not None else None
    )
    talker_model = getattr(talker, "model", None)
    original_talker_model_forward = (
        getattr(talker_model, "forward", None) if talker_model is not None else None
    )

    def hooked_forward(*args, **kwargs):
        if sync_each_step:
            _cuda_synchronize()
        started = time.perf_counter()
        outputs = original_forward(*args, **kwargs)
        if sync_each_step:
            _cuda_synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        frames = _extract_codec_frame_batch(outputs)
        batch_size = int(frames.shape[0]) if frames is not None else 0
        events.append(
            {
                "elapsed_ms": elapsed_ms,
                "batch_size": batch_size,
                "frame_shape": list(frames.shape) if frames is not None else None,
            }
        )
        return outputs

    def hooked_code_predictor_generate(*args, **kwargs):
        if sync_each_step:
            _cuda_synchronize()
        started = time.perf_counter()
        outputs = original_code_predictor_generate(*args, **kwargs)
        if sync_each_step:
            _cuda_synchronize()
        code_predictor_events.append(
            {
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "batch_size": _batch_size_from_call(args, kwargs, outputs),
                "frame_shape": _shape_from_output_sequences(outputs),
            }
        )
        return outputs

    def hooked_talker_model_forward(*args, **kwargs):
        if sync_each_step:
            _cuda_synchronize()
        started = time.perf_counter()
        outputs = original_talker_model_forward(*args, **kwargs)
        if sync_each_step:
            _cuda_synchronize()
        talker_model_events.append(
            {
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "batch_size": _batch_size_from_call(args, kwargs, outputs),
                "frame_shape": _shape_from_last_hidden_state(outputs),
            }
        )
        return outputs

    hooked_forward.__signature__ = inspect.signature(original_forward)  # type: ignore[attr-defined]
    if original_code_predictor_generate is not None:
        hooked_code_predictor_generate.__signature__ = inspect.signature(  # type: ignore[attr-defined]
            original_code_predictor_generate
        )
        code_predictor.generate = hooked_code_predictor_generate
    if original_talker_model_forward is not None:
        hooked_talker_model_forward.__signature__ = inspect.signature(  # type: ignore[attr-defined]
            original_talker_model_forward
        )
        talker_model.forward = hooked_talker_model_forward
    talker.forward = hooked_forward
    try:
        return run_generate()
    finally:
        talker.forward = original_forward
        if original_code_predictor_generate is not None:
            code_predictor.generate = original_code_predictor_generate
        if original_talker_model_forward is not None:
            talker_model.forward = original_talker_model_forward


def _normalise_codes(generation_result: Any) -> list[Any]:
    codes = generation_result[0] if isinstance(generation_result, tuple) else generation_result
    if isinstance(codes, np.ndarray):
        return [codes]
    return list(codes)


def _codec_ids_report(codes_list: list[Any]) -> dict[str, Any]:
    import torch

    combined = hashlib.sha256()
    item_sha256 = []
    shapes = []
    dtypes = []
    for codes in codes_list:
        if torch.is_tensor(codes):
            cpu_codes = codes.detach().to(device="cpu").contiguous()
        else:
            cpu_codes = torch.as_tensor(np.ascontiguousarray(codes)).contiguous()
        raw_bytes = cpu_codes.numpy().tobytes(order="C")
        combined.update(raw_bytes)
        item_sha256.append(hashlib.sha256(raw_bytes).hexdigest())
        shapes.append([int(size) for size in cpu_codes.shape])
        dtypes.append(str(cpu_codes.dtype))
    return {
        "combined_sha256": combined.hexdigest(),
        "item_sha256": item_sha256,
        "shapes": shapes,
        "dtypes": dtypes,
    }


def _outer_engine_mode(talker: Any) -> str:
    generate = getattr(talker, "generate", None)
    if getattr(generate, "_qav_active_prefix_outer_talker", False):
        return "active_prefix"
    if getattr(generate, "_qav_cuda_graph_outer_talker", False):
        return "cuda_graph"
    if getattr(generate, "_qav_static_outer_talker", False):
        return "static"
    if getattr(generate, "_qav_explicit_talker_step_engine", False):
        return "explicit"
    return "upstream"


def _outer_engine_snapshot(talker: Any) -> tuple[str, dict[str, Any]]:
    generate = getattr(talker, "generate", None)
    mode = _outer_engine_mode(talker)

    engine = getattr(generate, "_qav_outer_engine", None)
    metrics_snapshot = getattr(engine, "metrics_snapshot", None)
    metrics = metrics_snapshot() if callable(metrics_snapshot) else {}
    return mode, copy.deepcopy(metrics)


def _first_dim(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    return int(np.asarray(value).shape[0])


def _summarise_timed_events(
    events: list[dict[str, Any]],
    generation_ms: float,
) -> dict[str, Any]:
    values = [float(event["elapsed_ms"]) for event in events]
    batch_sizes = [int(event["batch_size"]) for event in events]
    sum_ms = sum(values)
    return {
        "steps": len(values),
        "sum_ms": round(sum_ms, 1),
        "first_ms": round(values[0], 1) if values else None,
        "mean_ms": round(sum(values) / len(values), 1) if values else None,
        "p50_ms": round(_percentile(values, 50), 1) if values else None,
        "p95_ms": round(_percentile(values, 95), 1) if values else None,
        "max_ms": round(max(values), 1) if values else None,
        "generation_pct": round((sum_ms / generation_ms) * 100.0, 1)
        if generation_ms > 0
        else None,
        "max_batch_size": max(batch_sizes) if batch_sizes else 0,
        "batch_sizes": sorted(set(batch_sizes)),
    }


def _batch_size_from_call(args: tuple[Any, ...], kwargs: dict[str, Any], outputs: Any) -> int:
    for name in ("batch_size", "inputs_embeds", "input_ids", "attention_mask"):
        if name in kwargs:
            value = kwargs[name]
            if isinstance(value, int):
                return value
            batch_size = _batch_size_from_shape(value)
            if batch_size is not None:
                return batch_size
    for value in args:
        batch_size = _batch_size_from_shape(value)
        if batch_size is not None:
            return batch_size
    for value in (
        getattr(outputs, "sequences", None),
        getattr(outputs, "last_hidden_state", None),
    ):
        batch_size = _batch_size_from_shape(value)
        if batch_size is not None:
            return batch_size
    return 0


def _batch_size_from_shape(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    return None


def _shape_from_output_sequences(outputs: Any) -> list[int] | None:
    sequences = getattr(outputs, "sequences", None)
    shape = getattr(sequences, "shape", None)
    return [int(item) for item in shape] if shape is not None else None


def _shape_from_last_hidden_state(outputs: Any) -> list[int] | None:
    hidden = getattr(outputs, "last_hidden_state", None)
    shape = getattr(hidden, "shape", None)
    return [int(item) for item in shape] if shape is not None else None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(np.floor(index))
    upper = int(np.ceil(index))
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _frames_per_second(codes_list: list[Any], generation_ms: float) -> float | None:
    if generation_ms <= 0:
        return None
    frames = sum(_first_dim(codes) for codes in codes_list)
    return round(frames / (generation_ms / 1000.0), 2)


def _rtf(elapsed_ms: float, durations_s: list[float]) -> float | None:
    total_duration = sum(durations_s)
    if total_duration <= 0:
        return None
    return round((elapsed_ms / 1000.0) / total_duration, 3)


def _cuda_synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _parse_batch_sizes(value: str) -> list[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--batch-sizes must contain positive integers")
    return sizes


def _texts_for_batch(texts: list[str], batch_size: int) -> list[str]:
    if len(texts) >= batch_size:
        return texts[:batch_size]
    return [texts[index % len(texts)] for index in range(batch_size)]


def build_parser() -> argparse.ArgumentParser:
    defaults = default_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=defaults.get("model_path"))
    parser.add_argument("--device", default=defaults.get("device", "cuda:1"))
    parser.add_argument("--language", default=defaults.get("language", "chinese"))
    parser.add_argument("--speaker", default=defaults.get("speaker", ""))
    parser.add_argument(
        "--texts",
        nargs="+",
        default=[
            "I'm here to help.",
            "What can I do for you?",
            "The system is ready.",
            "Please send the next request.",
        ],
    )
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--left-context-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--non-streaming-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sync-each-step", action="store_true")
    parser.add_argument("--fast-code-predictor", action="store_true")
    parser.add_argument("--static-code-predictor", action="store_true")
    parser.add_argument("--cuda-graph-code-predictor", action="store_true")
    parser.add_argument("--cuda-graph-fixed-slots", type=int, default=1)
    parser.add_argument("--cuda-graph-batch-window-ms", type=float, default=0.0)
    parser.add_argument("--fast-code-predictor-batch-window-ms", type=float, default=0.0)
    parser.add_argument("--fast-code-predictor-max-batch-size", type=int, default=8)
    parser.add_argument("--outer-static-talker-engine", action="store_true")
    parser.add_argument("--outer-active-prefix-talker-engine", action="store_true")
    parser.add_argument("--outer-graph-max-cache-len", type=int, default=1024)
    parser.add_argument("--outer-cuda-graph-talker-engine", action="store_true")
    parser.add_argument("--outer-graph-fixed-slots", type=int, default=2)
    parser.add_argument("--outer-graph-fuse-qkv", action="store_true")
    parser.add_argument("--outer-graph-fuse-attention", action="store_true")
    parser.add_argument("--outer-graph-fused-projection-kernel", action="store_true")
    parser.add_argument(
        "--outer-graph-fusion-policy",
        choices=("allow", "strict"),
        default="allow",
    )
    parser.add_argument("--outer-graph-fusion-max-abs-error", type=float, default=2.0e-3)
    parser.add_argument(
        "--outer-graph-fusion-max-relative-l2", type=float, default=2.0e-4
    )
    parser.add_argument("--explicit-talker-step-engine", action="store_true")
    parser.add_argument("--compile-step-engine", action="store_true")
    parser.add_argument("--compile-step-engine-mode", default="reduce-overhead")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/qwen_tts_internal_profile.json"),
    )
    return parser


def _warning_messages(captured_warnings: list[Any]) -> list[str]:
    messages = []
    seen = set()
    for captured in captured_warnings:
        message = str(captured.message).strip()
        if message and message not in seen:
            seen.add(message)
            messages.append(message)
    return messages


def _is_cache_overflow_error(error: Exception) -> bool:
    message = " ".join(str(error).lower().split())
    if isinstance(error, OuterTalkerEngineError):
        return "cache capacity" in message or "cache overflow" in message
    return "requested prompt and decode tokens exceed static cache capacity" in message


def _intended_outer_engine(args: argparse.Namespace) -> str:
    if args.outer_active_prefix_talker_engine:
        return "active_prefix"
    if args.outer_cuda_graph_talker_engine:
        return "cuda_graph"
    if args.explicit_talker_step_engine:
        return "explicit"
    if args.outer_static_talker_engine:
        return "static"
    return "upstream"


def _validate_outer_engine_flags(args: argparse.Namespace) -> None:
    if args.outer_active_prefix_talker_engine and (
        args.outer_static_talker_engine
        or args.outer_cuda_graph_talker_engine
        or args.explicit_talker_step_engine
    ):
        raise ValueError(
            "active-prefix, fixed-static, and explicit outer engines are "
            "mutually exclusive"
        )


def _backend_runtime_metrics(backend: QwenTtsBackend | None) -> dict[str, Any]:
    snapshot = getattr(backend, "runtime_metrics", None)
    if not callable(snapshot):
        return {}
    return copy.deepcopy(snapshot())


def _format_error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _build_profile_report(
    args: argparse.Namespace,
    batch_sizes: list[int],
    reports: list[dict[str, Any]],
    *,
    outer_engine: str,
    runtime_metrics: dict[str, Any],
    status: str,
    warning_messages: list[str],
    errors: list[str],
    cache_overflow: bool,
) -> dict[str, Any]:
    return {
        "stage": "qwen_tts_internal_profile",
        "model_path": args.model_path,
        "device": args.device,
        "language": args.language,
        "speaker": args.speaker,
        "texts": list(args.texts),
        "batch_sizes": batch_sizes,
        "chunk_size": int(args.chunk_size),
        "left_context_size": int(args.left_context_size),
        "max_new_tokens": args.max_new_tokens,
        "non_streaming_mode": bool(args.non_streaming_mode),
        "sync_each_step": bool(args.sync_each_step),
        "warmup": not args.no_warmup,
        "status": status,
        "warnings": list(warning_messages),
        "errors": list(errors),
        "cache_overflow": bool(cache_overflow),
        "fast_code_predictor": bool(args.fast_code_predictor),
        "static_code_predictor": bool(args.static_code_predictor),
        "cuda_graph_code_predictor": bool(args.cuda_graph_code_predictor),
        "cuda_graph_fixed_slots": int(args.cuda_graph_fixed_slots),
        "cuda_graph_batch_window_ms": float(args.cuda_graph_batch_window_ms),
        "fast_code_predictor_batch_window_ms": float(
            args.fast_code_predictor_batch_window_ms
        ),
        "fast_code_predictor_max_batch_size": int(
            args.fast_code_predictor_max_batch_size
        ),
        "outer_engine": outer_engine,
        "outer_static_talker_engine": bool(args.outer_static_talker_engine),
        "outer_active_prefix_talker_engine": bool(
            args.outer_active_prefix_talker_engine
        ),
        "outer_graph_max_cache_len": int(args.outer_graph_max_cache_len),
        "outer_cuda_graph_talker_engine": bool(args.outer_cuda_graph_talker_engine),
        "outer_graph_fixed_slots": int(args.outer_graph_fixed_slots),
        "outer_graph_fuse_qkv": bool(args.outer_graph_fuse_qkv),
        "outer_graph_fuse_attention": bool(args.outer_graph_fuse_attention),
        "outer_graph_fused_projection_kernel": bool(
            args.outer_graph_fused_projection_kernel
        ),
        "outer_graph_fusion_policy": args.outer_graph_fusion_policy,
        "outer_graph_fusion_max_abs_error": float(
            args.outer_graph_fusion_max_abs_error
        ),
        "outer_graph_fusion_max_relative_l2": float(
            args.outer_graph_fusion_max_relative_l2
        ),
        "explicit_talker_step_engine": bool(args.explicit_talker_step_engine),
        "compile_step_engine": bool(args.compile_step_engine),
        "compile_step_engine_mode": args.compile_step_engine_mode,
        "seed": int(args.seed),
        "greedy": bool(args.greedy),
        "runtime_metrics": copy.deepcopy(runtime_metrics),
        "reports": list(reports),
    }


def _write_profile_report(path: Path, report: dict[str, Any]) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    print(payload, end="")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_outer_engine_flags(args)
    if not args.model_path:
        raise SystemExit("--model-path is required")
    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    backend: QwenTtsBackend | None = None
    reports: list[dict[str, Any]] = []
    primary_error: Exception | None = None
    primary_traceback = None
    secondary_errors: list[Exception] = []
    outer_engine = _intended_outer_engine(args)
    runtime_metrics: dict[str, Any] = {}

    def record_lifecycle_error(error: Exception) -> None:
        nonlocal primary_error, primary_traceback
        if primary_error is None:
            primary_error = error
            primary_traceback = error.__traceback__
        else:
            secondary_errors.append(error)

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        try:
            backend = QwenTtsBackend(
                model_path=args.model_path,
                device=args.device,
                language=args.language,
                speaker=args.speaker,
                warmup=not args.no_warmup,
                fast_code_predictor=args.fast_code_predictor,
                static_code_predictor=args.static_code_predictor,
                cuda_graph_code_predictor=args.cuda_graph_code_predictor,
                cuda_graph_fixed_slots=args.cuda_graph_fixed_slots,
                cuda_graph_batch_window_ms=args.cuda_graph_batch_window_ms,
                fast_code_predictor_batch_window_ms=(
                    args.fast_code_predictor_batch_window_ms
                ),
                fast_code_predictor_max_batch_size=(
                    args.fast_code_predictor_max_batch_size
                ),
                outer_static_talker_engine=args.outer_static_talker_engine,
                outer_active_prefix_talker_engine=(
                    args.outer_active_prefix_talker_engine
                ),
                outer_graph_max_cache_len=args.outer_graph_max_cache_len,
                outer_cuda_graph_talker_engine=args.outer_cuda_graph_talker_engine,
                outer_graph_fixed_slots=args.outer_graph_fixed_slots,
                outer_graph_fuse_qkv=args.outer_graph_fuse_qkv,
                outer_graph_fuse_attention=args.outer_graph_fuse_attention,
                outer_graph_fused_projection_kernel=(
                    args.outer_graph_fused_projection_kernel
                ),
                outer_graph_fusion_policy=args.outer_graph_fusion_policy,
                outer_graph_fusion_max_abs_error=args.outer_graph_fusion_max_abs_error,
                outer_graph_fusion_max_relative_l2=(
                    args.outer_graph_fusion_max_relative_l2
                ),
                explicit_talker_step_engine=args.explicit_talker_step_engine,
                compile_step_engine=args.compile_step_engine,
                compile_step_engine_mode=args.compile_step_engine_mode,
            )
            if args.greedy:
                backend._model.generate_defaults.update(
                    do_sample=False,
                    subtalker_dosample=False,
                )
            _set_seed(args.seed)
            for batch_size in batch_sizes:
                reports.append(
                    profile_qwen_tts_batch(
                        backend._model,
                        texts=_texts_for_batch(list(args.texts), batch_size),
                        chunk_size=args.chunk_size,
                        left_context_size=args.left_context_size,
                        non_streaming_mode=args.non_streaming_mode,
                        max_new_tokens=args.max_new_tokens,
                        sync_each_step=args.sync_each_step,
                    )
                )
        except Exception as error:
            primary_error = error
            primary_traceback = error.__traceback__
        finally:
            if backend is not None:
                try:
                    outer_engine = _outer_engine_mode(backend._model.model.talker)
                except Exception as mode_error:
                    record_lifecycle_error(mode_error)
                try:
                    runtime_metrics = _backend_runtime_metrics(backend)
                except Exception as snapshot_error:
                    record_lifecycle_error(snapshot_error)
                try:
                    backend.close()
                except Exception as close_error:
                    record_lifecycle_error(close_error)

    errors = ([primary_error] if primary_error is not None else []) + secondary_errors
    failed = primary_error is not None
    report = _build_profile_report(
        args,
        batch_sizes,
        reports,
        outer_engine=outer_engine,
        runtime_metrics=runtime_metrics,
        status="failure" if failed else "success",
        warning_messages=_warning_messages(captured_warnings),
        errors=[_format_error(error) for error in errors],
        cache_overflow=(
            _is_cache_overflow_error(primary_error)
            if primary_error is not None
            else False
        ),
    )
    _write_profile_report(args.out, report)
    if primary_error is not None:
        raise primary_error.with_traceback(primary_traceback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
