"""Stage-A Qwen-TTS codec chunk decode probe.

This probe still generates the full codec sequence first. It then decodes those
codes in chunks to test audio continuity and codec-to-waveform first chunk time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_asr_vllm.agent.local_tts import QwenTtsBackend
from qwen_asr_vllm.agent.qwen_tts_streaming import (
    chunked_decode_codec_codes,
    concatenate_audio_chunks,
    numpy_to_wav,
    run_with_codec_frame_hook,
    stream_decode_codec_frames,
)


def ensure_voice_app_path() -> None:
    app_root = Path("/home/voice_assistant_app")
    if app_root.is_dir() and str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))


def default_config() -> dict[str, Any]:
    ensure_voice_app_path()
    try:
        from src import config
    except Exception:
        return {}
    return {
        "model_path": config.TTS_MODEL_PATH,
        "device": config.TTS_DEVICE,
        "language": config.TTS_LANGUAGE,
        "speaker": config.TTS_SPEAKER,
    }


def generate_custom_voice_codes(
    wrapper: Any,
    *,
    text: str,
    speaker: str,
    language: str,
    non_streaming_mode: bool,
    max_new_tokens: int | None = None,
) -> list[Any]:
    direct = getattr(wrapper.model, "get_custom_voice_codes", None)
    if callable(direct):
        return list(
            direct(
                text=text,
                speaker=speaker,
                language=language,
                non_streaming_mode=non_streaming_mode,
                max_new_tokens=max_new_tokens,
            )
        )

    model = wrapper.model
    if getattr(model, "tts_model_type", None) != "custom_voice":
        raise ValueError("stage-A probe currently supports custom_voice Qwen-TTS only")

    texts = wrapper._ensure_list(text)
    languages = (
        wrapper._ensure_list(language)
        if isinstance(language, list)
        else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
    )
    speakers = wrapper._ensure_list(speaker)
    instructs = [""] * len(texts)

    if len(languages) == 1 and len(texts) > 1:
        languages = languages * len(texts)
    if len(speakers) == 1 and len(texts) > 1:
        speakers = speakers * len(texts)
    if len(speakers) != len(texts) or len(languages) != len(texts):
        raise ValueError("text/language/speaker batch sizes do not match")

    wrapper._validate_languages(languages)
    wrapper._validate_speakers(speakers)

    input_ids = wrapper._tokenize_texts([wrapper._build_assistant_text(t) for t in texts])
    instruct_ids = [None for _ in instructs]
    gen_kwargs = wrapper._merge_generate_kwargs(
        max_new_tokens=max_new_tokens,
    )
    codes, _ = model.generate(
        input_ids=input_ids,
        instruct_ids=instruct_ids,
        languages=languages,
        speakers=speakers,
        non_streaming_mode=non_streaming_mode,
        **gen_kwargs,
    )
    return list(codes)


def _run_custom_voice_generate(
    wrapper: Any,
    *,
    text: str,
    speaker: str,
    language: str,
    non_streaming_mode: bool,
    max_new_tokens: int | None = None,
):
    model = wrapper.model
    if getattr(model, "tts_model_type", None) != "custom_voice":
        raise ValueError("stage-B probe currently supports custom_voice Qwen-TTS only")

    texts = wrapper._ensure_list(text)
    languages = (
        wrapper._ensure_list(language)
        if isinstance(language, list)
        else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
    )
    speakers = wrapper._ensure_list(speaker)
    instructs = [""] * len(texts)

    if len(languages) == 1 and len(texts) > 1:
        languages = languages * len(texts)
    if len(speakers) == 1 and len(texts) > 1:
        speakers = speakers * len(texts)
    if len(speakers) != len(texts) or len(languages) != len(texts):
        raise ValueError("text/language/speaker batch sizes do not match")

    wrapper._validate_languages(languages)
    wrapper._validate_speakers(speakers)

    input_ids = wrapper._tokenize_texts([wrapper._build_assistant_text(t) for t in texts])
    instruct_ids = [None for _ in instructs]
    gen_kwargs = wrapper._merge_generate_kwargs(max_new_tokens=max_new_tokens)
    return model.generate(
        input_ids=input_ids,
        instruct_ids=instruct_ids,
        languages=languages,
        speakers=speakers,
        non_streaming_mode=non_streaming_mode,
        **gen_kwargs,
    )


def run_probe(
    wrapper: Any,
    *,
    text: str,
    chunk_size: int,
    left_context_size: int,
    out_dir: Path,
    non_streaming_mode: bool = True,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    speaker = getattr(wrapper, "_speaker", "")
    language = getattr(wrapper, "_language", "auto")

    full_started = time.perf_counter()
    api_full_audios, api_full_sample_rate = wrapper.generate_custom_voice(
        text=text,
        speaker=speaker,
        language=language,
        non_streaming_mode=non_streaming_mode,
        **({"max_new_tokens": max_new_tokens} if max_new_tokens is not None else {}),
    )
    api_full_total_ms = (time.perf_counter() - full_started) * 1000
    api_full_audio = np.asarray(api_full_audios[0], dtype=np.float32).reshape(-1)

    codes_started = time.perf_counter()
    codes = generate_custom_voice_codes(
        wrapper,
        text=text,
        speaker=speaker,
        language=language,
        non_streaming_mode=non_streaming_mode,
        max_new_tokens=max_new_tokens,
    )[0]
    codes_ms = (time.perf_counter() - codes_started) * 1000

    full_decode_started = time.perf_counter()
    full_decoded, full_sample_rate = wrapper.model.speech_tokenizer.decode(
        [{"audio_codes": codes}]
    )
    full_decode_ms = (time.perf_counter() - full_decode_started) * 1000
    full_audio = np.asarray(full_decoded[0], dtype=np.float32).reshape(-1)

    decode_started = time.perf_counter()
    chunks = list(
        chunked_decode_codec_codes(
            wrapper.model.speech_tokenizer,
            codes,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
        )
    )
    decode_total_ms = (time.perf_counter() - decode_started) * 1000
    first_audio_ms = chunks[0].decode_ms if chunks else None
    chunked_audio = concatenate_audio_chunks(chunks)

    full_wav = out_dir / "full.wav"
    api_full_wav = out_dir / "api_full.wav"
    chunked_wav = out_dir / "chunked.wav"
    report_json = out_dir / "report.json"
    full_wav.write_bytes(numpy_to_wav(full_audio, int(full_sample_rate)))
    api_full_wav.write_bytes(numpy_to_wav(api_full_audio, int(api_full_sample_rate)))
    sample_rate = chunks[0].sample_rate if chunks else int(full_sample_rate)
    chunked_wav.write_bytes(numpy_to_wav(chunked_audio, sample_rate))

    min_len = min(full_audio.size, chunked_audio.size)
    mean_abs_diff = None
    if min_len:
        mean_abs_diff = float(np.mean(np.abs(full_audio[:min_len] - chunked_audio[:min_len])))

    report = {
        "stage": "codec_chunk_decode",
        "text_chars": len(text),
        "codec_frames": int(codes.shape[0]),
        "chunk_size": int(chunk_size),
        "left_context_size": int(left_context_size),
        "chunks": len(chunks),
        "api_full_total_ms": round(api_full_total_ms, 1),
        "codec_generate_ms": round(codes_ms, 1),
        "full_decode_ms": round(full_decode_ms, 1),
        "chunked_decode_total_ms": round(decode_total_ms, 1),
        "chunked_first_audio_ms": round(first_audio_ms, 1)
        if first_audio_ms is not None
        else None,
        "full_duration_s": round(full_audio.size / int(full_sample_rate), 3),
        "chunked_duration_s": round(chunked_audio.size / sample_rate, 3)
        if sample_rate
        else 0,
        "duration_diff_s": round(
            chunked_audio.size / sample_rate - full_audio.size / int(full_sample_rate),
            3,
        )
        if sample_rate
        else None,
        "mean_abs_diff_overlap": round(mean_abs_diff, 6)
        if mean_abs_diff is not None
        else None,
        "chunk_decode_ms": [round(chunk.decode_ms, 1) for chunk in chunks],
        "chunk_samples": [int(chunk.audio.size) for chunk in chunks],
        "full_wav": str(full_wav),
        "api_full_wav": str(api_full_wav),
        "chunked_wav": str(chunked_wav),
        "report_json": str(report_json),
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def run_streaming_probe(
    wrapper: Any,
    *,
    text: str,
    chunk_size: int,
    first_chunk_size: int | None = None,
    left_context_size: int,
    out_dir: Path,
    non_streaming_mode: bool = True,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    speaker = getattr(wrapper, "_speaker", "")
    language = getattr(wrapper, "_language", "auto")
    eos_token_id = getattr(
        getattr(getattr(wrapper.model, "config", None), "talker_config", None),
        "codec_eos_token_id",
        None,
    )

    def run_generate(on_codec_frame):
        return run_with_codec_frame_hook(
            wrapper.model.talker,
            lambda: _run_custom_voice_generate(
                wrapper,
                text=text,
                speaker=speaker,
                language=language,
                non_streaming_mode=non_streaming_mode,
                max_new_tokens=max_new_tokens,
            ),
            on_codec_frame,
        )

    started = time.perf_counter()
    first_audio_ms = None
    chunks = []
    for chunk in stream_decode_codec_frames(
        run_generate,
        wrapper.model.speech_tokenizer,
        chunk_size=chunk_size,
        first_chunk_size=first_chunk_size,
        left_context_size=left_context_size,
        eos_token_id=eos_token_id,
    ):
        if first_audio_ms is None:
            first_audio_ms = (time.perf_counter() - started) * 1000
        chunks.append(chunk)
    total_ms = (time.perf_counter() - started) * 1000
    audio = concatenate_audio_chunks(chunks)
    sample_rate = chunks[0].sample_rate if chunks else 0

    streaming_wav = out_dir / "streaming.wav"
    report_json = out_dir / "streaming_report.json"
    if sample_rate:
        streaming_wav.write_bytes(numpy_to_wav(audio, sample_rate))
    else:
        streaming_wav.write_bytes(b"")

    report = {
        "stage": "codec_step_stream_decode",
        "text_chars": len(text),
        "codec_frames": int(chunks[-1].code_end) if chunks else 0,
        "chunk_size": int(chunk_size),
        "first_chunk_size": int(first_chunk_size or chunk_size),
        "left_context_size": int(left_context_size),
        "chunks": len(chunks),
        "stream_first_audio_ms": round(first_audio_ms, 1)
        if first_audio_ms is not None
        else None,
        "stream_total_ms": round(total_ms, 1),
        "stream_duration_s": round(audio.size / sample_rate, 3) if sample_rate else 0,
        "first_chunk_duration_s": round(chunks[0].audio.size / chunks[0].sample_rate, 3)
        if chunks
        else 0,
        "chunk_decode_ms": [round(chunk.decode_ms, 1) for chunk in chunks],
        "chunk_codec_frames": [int(chunk.code_end - chunk.code_start) for chunk in chunks],
        "chunk_duration_s": [
            round(chunk.audio.size / chunk.sample_rate, 3) for chunk in chunks
        ],
        "chunk_samples": [int(chunk.audio.size) for chunk in chunks],
        "streaming_wav": str(streaming_wav),
        "report_json": str(report_json),
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    defaults = default_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=defaults.get("model_path"))
    parser.add_argument("--device", default=defaults.get("device", "cuda:1"))
    parser.add_argument("--language", default=defaults.get("language", "chinese"))
    parser.add_argument("--speaker", default=defaults.get("speaker", ""))
    parser.add_argument("--text", default="你好，请用一句话确认系统已经准备好。")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--first-chunk-size", type=int, default=None)
    parser.add_argument("--left-context-size", type=int, default=4)
    parser.add_argument("--stage", choices=["a", "b", "both"], default="a")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--non-streaming-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/qwen_tts_streaming_probe"))
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model_path:
        raise SystemExit("--model-path is required")
    backend = QwenTtsBackend(
        model_path=args.model_path,
        device=args.device,
        language=args.language,
        speaker=args.speaker,
        warmup=not args.no_warmup,
    )
    reports = []
    if args.stage in {"a", "both"}:
        reports.append(
            run_probe(
                backend._model,
                text=args.text,
                chunk_size=args.chunk_size,
                first_chunk_size=args.first_chunk_size,
                left_context_size=args.left_context_size,
                out_dir=args.out_dir,
                non_streaming_mode=args.non_streaming_mode,
                max_new_tokens=args.max_new_tokens,
            )
        )
    if args.stage in {"b", "both"}:
        reports.append(
            run_streaming_probe(
                backend._model,
                text=args.text,
                chunk_size=args.chunk_size,
                left_context_size=args.left_context_size,
                out_dir=args.out_dir,
                non_streaming_mode=args.non_streaming_mode,
                max_new_tokens=args.max_new_tokens,
            )
        )
    report = reports[0] if len(reports) == 1 else {"reports": reports}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
