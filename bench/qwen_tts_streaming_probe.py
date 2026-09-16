"""Probe whether local Qwen-TTS can emit real partial audio before completion.

Example:

    /opt/conda/envs/nano-vllm/bin/python bench/qwen_tts_streaming_probe.py
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_asr_vllm.agent.local_tts import QwenTtsBackend


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


def wav_bytes_from_audio(backend: QwenTtsBackend, audio, sample_rate: int) -> bytes:
    return backend._numpy_to_wav(audio, sample_rate)


def measure_mode(
    backend: QwenTtsBackend,
    *,
    text: str,
    non_streaming_mode: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = backend._model.generate_custom_voice(
        text=text,
        speaker=backend._speaker,
        language=backend._language,
        non_streaming_mode=non_streaming_mode,
    )
    returned = time.perf_counter()

    if inspect.isgenerator(result) or (
        hasattr(result, "__iter__") and not isinstance(result, tuple | list)
    ):
        first_audio_ms = None
        chunks = []
        sample_rate = None
        for item in result:
            now = time.perf_counter()
            if first_audio_ms is None:
                first_audio_ms = (now - started) * 1000
            chunks.append(item)
        ended = time.perf_counter()
        return {
            "non_streaming_mode": non_streaming_mode,
            "result_kind": "iterator",
            "first_audio_ms": round(first_audio_ms, 1) if first_audio_ms else None,
            "return_ms": round((returned - started) * 1000, 1),
            "total_ms": round((ended - started) * 1000, 1),
            "chunks": len(chunks),
            "sample_rate": sample_rate,
            "true_streaming": bool(first_audio_ms and first_audio_ms < (ended - started) * 900),
        }

    audios, sample_rate = result
    encoded = [
        wav_bytes_from_audio(backend, audio, sample_rate)
        for audio in audios
        if audio is not None and len(audio) > 0
    ]
    ended = time.perf_counter()
    total_ms = (ended - started) * 1000
    return {
        "non_streaming_mode": non_streaming_mode,
        "result_kind": "tuple",
        "first_audio_ms": round(total_ms, 1) if encoded else None,
        "return_ms": round((returned - started) * 1000, 1),
        "total_ms": round(total_ms, 1),
        "chunks": len(encoded),
        "bytes": [len(item) for item in encoded],
        "sample_rate": sample_rate,
        "true_streaming": False,
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = default_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=defaults.get("model_path"))
    parser.add_argument("--device", default=defaults.get("device", "cuda:1"))
    parser.add_argument("--language", default=defaults.get("language", "chinese"))
    parser.add_argument("--speaker", default=defaults.get("speaker", ""))
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="Text to synthesize; may be repeated.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model_path:
        raise SystemExit("--model-path is required")
    texts = args.text or [
        "你好，请用一句简短的话确认系统已经准备好。",
        "这是一个稍长的语音合成探针，用来观察 Qwen TTS 是否能够在整段生成完成之前返回第一段可播放音频。",
    ]
    backend = QwenTtsBackend(
        model_path=args.model_path,
        device=args.device,
        language=args.language,
        speaker=args.speaker,
        warmup=not args.no_warmup,
    )
    try:
        rows = []
        for text in texts:
            for mode in (True, False):
                row = measure_mode(
                    backend,
                    text=text,
                    non_streaming_mode=mode,
                )
                row["text_chars"] = len(text)
                rows.append(row)
        payload = {"results": rows}
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
