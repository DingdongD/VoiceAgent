"""Serve streaming Qwen-TTS over TCP for a cloud-edge voice agent."""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
try:
    import torch._dynamo

    torch._dynamo.config.disable = True
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_asr_vllm.agent.local_tts import QwenTtsBackend  # noqa: E402
from qwen_asr_vllm.agent.remote_tts import serve_tts_backend  # noqa: E402


def build_backend(args: argparse.Namespace) -> QwenTtsBackend:
    return QwenTtsBackend(
        model_path=args.model,
        device=args.device,
        language=args.language,
        speaker=args.speaker or "",
        warmup=True,
        check_runtime=False,
        streaming=True,
        stream_chunk_size=args.chunk_size,
        stream_first_chunk_size=args.first_chunk_size,
        stream_left_context_size=args.left_context_size,
        stream_batch_exact_parity=False,
        cuda_graph_code_predictor=True,
        cuda_graph_fixed_slots=2,
        do_sample=False,
        subtalker_dosample=False,
        temperature=0.0,
        max_new_tokens=args.max_new_tokens,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18766)
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "TTS_MODEL_PATH",
            "/home/xuliangyu/cloudedge/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        ),
    )
    parser.add_argument("--device", default=os.environ.get("TTS_DEVICE", "cuda:1"))
    parser.add_argument("--language", default="chinese")
    parser.add_argument("--speaker", default="")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--first-chunk-size", type=int, default=2)
    parser.add_argument("--left-context-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    backend = build_backend(args)
    listener = serve_tts_backend(backend, args.host, args.port)
    host, port = listener.getsockname()
    print(
        f"edge TTS listening on {host}:{port} model={args.model} device={args.device}",
        flush=True,
    )
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()
        close = getattr(backend, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
