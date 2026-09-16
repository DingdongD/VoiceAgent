"""Serve a streaming ASR engine over TCP for cloud-edge voice agents."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must run before any qwen_asr_vllm import: rotary/RMSNorm are decorated with
# torch.compile at module import, and Inductor then demands a `libcuda.so`.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
try:
    import torch._dynamo

    torch._dynamo.config.disable = True
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_asr_vllm.agent.remote_asr import serve_asr_engine  # noqa: E402


def build_engine(args: argparse.Namespace):
    from qwen_asr_vllm.engine.async_engine import AsyncAsrEngine

    engine = AsyncAsrEngine(
        model=args.model,
        device=args.device,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        num_kvcache_blocks=args.num_kvcache_blocks,
        enforce_eager=args.enforce_eager,
    )
    import numpy as np

    engine.transcribe(np.zeros(16000, dtype=np.float32), sample_rate=16000)
    return engine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "ASR_MODEL_PATH", "/home/xuliangyu/cloudedge/Qwen3-ASR-0.6B"
        ),
    )
    parser.add_argument("--device", default=os.environ.get("ASR_DEVICE", "cuda:0"))
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--num-kvcache-blocks", type=int, default=128)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="skip CUDA-graph capture; useful when the edge driver layout cannot capture",
    )
    args = parser.parse_args()

    engine = build_engine(args)
    listener = serve_asr_engine(engine, args.host, args.port)
    host, port = listener.getsockname()
    print(f"edge ASR listening on {host}:{port} model={args.model} device={args.device}", flush=True)
    try:
        threading_event_wait()
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()
        close = getattr(engine, "close", None)
        if callable(close):
            close()
    return 0


def threading_event_wait() -> None:
    import threading

    threading.Event().wait()


if __name__ == "__main__":
    raise SystemExit(main())
