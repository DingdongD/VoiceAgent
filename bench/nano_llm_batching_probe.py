"""Measure concurrent nano-vLLM chat streams through the step-batching adapter."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_asr_vllm.agent.nano_llm import NanoVllmStepBatchingBackend


def ensure_voice_app_path() -> None:
    app_root = Path("/home/voice_assistant_app")
    if app_root.is_dir() and str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))


def default_config() -> dict:
    ensure_voice_app_path()
    try:
        from src import config
    except Exception:
        return {}
    return {
        "model_path": config.LLM_MODEL_PATH,
        "nanovllm_root": config.NANOVLLM_ROOT,
        "device": config.LLM_DEVICE,
        "max_model_len": config.LLM_MAX_MODEL_LEN,
        "max_num_seqs": config.LLM_MAX_NUM_SEQS,
        "gpu_memory_utilization": config.LLM_GPU_MEMORY_UTILIZATION,
        "num_kvcache_blocks": getattr(config, "LLM_NUM_KVCACHE_BLOCKS", -1),
        "warmup_batch_sizes": getattr(config, "LLM_WARMUP_BATCH_SIZES", (1,)),
        "enforce_eager": config.LLM_ENFORCE_EAGER,
        "system_prompt": config.LLM_SYSTEM_PROMPT,
        "temperature": config.LLM_TEMPERATURE,
        "max_new_tokens": config.LLM_MAX_NEW_TOKENS,
    }


def run_one(backend: NanoVllmStepBatchingBackend, prompt: str) -> dict:
    started = time.perf_counter()
    first = None
    chunks = []
    for chunk in backend.chat_stream(prompt):
        if first is None:
            first = time.perf_counter()
        chunks.append(chunk)
    ended = time.perf_counter()
    return {
        "first_token_ms": round((first - started) * 1000, 1) if first else None,
        "total_ms": round((ended - started) * 1000, 1),
        "chunks": len(chunks),
        "chars": len("".join(chunks)),
    }


def backend_metrics(backend: NanoVllmStepBatchingBackend) -> dict:
    return {
        "backend_stats": backend.stats,
        "warmup_stats": getattr(backend, "warmup_stats", {}),
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = default_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=defaults.get("model_path"))
    parser.add_argument("--nanovllm-root", default=defaults.get("nanovllm_root", "/home/nano-vllm"))
    parser.add_argument("--device", default=defaults.get("device", "cuda:2"))
    parser.add_argument("--max-model-len", type=int, default=int(defaults.get("max_model_len", 4096)))
    parser.add_argument("--max-num-seqs", type=int, default=int(defaults.get("max_num_seqs", 4)))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(defaults.get("gpu_memory_utilization", 0.85)))
    parser.add_argument("--num-kvcache-blocks", type=int, default=int(defaults.get("num_kvcache_blocks", -1)))
    parser.add_argument(
        "--warmup-batch-sizes",
        nargs="+",
        type=int,
        default=list(defaults.get("warmup_batch_sizes", (1,))),
    )
    parser.add_argument("--enforce-eager", action="store_true", default=bool(defaults.get("enforce_eager", False)))
    parser.add_argument("--temperature", type=float, default=float(defaults.get("temperature", 0.7)))
    parser.add_argument("--max-new-tokens", type=int, default=int(defaults.get("max_new_tokens", 128)))
    parser.add_argument("--system-prompt", default=defaults.get("system_prompt", "You are a helpful voice assistant."))
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model_path:
        raise SystemExit("--model-path is required")
    prompts = args.prompt or [
        "用一句中文回答：实时语音助手为什么需要流式调度？",
        "用一句中文回答：批处理解码可以降低什么开销？",
        "用一句中文回答：首 token 延迟为什么重要？",
        "用一句中文回答：TTS 合并短片段有什么收益？",
    ]
    prompts = [prompts[index % len(prompts)] for index in range(args.concurrency)]
    backend = NanoVllmStepBatchingBackend(
        model_path=args.model_path,
        nanovllm_root=args.nanovllm_root,
        device=args.device,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max(args.max_model_len, 2048),
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
        enforce_eager=args.enforce_eager,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        warmup=not args.no_warmup,
        warmup_batch_sizes=tuple(args.warmup_batch_sizes),
    )
    results = [None] * len(prompts)

    def worker(index: int, prompt: str) -> None:
        try:
            results[index] = run_one(backend, prompt)
        except Exception as exc:
            results[index] = {"error": str(exc)}

    started = time.perf_counter()
    threads = [
        threading.Thread(target=worker, args=(index, prompt))
        for index, prompt in enumerate(prompts)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        total_ms = (time.perf_counter() - started) * 1000
        payload = {
            "concurrency": len(prompts),
            "wall_ms": round(total_ms, 1),
            "requests": results,
        }
        payload.update(backend_metrics(backend))
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
