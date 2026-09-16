"""Synthetic/real-device probe for the opt-in varlen outer-talker reference."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_asr_vllm.agent.qwen_tts_outer_paged_cache import PagedOuterCache
from qwen_asr_vllm.agent.qwen_tts_outer_varlen import (
    OuterVarlenRequest,
    OuterVarlenScheduler,
    PackedOuterOutput,
    ReferencePagedAttentionAdapter,
)


def _make_request(cache, request_id: str, seq_len: int):
    table = cache.allocate(request_id, max_seq_len=seq_len + 8)
    keys = torch.arange(seq_len, dtype=torch.float32).reshape(1, 1, 1, seq_len, 1)
    keys = keys + (1.0 if request_id == "short" else 100.0)
    cache.append(request_id, keys, keys, torch.arange(seq_len, dtype=torch.long))
    return OuterVarlenRequest(
        request_id=request_id,
        page_table=table,
        seq_len=seq_len,
        max_seq_len=seq_len + 8,
        rope_delta=torch.zeros((1, 1), dtype=torch.long),
        past_hidden=torch.zeros((1, 1, 1)),
        trailing_text_hidden=torch.zeros((1, 1, 1)),
        tts_pad_embed=torch.zeros((1, 1, 1)),
    )


def run_reference_probe(
    *,
    requests: int = 2,
    steps: int = 8,
    page_size: int = 16,
    num_pages: int = 256,
    device: str = "cpu",
) -> dict[str, Any]:
    if int(requests) != 2:
        raise ValueError("reference probe requires exactly two request IDs")
    if int(steps) <= 0 or int(page_size) <= 0 or int(num_pages) <= 0:
        raise ValueError("steps, page_size, and num_pages must be positive")
    target = torch.device(device)
    cache = PagedOuterCache(num_layers=1, num_pages=num_pages, page_size=page_size)
    short = _make_request(cache, "short", 3)
    long = _make_request(cache, "long", 5)
    adapter = ReferencePagedAttentionAdapter(cache)
    callback_steps = 0
    tick_times = []

    def decode(step):
        nonlocal callback_steps
        callback_steps += 1
        started = time.perf_counter()
        queries = torch.tensor(
            [[[[2.0 + callback_steps]]], [[[200.0 + callback_steps]]]],
            dtype=torch.float32,
            device=target,
        )
        for row, request in enumerate(step.requests):
            query = queries[row : row + 1]
            key = query.reshape(1, 1, 1, 1, 1)
            cache.append(
                request.request_id,
                key,
                key,
                torch.tensor([request.seq_len], dtype=torch.long, device=target),
            )
        output = adapter.decode_packed(step, queries)
        tick_times.append((time.perf_counter() - started) * 1000.0)
        done = step.request_ids if callback_steps >= int(steps) else ()
        return PackedOuterOutput(
            request_ids=output.request_ids,
            logits=output.logits,
            past_hidden=output.past_hidden,
            done_request_ids=done,
            metrics=output.metrics,
        )

    scheduler = OuterVarlenScheduler(
        cache=cache,
        prefill_fn=lambda request: None,
        decode_packed_fn=decode,
    )
    scheduler.add_request(short)
    scheduler.add_request(long)
    started = time.perf_counter()
    scheduler.run_until_idle(max_steps=int(steps) + 1)
    elapsed = (time.perf_counter() - started) * 1000.0
    cache_snapshot = cache.snapshot()
    parity = (
        short.error is None
        and long.error is None
        and short.result is not None
        and long.result is not None
        and short.seq_len == 3 + int(steps)
        and long.seq_len == 5 + int(steps)
    )
    return {
        "schema": "qwen_tts_outer_varlen_probe",
        "status": "completed" if parity and cache_snapshot["live_pages"] == 0 else "failure",
        "parity_passed": parity,
        "request_ids": ["short", "long"],
        "different_initial_lengths": True,
        "steps": int(steps),
        "device": str(target),
        "scheduler": {
            **scheduler.metrics(),
            "packed_ticks": scheduler.ticks,
            "tick_time_ms": tick_times,
        },
        "cache": cache_snapshot,
        "timing_ms": {
            "scheduler_overhead": max(0.0, elapsed - sum(tick_times)),
            "total": elapsed,
        },
        "errors": [str(request.error) for request in (short, long) if request.error],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--num-pages", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("results/qwen_tts_outer_varlen_cuda2.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_reference_probe(
        requests=args.requests,
        steps=args.steps,
        page_size=args.page_size,
        num_pages=args.num_pages,
        device=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
