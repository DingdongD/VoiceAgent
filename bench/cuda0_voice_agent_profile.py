"""Run the fixed-budget voice-agent profile only when physical CUDA 0 is idle."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path


def ensure_repo_import_path() -> None:
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


ensure_repo_import_path()


def activate_cuda0_profile_environment() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VOICE_RUNTIME_PROFILE"] = "cuda0-throughput"
    os.environ.setdefault("LLM_TEMPERATURE", "0")


def parse_gpu_snapshot(output: str) -> dict[str, int]:
    line = output.strip().splitlines()[0]
    values = [int(value.strip()) for value in line.split(",")]
    if len(values) != 4:
        raise ValueError(f"expected four nvidia-smi fields, got: {line!r}")
    return {
        "index": values[0],
        "memory_used_mib": values[1],
        "memory_free_mib": values[2],
        "utilization_percent": values[3],
    }


def query_cuda0_snapshot() -> dict[str, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_gpu_snapshot(completed.stdout)


def require_idle_cuda(
    snapshot: dict[str, int],
    *,
    max_memory_mib: int,
    max_utilization: int,
) -> None:
    if snapshot["memory_used_mib"] > int(max_memory_mib):
        raise RuntimeError(
            "CUDA 0 memory is not idle: "
            f"{snapshot['memory_used_mib']} MiB > {int(max_memory_mib)} MiB"
        )
    if snapshot["utilization_percent"] > int(max_utilization):
        raise RuntimeError(
            "CUDA 0 utilization is not idle: "
            f"{snapshot['utilization_percent']}% > {int(max_utilization)}%"
        )


def _write_report(path: Path, payload: dict) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered)
    print(rendered, end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-idle-memory-mib", type=int, default=64)
    parser.add_argument("--max-idle-utilization", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--tts-max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--llm-trigger", choices=("committed", "final"), default="committed"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snapshot = query_cuda0_snapshot()
    try:
        require_idle_cuda(
            snapshot,
            max_memory_mib=args.max_idle_memory_mib,
            max_utilization=args.max_idle_utilization,
        )
    except RuntimeError as error:
        _write_report(
            args.out,
            {"status": "blocked", "preflight": snapshot, "error": str(error)},
        )
        return 2

    activate_cuda0_profile_environment()

    from bench import voice_agent_timing

    timing_args = voice_agent_timing.parse_profiled_args(
        voice_agent_timing.build_parser(),
        [
            "--runtime-profile",
            "cuda0-throughput",
            "--mode",
            "real",
            "--real-target",
            "async",
            "--llm-trigger",
            args.llm_trigger,
            "--audio",
            str(args.audio),
            "--timeout",
            str(args.timeout),
            "--tts-max-new-tokens",
            str(args.tts_max_new_tokens),
        ],
    )
    voice_agent_timing.activate_runtime_profile_environment(timing_args)
    pcm = voice_agent_timing.read_wav(args.audio)
    result = asyncio.run(voice_agent_timing.run_real_async(pcm, timing_args))
    payload = {
        "status": "completed",
        "preflight": snapshot,
        "runtime_profile": "cuda0-throughput",
        "result": result.as_dict(),
    }
    _write_report(args.out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
