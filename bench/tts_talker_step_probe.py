"""Locate the missing time in a Qwen-TTS talker step.

The step costs 85.4 ms and produces 80 ms of audio, while the memory-bandwidth
roofline for the weights it touches is 3.11 ms, so it runs about 27x off any
hardware limit. Averages cannot say why. This probe separates the three
candidates by measuring, per talker step, how long the GPU is actually busy,
how many kernels are launched, and how much wall time is neither.

A GPU busy fraction far below 1 with thousands of tiny kernels means the step is
bound by launch and framework overhead, which is fixable without new math. A
busy fraction near 1 would instead mean the kernels themselves are slow and the
roofline estimate is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_asr_vllm.agent.local_tts import QwenTtsBackend  # noqa: E402

DEFAULT_TEXT = "Yes, the boy wills the item, and Montfiche feels too ill to oppose it."


def summarise(prof: profile, wall_ms: float, steps: int) -> dict:
    """Split profiler events into GPU-side and launch-side totals."""
    kernel_ms = 0.0
    kernel_count = 0
    by_name: Counter[str] = Counter()
    time_by_name: Counter[str] = Counter()

    for event in prof.key_averages():
        # `self_device_time_total` is GPU time attributed to this op alone, so
        # summing it over all ops counts each kernel exactly once.
        device_us = getattr(event, "self_device_time_total", 0) or 0
        if device_us > 0:
            kernel_ms += device_us / 1000.0
            kernel_count += event.count
            by_name[event.key] += event.count
            time_by_name[event.key] += device_us / 1000.0

    return {
        "steps": steps,
        "wall_ms": round(wall_ms, 1),
        "wall_ms_per_step": round(wall_ms / steps, 2) if steps else None,
        "gpu_kernel_ms": round(kernel_ms, 1),
        "gpu_kernel_ms_per_step": round(kernel_ms / steps, 2) if steps else None,
        "gpu_busy_fraction": round(kernel_ms / wall_ms, 3) if wall_ms else None,
        "kernel_launches": kernel_count,
        "kernel_launches_per_step": round(kernel_count / steps, 1) if steps else None,
        "mean_kernel_us": round(kernel_ms * 1000 / kernel_count, 1) if kernel_count else None,
        "top_by_count": by_name.most_common(12),
        "top_by_gpu_ms": [(name, round(ms, 1)) for name, ms in time_by_name.most_common(12)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--language", default="en")
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--cuda-graph-code-predictor", action="store_true")
    parser.add_argument("--cuda-graph-fixed-slots", type=int, default=2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    backend = QwenTtsBackend(
        model_path=args.model_path,
        device=args.device,
        language=args.language,
        speaker=args.speaker,
        warmup=True,
        cuda_graph_code_predictor=args.cuda_graph_code_predictor,
        cuda_graph_fixed_slots=args.cuda_graph_fixed_slots,
    )
    try:
        model = backend._model
        model.generate_defaults.update(do_sample=False, subtalker_dosample=False)

        # Warmup is already done by the backend, but the profiler itself perturbs
        # the first iteration, so discard one more generation before measuring.
        list(backend.synthesize_stream(args.text))

        torch.cuda.synchronize(args.device)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            audio = list(backend.synthesize_stream(args.text))
            end.record()
            torch.cuda.synchronize(args.device)
        wall_ms = start.elapsed_time(end)

        # One talker step per codec frame; recover the count from the codec cache
        # the run just produced rather than assuming a frame rate.
        steps = getattr(model, "_last_codec_frames", None)
        if not steps:
            # 12 Hz codec: each frame is 80 ms of audio.
            total_audio_ms = sum(len(chunk) for chunk in audio) / (24000 * 2) * 1000
            steps = max(1, round(total_audio_ms / 80.0))

        report = summarise(prof, wall_ms, int(steps))
        report["device"] = args.device
        report["cuda_graph_code_predictor"] = bool(args.cuda_graph_code_predictor)
        print(json.dumps(report, indent=2))
        if args.out:
            args.out.write_text(json.dumps(report, indent=2))
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
