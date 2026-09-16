"""Measure real per-device compute availability instead of trusting nvidia-smi utilization.

The host reports `100%` utilization on every card from a PID outside this
container's namespace, while power draw disagrees. A fixed matmul workload
tells us which devices can actually deliver full throughput.

Read `tflops_*` with the known limitation below. A sustained large matmul heats
the card into thermal slowdown before it finishes, so on a thermally saturated
node it reports the *hot* steady state no matter what state it started from: it
measured a flat ~90 TFLOPs across every arm here while real workloads varied by
1.9x. That is why `smi_before.clock_sm_mhz` and `throttle_reasons` are recorded;
on this node they, not the TFLOPs figure, are what distinguishes the regimes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch


# NVML clocksThrottleReasons bits that explain a low SM clock.
THROTTLE_REASONS = {
    0x0001: "gpu_idle",
    0x0002: "applications_clocks_setting",
    0x0004: "sw_power_cap",
    0x0008: "hw_slowdown",
    0x0010: "sync_boost",
    0x0020: "sw_thermal_slowdown",
    0x0040: "hw_thermal_slowdown",
    0x0080: "hw_power_brake_slowdown",
    0x0100: "display_clock_setting",
}


def decode_throttle(mask: int) -> list[str]:
    return [name for bit, name in THROTTLE_REASONS.items() if mask & bit]


def query_smi() -> list[dict[str, object]]:
    """Snapshot per-device state, including why a clock might be low.

    SM clock and throttle reasons are the load-bearing fields. On this node all
    four cards sit in `sw_thermal_slowdown` at 46-53% of max clock, which changes
    measured throughput by ~1.9x and is invisible in `utilization.gpu`.
    """
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw,"
            "temperature.gpu,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in completed.stdout.strip().splitlines():
        values = [value.strip() for value in line.split(",")]
        clock_sm = int(values[6])
        clock_max = int(values[7])
        mask = int(values[8], 16)
        rows.append(
            {
                "index": int(values[0]),
                "memory_used_mib": int(values[1]),
                "memory_free_mib": int(values[2]),
                "utilization_percent": int(values[3]),
                "power_draw_w": float(values[4]),
                "temperature_c": int(values[5]),
                "clock_sm_mhz": clock_sm,
                "clock_max_sm_mhz": clock_max,
                "clock_ratio": round(clock_sm / clock_max, 3) if clock_max else None,
                "throttle_reasons": decode_throttle(mask),
            }
        )
    return rows


def measure_device(index: int, *, size: int, iters: int, repeats: int) -> dict[str, object]:
    device = torch.device(f"cuda:{index}")
    torch.cuda.set_device(device)
    left = torch.randn(size, size, device=device, dtype=torch.float16)
    right = torch.randn(size, size, device=device, dtype=torch.float16)

    for _ in range(5):
        left @ right
    torch.cuda.synchronize(device)

    flops_per_iter = 2.0 * size**3
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iters):
            left @ right
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        samples.append(flops_per_iter * iters / elapsed / 1e12)

    del left, right
    torch.cuda.empty_cache()
    return {
        "index": index,
        "tflops_samples": [round(value, 2) for value in samples],
        "tflops_best": round(max(samples), 2),
        "tflops_median": round(sorted(samples)[len(samples) // 2], 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/gpu_contention_probe.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this interpreter")

    payload = {
        "torch_version": torch.__version__,
        "device_count": torch.cuda.device_count(),
        "matmul_size": args.size,
        "smi_before": query_smi(),
        "devices": [],
    }
    for index in range(torch.cuda.device_count()):
        payload["devices"].append(
            measure_device(index, size=args.size, iters=args.iters, repeats=args.repeats)
        )
    payload["smi_after"] = query_smi()

    best = max(entry["tflops_best"] for entry in payload["devices"])
    for entry in payload["devices"]:
        entry["relative_to_best"] = round(entry["tflops_best"] / best, 3)

    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
