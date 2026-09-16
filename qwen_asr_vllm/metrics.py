"""Metrics collection and Prometheus text export.

Deliberately dependency-free. Pulling in ``prometheus_client`` would make it a hard
requirement of the engine for the sake of a few counters and one histogram
implementation, and it would fight the process model: the engine may be embedded in
a process that already owns a global registry.

Histograms use explicit bucket boundaries rather than reservoir sampling so that
concurrent scrapes are cheap and exact at the quantiles that matter for serving.
"""
from __future__ import annotations

import bisect
import threading
from dataclasses import dataclass, field

# Latency buckets in seconds. Dense below a second because a short clip should
# transcribe in well under one, sparse above because anything past ten is already a
# problem and the exact value stops informing the decision.
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
# Audio duration buckets in seconds, spanning a word to a long-form recording.
DURATION_BUCKETS = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0)


@dataclass
class Histogram:
    """Cumulative-bucket histogram in the Prometheus sense."""

    buckets: tuple[float, ...]
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * (len(self.buckets) + 1)

    def observe(self, value: float) -> None:
        self.counts[bisect.bisect_left(self.buckets, value)] += 1
        self.total += value
        self.count += 1

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def quantile(self, q: float) -> float:
        """Upper bucket bound containing the q-th quantile.

        Bucketed, so this is an upper bound on the true quantile, not the value
        itself. Reported as such rather than interpolated, because interpolation
        invents precision the data does not have.
        """
        if not self.count:
            return 0.0
        target = q * self.count
        seen = 0
        for index, bucket_count in enumerate(self.counts):
            seen += bucket_count
            if seen >= target:
                return self.buckets[index] if index < len(self.buckets) else float("inf")
        return float("inf")


class Metrics:
    """Engine-wide counters and histograms. Safe to read from a scrape thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

        self.requests_received = 0
        self.requests_finished = 0
        self.requests_aborted = 0
        self.requests_cancelled = 0
        self.requests_timed_out = 0
        self.requests_failed = 0

        self.audio_seconds_total = 0.0
        self.prompt_tokens_total = 0
        self.output_tokens_total = 0

        self.finish_reasons: dict[str, int] = {}

        self.request_latency = Histogram(LATENCY_BUCKETS)
        self.queue_latency = Histogram(LATENCY_BUCKETS)
        self.encode_latency = Histogram(LATENCY_BUCKETS)
        self.audio_duration = Histogram(DURATION_BUCKETS)

    # ------------------------------------------------------------------ record

    def record_received(self, audio_seconds: float) -> None:
        with self._lock:
            self.requests_received += 1
            self.audio_seconds_total += audio_seconds
            self.audio_duration.observe(audio_seconds)

    def record_finished(self, output) -> None:
        """Account for one completed request, successful or aborted."""
        reason = output.finish_reason or "unknown"
        with self._lock:
            self.finish_reasons[reason] = self.finish_reasons.get(reason, 0) + 1
            self.prompt_tokens_total += output.num_prompt_tokens
            self.output_tokens_total += output.num_output_tokens
            if reason.startswith("aborted"):
                self.requests_aborted += 1
            elif reason != "cancelled":
                # Cancellations are counted where they are requested, not here;
                # the engine still emits an output for them.
                self.requests_finished += 1

            timings = output.timings
            if timings.total_seconds:
                self.request_latency.observe(timings.total_seconds)
            if timings.queue_seconds:
                self.queue_latency.observe(timings.queue_seconds)
            if timings.encode_seconds:
                self.encode_latency.observe(timings.encode_seconds)

    def record_cancelled(self) -> None:
        with self._lock:
            self.requests_cancelled += 1

    def record_timed_out(self) -> None:
        with self._lock:
            self.requests_timed_out += 1

    def record_failed(self) -> None:
        with self._lock:
            self.requests_failed += 1

    # ------------------------------------------------------------------ export

    def snapshot(self) -> dict:
        """Plain-dict view, for logging or a JSON status endpoint."""
        with self._lock:
            return {
                "requests": {
                    "received": self.requests_received,
                    "finished": self.requests_finished,
                    "aborted": self.requests_aborted,
                    "cancelled": self.requests_cancelled,
                    "timed_out": self.requests_timed_out,
                    "failed": self.requests_failed,
                },
                "finish_reasons": dict(self.finish_reasons),
                "totals": {
                    "audio_seconds": self.audio_seconds_total,
                    "prompt_tokens": self.prompt_tokens_total,
                    "output_tokens": self.output_tokens_total,
                },
                "latency_seconds": {
                    "mean": self.request_latency.mean,
                    "p50": self.request_latency.quantile(0.5),
                    "p90": self.request_latency.quantile(0.9),
                    "p99": self.request_latency.quantile(0.99),
                },
            }

    def render_prometheus(self, scheduler_stats=None, extra: dict | None = None) -> str:
        lines: list[str] = []

        def counter(name: str, value, help_text: str, labels: str = "") -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name}{labels} {value}")

        def histogram(name: str, hist: Histogram, help_text: str) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} histogram")
            cumulative = 0
            for bound, count in zip(hist.buckets, hist.counts):
                cumulative += count
                lines.append(f'{name}_bucket{{le="{bound}"}} {cumulative}')
            lines.append(f'{name}_bucket{{le="+Inf"}} {hist.count}')
            lines.append(f"{name}_sum {hist.total}")
            lines.append(f"{name}_count {hist.count}")

        with self._lock:
            counter("asr_requests_received_total", self.requests_received, "Requests admitted.")
            counter("asr_requests_finished_total", self.requests_finished, "Requests completed.")
            counter("asr_requests_aborted_total", self.requests_aborted, "Requests given up on.")
            counter(
                "asr_requests_cancelled_total", self.requests_cancelled, "Requests cancelled."
            )
            counter("asr_requests_timed_out_total", self.requests_timed_out, "Requests timed out.")
            counter("asr_requests_failed_total", self.requests_failed, "Requests raising.")
            counter(
                "asr_audio_seconds_total", self.audio_seconds_total, "Audio submitted, seconds."
            )
            counter("asr_prompt_tokens_total", self.prompt_tokens_total, "Prompt tokens.")
            counter("asr_output_tokens_total", self.output_tokens_total, "Generated tokens.")

            lines.append("# HELP asr_finish_reason_total Completions by finish reason.")
            lines.append("# TYPE asr_finish_reason_total counter")
            for reason, count in sorted(self.finish_reasons.items()):
                lines.append(f'asr_finish_reason_total{{reason="{reason}"}} {count}')

            histogram(
                "asr_request_duration_seconds", self.request_latency, "Arrival to completion."
            )
            histogram("asr_queue_duration_seconds", self.queue_latency, "Arrival to encode start.")
            histogram("asr_encode_duration_seconds", self.encode_latency, "Audio encode stage.")
            histogram("asr_audio_duration_seconds", self.audio_duration, "Submitted clip length.")

        if scheduler_stats is not None:
            for field_name, value in vars(scheduler_stats).items():
                counter(f"asr_scheduler_{field_name}", value, f"Scheduler {field_name}.")

        for name, value in (extra or {}).items():
            lines.append(f"# TYPE asr_{name} gauge")
            lines.append(f"asr_{name} {value}")

        return "\n".join(lines) + "\n"
