"""Counters, bucketed histograms and the Prometheus rendering.

Worth testing directly because these numbers are what an operator makes decisions
from: a histogram whose buckets are not cumulative, or a counter that double-counts
a cancelled request, produces a dashboard that quietly lies.
"""
import math

from qwen_asr_vllm.engine.request import RequestTimings
from qwen_asr_vllm.engine.scheduler import SchedulerStats
from qwen_asr_vllm.metrics import LATENCY_BUCKETS, Histogram, Metrics


def make_output(finish_reason="stop", total_seconds=0.4, prompt_tokens=16, output_tokens=5):
    timings = RequestTimings(arrival=0.0, encode_start=0.1, encode_end=0.2, finish=total_seconds)
    return type(
        "Output",
        (),
        {
            "finish_reason": finish_reason,
            "num_prompt_tokens": prompt_tokens,
            "num_output_tokens": output_tokens,
            "timings": timings,
        },
    )()


class TestHistogram:
    def test_observations_land_in_the_right_bucket(self):
        hist = Histogram((1.0, 2.0, 5.0))
        for value in (0.5, 1.5, 1.9, 7.0):
            hist.observe(value)

        assert hist.counts == [1, 2, 0, 1]
        assert hist.count == 4
        assert hist.total == 10.9

    def test_bucket_boundary_is_inclusive(self):
        """Prometheus buckets are ``le``, so an exact bound belongs to that bucket."""
        hist = Histogram((1.0, 2.0))
        hist.observe(1.0)
        assert hist.counts == [1, 0, 0]

    def test_quantile_is_an_upper_bound(self):
        hist = Histogram((1.0, 2.0, 5.0))
        for _ in range(99):
            hist.observe(0.5)
        hist.observe(4.0)

        assert hist.quantile(0.5) == 1.0
        assert hist.quantile(0.99) == 1.0
        assert hist.quantile(1.0) == 5.0

    def test_overflow_quantile_is_infinite(self):
        hist = Histogram((1.0,))
        hist.observe(100.0)
        assert math.isinf(hist.quantile(0.5))

    def test_empty_histogram_reports_zero(self):
        hist = Histogram(LATENCY_BUCKETS)
        assert hist.quantile(0.5) == 0.0
        assert hist.mean == 0.0


class TestCounters:
    def test_success_and_abort_are_separated(self):
        metrics = Metrics()
        metrics.record_finished(make_output("stop"))
        metrics.record_finished(make_output("length"))
        metrics.record_finished(make_output("aborted:out_of_memory"))

        assert metrics.requests_finished == 2
        assert metrics.requests_aborted == 1
        assert metrics.finish_reasons == {
            "stop": 1,
            "length": 1,
            "aborted:out_of_memory": 1,
        }

    def test_cancellation_is_not_counted_as_a_completion(self):
        """The engine still emits an output for a cancelled request."""
        metrics = Metrics()
        metrics.record_cancelled()
        metrics.record_finished(make_output("cancelled"))

        assert metrics.requests_cancelled == 1
        assert metrics.requests_finished == 0
        assert metrics.requests_aborted == 0

    def test_token_and_audio_totals_accumulate(self):
        metrics = Metrics()
        metrics.record_received(3.0)
        metrics.record_received(7.0)
        metrics.record_finished(make_output(prompt_tokens=16, output_tokens=5))

        assert metrics.requests_received == 2
        assert metrics.audio_seconds_total == 10.0
        assert metrics.prompt_tokens_total == 16
        assert metrics.output_tokens_total == 5

    def test_snapshot_is_json_ready(self):
        metrics = Metrics()
        metrics.record_received(1.0)
        metrics.record_finished(make_output(total_seconds=0.4))

        snapshot = metrics.snapshot()
        assert snapshot["requests"]["finished"] == 1
        assert snapshot["latency_seconds"]["p50"] == 0.5


class TestPrometheusRendering:
    def test_buckets_are_cumulative_and_end_at_inf(self):
        metrics = Metrics()
        for seconds in (0.02, 0.3, 3.0):
            metrics.record_finished(make_output(total_seconds=seconds))

        lines = metrics.render_prometheus().splitlines()
        buckets = [
            line for line in lines if line.startswith("asr_request_duration_seconds_bucket")
        ]
        values = [int(line.rsplit(" ", 1)[1]) for line in buckets]

        assert values == sorted(values)
        assert values[-1] == 3
        assert buckets[-1].startswith('asr_request_duration_seconds_bucket{le="+Inf"}')

    def test_every_metric_declares_a_type(self):
        body = metrics_body()
        names = {
            line.split()[0]
            for line in body.splitlines()
            if line and not line.startswith("#")
        }
        declared = {line.split()[2] for line in body.splitlines() if line.startswith("# TYPE")}
        for name in names:
            base = name.split("{")[0]
            for suffix in ("_bucket", "_sum", "_count"):
                base = base.removesuffix(suffix)
            assert base in declared, f"{name} has no # TYPE line"

    def test_scheduler_stats_are_exported(self):
        body = metrics_body()
        assert "asr_scheduler_num_model_ooms 0" in body
        assert "asr_scheduler_num_preemptions 0" in body

    def test_labels_are_quoted(self):
        metrics = Metrics()
        metrics.record_finished(make_output("aborted:kv_cache_too_small"))
        assert (
            'asr_finish_reason_total{reason="aborted:kv_cache_too_small"} 1'
            in metrics.render_prometheus()
        )

    def test_output_ends_with_a_newline(self):
        assert metrics_body().endswith("\n")


def metrics_body() -> str:
    metrics = Metrics()
    metrics.record_received(1.0)
    metrics.record_finished(make_output())
    return metrics.render_prometheus(
        scheduler_stats=SchedulerStats(), extra={"kv_cache_blocks": 128}
    )
