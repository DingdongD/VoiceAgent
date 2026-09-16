import json

from bench.qwen_tts_outer_varlen_probe import build_parser, run_reference_probe


def test_reference_probe_reports_two_request_varlen_lifecycle(tmp_path):
    output = tmp_path / "probe.json"
    report = run_reference_probe(
        requests=2,
        steps=3,
        page_size=2,
        num_pages=16,
        device="cpu",
    )

    assert report["status"] == "completed"
    assert report["parity_passed"] is True
    assert report["request_ids"] == ["short", "long"]
    assert report["different_initial_lengths"] is True
    assert report["scheduler"]["packed_ticks"] == 3
    assert report["cache"]["live_pages"] == 0
    assert report["cache"]["allocations"] == 2
    assert report["cache"]["releases"] == 2
    assert report["timing_ms"]["scheduler_overhead"] >= 0

    output.write_text(json.dumps(report))
    assert json.loads(output.read_text())["parity_passed"] is True


def test_probe_parser_has_explicit_reference_options():
    args = build_parser().parse_args([])

    assert args.requests == 2
    assert args.steps == 8
    assert args.page_size == 16
    assert args.num_pages == 256
