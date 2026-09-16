import asyncio
import json

import pytest

from bench import voice_agent_concurrency_probe


def test_fake_voice_concurrency_sweep_reports_all_levels_and_output_parity():
    args = voice_agent_concurrency_probe.build_parser().parse_args(
        [
            "--mode",
            "fake",
            "--sessions",
            "1",
            "2",
            "4",
            "8",
            "--llm-trigger",
            "final",
            "--llm-delay",
            "0",
            "--tts-delay",
            "0",
        ]
    )

    report = asyncio.run(voice_agent_concurrency_probe.run_sweep(args))

    assert report["mode"] == "fake"
    assert [level["session_count"] for level in report["levels"]] == [1, 2, 4, 8]
    assert report["baseline_signature"]
    for level in report["levels"]:
        assert len(level["sessions"]) == level["session_count"]
        assert level["errors"] == []
        assert level["parity"] == {"eligible": True, "matched": True}
        assert level["latency_ms"]["first_audio_p95"] is not None
        assert level["throughput"]["sessions_per_s"] > 0
        assert all(session["asr_hypothesis"] == "status" for session in level["sessions"])
        assert all(session["llm_output"] == "First. Second." for session in level["sessions"])


def test_concurrency_probe_main_writes_json_report(tmp_path):
    output = tmp_path / "sweep.json"

    assert voice_agent_concurrency_probe.main(
        [
            "--mode",
            "fake",
            "--sessions",
            "1",
            "--llm-trigger",
            "final",
            "--llm-delay",
            "0",
            "--tts-delay",
            "0",
            "--output",
            str(output),
        ]
    ) == 0

    report = json.loads(output.read_text())
    assert report["levels"][0]["session_count"] == 1


@pytest.mark.parametrize("counts", [[], [0], [1, 1], [2, -1]])
def test_normalize_session_counts_rejects_invalid_levels(counts):
    with pytest.raises(ValueError):
        voice_agent_concurrency_probe.normalize_session_counts(counts)
