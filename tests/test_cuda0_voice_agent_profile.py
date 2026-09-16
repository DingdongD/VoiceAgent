import os
import sys
from pathlib import Path

import pytest


def test_cuda0_profile_bootstraps_repo_root_for_direct_script(monkeypatch):
    from bench import cuda0_voice_agent_profile

    repo_root = Path(cuda0_voice_agent_profile.__file__).resolve().parents[1]
    monkeypatch.setattr(sys, "path", [str(repo_root / "bench")])

    cuda0_voice_agent_profile.ensure_repo_import_path()

    assert sys.path[0] == str(repo_root)


def test_cuda0_profile_environment_defaults_to_deterministic_llm(monkeypatch):
    from bench.cuda0_voice_agent_profile import activate_cuda0_profile_environment

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("VOICE_RUNTIME_PROFILE", raising=False)
    monkeypatch.delenv("VOICE_TTS_REQUEST_STEP_SCHEDULER", raising=False)
    monkeypatch.delenv("VOICE_TTS_LEGACY_STREAM_BATCH", raising=False)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)

    activate_cuda0_profile_environment()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ["VOICE_RUNTIME_PROFILE"] == "cuda0-throughput"
    assert os.environ["LLM_TEMPERATURE"] == "0"


def test_cuda0_profile_preserves_explicit_llm_temperature(monkeypatch):
    from bench.cuda0_voice_agent_profile import activate_cuda0_profile_environment

    monkeypatch.delenv("VOICE_TTS_REQUEST_STEP_SCHEDULER", raising=False)
    monkeypatch.delenv("VOICE_TTS_LEGACY_STREAM_BATCH", raising=False)
    monkeypatch.setenv("LLM_TEMPERATURE", "0.3")

    activate_cuda0_profile_environment()

    assert os.environ["LLM_TEMPERATURE"] == "0.3"


def test_cuda0_preflight_parses_gpu_snapshot():
    from bench.cuda0_voice_agent_profile import parse_gpu_snapshot

    assert parse_gpu_snapshot("0, 16, 40944, 0\n") == {
        "index": 0,
        "memory_used_mib": 16,
        "memory_free_mib": 40944,
        "utilization_percent": 0,
    }


@pytest.mark.parametrize(
    ("snapshot", "message"),
    [
        (
            {
                "index": 0,
                "memory_used_mib": 2488,
                "memory_free_mib": 37850,
                "utilization_percent": 0,
            },
            "memory",
        ),
        (
            {
                "index": 0,
                "memory_used_mib": 16,
                "memory_free_mib": 40944,
                "utilization_percent": 100,
            },
            "utilization",
        ),
    ],
)
def test_cuda0_preflight_rejects_busy_gpu(snapshot, message):
    from bench.cuda0_voice_agent_profile import require_idle_cuda

    with pytest.raises(RuntimeError, match=message):
        require_idle_cuda(snapshot, max_memory_mib=64, max_utilization=0)


def test_blocked_preflight_writes_report(monkeypatch, tmp_path):
    from bench import cuda0_voice_agent_profile

    monkeypatch.setattr(
        cuda0_voice_agent_profile,
        "query_cuda0_snapshot",
        lambda: {
            "index": 0,
            "memory_used_mib": 2488,
            "memory_free_mib": 37850,
            "utilization_percent": 100,
        },
    )
    output = tmp_path / "blocked.json"

    status = cuda0_voice_agent_profile.main(
        ["--audio", str(tmp_path / "unused.wav"), "--out", str(output)]
    )

    assert status == 2
    assert '"status": "blocked"' in output.read_text()
