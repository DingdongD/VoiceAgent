import os
from types import SimpleNamespace

from qwen_asr_vllm.agent import service_runners
from qwen_asr_vllm.agent.service_runners import (
    ProcessAsrEngine,
    ProcessConcurrentTtsBackend,
    ProcessNanoLlmBackend,
)


class MetricsBackend:
    supports_streaming_tts = True

    def runtime_metrics(self):
        return {"backend_ready": True}

    def close(self):
        return None


def make_metrics_backend():
    return MetricsBackend()


def test_cuda_memory_snapshot_does_not_initialize_cuda():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: False,
        )
    )

    assert service_runners.cuda_memory_snapshot(torch_module=fake_torch) == {
        "available": True,
        "initialized": False,
    }


def test_cuda_memory_snapshot_reports_initialized_allocator():
    gib = 2**30
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        current_device=lambda: 0,
        memory_allocated=lambda _device: 4 * gib,
        memory_reserved=lambda _device: 5 * gib,
        max_memory_allocated=lambda _device: 6 * gib,
        mem_get_info=lambda _device: (30 * gib, 40 * gib),
    )

    assert service_runners.cuda_memory_snapshot(
        "cuda:0",
        torch_module=SimpleNamespace(cuda=fake_cuda),
    ) == {
        "available": True,
        "initialized": True,
        "device": "cuda:0",
        "allocated_bytes": 4 * gib,
        "reserved_bytes": 5 * gib,
        "peak_allocated_bytes": 6 * gib,
        "device_used_bytes": 10 * gib,
        "device_free_bytes": 30 * gib,
        "device_total_bytes": 40 * gib,
    }


def test_process_wrappers_expose_startup_metrics(monkeypatch):
    monkeypatch.setenv("VOICE_TTS_REQUEST_STEP_SCHEDULER", "0")
    monkeypatch.setenv("VOICE_TTS_LEGACY_STREAM_BATCH", "0")
    wrappers = [
        ProcessAsrEngine(make_metrics_backend, context="fork", timeout=5),
        ProcessNanoLlmBackend(make_metrics_backend, context="fork", timeout=5),
        ProcessConcurrentTtsBackend(
            make_metrics_backend,
            context="fork",
            timeout=5,
        ),
    ]
    try:
        for wrapper in wrappers:
            metrics = wrapper.runtime_metrics()
            assert metrics["pid"] != os.getpid()
            assert metrics["backend"] == {"backend_ready": True}
            assert "cuda_memory" in metrics
    finally:
        for wrapper in wrappers:
            wrapper.close()
