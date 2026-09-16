import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


VOICE_APP_ROOT = Path("/home/voice_assistant_app")


def _load_voice_config():
    spec = importlib.util.spec_from_file_location(
        "voice_config_resource_test",
        VOICE_APP_ROOT / "src" / "config.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_voice_config_reads_fixed_cache_environment(monkeypatch):
    monkeypatch.setenv("ASR_NUM_KVCACHE_BLOCKS", "128")
    monkeypatch.setenv("LLM_NUM_KVCACHE_BLOCKS", "64")
    monkeypatch.setenv("LLM_WARMUP_BATCH_SIZES", "1,2,4")
    module = _load_voice_config()

    assert module.ASR_NUM_KVCACHE_BLOCKS == 128
    assert module.LLM_NUM_KVCACHE_BLOCKS == 64
    assert module.LLM_WARMUP_BATCH_SIZES == (1, 2, 4)


def test_cuda0_profile_supplies_single_gpu_resource_defaults(monkeypatch):
    monkeypatch.setenv("VOICE_RUNTIME_PROFILE", "cuda0-throughput")
    for name in (
        "ASR_NUM_KVCACHE_BLOCKS",
        "LLM_NUM_KVCACHE_BLOCKS",
        "LLM_WARMUP_BATCH_SIZES",
        "TTS_DEVICE",
        "LLM_DEVICE",
    ):
        monkeypatch.delenv(name, raising=False)

    module = _load_voice_config()

    assert module.TTS_DEVICE == "cuda:0"
    assert module.LLM_DEVICE == "cuda:0"
    assert module.LLM_MAX_NUM_SEQS == 4
    assert module.ASR_NUM_KVCACHE_BLOCKS == 128
    assert module.LLM_NUM_KVCACHE_BLOCKS == 64
    assert module.LLM_WARMUP_BATCH_SIZES == (1, 2, 4)


def test_voice_asr_backend_passes_fixed_kv_blocks(monkeypatch):
    if str(VOICE_APP_ROOT) not in sys.path:
        sys.path.append(str(VOICE_APP_ROOT))
    asr_backend = importlib.import_module("src.asr_backend")
    monkeypatch.setattr(asr_backend.config, "ASR_NUM_KVCACHE_BLOCKS", 128, raising=False)
    captured = {}

    class FakeAsyncAsrEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def transcribe(self, *_args, **_kwargs):
            return []

    engine_module = ModuleType("qwen_asr_vllm.engine.async_engine")
    engine_module.AsyncAsrEngine = FakeAsyncAsrEngine
    monkeypatch.setitem(sys.modules, "qwen_asr_vllm.engine.async_engine", engine_module)

    asr_backend.QwenAsrVllmBackend()

    assert captured["num_kvcache_blocks"] == 128
