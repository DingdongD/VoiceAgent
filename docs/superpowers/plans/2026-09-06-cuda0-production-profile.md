# CUDA 0 Voice-Agent Production Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run ASR, nano-vLLM, and parity-gated streaming Qwen-TTS predictably on one CUDA 0 device with fixed KV budgets, startup graph warmup, and reproducible latency reporting.

**Architecture:** Keep each model in its existing resident process. Add explicit cache budgets at the engine boundaries, warm nano-vLLM graph shapes before readiness, and resolve a named runtime profile before factories import voice-app configuration. Preserve the current compatibility defaults and keep experimental outer-talker engines opt-in.

**Tech Stack:** Python 3.11, PyTorch/CUDA Graphs, nano-vLLM, Qwen3-ASR, Qwen3-TTS, argparse, multiprocessing queues, pytest.

**Spec:** `docs/superpowers/specs/2026-09-06-cuda0-production-profile-design.md`

## Global Constraints

- `compat` remains the default runtime profile.
- Fixed cache values accept `-1` for automatic sizing or a positive integer; `0` and values below `-1` fail before model loading.
- The CUDA 0 profile starts with ASR 128 blocks and LLM 64 blocks.
- Explicit CLI arguments and device/cache environment variables override profile-derived values.
- Exact scalar TTS parity remains available and continues to force one worker with no outer native batching.
- Inner predictor CUDA Graph is production-eligible; outer CUDA Graph, QKV fusion, and varlen paged attention remain experimental.
- Performance claims require matching ASR text, LLM text, rendered TTS input, PCM byte count, and PCM duration.
- The repository has no Git baseline. Record task checkpoints in `progress.md`; do not create an unrelated initial commit.

## File Structure

- `qwen_asr_vllm/config.py`: validate ASR fixed-cache configuration.
- `qwen_asr_vllm/agent/nano_llm.py`: propagate LLM fixed blocks and run startup batch-shape warmup.
- `qwen_asr_vllm/agent/runtime_profiles.py`: own named profile defaults and explicit-option precedence.
- `/home/nano-vllm/nanovllm/engine/model_runner.py`: honor an explicit positive LLM block count.
- `/home/voice_assistant_app/src/config.py`: map environment/profile values into service configuration.
- `/home/voice_assistant_app/src/asr_backend.py`: pass the ASR cache budget to `AsyncAsrEngine`.
- `bench/voice_agent_timing.py`: propagate LLM cache/warmup values, resolve profiles, and report service metrics.
- `bench/serve_voice_agent_ui.py`: resolve the same production profile for the live UI.
- `bench/cuda0_voice_agent_profile.py`: preflight and run the clean-card acceptance workload.
- `tests/test_engine.py`: ASR cache validation.
- `tests/test_agent_service_runners.py`: nano-vLLM cache propagation and graph-shape warmup.
- `tests/test_voice_agent_tts_streaming_flags.py`: profile resolution and CLI precedence.
- `tests/test_voice_agent_resource_metrics.py`: process-ready memory and cache telemetry.

---

### Task 1: Deterministic ASR And LLM KV Budgets (Complete)

**Files:**
- Modify: `qwen_asr_vllm/config.py`
- Modify: `qwen_asr_vllm/agent/nano_llm.py`
- Modify: `/home/nano-vllm/nanovllm/engine/model_runner.py`
- Modify: `/home/voice_assistant_app/src/config.py`
- Modify: `/home/voice_assistant_app/src/asr_backend.py`
- Modify: `bench/voice_agent_timing.py`
- Test: `tests/test_engine.py`
- Test: `tests/test_agent_service_runners.py`
- Test: `tests/test_voice_agent_tts_streaming_flags.py`

**Interfaces:**
- Consumes: `EngineConfig.num_kvcache_blocks`, nano-vLLM `Config.num_kvcache_blocks`, and voice-app environment configuration.
- Produces: `NanoVllmStepBatchingBackend(..., num_kvcache_blocks: int = -1)` and voice config constants `ASR_NUM_KVCACHE_BLOCKS`, `LLM_NUM_KVCACHE_BLOCKS`.

- [x] **Step 1: Write ASR cache-value validation tests**

Add parameterized tests which construct `EngineConfig` with a stub local model config and assert that `-1`, `1`, and `128` are accepted while `0` and `-2` raise the exact message.

```python
@pytest.mark.parametrize("blocks", [0, -2])
def test_engine_config_rejects_invalid_fixed_kv_blocks(model_dir, model_config, blocks):
    with pytest.raises(ValueError, match="num_kvcache_blocks must be -1 or positive"):
        EngineConfig(
            model=str(model_dir),
            model_config=model_config,
            num_kvcache_blocks=blocks,
        )
```

- [x] **Step 2: Run the ASR validation test and confirm RED**

Run: `pytest -q tests/test_engine.py -k fixed_kv_blocks`

Expected: FAIL because `EngineConfig.__post_init__` currently accepts `0` and values below `-1`.

- [x] **Step 3: Implement ASR cache validation**

Add this guard before model loading-dependent capacity work:

```python
if self.num_kvcache_blocks == 0 or self.num_kvcache_blocks < -1:
    raise ValueError("num_kvcache_blocks must be -1 or positive")
```

- [x] **Step 4: Write LLM fixed-block propagation tests**

Inject fake `nanovllm.llm.LLM` and `SamplingParams` modules, construct the backend with `num_kvcache_blocks=64`, and assert the LLM constructor receives that exact value. Add a source-level unit around `ModelRunner.allocate_kv_cache` with fake CUDA statistics proving a positive configured count is not overwritten.

```python
backend = NanoVllmStepBatchingBackend(
    model_path="/model",
    num_kvcache_blocks=64,
    warmup=False,
)
assert fake_llm.kwargs["num_kvcache_blocks"] == 64
```

- [x] **Step 5: Run LLM propagation tests and confirm RED**

Run: `pytest -q tests/test_agent_service_runners.py -k 'num_kvcache_blocks or fixed_kv'`

Expected: FAIL because the backend has no parameter and nano-vLLM currently recomputes the block count unconditionally.

- [x] **Step 6: Implement LLM fixed-block handling**

Pass `num_kvcache_blocks` into `LLM(...)`. In nano-vLLM, compute a memory-derived count only when the configured value is `-1`; retain the existing positive value otherwise.

```python
if config.num_kvcache_blocks == -1:
    config.num_kvcache_blocks = (
        int(total * config.gpu_memory_utilization - used - peak + current)
        // block_bytes
    )
if config.num_kvcache_blocks <= 0:
    raise RuntimeError("no memory available for nano-vLLM KV cache")
```

- [x] **Step 7: Write voice-factory configuration tests**

Test that environment values become integers and that factories propagate ASR `128` and LLM `64` without relying on GPU imports. Reuse fake modules through `sys.modules` as existing voice-agent flag tests do.

- [x] **Step 8: Run voice-factory tests and confirm RED**

Run: `pytest -q tests/test_voice_agent_tts_streaming_flags.py -k 'cache or kvcache'`

Expected: FAIL because the constants and factory arguments do not exist.

- [x] **Step 9: Wire voice configuration and factories**

Add:

```python
ASR_NUM_KVCACHE_BLOCKS = int(os.getenv("ASR_NUM_KVCACHE_BLOCKS", "-1"))
LLM_NUM_KVCACHE_BLOCKS = int(os.getenv("LLM_NUM_KVCACHE_BLOCKS", "-1"))
```

Pass ASR blocks into `AsyncAsrEngine` and LLM blocks through `create_real_llm_agent()` into `NanoVllmStepBatchingBackend`.

- [x] **Step 10: Verify Task 1**

Run: `pytest -q tests/test_engine.py tests/test_agent_service_runners.py tests/test_voice_agent_tts_streaming_flags.py`

Expected: all selected tests pass. Record exact counts in `progress.md`.

### Task 2: Pre-Capture Nano-vLLM Batch Shapes During Startup (Complete)

**Files:**
- Modify: `qwen_asr_vllm/agent/nano_llm.py`
- Modify: `/home/voice_assistant_app/src/config.py`
- Modify: `bench/voice_agent_timing.py`
- Modify: `bench/nano_llm_batching_probe.py`
- Test: `tests/test_agent_service_runners.py`

**Interfaces:**
- Consumes: `max_num_seqs`, existing `chat_stream()` and `add_request`/`step` scheduling.
- Produces: `warmup_batch_sizes: Sequence[int] = (1,)`, `warmup_stats`, and `LLM_WARMUP_BATCH_SIZES`.

- [x] **Step 1: Write warmup normalization tests**

Test clipping, stable deduplication, and validation:

```python
assert normalize_warmup_batch_sizes((1, 2, 4, 4, 8), max_num_seqs=4) == (1, 2, 4)
with pytest.raises(ValueError, match="warmup batch sizes must be positive"):
    normalize_warmup_batch_sizes((0, 2), max_num_seqs=4)
```

- [x] **Step 2: Run normalization tests and confirm RED**

Run: `pytest -q tests/test_agent_service_runners.py -k warmup_batch_sizes`

Expected: FAIL because the helper does not exist.

- [x] **Step 3: Implement normalization**

Create a public pure helper in `nano_llm.py` that preserves input order, removes duplicates, and clips sizes above `max_num_seqs` to that limit.

- [x] **Step 4: Write concurrent warmup behavior tests**

Construct `NanoVllmStepBatchingBackend` with `warmup_batch_sizes=(1, 2, 4)` and a fake step engine. Assert warmup reaches batch sizes 1, 2, and 4, online `stats` starts at zero, and `warmup_stats` reports the captured maximum.

```python
assert backend.warmup_stats["batch_sizes"] == [1, 2, 4]
assert backend.warmup_stats["max_step_batch_size"] == 4
assert backend.stats == {"step_calls": 0, "max_step_batch_size": 0}
```

- [x] **Step 5: Run concurrent warmup test and confirm RED**

Run: `pytest -q tests/test_agent_service_runners.py -k startup_batch_warmup`

Expected: FAIL because initialization currently warms only one request and includes it in online metrics.

- [x] **Step 6: Implement cohort warmup**

For each normalized size, start that many daemon threads behind a barrier, consume one-token `chat_stream()` calls, join all threads, save a stats snapshot, reset online counters, and reset conversation history. Propagate any worker exception from the constructor.

- [x] **Step 7: Wire config and probe reporting**

Parse `LLM_WARMUP_BATCH_SIZES` from comma-separated integers, defaulting to `1`. Pass it through `create_real_llm_agent`. Include `warmup_stats` beside online `backend_stats` in `nano_llm_batching_probe.py`.

- [x] **Step 8: Verify Task 2**

Run: `pytest -q tests/test_agent_service_runners.py tests/test_voice_agent_tts_streaming_flags.py`

Expected: all selected tests pass and no worker thread remains alive.

### Task 3: Named CUDA 0 Throughput Profile (Complete)

**Files:**
- Create: `qwen_asr_vllm/agent/runtime_profiles.py`
- Modify: `/home/voice_assistant_app/src/config.py`
- Modify: `bench/voice_agent_timing.py`
- Modify: `bench/serve_voice_agent_ui.py`
- Test: `tests/test_voice_agent_tts_streaming_flags.py`

**Interfaces:**
- Consumes: argparse parser plus raw argument vector.
- Produces: `parse_profiled_args(parser, argv=None)` and profile names `compat`, `cuda0-throughput`.

- [x] **Step 1: Write profile default tests**

Test that `compat` preserves current values and `cuda0-throughput` resolves codec streaming, graph slots, batching, and greedy benchmark controls:

```python
args = parse_profiled_args(
    voice_agent_timing.build_parser(),
    ["--runtime-profile", "cuda0-throughput", "--mode", "real", "--audio", "x.wav"],
)
assert args.tts_streaming_engine == "codec-step"
assert args.tts_cuda_graph_code_predictor is True
assert args.tts_cuda_graph_fixed_slots == 2
assert args.tts_process_stream_workers == 2
assert args.tts_process_batch_window_ms == 10.0
assert args.tts_stream_batch_exact_parity is False
assert args.tts_do_sample is False
assert args.tts_subtalker_do_sample is False
```

- [x] **Step 2: Write explicit precedence tests**

Pass `--tts-streaming-engine off`, `--no-tts-cuda-graph-code-predictor`, and `--tts-stream-batch-exact-parity` after the profile. Assert those explicit values win independent of argument order.

- [x] **Step 3: Run profile tests and confirm RED**

Run: `pytest -q tests/test_voice_agent_tts_streaming_flags.py -k runtime_profile`

Expected: FAIL because neither the profile module nor CLI option exists.

- [x] **Step 4: Implement profile-aware parsing**

Map option strings to argparse destinations, collect explicitly supplied destinations from the raw argument vector, parse normally, then apply only profile values whose destinations were not explicit.

```python
PROFILE_DEFAULTS = {
    "compat": {},
    "cuda0-throughput": {
        "tts_streaming_engine": "codec-step",
        "tts_cuda_graph_code_predictor": True,
        "tts_cuda_graph_fixed_slots": 2,
        "tts_process_stream_workers": 2,
        "tts_process_batch_window_ms": 10.0,
        "tts_stream_batch_exact_parity": False,
    },
}
```

The timing CLI adds deterministic `tts_do_sample=False`, `tts_subtalker_do_sample=False`, and `tts_temperature=0.0`; the live UI does not force generation semantics.

- [x] **Step 5: Apply the profile before importing service config**

Both `main()` entry points call `parse_profiled_args`. When selected, set `VOICE_RUNTIME_PROFILE=cuda0-throughput` before constructing factories. In voice config, use profile-derived defaults only when the corresponding environment variable is absent:

```python
_cuda0 = os.getenv("VOICE_RUNTIME_PROFILE", "compat") == "cuda0-throughput"
TTS_DEVICE = os.getenv("TTS_DEVICE", "cuda:0" if _cuda0 else "cuda:1")
LLM_DEVICE = os.getenv("LLM_DEVICE", "cuda:0" if _cuda0 else "cuda:2")
ASR_NUM_KVCACHE_BLOCKS = int(os.getenv("ASR_NUM_KVCACHE_BLOCKS", "128" if _cuda0 else "-1"))
LLM_NUM_KVCACHE_BLOCKS = int(os.getenv("LLM_NUM_KVCACHE_BLOCKS", "64" if _cuda0 else "-1"))
```

- [x] **Step 6: Verify profile and exact-parity interaction**

Run: `pytest -q tests/test_voice_agent_tts_streaming_flags.py`

Expected: all tests pass, including the existing assertion that exact parity resolves `(1, 0.0, 1)`.

### Task 4: Resident Resource Telemetry And CUDA 0 Preflight (Complete)

**Files:**
- Modify: `qwen_asr_vllm/agent/service_runners.py`
- Modify: `qwen_asr_vllm/agent/local_tts.py`
- Create: `bench/cuda0_voice_agent_profile.py`
- Create: `tests/test_voice_agent_resource_metrics.py`
- Modify: `bench/voice_agent_timing.py`

**Interfaces:**
- Consumes: child-process ready messages and backend runtime metrics.
- Produces: `cuda_memory_snapshot(device=None) -> dict`, wrapper `runtime_metrics()` methods, and a JSON acceptance report.

- [x] **Step 1: Write CUDA snapshot unit tests**

Inject a fake torch CUDA object and assert the snapshot contains logical device, allocated, reserved, peak allocated, driver used, and free bytes. Assert CPU/unavailable CUDA returns `{"available": False}`.

- [x] **Step 2: Run snapshot tests and confirm RED**

Run: `pytest -q tests/test_voice_agent_resource_metrics.py -k cuda_memory_snapshot`

Expected: FAIL because the helper does not exist.

- [x] **Step 3: Implement the side-effect-free snapshot helper**

The helper reads allocator and `mem_get_info` values without synchronizing or emptying the cache. It must not initialize CUDA when the backend is CPU-only.

- [x] **Step 4: Write process-ready telemetry tests**

Use lightweight child factories exposing `runtime_metrics()`. Assert ASR, LLM, and TTS wrappers retain PID/startup metrics and merge backend metrics without changing existing stream protocols.

- [x] **Step 5: Run process telemetry tests and confirm RED**

Run: `pytest -q tests/test_voice_agent_resource_metrics.py -k process`

Expected: FAIL because wrappers currently discard ready-message resource data.

- [x] **Step 6: Propagate startup metrics**

Have each child send a dictionary in its ready payload:

```python
{
    "pid": os.getpid(),
    "supports_streaming": supports_streaming,
    "cuda_memory": cuda_memory_snapshot(),
    "backend": backend.runtime_metrics() if available else {},
}
```

Retain backward-compatible tuple parsing for test doubles, expose a read-only `runtime_metrics()` method, and never use telemetry failure as a model fallback.

- [x] **Step 7: Implement clean CUDA 0 preflight**

Create a CLI that checks `nvidia-smi` before model startup and refuses timing when memory use exceeds a configurable idle threshold or utilization is nonzero. It then invokes the existing real timing path with `cuda0-throughput`, captures stage offsets and resident metrics, and writes one JSON report.

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/nano-vllm/bin/python \
  bench/cuda0_voice_agent_profile.py \
  --audio reference/librispeech_sample.wav \
  --out results/cuda0_voice_agent_production_profile.json
```

- [x] **Step 8: Verify Task 4**

Run: `pytest -q tests/test_voice_agent_resource_metrics.py tests/test_agent_service_runners.py`

Expected: all tests pass and process teardown tests find no live children.

### Task 5: Regression And Real CUDA Acceptance (Complete)

**Files:**
- Modify: `progress.md`
- Modify: `findings.md`
- Modify: `task_plan.md`
- Output: `results/cuda0_voice_agent_production_profile.json`
- Output: `results/cuda0_nano_llm_precaptured_c4.json`
- Output: `results/cuda0_qwen_tts_production_profile_n16.json`

**Interfaces:**
- Consumes: Tasks 1-4 and a clean physical CUDA 0.
- Produces: parity-qualified stage and full-chain timing evidence.

- [x] **Step 1: Run focused CPU regression**

Run:

```bash
pytest -q \
  tests/test_engine.py \
  tests/test_agent_service_runners.py \
  tests/test_voice_agent_tts_streaming_flags.py \
  tests/test_voice_agent_resource_metrics.py
```

Expected: zero failures and no leaked child process.

- [x] **Step 2: Run full non-GPU regression**

Run: `pytest -q -m 'not gpu'`

Expected: zero failures. Record any environment-only HTTP fixture failure separately; do not classify it as passing.

- [x] **Step 3: Check CUDA 0 cleanliness**

Run:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
  --format=csv,noheader,nounits
```

Expected: physical CUDA 0 has only driver baseline memory and 0% utilization. If the external PID remains, stop GPU acceptance and report the exact blocker.

- [x] **Step 4: Measure pre-captured nano-vLLM**

Run four concurrent requests twice in one resident service with fixed 64 blocks and warmup sizes `1,2,4`. Require zero request-path graph captures if exposed, second-cohort TTFT at or below `45 ms`, and write the JSON artifact.

- [x] **Step 5: Measure parity-gated TTS production path**

Run the existing two-request greedy outer batching probe with inner graph slots 2. Require PCM byte counts `[38400, 53760]` for the controlled prompts and compare native/service walls to the clean CUDA 0 baseline.

- [x] **Step 6: Run the complete same-GPU chain**

Use the LibriSpeech sample, process isolation, fixed ASR/LLM blocks, codec-step streaming, graph slots 2, and native parity gate. Require all services resident together, no OOM, no errors, and all latency offsets present.

- [x] **Step 7: Attribute and report**

Report ASR compute, endpoint wait, LLM cold/warm TTFT, sentence-ready delay, TTS queue wait, first chunk, total drain, PCM duration, resident memory by service, and throughput. Compute speedup only against matching outputs.

- [x] **Step 8: Clean up and verify no benchmark processes remain**

Run:

```bash
ps -C python -C python3 -o pid=,etime=,args= | \
  rg 'cuda0_voice_agent_profile|voice_agent_timing|qwen_tts_outer|nano_llm' || true
```

Expected: no matching process. Update planning files with measured results or the clean-GPU blocker.
