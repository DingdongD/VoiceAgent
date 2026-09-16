# Post-Profile Voice Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete lifecycle-safe profiling, production outer-talker selection, dynamic TTS request scheduling, shared-memory PCM IPC, and full-session concurrency measurement.

**Architecture:** Preserve resident process ownership and queue control planes. Add strict evidence/lifecycle helpers, select only parity-qualified outer engines, enrich the existing unified TTS batch scheduler, and transport large byte payloads through bounded shared memory without changing public backend APIs.

**Tech Stack:** Python 3.11, asyncio, multiprocessing, shared_memory, PyTorch/CUDA, nano-vLLM, Qwen3-TTS, pytest.

**Spec:** `docs/superpowers/specs/2026-09-06-post-profile-voice-optimization-design.md`

## Global Constraints

- `compat` behavior remains unchanged.
- No arbitrary runtime fallback after model execution begins.
- QKV fusion and outer per-position CUDA Graph remain experimental and disabled.
- Performance ratios require matching final text and PCM output.
- GPU tests run serially only after an idle-device preflight.

---

### Task 1: Lifecycle-Safe Timing And Parity Evidence

**Files:**
- Modify: `bench/voice_agent_timing.py`
- Test: `tests/test_voice_agent_timing_lifecycle.py`

**Interfaces:**
- Produces: collector input-complete barrier and `compare_timing_outputs(base, candidate) -> dict`.

- [x] Write failing tests for early `done` followed by ASR final and output eligibility.
- [x] Run focused tests and confirm RED.
- [x] Implement the input-complete barrier and comparison helper.
- [x] Run focused and coordinator regressions (`30 passed`).

### Task 2: Production ActivePrefix Outer Mode

**Files:**
- Modify: `qwen_asr_vllm/agent/local_tts.py`
- Modify: `qwen_asr_vllm/agent/runtime_profiles.py`
- Modify: `bench/voice_agent_timing.py`
- Modify: `bench/serve_voice_agent_ui.py`
- Test: `tests/test_agent_local_tts.py`
- Test: `tests/test_voice_agent_tts_streaming_flags.py`

**Interfaces:**
- Produces: `outer_active_prefix_talker_engine` factory/CLI flag and executed-engine metrics.

- [x] Write failing propagation, exclusivity, and metrics tests.
- [x] Run focused tests and confirm RED.
- [x] Wire the existing ActivePrefix installer through resident factories/profile.
- [x] Verify compatibility defaults and production profile selection (`26 passed`).

### Task 3: Dynamic TTS Slot Admission

**Files:**
- Modify: `qwen_asr_vllm/agent/service_runners.py`
- Test: `tests/test_agent_service_runners.py`

**Interfaces:**
- Produces: bounded pending queue, compatibility-aware cohorts, refill and occupancy metrics.

- [x] Write failing staggered request/refill/cancellation tests.
- [x] Run focused tests and confirm RED.
- [x] Implement cohort refill and metrics without concurrent model calls.
- [x] Verify ordering, cancellation, and shutdown.

### Task 4: Shared-Memory TTS Payloads

**Files:**
- Create: `qwen_asr_vllm/agent/shared_bytes.py`
- Modify: `qwen_asr_vllm/agent/service_runners.py`
- Test: `tests/test_shared_bytes.py`
- Test: `tests/test_agent_service_runners.py`

**Interfaces:**
- Produces: `SharedBytesTransport`, descriptor encode/decode, cleanup metrics.

- [x] Write failing round-trip and cleanup tests.
- [x] Run focused tests and confirm RED.
- [x] Implement thresholded descriptors and deterministic unlink.
- [x] Integrate TTS response emission/demultiplexing and verify public bytes output.

### Task 5: Full Voice Concurrency Sweep

**Files:**
- Create: `bench/voice_agent_concurrency_probe.py`
- Test: `tests/test_voice_agent_concurrency_probe.py`
- Modify: `progress.md`
- Modify: `findings.md`

**Interfaces:**
- Produces: fake/real `1/2/4/8` JSON report with latency, throughput, parity, and scheduler metrics.

- [x] Write failing fake-mode aggregation and validation tests.
- [x] Run focused tests and confirm RED.
- [x] Implement shared-service session runner and report schema.
- [ ] Run fake sweep, non-GPU regression, and clean-card CUDA acceptance when available. Fake and non-GPU passed; CUDA0 was reacquired by an external 100%-utilization job before the correct Python 3.11 run.
