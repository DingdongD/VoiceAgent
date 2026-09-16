# Qwen-TTS Outer Talker Static/Graph Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in static-KV and CUDA Graph decode engine for the Qwen-TTS outer talker, add safe batch-2 to batch-1 slot compaction, and validate it with controlled concurrent load tests.

**Architecture:** Variable-length prefill remains eager and populates a cohort-owned `StaticCache`. Each decode step runs the existing code predictor graph first, then replays a separate fixed-shape outer model graph over prepared codec IDs and explicit per-request position state. The service forms fixed cohorts; completed batch-2 rows may compact into a batch-1 bundle, while arbitrary mid-cohort refill remains deferred.

**Tech Stack:** Python 3.11/3.13, PyTorch 2.5.1 CUDA Graphs, Transformers `StaticCache`, local `qwen_tts`, pytest, multiprocessing request-id services.

**Spec:** `docs/superpowers/specs/2026-08-26-qwen-tts-outer-talker-graph-design.md`

## Global Constraints

- Do not modify files inside `/opt/conda/envs/nano-vllm/lib/python3.11/site-packages/qwen_tts`.
- Do not capture the code predictor graph inside the outer graph.
- Do not silently fall back from an enabled outer graph engine to upstream eager generation.
- Keep the new engine opt-in until all controlled CUDA acceptance gates pass.
- Use greedy, fixed-work comparisons for parity and speedup claims.
- Use CUDA 2 for primary timing and CUDA 3 only as a contention-affected consistency run.
- The repository currently has no baseline commit and all project files are untracked. Do not create partial commits that imply a false baseline; record task checkpoints in `progress.md` until the user establishes repository history.

## File Structure

- Create `qwen_asr_vllm/agent/qwen_tts_outer_static_engine.py`: request/cohort state, static prefill, explicit decode orchestration, static installer, and metrics.
- Create `qwen_asr_vllm/agent/qwen_tts_outer_graph_engine.py`: shape-keyed outer graph bundles, capture/replay, graph installer, and graph metrics.
- Create `qwen_asr_vllm/agent/qwen_tts_outer_slots.py`: compaction policy and static-cache row-copy utilities.
- Create `tests/test_qwen_tts_outer_static_engine.py`: state isolation, prefill/decode, parity-contract, and installer tests.
- Create `tests/test_qwen_tts_outer_graph_engine.py`: bundle reuse, capture boundary, replay, and error-reset tests.
- Create `tests/test_qwen_tts_outer_slots.py`: grace policy, cache-row copy, and state compaction tests.
- Modify `qwen_asr_vllm/agent/local_tts.py`: backend flags, installer precedence, warmup, close, and metrics access.
- Modify `bench/voice_agent_timing.py`: real-mode outer-engine flags and factory wiring.
- Modify `bench/serve_voice_agent_ui.py`: UI-service outer-engine flags and wiring.
- Modify `bench/qwen_tts_internal_profile.py`: static/graph outer modes, greedy parity output, and engine metrics.
- Create `bench/qwen_tts_outer_load_probe.py`: simultaneous/staggered concurrent load driver and JSON report.
- Modify existing backend/CLI tests and remove `qwen_tts_outer_engine.py` only after replacement acceptance passes.

---

### Task 1: Define Isolated Outer Request And Cohort State

**Files:**
- Create: `qwen_asr_vllm/agent/qwen_tts_outer_static_engine.py`
- Create: `tests/test_qwen_tts_outer_static_engine.py`
- Modify: `progress.md`

**Interfaces:**
- Produces: `OuterTalkerEngineError`, `OuterRequestState`, `OuterCohortState`, `build_decode_position_ids`, `select_text_condition`.
- Consumes: tensor-like objects with PyTorch batch/sequence semantics.

- [ ] **Step 1: Write failing request-state isolation tests**

```python
def test_outer_request_state_keeps_rope_and_text_condition_per_slot():
    first = OuterRequestState(
        request_id=11,
        rope_delta=torch.tensor([[2]]),
        past_hidden=torch.full((1, 1, 4), 1.0),
        generation_step=0,
        trailing_text_hidden=torch.full((1, 2, 4), 3.0),
        tts_pad_embed=torch.full((1, 1, 4), 9.0),
    )
    second = OuterRequestState(
        request_id=12,
        rope_delta=torch.tensor([[7]]),
        past_hidden=torch.full((1, 1, 4), 5.0),
        generation_step=3,
        trailing_text_hidden=torch.full((1, 1, 4), 6.0),
        tts_pad_embed=torch.full((1, 1, 4), 8.0),
    )

    positions = build_decode_position_ids(10, [first, second])
    conditions = select_text_condition([first, second])

    assert positions[:, 0, 0].tolist() == [12, 12, 12]
    assert positions[:, 1, 0].tolist() == [17, 17, 17]
    assert conditions[:, 0, :].tolist() == [[3.0] * 4, [8.0] * 4]
```

- [ ] **Step 2: Run RED test**

Run: `pytest tests/test_qwen_tts_outer_static_engine.py -q`

Expected: collection fails because the new state types and helpers do not exist.

- [ ] **Step 3: Implement minimal state types and helpers**

```python
@dataclass
class OuterRequestState:
    request_id: int
    rope_delta: Any
    past_hidden: Any
    generation_step: int
    trailing_text_hidden: Any
    tts_pad_embed: Any
    active: bool = True
    done_steps: int = 0
    first_codebook_history: list[Any] = field(default_factory=list)
    hidden_history: list[Any] = field(default_factory=list)


@dataclass
class OuterCohortState:
    cache: Any
    cache_position: int
    requests: list[OuterRequestState]
    last_logits: Any

    @property
    def active_indices(self) -> list[int]:
        return [index for index, state in enumerate(self.requests) if state.active]
```

`build_decode_position_ids` must combine shared physical cache position with each request's rope delta. `select_text_condition` must choose `trailing_text_hidden[:, generation_step]` while in range and `tts_pad_embed` afterward.

- [ ] **Step 4: Run GREEN test and focused static checks**

Run: `pytest tests/test_qwen_tts_outer_static_engine.py -q`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Record checkpoint**

Append the interfaces and test result to `progress.md` and mark Phase 27 in progress in `task_plan.md`.

---

### Task 2: Implement Eager Prefill And Static Outer Decode

**Files:**
- Modify: `qwen_asr_vllm/agent/qwen_tts_outer_static_engine.py`
- Modify: `tests/test_qwen_tts_outer_static_engine.py`
- Modify: `progress.md`

**Interfaces:**
- Produces: `OuterRuntimeLease`, `EagerOuterStepRuntime`, `OuterTalkerStaticEngine(talker, max_cache_len=1024, step_runtime=None)`, and `install_static_outer_talker(talker, *, max_cache_len=1024) -> bool`.
- Consumes: Task 1 state types, `talker.model`, `talker.codec_head`, `talker.code_predictor.generate`, and existing `_sample_next_token`.

- [ ] **Step 1: Write failing static-engine behavior tests**

Add a fake talker whose `model` records `past_key_values`, `cache_position`, `position_ids`, and 4D attention masks. Assert:

```python
def test_static_outer_engine_runs_prefill_then_explicit_decode_without_shared_rope_state():
    talker = FakeStaticTalker()
    engine = OuterTalkerStaticEngine(talker, step_runtime=FakeOuterStepRuntime())

    result = engine.generate(**fake_generation_kwargs(batch_size=2, max_new_tokens=3))

    assert talker.rope_deltas is None
    assert talker.model.cache_positions == [(0, 1, 2), (3,), (4,), (5,)]
    assert talker.code_predictor.batch_sizes == [2, 2, 2]
    assert all(mask.ndim == 4 for mask in talker.model.decode_masks)
    assert len(result.hidden_states) == 4
```

Add separate tests for unsupported kwargs, cache overflow, minimum token handling, EOS masking, and `return_dict_in_generate=False`.

- [ ] **Step 2: Run RED test**

Run: `pytest tests/test_qwen_tts_outer_static_engine.py -q`

Expected: failures show `OuterTalkerStaticEngine` and installer are missing.

- [ ] **Step 3: Implement eager prefill**

The prefill method must:

1. Validate `inputs_embeds`, `attention_mask`, `trailing_text_hidden`, `tts_pad_embed`, and cache capacity.
2. Acquire `lease = step_runtime.acquire(batch_size, device, dtype, max_cache_len)` and reset `lease.cache`.
3. Compute `(position_ids, rope_deltas) = talker.get_rope_index(attention_mask)` without assigning `talker.rope_deltas`.
4. Call `talker.model(inputs_embeds=..., attention_mask=..., position_ids=..., past_key_values=lease.cache, cache_position=torch.arange(prompt_length), use_cache=True)`.
5. Compute initial logits with `talker.codec_head`, initialize one `OuterRequestState` per row, and retain the lease in `OuterCohortState` until completion.

- [ ] **Step 4: Implement one static decode step**

The decode method must:

1. Sample first-codebook IDs from prior logits.
2. Call `talker.code_predictor.generate` outside the main step runner.
3. Concatenate all codec IDs.
4. Build conditional embeddings from Task 1 state.
5. Build fixed-length 4D masks for the static cache.
6. Call `cohort.lease.run(prepared_decode)`; the eager lease invokes `talker.model` plus `talker.codec_head` and the graph lease replays its bundle.
7. Update each request's `past_hidden`, generation step, histories, and EOS state.
8. Advance shared physical cache position once.

- [ ] **Step 5: Implement installer and metrics**

Installer attributes:

```python
static_generate._qav_static_outer_talker = True
static_generate._qav_original_generate = original_generate
static_generate._qav_outer_engine = engine
```

Metrics snapshot must include `prefill_calls`, `static_steps_by_batch`, `active_slots_per_step`, `slot_occupancy`, and `errors`.

- [ ] **Step 6: Run focused and regression tests**

Run:

```bash
pytest tests/test_qwen_tts_outer_static_engine.py tests/test_qwen_tts_outer_engine.py tests/test_qwen_tts_cuda_graph_predictor.py -q
```

Expected: all pass; the existing code predictor graph tests remain unchanged.

- [ ] **Step 7: Record checkpoint**

Update `progress.md` with static engine behavior and test count.

---

### Task 3: Add A Real Static-Outer Parity Probe

**Files:**
- Modify: `bench/qwen_tts_internal_profile.py`
- Modify: `qwen_asr_vllm/agent/local_tts.py`
- Modify: `tests/test_qwen_tts_internal_profile.py`
- Modify: `tests/test_agent_local_tts.py`
- Modify: `progress.md`

**Interfaces:**
- Produces backend argument `outer_static_talker_engine: bool = False` and profiler flag `--outer-static-talker-engine`.
- Consumes `install_static_outer_talker` from Task 2.

- [ ] **Step 1: Write failing wiring tests**

```python
def test_qwen_tts_backend_installs_static_outer_engine(monkeypatch):
    installed = []
    monkeypatch.setattr(local_tts, "install_static_outer_talker", lambda talker, **kw: installed.append((talker, kw)))

    QwenTtsBackend(
        model_path="/models/tts",
        device="cpu",
        warmup=False,
        check_runtime=False,
        outer_static_talker_engine=True,
        outer_graph_max_cache_len=512,
    )

    assert installed[0][1] == {"max_cache_len": 512}
```

Parser tests must assert the new profiler flag and cache length.

- [ ] **Step 2: Run RED tests**

Run: `pytest tests/test_agent_local_tts.py tests/test_qwen_tts_internal_profile.py -q`

Expected: constructor/parser failures for missing arguments.

- [ ] **Step 3: Wire static mode and codec-ID reporting**

The internal profile JSON must include:

```json
{
  "outer_engine": "static",
  "codec_ids_sha256": "...",
  "outer_engine_metrics": {}
}
```

Hash the generated codec tensor bytes after moving them to CPU. Do not compare waveform hashes because codec decoding may introduce backend-specific floating-point variation.

- [ ] **Step 4: Run unit verification**

Run: `pytest tests/test_agent_local_tts.py tests/test_qwen_tts_internal_profile.py -q`

Expected: all pass.

- [ ] **Step 5: Run controlled CUDA 2 static parity**

Run upstream and static profiles with `--greedy --max-new-tokens 64 --batch-sizes 1,2` and identical text/seed. Save:

- `results/qwen_tts_outer_upstream_cuda2.json`
- `results/qwen_tts_outer_static_cuda2.json`

Acceptance: same codec frame counts and codec hashes per batch shape; finite non-empty decoded audio; no cache overflow or shared-state warning.

- [ ] **Step 6: Record results and decision**

If parity fails, stop graph work and fix static state ownership first. If it passes, mark Phase 27 complete and Phase 28 in progress.

---

### Task 4: Capture Shape-Keyed Outer Decode CUDA Graphs

**Files:**
- Create: `qwen_asr_vllm/agent/qwen_tts_outer_graph_engine.py`
- Create: `tests/test_qwen_tts_outer_graph_engine.py`
- Modify: `qwen_asr_vllm/agent/qwen_tts_outer_static_engine.py`
- Modify: `progress.md`

**Interfaces:**
- Produces: `OuterGraphKey`, `OuterGraphBundle`, `CUDAGraphOuterStepRunner`, `install_cuda_graph_outer_talker`.
- Consumes: `OuterTalkerStaticEngine` through the `OuterRuntimeLease`/step-runtime interface from Task 2.

- [ ] **Step 1: Write failing graph-boundary tests**

Use injected fake bundles to prove bundle reuse and capture scope:

```python
def test_outer_graph_runner_reuses_batch_shape_and_keeps_predictor_outside_capture():
    events = []
    runner = CUDAGraphOuterStepRunner(
        talker=FakeTalker(events),
        bundle_factory=lambda **kw: FakeOuterBundle(events, kw),
        allow_non_cuda_for_testing=True,
    )

    runner.run(prepared_decode(batch_size=2, marker=1))
    runner.run(prepared_decode(batch_size=2, marker=2))

    assert events.count("capture-main-model") == 1
    assert events.count("replay-main-model") == 2
    assert "code-predictor" not in events
```

Add tests for batch-1/batch-2 keys, cache length keys, device context, replay serialization, reset after exception, and rejection above configured slots.

- [ ] **Step 2: Run RED test**

Run: `pytest tests/test_qwen_tts_outer_graph_engine.py -q`

Expected: import failure for the new graph module.

- [ ] **Step 3: Implement graph key and injected runner**

`OuterGraphKey` fields must include `batch_size`, `device`, `dtype`, `hidden_size`, `num_code_groups`, `max_cache_len`, and attention implementation.

`CUDAGraphOuterStepRunner.acquire(...)` must fetch/create and lock a bundle before prefill, returning an `OuterRuntimeLease` whose `cache` is the bundle-owned cache. `lease.run(prepared)` copies codec IDs/condition/position/mask into fixed buffers, replays, and returns cloned output views required by the static engine. `lease.release()` resets state and unlocks the bundle.

- [ ] **Step 4: Implement real bundle capture**

Capture only:

```python
codec_embeds = talker.get_input_embeddings()(static_codec_ids[:, :1])
for index, embedding in enumerate(talker.code_predictor.get_input_embeddings()):
    codec_embeds = codec_embeds + embedding(static_codec_ids[:, index + 1:index + 2])
inputs_embeds = codec_embeds + static_condition
outputs = talker.model(
    inputs_embeds=inputs_embeds,
    attention_mask=static_4d_mask,
    position_ids=static_position_ids,
    past_key_values=static_cache,
    cache_position=static_cache_position,
    use_cache=True,
)
static_logits = talker.codec_head(outputs.last_hidden_state)
```

Warm up eager once, reset cache, capture under the input CUDA device context, reset again, and synchronize. Do not call `talker.forward` or `code_predictor.generate` inside capture.

- [ ] **Step 5: Integrate graph runner into static engine**

`install_cuda_graph_outer_talker` creates `OuterTalkerStaticEngine(step_runtime=graph_runner)` and marks:

```python
graph_generate._qav_static_outer_talker = True
graph_generate._qav_cuda_graph_outer_talker = True
graph_generate._qav_outer_engine = engine
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest tests/test_qwen_tts_outer_graph_engine.py tests/test_qwen_tts_outer_static_engine.py tests/test_qwen_tts_cuda_graph_predictor.py -q
python -m py_compile qwen_asr_vllm/agent/qwen_tts_outer_graph_engine.py
```

Expected: all pass without graph warnings.

- [ ] **Step 7: Record checkpoint**

Update `progress.md` with graph interfaces, capture boundary, and test results.

---

### Task 5: Add Batch-2 To Batch-1 Cohort Compaction

**Files:**
- Create: `qwen_asr_vllm/agent/qwen_tts_outer_slots.py`
- Create: `tests/test_qwen_tts_outer_slots.py`
- Modify: `qwen_asr_vllm/agent/qwen_tts_outer_static_engine.py`
- Modify: `qwen_asr_vllm/agent/qwen_tts_outer_graph_engine.py`
- Modify: `progress.md`

**Interfaces:**
- Produces: `OuterCompactionPolicy(grace_steps: int)`, `copy_static_cache_row(source, target, row) -> int`, and `compact_cohort(source, target, survivor_index) -> int`.
- Consumes Task 1 state and Task 4 batch-1/batch-2 bundles.

- [ ] **Step 1: Write failing policy and cache-copy tests**

```python
def test_compaction_waits_for_grace_then_selects_survivor():
    policy = OuterCompactionPolicy(grace_steps=2)
    assert policy.observe([True, False]) is None
    assert policy.observe([True, False]) == 0


def test_copy_static_cache_row_copies_only_live_sequence_prefix():
    source = fake_static_cache(batch=2, length=16, marker_rows=(3, 7))
    target = fake_static_cache(batch=1, length=16, marker_rows=(0,))

    copied = copy_static_cache_row(source, target, row=1, used_length=6)

    assert copied == expected_kv_bytes(source, used_length=6)
    assert torch.equal(target.layers[0].keys[0, :, :6], source.layers[0].keys[1, :, :6])
    assert torch.count_nonzero(target.layers[0].keys[0, :, 6:]) == 0
```

Add a test proving rope delta, hidden state, trailing text, history, EOS state, and request ID all follow the survivor.

- [ ] **Step 2: Run RED tests**

Run: `pytest tests/test_qwen_tts_outer_slots.py -q`

Expected: import failure for the slots module.

- [ ] **Step 3: Implement policy and atomic state copy**

The policy resets when active count changes away from one. Cache copy iterates cache layers, copies only `:used_length`, resets target storage first, and returns exact bytes copied from tensor element size and dimensions.

- [ ] **Step 4: Integrate compaction into generation loop**

After each EOS update, call the policy. On compaction:

1. Acquire source and target bundle locks in deterministic batch-size order.
2. Acquire the target bundle lease and reset its cache.
3. Copy survivor KV prefix and request state.
4. Switch cohort batch size and release source bundle.
5. Record copy bytes/time and continue from the same physical cache position.

- [ ] **Step 5: Add metrics assertions**

Assert `compactions`, `compaction_copy_bytes`, `compaction_copy_ms`, and `padded_tail_steps` update correctly for compaction enabled and disabled modes.

- [ ] **Step 6: Run focused regressions**

Run:

```bash
pytest tests/test_qwen_tts_outer_slots.py tests/test_qwen_tts_outer_static_engine.py tests/test_qwen_tts_outer_graph_engine.py -q
```

Expected: all pass.

- [ ] **Step 7: Record checkpoint**

Update `progress.md` with compaction policy and mark any copy-cost assumptions as unverified until CUDA timing.

---

### Task 6: Wire Backend, Service, Timing CLI, And UI

**Files:**
- Modify: `qwen_asr_vllm/agent/local_tts.py`
- Modify: `bench/voice_agent_timing.py`
- Modify: `bench/serve_voice_agent_ui.py`
- Modify: `bench/qwen_tts_internal_profile.py`
- Modify: `tests/test_agent_local_tts.py`
- Modify: `tests/test_voice_agent_tts_streaming_flags.py`
- Modify: `tests/test_qwen_tts_internal_profile.py`
- Modify: `progress.md`

**Interfaces:**
- Produces flags from the spec: `outer_static_talker_engine`, `outer_cuda_graph_talker_engine`, `outer_graph_fixed_slots`, `outer_graph_max_cache_len`, `outer_graph_compaction_grace_steps`.
- Consumes installers from Tasks 2 and 4.

- [ ] **Step 1: Write failing precedence and propagation tests**

Assert graph outer mode wins over static outer mode and the old explicit mode cannot be combined:

```python
with pytest.raises(ValueError, match="outer talker engines are mutually exclusive"):
    QwenTtsBackend(
        model_path="/models/tts",
        check_runtime=False,
        warmup=False,
        outer_cuda_graph_talker_engine=True,
        explicit_talker_step_engine=True,
    )
```

Parser and factory tests must assert every value reaches `QwenTtsBackend` in timing and UI process factories.

- [ ] **Step 2: Run RED tests**

Run:

```bash
pytest tests/test_agent_local_tts.py tests/test_voice_agent_tts_streaming_flags.py tests/test_qwen_tts_internal_profile.py -q
```

Expected: missing arguments and validation failures.

- [ ] **Step 3: Implement backend precedence and warmup**

Install code predictor graph first, then install the selected outer engine. Resident warmup must capture batch-1 and configured fixed-slot outer bundles. `close()` must close/reset both predictor and outer schedulers.

- [ ] **Step 4: Add CLI flags and process-service resolution**

Graph outer slots above one must reuse `resolve_tts_process_batching` so service request cohorts match outer graph capacity. Keep outer predictor-level wait at zero.

- [ ] **Step 5: Expose engine metrics**

Add `QwenTtsBackend.runtime_metrics()` returning nested `code_predictor` and `outer_talker` snapshots without exposing mutable engine objects.

- [ ] **Step 6: Run wiring regressions**

Run the RED command again plus:

```bash
pytest tests/test_agent_service_runners.py tests/test_qwen_tts_cuda_graph_predictor.py -q
```

Expected: all pass.

- [ ] **Step 7: Record checkpoint**

Update `progress.md` with flags, precedence, and warmup behavior.

---

### Task 7: Build Complete Concurrent Load Probe

**Files:**
- Create: `bench/qwen_tts_outer_load_probe.py`
- Create: `tests/test_qwen_tts_outer_load_probe.py`
- Modify: `progress.md`

**Interfaces:**
- Produces a JSON report for load levels 1/2/4/8 and arrival modes `simultaneous` and `staggered`.
- Consumes `ProcessConcurrentTtsBackend`, `QwenTtsBackend.runtime_metrics`, and real CUDA memory APIs.

- [ ] **Step 1: Write failing report-aggregation tests**

```python
def test_load_report_computes_latency_percentiles_and_slot_occupancy():
    report = summarize_load_run(
        request_timings=[
            RequestTiming(1, 10.0, 40.0, 0.0),
            RequestTiming(2, 20.0, 60.0, 5.0),
        ],
        codec_frames=20,
        wall_ms=100.0,
        runtime_metrics={"outer_talker": {"active_slots_per_step": [2, 2, 1, 1]}},
    )

    assert report["first_audio_p50_ms"] == 15.0
    assert report["total_p95_ms"] == 59.0
    assert report["codec_frames_per_s"] == 200.0
    assert report["slot_occupancy"] == 0.75
```

Add tests for failures, empty runs, stagger scheduling, and JSON serialization.

- [ ] **Step 2: Run RED test**

Run: `pytest tests/test_qwen_tts_outer_load_probe.py -q`

Expected: import failure for the probe module.

- [ ] **Step 3: Implement deterministic workload generation**

Use fixed short/medium/long texts, repeat them deterministically, support a seed, and schedule stagger delays from an explicit list. Capture request queue time, first audio, total time, audio bytes, WAV duration, and errors.

- [ ] **Step 4: Implement report metrics**

Report per load/mode:

- p50/p95 queue, first-audio, and total latency.
- Requests/s and codec frames/s.
- Graph captures and batch-1/batch-2 replay counts.
- Mean slot occupancy and padded-tail percentage.
- Compaction count, bytes, and milliseconds.
- Peak allocated/reserved CUDA memory.
- Exact command/configuration and error list.

- [ ] **Step 5: Run probe unit tests and compile check**

Run:

```bash
pytest tests/test_qwen_tts_outer_load_probe.py -q
python -m py_compile bench/qwen_tts_outer_load_probe.py
```

Expected: all pass.

- [ ] **Step 6: Record checkpoint**

Update `progress.md` and mark Phase 28 complete once graph and compaction unit suites are green.

---

### Task 8: Run Real CUDA Acceptance And Clean Historical Outer Path

**Files:**
- Modify: `findings.md`
- Modify: `progress.md`
- Modify: `task_plan.md`
- Delete after acceptance: `qwen_asr_vllm/agent/qwen_tts_outer_engine.py`
- Delete after acceptance: `tests/test_qwen_tts_outer_engine.py`
- Modify references to old explicit flags in backend and bench files.

**Interfaces:**
- Consumes all prior tasks.
- Produces final timing artifacts under `results/` and a paged-slot go/no-go decision.

- [ ] **Step 1: Run controlled batch-1/batch-2 CUDA 2 profiles**

For upstream, outer static, and outer graph, run two repetitions after warmup with:

```bash
TTS_DEVICE=cuda:2 /opt/conda/envs/nano-vllm/bin/python \
  bench/qwen_tts_internal_profile.py \
  --device cuda:2 --greedy --seed 7 --max-new-tokens 64 \
  --batch-sizes 1,1,2,2 \
  --texts '你好，请确认实时语音系统已经准备好。' \
          '你好，请确认实时语音系统已经准备好。'
```

Add the applicable outer engine flag and unique `--out` path for each mode.

- [ ] **Step 2: Enforce parity and latency gates**

Reject graph activation if codec hashes differ from static eager for the same batch shape, batch-1 regresses more than 5%, or batch-2 aggregate throughput improves less than 15%.

- [ ] **Step 3: Measure compaction on controlled long-tail pairs**

Run compaction enabled and disabled against one short plus one long text. Keep compaction only if total wall time or padded-tail work improves without codec mismatch. Record exact KV copy bytes and milliseconds.

- [ ] **Step 4: Run complete load matrix**

Run `bench/qwen_tts_outer_load_probe.py` on CUDA 2 for:

- Sessions: 1, 2, 4, 8.
- Arrival modes: simultaneous, staggered.
- Engines: current predictor-only graph baseline, outer static, outer graph with compaction.
- Stability: at least 100 requests or 10 minutes for the winning mode.

- [ ] **Step 5: Run CUDA 3 consistency sample**

Repeat 1/2-session greedy profiles on CUDA 3. Label absolute values as contention affected and compare only correctness plus broad behavior.

- [ ] **Step 6: Run one full ASR/LLM/TTS timing sample**

Use the existing deterministic audio and current production coalescing settings. Compare first audio, total latency, TTS compute wait, ASR hypothesis, LLM output, TTS fragments, and first audio bytes against the current predictor-graph path.

- [ ] **Step 7: Remove superseded explicit Python engine after acceptance**

Only after Steps 1–6 pass, remove `qwen_tts_outer_engine.py`, its tests, old explicit flags, and all imports. Run `rg -n "explicit_talker_step_engine|qwen_tts_outer_engine" .` and require no runtime references.

- [ ] **Step 8: Run final verification**

Run:

```bash
pytest -q
python -m py_compile \
  qwen_asr_vllm/agent/qwen_tts_outer_static_engine.py \
  qwen_asr_vllm/agent/qwen_tts_outer_graph_engine.py \
  qwen_asr_vllm/agent/qwen_tts_outer_slots.py \
  bench/qwen_tts_outer_load_probe.py \
  bench/qwen_tts_internal_profile.py \
  bench/voice_agent_timing.py \
  bench/serve_voice_agent_ui.py
```

Check no benchmark/service process remains and record post-test GPU memory.

- [ ] **Step 9: Make paged-slot decision and close phases**

If staggered slot occupancy is below 70% or padded-tail work exceeds 20% after compaction, add a new design phase for a custom per-slot cache/attention adapter. Otherwise record that arbitrary refill is not justified. Mark Phase 29 complete only when timing artifacts and the decision are written to `findings.md` and `progress.md`.

## Plan Self-Review

- Spec coverage: static state, graph capture boundary, compaction, cancellation/error reset, observability, runtime wiring, concurrency matrix, acceptance gates, cleanup, and paged-slot decision are each assigned to a task.
- Type consistency: Tasks 2–6 consistently consume Task 1 state types; Task 2 defines the runtime lease contract, Task 4 supplies graph leases backed by bundle-owned cache, and Task 5 compacts between leases from Task 4.
- Scope: arbitrary row-specific cache positions and paged attention remain outside this implementation and require a separate design only when measured occupancy triggers the gate.
- Placeholder scan: no implementation step is left as `TBD`, `TODO`, or an unspecified test/fix action.
