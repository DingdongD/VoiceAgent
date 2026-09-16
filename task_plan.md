# Voice Agent Streaming Migration Task Plan

Goal: migrate the prior ASR/LLM/TTS voice agent shape into this repo as a streaming runtime layer around the existing ASR engine.

## Phases

- [x] Phase 1: Locate the prior agent implementation and summarize usable pieces.
- [x] Phase 2: Define migration scope and implementation plan.
- [x] Phase 3: Add tests for streaming event protocol and coordinator behavior.
- [x] Phase 4: Implement `qwen_asr_vllm.agent` runtime modules.
- [x] Phase 5: Expose the runtime through a WebSocket endpoint without changing ASR endpoints.
- [x] Phase 6: Run targeted tests and update progress.
- [x] Phase 7: Add async coordinator tests that prove LLM/TTS overlap.
- [x] Phase 8: Implement async coordinator and WebSocket async-session support.
- [x] Phase 9: Add fake/real voice timing benchmark harness.
- [x] Phase 10: Run fake timing and attempt real timing with available local models.
- [x] Phase 11: Add fixed service runners for ASR/LLM/TTS and expose them through the factory.
- [x] Phase 12: Add process-isolated runners and rerun real ASR+LLM/TTS timing.
- [x] Phase 13: Add low-latency TTS fragment flushing and rerun real timing.
- [x] Phase 14: Repair local Qwen-TTS runtime and run full ASR+LLM+TTS timing.
- [x] Phase 15: Add TTS fragment coalescing and request-id multiplexed runners.
- [x] Phase 16: Add nano-vLLM `add_request`/`step` cross-session LLM batching adapter and Qwen-TTS streaming probe.
- [x] Phase 17: Use a dedicated request-id IPC service for real LLM process timing.
- [x] Phase 18: Profile Qwen-TTS `talker.forward` internals and test attention/cache optimization options.
- [x] Phase 19: Run CUDA 3 constrained timing checks and reassess remaining acceleration potential.
- [x] Phase 20: Add request-id level batched Qwen-TTS code predictor scheduler.
- [x] Phase 21: Add explicit outer Qwen-TTS talker step engine prototype.
- [x] Phase 22: Add torch.compile step-engine opt-in and measure compile viability.
- [x] Phase 23: Build and benchmark a static-cache Qwen-TTS code predictor step engine.
- [x] Phase 24: Profile and fuse the static predictor sampling path when its measured share justifies a custom kernel.
- [x] Phase 25: Prototype per-codebook-step CUDA Graph replay on the stable static-cache engine.
- [x] Phase 26: Add fixed-slot request-id batching to the CUDA Graph predictor engine and benchmark concurrent sessions.
- [x] Phase 27: Design and validate an outer talker static-cache decode interface. (fixed-full prototype retained as failed control; exact-parity ActivePrefixCache supersedes it)
- [x] Phase 28: Add outer decode CUDA Graph bundles and safe cohort slot compaction.
- [ ] Phase 29: Add concurrent load probes and decide whether arbitrary slot refill justifies a custom attention/cache adapter.
- [x] Phase 30: Validate a preallocated ActivePrefixCache against exact DynamicCache outer-talker codec parity on CUDA 2. (passed twice; Task 6 stability accepted with 150 requests/600s)
- [x] Phase 31: Fuse outer decode KV write with paged attention, stage page metadata per tick, and prototype QKV projection fusion.
- [x] Phase 32: Add an incremental codec frame buffer and preserve the package decoder's exact chunk semantics.
- [x] Phase 33: Add first-sentence-immediate TTS admission and expose it to real timing/UI paths.
- [x] Phase 34: Add a deterministic native-stream batch parity gate with scalar fallback and runtime counters.
- [x] Phase 35: Extend the real TTS probe with codec token/s metrics and concurrent load sweep.
- [x] Phase 36: Capture a clean CUDA 0 resource baseline and inventory the current stage/full-chain profilers.
- [x] Phase 37: Profile ASR, nano-vLLM, and Qwen-TTS separately on CUDA 0 with fixed workloads.
- [ ] Phase 38: Measure a comparable end-to-end chain and attribute latency to compute, queue, IPC, and output length. Fixed ASR/LLM KV budgets are now exposed; trustworthy timing remains blocked only by the external CUDA 0 workload.
- [x] Phase 39: Rank the remaining optimization space by expected gain, implementation risk, and parity impact.
- [x] Phase 40: Add explicit ASR/nano-vLLM KV block budgets and propagate them through resident service factories.
- [x] Phase 41: Pre-capture nano-vLLM batch `1/2/4` shapes and separate startup metrics from online metrics.
- [x] Phase 42: Add the named, explicit-override-safe `cuda0-throughput` production profile.
- [x] Phase 43: Add process startup resource telemetry and a no-model-load CUDA 0 preflight gate.
- [x] Phase 44: Run full CUDA 0 residency and latency acceptance; all three services fit and completed without errors. A matched-output graph/control rerun remains under Phase 38 because the external workload returned after true LLM greedy sampling was repaired.
- [x] Phase 45: Make prerecorded full-duplex timing retain ASR final/golden output after an early agent `done`, and emit parity-comparison eligibility.
- [x] Phase 46: Expose the exact-parity ActivePrefix/paged outer-talker adapter through the resident TTS service and production profile without enabling divergent QKV fusion.
- [x] Phase 47: Add dynamic request-id slot admission/refill metrics for compatible outer-talker cohorts and preserve scalar fallback for incompatible requests.
- [x] Phase 48: Move large TTS PCM payloads from multiprocessing queue pickles to bounded shared-memory descriptors with deterministic cleanup.
- [ ] Phase 49: Add `1/2/4/8` shared-service voice-session load sweeps with output/parity gates and run CPU plus available CUDA acceptance.
- [x] Phase 50: Make the production profile select one strict request-id TTS scheduler, reject incompatible legacy downgrades at startup, and keep the old stream-batch adapter explicit-only.
- [x] Phase 51: Reduce outer joined-KV traffic with incremental source synchronization and shortest-prefix-first cross-session cohorts; validate real CUDA timing and exact single-session parity.
- [ ] Phase 52: Add fixed-size outer slot pools with explicit global request-id mapping, validate slot reuse/output routing, and run clean-GPU parity/timing acceptance.
- [x] Phase 53: Replace the blocking "clean card" gate with an interleaved-repeat A/B methodology, because every GPU is permanently SW-thermal-throttled and shares one external tenant.
- [x] Phase 54: Measure the real default end-to-end voice chain and attribute wall time to ASR re-decode, endpointing/admission policy, LLM, TTS codec generation, and codec->waveform decode.
- [x] Phase 55: Rank the remaining gain by *streaming policy* versus *kernel compute*, and state the Amdahl ceiling for each.
- [x] Phase 56: Close the ASR streaming gap. **Measured and rejected as a latency fix**: incremental ingest regressed first audio 1716.5 -> 1913.5 ms because ASR leaves the critical path after ~390 ms. Retained as a throughput/concurrency option, not a default.
- [x] Phase 57: Close the perceived-latency gap: `defer_tts_audio_until_asr_final` and `tts_streaming_engine=off` are on/off in the shipped defaults, which discards the already-implemented streaming path.
- [x] Phase 58: Sweep the ASR commit policy and chunk size. `--chunk-ms 800` is the optimum (~1.6x median ASR ingest compute); `speculate` beats `retranscribe` 4.6x. **The commit-gate conclusion here was wrong** and is superseded by Phase 61: the sweep compared latency without ever recording the prompt, and the permissive gate prompts the LLM with a single word.
- [x] Phase 59: Quantify whole-system per-turn cost and the speedup over a naive serialized implementation: 18198.2 -> 573.6 ms first audio (31.7x, not output-matched; 6.8x output-matched). TTS is 76% of per-turn compute at RTF 1.25-1.85.
- [x] Phase 60: Drive TTS real-time factor below 1.0. **Met, by deleting a regression rather than adding an optimization**: both custom outer talker engines are slower than upstream on output-matched profiling (RTF 1.390 upstream vs 3.141 active-prefix vs 8.659 CUDA-graph talker), and removing active-prefix from the profile measured TTS RTF `0.752` and `0.936` with flat per-chunk cost. Chunk size and static-cache size were tested as explanations and both refuted. On a contended host the same profile measures `1.259-1.337`, so the crossing of 1.0 is host-dependent while the `2.39x` ratio is not.
- [x] Phase 61: Audit the pipeline for semantic distortion under acceleration. **Found and rejected the largest reported speedup**: `--llm-trigger committed` with the default gate prompts the LLM with the first committed word (`"Have"` out of a 22-word utterance), so the 3194.2 -> 994.3 ms first-audio win is the agent answering a different question. Gating to 15 of 22 words still distorts the reply and gives back most of the latency. The harness now records `details.llm_prompt` and the parity gate compares it.
- [x] Phase 62: Re-establish the honest latency and RTF baseline on `llm_trigger=final` and define the shipping profile against it. `low-latency` now pins `llm_trigger=final`, disables both custom outer talker engines, and measures first audio `1755.3-2221.7 ms`, total `5224.4-6556.3 ms`, TTS RTF `0.752-0.936`. README's withdrawn `17.6x`/`31.7x` rows are replaced with an output-matched `8.2-10.4x` over the naive serialized path.
- [ ] Phase 63: Make TTS RTF hold below 1.0 on a contended host, where it currently measures `1.259-1.337`. Talker generation is the only remaining term (decode is 1-6%, the code predictor is already graphed, and both hand-written talker engines are slower than upstream), so this needs a cheaper talker step: quantization, a smaller talker, or kernel work.
- [ ] Phase 64: Decide whether an early LLM trigger can ever be made safe. It would need a semantic completeness signal plus a revision path that can replace audio already emitted; a word-count gate provably cannot do it.

## Decisions

- Keep `AsrEngine` and `AsyncAsrEngine` focused on transcription.
- Add a separate `qwen_asr_vllm.agent` package for ASR -> LLM -> TTS orchestration.
- Port old app ideas, not the whole app: no wake word, robot audio device management, or Nuitka deployment.
- Use injectable LLM and TTS backends so unit tests do not require `nano-vllm` or `qwen_tts`.
- Keep sync coordinator for compatibility; add async coordinator for actual overlap.
- Real qwen-asr-vLLM/nano-vLLM/TTS timing uses process-isolated runners because same-process worker-thread CUDA paths hit backend errors.
- `qwen_tts` is not an external service in this migration; it is the Python module imported by the old local `TTSAgent` wrapper around the local Qwen3-TTS checkpoint.
- Thread service runners avoid arbitrary `to_thread` model calls, but real ASR+LLM still share Torch/nano-vLLM process state; process runners are required for the current real backend mix.
- Removed early inline ASR/LLM event-loop fallback knobs from the async coordinator; real backends should use resident thread/process runners rather than switching blocking model calls onto the event loop.
- Keep sentence-boundary flushing as the default. Use `tts_flush_chars` for latency-sensitive sessions; current real timing favors `20` over `10` on the 5s sample.
- `qwen-tts` belongs in the nano-vLLM environment with `--no-deps`; the separate `qwen3-asr` environment has an incompatible torch/torchaudio pair.
- Current runner separation supports stage overlap and request-id multiplexing. `NanoVllmStepBatchingBackend` now provides backend-specific cross-session LLM batching over nano-vLLM `add_request`/`step`, and `ProcessNanoLlmBackend` preserves that behavior across the LLM service-process boundary.
- LLM process isolation now uses only `ProcessNanoLlmBackend`.
- Current Qwen-TTS package does not expose true streaming generation: `non_streaming_mode=False` returns after whole generation and decode, matching the package note that it only simulates streaming text input.
- Qwen-TTS `talker.forward` already uses HF `DynamicCache`; the next low-risk experiment is selecting the compiled attention backend (`flash_attention_2`/SDPA) and measuring whether it helps before attempting a custom paged-KV scheduler.
- Qwen-TTS default load already selects SDPA. For this short online sample, forcing `flash_attention_2` slowed generation, so it is not the recommended backend.
- The useful internal optimization is the nested `code_predictor.generate` path: replacing the fixed 15-token HF GenerationMixin call with a local step loop improves TTS-only codec throughput, but only gives about 1.02x end-to-end gain on the current first-chunk-heavy ASR/LLM/TTS sample.
- Full single-GPU ASR+LLM+TTS on `cuda:3` is not the right target: the resident ASR, LLM and TTS footprints previously add up beyond one 40GB A100. Use `cuda:3` mainly for the TTS service while keeping process-isolated ASR/LLM on separate devices.
- Request-id batched code predictor is implemented as an opt-in wait-window scheduler. Keep the default batch window at 0ms because concurrent outer Qwen-TTS generation threads currently add more contention than the inner predictor batching saves on the tested two-fragment sample.
- Explicit outer `talker.generate` replacement is implemented as an opt-in prototype. It runs correctly on the local Qwen-TTS model, but is slower than the package HF generate path in current Python form, so keep it disabled by default.
- Torch compile step-engine is implemented as an opt-in prototype. Real Qwen-TTS predictor compile showed large first-run compile/recompile cost and is not suitable for the current realtime path.
- Optimize the dominant inner code predictor before the outer talker. Establish a fixed-shape static KV-cache step interface first; only add CUDA Graph or custom sampling kernels after real parity and timing checks show the static engine is viable.
- Phase 23 validates the static engine and keeps it opt-in via `--tts-static-code-predictor`. Continue with kernel/graph work only against fixed-work A/B controls, because stochastic codec EOS timing can otherwise hide regressions.
- Phase 24 measured sampling at about 3.5% of a static predictor call, with candidate-domain PyTorch saving under 0.7% of predictor time. Do not add a custom sampling kernel; move effort to CUDA Graph replay of model steps.
- Exact codec parity takes precedence over fixed-full-extent outer graph capture. The selected follow-up is a preallocated active-prefix cache that exposes only live K/V views and matches DynamicCache mask/GQA dispatch before any new kernel work.
- ActivePrefixCache implementation is authorized under a strict stop gate: no hidden contiguous K/V copy, and any stride-driven attention-dispatch divergence requires a separate adapter design.
- Phase 25 keeps CUDA Graph predictor opt-in and batch-1 focused. Capture occurs in resident backend warmup; batch sizes above the graph limit fall back to static eager. Do not report capture-inclusive first-call profiler timing as online latency.
- ActivePrefixCache stability acceptance passed on CUDA 2: 150 alternating batch-1/2 requests over 600.545s, exact codec hashes for every request, real batch-2 left padding, stable existing cache backing addresses, zero cache errors/overflows, and only 16 MiB reserved-memory growth. The next optimization remains a separately designed varlen/paged-attention adapter; no stock StaticCache shortcut is enabled.
- Triton paged outer decode now reads request-owned KV pages directly and fixes the heterogeneous-padding codec divergence: CUDA 2 full-codec A/B is 4/4 frame exact for both requests. The final repeated median measures `1.264x` generation and `1.260x` codec-plus-audio versus ActivePrefix; raw samples remain in the report because GPU scheduling variance is visible, so the path remains opt-in while launch/projection overhead is optimized.
- Phase 31 moves decode KV writes into the same Triton launch as paged attention and commits request sequence lengths once after the complete model forward. The Qwen cache stages page table, valid mask, kernel extent, and page count once per bind tick; the per-layer path uses trusted metadata without repeated host synchronization or full page validation. CUDA 2 full-codec A/B with the kernel path and QKV fusion disabled remains 4/4 frame exact and measures `1.081x` generation (`1471.25 ms -> 1361.25 ms`) and `1.078x` codec-plus-audio (`1494.22 ms -> 1386.03 ms`). Report: `results/qwen_tts_outer_varlen_full_codec_kernel_metadata_ab_cuda2.json`.
- QKV projection fusion is implemented as an explicit `--fuse-qkv` experiment. On the same CUDA 2 probe it measures only `1.021x` generation and `1.020x` total, while the long request diverges from frame 1 because the wider half-precision GEMM has different rounding/algorithm behavior. It is therefore not enabled by default and must not be used for exact-parity production output. Report: `results/qwen_tts_outer_varlen_full_codec_kernel_metadata_qkv_ab_cuda2.json`.
- A CUDA 2 tile sweep showed that isolated kernel timing does not predict full model timing: `BLOCK_N=32` reduced end-to-end gain to `1.026x`, while the exact-parity `BLOCK_N=128, num_warps=4` configuration remains the best measured full-chain default at about `1.08x`. Tile and warp settings remain explicit CLI tuning knobs.
- CUDA 0 stage profiling identifies configuration before new kernels as the next priority: expose fixed ASR and nano-vLLM KV block budgets, pre-capture expected LLM graph batches, and make the parity-validated inner predictor CUDA Graph plus unified native batching an explicit production profile. Do not make experimental outer graph/QKV fusion defaults.

## Outer CUDA Graph Follow-up

- Added shape-keyed `StaticCache` CUDA Graph bundles with explicit CUDA-only
  behavior, post-prefill lazy capture, replay metrics, and backend/profiler
  flags. Repeated CUDA 2 timing measured `1.232x` steady-state over static
  eager (`1187.52 ms -> 963.74 ms`); capture-inclusive first calls are
  reported separately.
- Optional StaticCache QKV+SDPA fusion adds about `1.022x` over graph replay
  but changes the greedy audio hash, so it remains opt-in and is not an
  exact-parity default.

## Projection Error and Custom Kernel Follow-up

- Added per-layer max-absolute/relative-L2/top-1 calibration and a strict
  startup tolerance gate, plus a single-launch FP32-accumulation Triton QKV
  projection prototype. Real CUDA 2 calibration measured `1.953125e-3` max
  absolute error and `5.28e-5` relative-L2; the custom kernel measured
  `1637.91 ms` steady-state versus `943.18 ms` for the torch-fused graph.
  Keep exact unfused graph as default and retain both fusion mechanisms as
  explicit experiments.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| `git log` failed because `master` has no commits | Baseline check | Work in current checkout and avoid destructive git operations |
| WebSocket test failed with `Cannot call "receive" once a disconnect message has been received` | First voice route implementation | Added explicit `websocket.disconnect` break and final session close |
| Full real timing failed with `ModuleNotFoundError: qwen_tts` | First real full-chain attempt | Repaired nano-vLLM env with `qwen-tts==0.1.1`, `onnxruntime`, and system `sox`; removed real-mode stub TTS fallback |
| Default Python real timing failed with `flash-attn is required` | Real ASR attempt | Used `/opt/conda/envs/nano-vllm/bin/python` |
| nano-vLLM graph capture failed in real timing | Real LLM attempt | Used `LLM_ENFORCE_EAGER=true` |
| Worker-thread ASR feed hit torch/FX/dynamo errors | Real async committed attempt | Removed benchmark event-loop workaround and made real async timing process-isolated |
| Combined static-predictor wiring patch missed an import anchor | Phase 23 integration edit | Re-read exact file sections and split the patch by file-specific anchors |
| `StaticCache.reset()` rejected an inference tensor update | First Phase 23 CUDA 3 probe | Add a regression test and move cache reset inside `torch.inference_mode()` |
| CUDA Graph profiler warned that captured graph was empty | First Phase 25 real profile on cuda:2 | Bind capture and replay to the input tensor's CUDA device; reject the apparent stale-logit timing |
| Initial outer-model source path did not exist | Phase 27 architecture review | Located the installed source under `qwen_tts/core/models/modeling_qwen3_tts.py` and continued read-only analysis |
| Findings patch did not match its anchor because the command contained a misspelled `active-prefix` phrase | CUDA 0 Task 1 notes | Reissued a narrowly anchored patch with the exact existing line; no source code was affected |
| New nano-vLLM allocator test caused pytest collection to import `/home/nano-vllm/bench` instead of the repository `bench` package | CUDA 0 Task 1 RED | Changed the test path setup from `sys.path.insert(0, ...)` to append, preserving repository import precedence |
| Report-writing patch split `require_idle_cuda()` and moved its utilization check into `_write_report()` | CUDA 0 Task 4 | Reproduced the missing utilization exception, traced the bad patch anchor, and moved only that condition back into the preflight function |
| Synthetic `StaticCache` inspection config lacked `get_text_config` | Phase 27 architecture review | Inspected `StaticLayer.update` directly; no runtime code or model state was affected |
| Thread service runners still hit torch/FX/dynamo in real ASR+LLM timing | Real async service attempt | Kept thread runners for lightweight/test backends; real timing now always isolates ASR, LLM and TTS by process |
| Process-isolated async still had late first audio | Real async committed timing | Added configurable `tts_flush_chars` so TTS can start before sentence punctuation |
| `qwen_tts` existed only in a broken qwen3-asr environment | Real TTS import | Installed `qwen-tts==0.1.1` and `onnxruntime` into nano-vLLM with `--no-deps`; installed system `sox` |
| Too many early TTS fragments increased total latency | `tts_flush_chars=10` timing | Added `tts_coalesce_chars` to merge queued fragments while TTS is busy |
| Real nano-vLLM step runner hit `Expected all tensors to be on the same device, cuda:2 and cuda:0` | First true threaded `engine.step()` probe | Set the CUDA current device inside the resident step runner thread before calling nano-vLLM; nano-vLLM uses thread-local `.cuda()` defaults |
- [x] Phase 65: Clean up redundant metrics and placeholders, and establish whether the pipeline overlaps LLM compute with speech. Removed three dead/incoherent fields (`tts_emit_rtf`/`tts_emit_span_ms` written but never read and disagreeing with the summarizer's definition; `tts_codec_tokens_per_s` dividing whole-reply tokens by first-chunk latency, overstating throughput ~36x; plus two unused codec estimates), replaced the hardcoded `INPUT_AUDIO_MS` with a recorded `input_audio_ms`, and fixed `--realtime-input` pacing that made feed wall 1.337x of audio. ASR is fully overlapped with speech (feed ratio 1.003); LLM is serial after `asr_final` by design and costs only 3.7% of the post-speech wait against TTS's 96.3%.

## Phase 66 - Cross-subsystem pipelining: what actually pays (complete)

Asked whether the subsystems, not just the models, are pipelined, and whether the
LLM should prefill on speech already spoken while deferring its output.

- [x] Quantify the speculative-prefill ceiling. Prefill is `~35.5 ms` of a
      `6990.6 ms` post-speech wait, `0.51%`. Decode is conditioned on the full
      prompt by definition and cannot move. Rejected: ASR revises committed
      prefixes (measured `"Montfichet"` -> `"Montfiche"`), so the cache needs
      revision detection and re-prefill, which is not worth `35 ms`.
- [x] Find the defect that RTF was hiding. Playback starting on the first chunk
      starves `809.7 ms` in and ends `1141.8 ms` in deficit whenever TTS runs
      above RTF 1.0, so the agent stutters through the whole reply.
- [x] Add `tts_playback_preroll_ms` and verify it paired. Gapless at 1200 ms for
      `1678.2 ms` of first-audio latency, identical prompt and total audio.
      Defaults to off: unnecessary on a quiet host, and the required preroll
      grows with reply length.
- [x] Surface `playback_worst_buffer_ms` / `playback_gapless` in the summarizer.

## Phase 67 - Remaining cross-subsystem work (open)

- [ ] Use the idle cards. ASR and the LLM are finished and their GPUs idle for
      the 96.5% of the post-speech wait that TTS occupies, while TTS uses one
      card. Sentence-level fragments across devices should be near-linear for
      multi-sentence replies. Needs a multi-sentence fixture first; boundaries
      must stay at sentence level because sub-sentence splitting inflates total
      audio by 19%.
- [ ] Phase 63 still governs. Preroll only mitigates short replies; RTF below
      1.0 is the only fix that scales with reply length.

## Phase 68 - Press the talker step cost directly (diagnosed, no fix yet)

- [x] Profile the step instead of averaging it (`bench/tts_talker_step_probe.py`).
      `85.4 ms` per step for `80 ms` of audio against a `3.11 ms` memory roofline:
      `27x` off, so RTF `1.42` is a software result, not a hardware ceiling.
- [x] Separate the two halves with `--sync-each-step`. `talker_model_forward` is
      `48 ms` of CPU time unchanged by a device sync, so it is launch-bound
      (`1953 cudaLaunchKernel` per step). `code_predictor_generate` is `15.8 ms`
      CPU / `38.9 ms` synced, so it is GPU-bound and already graphed.
- [x] Rule out quantization. GPU time alone is `18x` the bandwidth roofline, so
      weight bytes are not the constraint. This corrects the earlier note that
      listed quantization first.
- [x] Establish the code predictor graph is worth `1.9x` (RTF `1.250` on,
      `2.398` off) and must never be disabled.
- [x] Test `torch.compile` on the talker forward: `1.004x` at mode `default`,
      `0.98x` at `max-autotune-no-cudagraphs`, output-identical. No help, likely
      graph breaks from the HF `Cache` path. `reduce-overhead` cannot run at all
      (inference tensors), and compiling while the code predictor graph is live
      fails on RNG offset state.
- [x] Test tight `max_cache_len` on the hand-written engines. Recorded a `1.30x`
      win for `active_prefix` @ 160, then **failed to replicate it**: the same
      parameters measured `2.10x slower` in a second bracketed window, and 160
      through 1024 are flat. `active_prefix` is bimodal and stays rejected.

Net: the step is diagnosed and four independent attempts to cut it have failed
(`active_prefix`, `cuda_graph` talker, `torch.compile`, tight cache). Untried
levers that match the diagnosis:

- [ ] Attack the code predictor's 15 sequential codebook sub-steps, now the
      GPU-bound floor once the talker half is fixed.
- [ ] Hand-fuse the talker decode step (RMSNorm/rotary/elementwise) so kernel
      count drops without relying on Inductor or a static cache.
- [ ] Explain `active_prefix` bimodality; a component that varies `2.7x` under
      fixed parameters makes every other TTS measurement noisier.

## Phase 69 - Cleanup and SOTA regression (complete)

- [x] Confirm the talker phase changed no production code, so there is nothing to
      roll back from it.
- [x] Remove the dead compile probe and one-off sweep drivers.
- [x] Fix the real defect it surfaced: `cuda0-throughput` shipped
      `tts_outer_active_prefix_talker_engine: True`, and a test asserted that
      value. Both locked in a measured regression. New test asserts no profile
      enables either hand-written outer talker engine. Suite `630 passed`.
- [x] Re-measure the speedup with semantics checked, paired, naive bracketing:
      first audio `3.37-4.41x`, total turn `1.22-1.59x`, with all arms prompting
      the LLM with the complete transcript and producing identical replies.
- [x] Stop `playback_gapless` from flattering non-streaming arms; it now needs at
      least two chunks.

## Phase 70 - Cloud-edge simulation (assessed, blocked)

- [x] Measure the link to the intended 4090 edge: RTT `0.252 ms`, jitter
      `0.016 ms`, 0% loss. SSH refused on 22 and five other common ports.
- [x] Assess whether the topology attacks the bottleneck. It does not: ASR
      already overlaps speech at `1.004` and contributes nothing to the
      post-speech wait, which is `96.5%` TTS, and TTS stays in the cloud.
- [ ] **Blocked on access.** Cannot validate: host answers ICMP but no reachable
      sshd. Needs sshd started or the real port.
- [ ] Measure ASR RTF on the 4090. The only genuinely unknown quantity in the
      proposal; the A100 does `0.33` and the edge must stay below `1.0`.
- [ ] Inject `tc netem` delay and jitter before quoting any latency number, since
      `0.25 ms` is 40-200x faster than the WAN being modelled.
- [ ] Put the committed-prefix divergence check in the wire protocol; the prefix
      diverged from the final transcript at word 11 of 21.

## Phase 70 - Cloud-edge simulation (unblocked, measured)

- [x] SSH is on port 1212. Two idle 4090s and the 0.6B ASR checkpoint are in place.
- [x] Stream PCM to the edge over a JSON RPC (`RemoteAsrEngine` / `serve_edge_asr.py`).
- [x] 4090 ASR holds RTF `1.000` under `--realtime-input`.
- [x] Paired against local paced ASR: first audio `9023` vs `9082` ms, identical
      prompt and TTS bytes. TTS still starves. Topology does not attack the bottleneck.
- [ ] Inject `tc netem` before quoting this as a WAN result (`0.25 ms` is not one).

## Phase 71 - Put TTS on the edge (measured)

- [x] Capacity: 4.3 GB disk, ~5.1 GB VRAM on a 24 GB 4090. Fits with ASR on the
      sibling card (6.1 GB) or even on the same card.
- [x] Isolated 4090 RTF **0.752** on the output-matched sentence.
- [x] E2E edge ASR+TTS / cloud LLM: TTS RTF **0.747**, gapless, first audio after
      stop **498 ms** vs 771 ms with cloud TTS. Semantics unchanged.
- [ ] Play on the edge without shipping wav back (true speaker path).
- [ ] `tc netem` on the text/token path only, now that audio need not recross.
