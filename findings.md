# Voice Agent Streaming Migration Findings

## Prior Implementation

- `/home/voice_assistant_app` is the relevant old voice assistant app.
- `/home/nano-vllm/nanovllm/qwen_rare_asr_agent` is an ASR rare-word correction pipeline, not the ASR/LLM/TTS assistant.

## Useful Files From `/home/voice_assistant_app`

- `src/agents/pipeline.py`: ASR -> streaming LLM -> sentence-level parallel TTS orchestration.
- `src/agents/asr_agent.py`: ASR backend wrapper around `qwen-asr-vllm`.
- `src/asr_stream.py`: growing-prefix ASR wrapper and live utterance helper.
- `src/llm_backend.py`: `NanoVllmBackend.chat_stream()` and HF fallback.
- `src/agents/tts_agent.py`: Qwen3-TTS wrapper, blocking sentence synthesis.
- `server.py`: WebSocket endpoint receiving whole WAV bytes and returning events/audio.

## Current Repo State

- Current repo already has `AsyncAsrEngine.open_stream()` and `StreamingSession`.
- Current HTTP server only exposes `/v1/audio/transcriptions`, `/health`, `/metrics`, `/v1/models`.
- README states streaming sessions are a Python API for agent integration.
- There is no current `qwen_asr_vllm.agent` package, LLM adapter, TTS adapter, or voice WebSocket endpoint.

## Migration Boundary

- First migration should create a runtime layer that can overlap stages and be tested with stubs.
- Model-specific streaming improvements for real TTS can be added behind the same `TtsBackend` interface later.

## Qwen-TTS Internal Optimization

- Installed Qwen-TTS model code lives under `/opt/conda/envs/nano-vllm/lib/python3.11/site-packages/qwen_tts/`.
- `Qwen3TTSModel.from_pretrained()` accepts Hugging Face model kwargs, including `attn_implementation`.
- Local Qwen3-TTS config has `talker_config.use_cache=true` and `code_predictor_config.use_cache=true`; `_attn_implementation` is unset in both configs before loading.
- `Qwen3TTSTalkerAttention.forward()` updates `past_key_values` through HF `DynamicCache`, so the current package is not recomputing the full KV history each codec step.
- Each generated codec frame still runs a nested path: a 5-layer `code_predictor.generate(... max_new_tokens=num_code_groups-1)` call plus the 28-layer talker decoder step. That makes Python GenerationMixin overhead and inner codebook prediction likely bottleneck candidates, not only attention cache layout.
- Flash attention is installed in the nano-vLLM env (`flash_attn==2.8.3`), so `attn_implementation="flash_attention_2"` can be measured as a low-risk backend swap before deeper paged-attention style work.
- Real attention backend comparison on `cuda:1`, text `"I'm here to help! What can I do for you?"`: default resolved to SDPA and generated 29 codec frames in 5017.8ms; forced `flash_attention_2` generated 26 frames in 5990.7ms. Flash attention was slower in this Qwen-TTS decode shape.
- Real internal batch profile with default SDPA: batch=1 generation 5053.6ms, 5.54 codec frames/s; batch=2 generation 7280.9ms, 10.71 frames/s; batch=4 generation 8243.3ms, 17.95 frames/s. Decode stayed small: full decode 56-95ms.
- The same profile split shows `code_predictor.generate` dominates: batch=1 mean 124.9ms/step and 69.2% of generation; batch=4 mean 126.0ms/step and 70.3%. Main `talker.model.forward` is about 21-25%.
- Synchronized batch=1 profile confirmed the split: `code_predictor.generate` 68.2%, `talker.model.forward` 21.6%.
- Fast code predictor prototype (`qwen_asr_vllm.agent.qwen_tts_fast_predictor`) replaces the fixed inner HF generate call with a local step loop. TTS-only profile improved code predictor mean step from about 125ms to 87-89ms and codec throughput from 5.54 to 7.05 frames/s at batch=1, and from 17.95 to 24.82 frames/s at batch=4.
- Full process-isolated ASR/LLM/TTS timing with `LLM_TEMPERATURE=0`, codec-step streaming, first chunk 3, chunk size 8, coalesce wait 80ms: fast predictor first audio 2210.4ms and total 6042.1ms; no-fast control first audio 2257.5ms and total 6174.1ms. End-to-end gain is about 1.02x because first audio needs only a few codec frames and the remaining latency includes ASR/LLM/scheduling.

## CUDA 3 Test

- GPU3 still has a driver-level `[Not Found]` context using about 3.8GB and showing nonzero utilization, but it has about 36.5GB free and can run the local Qwen-TTS model.
- TTS-only default profile on `cuda:3`: batch=1 generated 54 codec frames in 7670.1ms, 7.04 frames/s; batch=4 generated 125 total frames in 6522.3ms, 19.17 frames/s. `code_predictor.generate` remained about 70% of generation.
- TTS-only fast predictor profile on `cuda:3`: batch=1 generated 30 frames in 5063.5ms, 5.92 frames/s; batch=4 generated 157 total frames in 8736.8ms, 17.97 frames/s. This run was not faster than default, likely due to GPU3 contention/orphan utilization and sampling length variance.
- Full async timing with `TTS_DEVICE=cuda:3`, default ASR/LLM devices, `LLM_TEMPERATURE=0`, codec-step streaming, first chunk 3, chunk size 8, coalesce wait 80ms, fast predictor: first audio 2301.6ms, total 5346.6ms, no errors.
- CUDA3 full-chain result is comparable to the latest cuda1 TTS result for first audio and faster in total on this sample, but it should not be treated as a clean hardware comparison because all GPUs currently report driver-level orphan contexts.

## Request-ID Batched Code Predictor

- Added a batched fast predictor scheduler that intercepts concurrent `code_predictor.generate()` calls and batches compatible predictor steps across request ids.
- The scheduler concatenates `inputs_embeds` for prefill steps and converts HF `DynamicCache` to/from legacy KV tuples to concatenate/split cached decode steps along batch dimension.
- Unit coverage verifies two concurrent request ids are merged into batch size 2 for every predictor step.
- Runtime integration keeps `fast_code_predictor=True`, but the batch wait window defaults to `0ms` to preserve the prior single-request fast path. Request-id batching is enabled explicitly through `--tts-fast-code-predictor-batch-window-ms`.
- Real full-chain test on `cuda:3` with `tts_process_stream_workers=2`, no static TTS batch, fast predictor batch window 8ms, max predictor batch size 2: first audio 2650.3ms, total 6444.3ms. This was worse than the single-worker cuda3 fast run (2301.6ms / 5346.6ms), likely because concurrent outer Qwen-TTS `talker.forward`/decode on one model added contention that outweighed inner predictor step batching.

## Explicit Outer Talker Step Engine

- Added an opt-in replacement for Qwen-TTS `talker.generate` that preserves upstream prompt/speaker/text preparation and replaces only the final HF GenerationMixin codec loop.
- The explicit loop performs one prefill `talker.forward`, then iterates codec first-codebook sampling through `talker.forward(input_ids=...)`. It returns the same `hidden_states` shape consumed by `Qwen3TTSForConditionalGeneration.generate`.
- Real compatibility issue found and fixed: passing an expanded full attention mask during decode doubled the effective causal-mask length with HF `DynamicCache`. The explicit loop now uses the prompt attention mask only for prefill and lets cached decode steps use cache position without a full mask.
- TTS-only real profile on `cuda:3` with fast predictor plus explicit outer engine: generation 8750.6ms for 34 codec frames, 3.89 frames/s. This is slower than the existing HF outer generate path on the same setup.
- Full async real timing on `cuda:3` with fast predictor plus explicit outer engine: first audio 2453.7ms, total 7356.0ms, no errors. This is slower than the latest single-worker fast path, so the explicit outer engine remains an experimental opt-in flag.

## Compiled Step Engine

- Added `--tts-compile-step-engine` / `--tts-compile-step-engine-mode` to compile the fast code predictor callable, and the explicit outer talker forward callable when the explicit outer engine is enabled.
- Unit tests use an injected compiler callable to verify both predictor and outer talker forward paths route through the compiled callable.
- Real TTS-only profile on `cuda:3` with fast predictor plus compiled predictor: generation 101997.3ms for 29 codec frames. The first compiled `code_predictor_generate` step took about 95.4s and TorchDynamo hit `config.cache_size_limit`, indicating recompilation/graph specialization churn.
- A profiler run with compiled explicit outer engine was interrupted after graph breaks caused by profiler hooks (`time.perf_counter`) and long Inductor compilation. Full-chain compilation was not run because the predictor-only result already shows the compile path is unsuitable for realtime startup latency without a separate ahead-of-time warmup/cache strategy.
- Conclusion: the compile path is available for further experiments, but should stay disabled by default. The next viable lower-level route would require static-shape cache management/CUDA graph capture or custom kernels, not plain `torch.compile` around HF wrapper callables.

## Static Code Predictor Engine

- The installed Transformers version exposes `StaticCache(config, max_cache_len, ...)` with `reset()`, `update()`, `get_seq_length()`, and `get_max_cache_shape()`; cache tensors are initialized lazily by the first model update.
- Qwen-TTS code predictor accepts the generic Transformers `Cache` interface, so `StaticCache` can be introduced without patching the installed package. The engine still needs explicit `cache_position` values for the two-token prefill and each one-token decode step.
- This phase will keep sampling in PyTorch initially. Stable cache storage and step shapes are prerequisites for a meaningful CUDA Graph or fused sampling-kernel experiment; combining all changes at once would make parity and timing regressions untraceable.
- First real CUDA 3 probe reached the model but failed on the second cache reuse: `StaticCache` backing tensors were allocated as inference tensors, while the engine called `reset()` before entering `torch.inference_mode()`. The fix must keep reset and all in-place cache updates inside the same inference-mode lifecycle.
- After moving reset into inference mode, the real CUDA 3 static path completed: 17-character text, 12 codec frames, 3173.2ms generation, and 95.3ms mean inner predictor time. This proves model/cache compatibility but is not yet a speedup claim because sampling and frame count were not controlled against a same-seed baseline.
- Fixed-work CUDA 3 A/B with seed 7 and outer `max_new_tokens=8` produced 7 codec frames in both paths. Fast DynamicCache loop: 3217.5ms generation, 167.3ms mean predictor call, 2.18 frames/s. StaticCache engine: 2473.1ms generation, 95.9ms mean predictor call, 2.83 frames/s. Observed gain is 1.30x generation and 1.74x inner predictor latency, despite GPU 3's pre-existing 99% orphan-context utilization.
- Reverse repeat remained favorable but showed contention variance: fast 3261.6ms / 176.5ms predictor mean versus static 2793.8ms / 121.4ms. Across two fixed-work runs, mean generation improved 3239.6ms to 2633.5ms (1.23x) and mean predictor step improved 171.9ms to 108.7ms (1.58x). Use these averaged ratios as the conservative CUDA 3 result.
- Same-environment full-chain control on CUDA 3 kept ASR hypothesis, LLM output, TTS input fragments, and first audio bytes identical. Fast predictor: first audio 2586.6ms, total 6691.1ms, TTS compute wait 917.2ms. Static predictor: first audio 2279.9ms, total 5537.8ms, TTS compute wait 648.9ms. Gains are 1.13x first audio, 1.21x total, and 1.41x isolated first-fragment TTS compute wait.

## Sampling Optimization

- The installed environment has Torch 2.5.1+cu121 and Triton 3.1.0. Qwen-TTS uses 16 code groups; relevant vocabulary sizes in the local config are 2048 and 3072.
- Current stochastic sampling computes top-k thresholding and then runs softmax/multinomial over the full vocabulary even though online settings keep only 50 candidates. Phase 24 first measures a candidate-domain PyTorch sampler before considering a custom Triton implementation.
- CUDA 3 microbenchmark at batch 1/vocab 2048 measured 0.2513ms for the current sampler and 0.2013ms for candidate-domain sampling (1.25x sampling-only). Across 15 codebook steps this saves only about 0.75ms, under 0.7% of the measured 108.7ms predictor call.
- Batch 1/4 and vocab 2048/3072 results stayed in the same range: current 0.243-0.251ms, candidate 0.197-0.216ms. Even a zero-cost ideal sampler caps predictor gain near 3.5%, so a custom Triton sampling kernel is not justified at this stage.

## CUDA Graph Predictor

- Real feasibility probe on CUDA 2 captured the fixed batch-1 predictor as one prefill graph plus 14 decode graphs sharing a graph memory pool and one `StaticCache`.
- For captured input shape `[1, 2, 2048]`, the 15-token greedy output matched static eager exactly: `[1159, 355, 22, 1174, 1093, 625, 1814, 1058, 905, 1846, 1247, 1677, 889, 812, 901]`.
- Production prototype should keep graph bundles shape-specific and serialized, hide capture in the existing backend warmup, and fall back to static eager for non-CUDA or batches above the configured graph batch limit.
- First integrated CUDA 2 profile emitted `The CUDA Graph is empty`: model tensors were on CUDA 2 while the process current device remained CUDA 0. The apparent 3.8ms steady-state predictor timing is invalid because replay could read capture-time logits. Capture and replay must explicitly enter the input tensor device context.
- After binding capture/replay to the input device, the warning disappeared. Fixed 7-frame CUDA 2 steady-state generation was 517.1ms with graph replay versus 1169.2ms static eager (2.26x). Codec throughput rose from 5.99 to 13.54 frames/s.
- Graph capture costs about 4.2s on the first predictor call. It must occur during resident TTS backend warmup; enabling it with `warmup=False` would directly add this cost to the first user request.
- Unsynchronized nested hook time (4.5ms graph predictor versus 114.4ms eager) mainly reflects CPU launch overhead, not complete GPU execution. Use synchronized whole-generation timing for the 2.26x claim.
- CUDA 3 also fit the graph bundle despite only about 7.9GB free: steady-state fixed 7-frame generation was 537.4ms.
- Same-parameter full-chain CUDA 3 A/B kept ASR hypothesis, LLM output, TTS input fragments, and first audio bytes identical. Graph: first audio 1999.3ms, total 3583.5ms, TTS compute wait 513.0ms. Static eager: 2297.9ms, 6131.4ms, and 726.9ms. Gains are 1.15x first audio, 1.71x total, and 1.42x isolated first-fragment TTS compute wait.

## Fixed-Slot CUDA Graph Request Batching

- The graph engine now pre-captures batch-1 and the configured fixed slot shape during resident TTS warmup. A request-id scheduler concatenates compatible complete code-predictor calls, replays one fixed-shape graph, and demultiplexes sequence rows back to callers.
- CUDA 3 fixed-work predictor probe with two request ids and 30 calls each: slots=1 took 1655.8ms (36.24 requests/s), while slots=2 took 913.9ms (65.66 requests/s). All 30 scheduler batches were batch-2, giving 1.81x predictor throughput.
- Independent outer talker threads do not automatically realize this gain. With a 3ms admission window they frequently miss each other; with 10ms they align but lose outer-model pipeline overlap. Greedy two-session wall time was 5.21s for slots=1 versus 5.53s for independent-thread slots=2, a 6% regression.
- Native outer batch-2 plus the batch-2 graph is the useful service schedule. Controlled greedy profiles generated exactly 41 codec frames per utterance: batch-1 took 2.761s at 14.85 frames/s, while batch-2 took 3.351s for two utterances at 24.47 frames/s. Aggregate throughput improved 1.65x, while per-utterance completion latency rose about 21% as the expected batching trade-off.
- Runtime policy keeps predictor admission at 0ms by default. When CUDA Graph fixed slots exceed one, the process TTS service automatically enables matching request workers, a one-time 10ms outer-request admission window, and a service batch size capped to the graph slot count.

## Outer Talker Static Engine Design Notes

- `Qwen3TTSTalkerModel.forward` already accepts generic Transformers `Cache` objects and explicit `cache_position`; its attention path forwards `cache_position` into cache updates, so an outer static-cache prototype does not require patching the installed Qwen-TTS package.
- The outer decode state is larger than KV alone: every session also owns `past_hidden`, `generation_step`, `trailing_text_hidden`, generated first-codebook history, and EOS state.
- `Qwen3TTSTalkerForConditionalGeneration` stores `rope_deltas` on the shared model object. Independent concurrent calls can overwrite it, so a true multi-session engine must move rope delta into request state rather than relying on the module attribute.
- Prefill is variable-shape because text/prompt lengths differ. CUDA Graph capture should initially target only the one-token decode step; prefill can remain eager and populate static cache before decode replay.
- Standard `StaticCache` and Transformers causal-mask construction assume one shared decode position tensor for the batch. Arbitrary slot refill with sessions at different cache positions therefore needs either position-bucketed scheduling or a custom per-slot cache/attention path; it cannot be safely represented by merely masking finished rows in the current HF forward.
- `StaticLayer.update` uses `index_copy_(2, cache_position, key_states)`, confirming that stock static cache writes the same sequence positions for every batch row. Transformers documents `cache_position` as `(query_length,)`, not one position per slot.
- A prebuilt 4D attention mask is accepted directly and bypasses generic causal-mask creation. This makes fixed-cohort decode capture feasible, but it does not solve per-slot KV writes for asynchronously positioned sessions.
- Controlled continuous batching therefore has two implementation levels: safe cohort compaction using batch-1/batch-2 graph bundles, or a deeper custom slot cache plus attention adapter for arbitrary refill. The latter is effectively a model-engine change comparable to adding a small paged-attention runtime.

## ActivePrefixCache CUDA 2 Acceptance

- Preflight used physical CUDA 2: NVIDIA A100-SXM4-40GB, UUID `GPU-51cd870f-27fe-4676-4d75-8d9474ec9763`, PCI `00000000:B1:00.0`, `16 MiB / 40960 MiB`, 0% utilization, and no compute process. Other GPUs were left untouched.
- The accepted probe ran twice in fresh processes with the local `Qwen3-TTS-12Hz-1.7B-CustomVoice` checkpoint, greedy seed 7, `max_new_tokens=64`, and batch sizes 1 and 2. Both reports have `status=completed`, `parity_passed=true`, zero warnings, and zero errors.
- Both batches matched five outer boundaries including every layer's K/V values and raw layouts, last hidden states, raw codec logits, processed logits, sampled first-codebook IDs, complete predictor/raw codec tensors, hashes, EOS positions, and decoded audio sample counts. Candidate K/V prefix views were non-contiguous but bitwise equal; no model-facing `.contiguous()` copy was added.
- First run timings (upstream -> active, milliseconds): batch 1 generation `2013.385 -> 1272.315` (`1.582x`), audio decode `407.201 -> 16.619` (`24.502x`); batch 2 generation `1456.576 -> 1324.126` (`1.100x`), audio decode `88.641 -> 16.408` (`5.402x`).
- Fresh repeat timings: batch 1 generation `2101.794 -> 1325.454` (`1.586x`), audio decode `455.067 -> 15.864` (`28.686x`); batch 2 generation `1428.560 -> 1342.587` (`1.064x`), audio decode `99.690 -> 16.053` (`6.210x`).
- Active-prefix is exact and improves generation on this workload, but batch-2 gain is modest and requires broader load testing. Decode timing is strongly affected by first-call/warmup behavior and is not the main end-to-end claim.

## 2026-08-30 Incremental Codec Buffer, Parity Gate, And Load Metrics

- The installed Qwen-TTS tokenizer exposes `decoder.chunked_decode()` but no public persistent waveform-decoder state. A true convolution/Transformer decoder-state implementation would require changing the package model and validating boundary samples. The local runtime therefore adds an exact-preserving growable codec frame buffer; each decode call receives only the generated prefix and avoids repeated `np.stack(all_frames)` allocations.
- Added `tts_first_sentence_immediate`: when enabled, the first segment of each LLM generation bypasses the coalesce wait, while later segments still merge under the configured character limit. Default behavior is unchanged.
- Added a runtime native-batch parity gate. Prompt-length grouping remains mandatory, and sampling requests fall back to scalar generation because their batch-local RNG state cannot be proven equivalent without duplicate generation. Gate counters are exposed in `runtime_metrics()`.
- Fixed ASR engine process-device initialization by selecting the configured CUDA device before model construction and graph capture. This repairs arbitrary `cuda:N` process startup where default-device operations could leave CPU or wrong-device tensors.
- Extended the real TTS outer-batching probe with PCM duration, estimated 12Hz codec frames, 16-codebook codec token/s, request/s, and optional concurrent request-count sweep.
- CUDA 3 real TTS, same prompt-length two-request cohort, `max_new_tokens=16`, greedy decoding, no inner graph: serial `4051.4 ms`, native batch `2158.3 ms` (`1.88x`), unified IPC `2703.0 ms` (`1.50x`). Codec token throughput was `91.0 -> 170.8 token/s` for native batch; the four-request service sweep reached `271.2 codec token/s`.
- CUDA 3 real TTS with different prompt lengths correctly triggered two scalar fallback groups; native throughput was `0.985x` and unified IPC `0.854x` versus serial. This is an intentional parity result, not a failed batch launch.
- Full ASR -> LLM -> TTS rerun used ASR CUDA 0, LLM CUDA 2, TTS CUDA 3, process isolation, and greedy `max_new_tokens=16`. It completed without errors with ASR hypothesis `You will be frank with me. I always am.` and LLM output `I'm here to help! What can I do for you?`; the observed first audio was `4812.1 ms` and total drain `9515.5 ms`. This single-chain result is not compared as a speedup against earlier runs because GPU placement and batching policy differed.

## 2026-09-06 CUDA 0 Optimization Audit

- CUDA 0 is the only clean card at audit start: `16 MiB / 40960 MiB`, `0%` utilization. CUDA 1-3 carry external `[Not Found]` contexts and high utilization, so all new stage baselines must run serially on CUDA 0.
- Existing coverage is sufficient for stage attribution: `profile_stages.py` separates ASR frontend/conv/audio-transformer/projector/prefill/decode, `nano_llm_batching_probe.py` exercises request-id LLM batching, `qwen_tts_internal_profile.py` separates talker/predictor/codec decode, and `voice_agent_timing.py` records end-to-end event offsets.
- There are no prior CUDA 0 JSON reports. CUDA 2/3 absolute latency and speedup values are not valid CUDA 0 baselines.
- Unbudgeted same-GPU colocation is unsafe because ASR and nano-vLLM independently size KV cache from currently free memory. Stage profiles will first run one resident engine at a time; a same-GPU full chain requires explicit cache/memory quotas rather than simply changing all three device strings to `cuda:0`.
- CUDA 0 ASR 0.6B steady-state profile: 2s audio `60.8 ms` wall / `41.7 ms` TTFT / `98.7 tok/s`; 10s audio `126.6 ms` wall / `40.7 ms` TTFT / `260.6 tok/s`. At 10s, text decode is `65.2%` of measured GPU time and `2.26 ms/step`; at 2s, prefill is `56.6%` and audio transformer `15.7%`.
- The ASR profiler reported `28.2 GiB` peak allocation despite a 0.6B model and tiny per-request KV use (`5.5-19.8 MiB`). Most of this is the engine's free-memory-based global KV reservation, making explicit per-service cache quotas the highest-priority prerequisite for CUDA 0 colocation.
- CUDA 0 nano-vLLM single request: `52.1 ms` first token, `125.3 ms` total, 22 streamed chunks, 24 engine steps. The first four-request run reached `max_step_batch_size=4` but showed about `491 ms` TTFT and `602.9 ms` wall. Because the probe only warms batch-1, this measurement may include lazy batch-4 graph capture and requires a same-process repeated run before throughput attribution.
- Same-process LLM repeat confirmed lazy graph capture: four-request round 1 was `647.6 ms` wall with about `509-511 ms` TTFT; round 2 was `152.3 ms` wall with `37.7-41.1 ms` TTFT and about `610.7` streamed chunks/s. The practical LLM optimization is startup pre-capture for expected batch sizes, not a new scheduler.
- CUDA 0 upstream TTS internal profile (`max_new_tokens=16`) attributed `70.2-70.3%` of generation to `code_predictor.generate` and `26.8-27.0%` to outer `talker_model_forward`; codec waveform decode was only `3-9%` of total. Batch-1 generation was `1694.9 ms` at `7.67 codec frames/s`; batch-2 was `2273.9 ms` at `12.75 frames/s`.
- Resident inner predictor CUDA Graph reduced the same profiler's batch-1 generation to `790.1 ms` and batch-2 to `925.9 ms`; predictor share fell to `4.5-6.1%`, leaving outer talker forward as the new dominant component (`65.8-66.4%`). However, codec hashes did not fully match between the two independent internal-profiler processes. Because that profiler mutates package `generate_defaults` after backend warmup, its ratio is a performance ceiling rather than a parity-grade production claim; a probe that passes greedy controls directly is required.
- The controlled greedy outer-batching probe provides the parity-grade TTS comparison. Without the inner graph, two same-length prompts took `4004.4 ms` serial, `2118.7 ms` native batch (`1.89x`), and `2604.8 ms` through the resident service (`1.54x`); all three paths emitted matching per-request PCM byte counts and durations.
- With resident batch-1/batch-2 inner predictor graphs, the native and service paths preserved those exact PCM byte counts. Native batch fell from `2118.7` to `1015.1 ms` (`2.09x`) and service wall fell from `2604.8` to `1147.7 ms` (`2.27x`); service throughput rose from `141.5` to `321.2` estimated codec token/s. The graph-run serial path emitted one extra codec frame for the first request, so no serial graph/non-graph parity claim is made.
- The inner graph recorded `47` replays, zero lazy captures, and batch-2 slot use for the native cohort. After this optimization, earlier internal attribution places outer talker forward at roughly two-thirds of generation time, making outer decode the next TTS compute target.
- The voice-agent ASR factory does not expose `gpu_memory_utilization` or `num_kvcache_blocks`, although `AsrEngine` supports both. Current deployment defaults place ASR on CUDA 0, TTS on CUDA 1, and LLM on CUDA 2. A true all-on-CUDA-0 run therefore first needs explicit per-service cache budgets; otherwise ASR alone reserves most free memory and the result depends on process startup order.
- CUDA 0 resident memory probes measured about `5.50 GiB` driver use for the warmed TTS inner-graph backend and `19.73 GiB` for nano-vLLM at `gpu_memory_utilization=0.50`. ASR with 256 fixed blocks used `7.0 GiB` of KV and about `11.63 GiB` driver memory; 128 blocks represent `3.5 GiB` of KV and are sufficient for four 8192-token maximum-length sequences. This confirms that explicit block counts, rather than free-memory fractions, are required for deterministic colocation.
- The UI and full-chain benchmark parsers currently default TTS streaming and the inner CUDA Graph off. The measured best path exists but is opt-in, so current launch defaults leave most of the demonstrated TTS gain unused.
- A later fixed-cache ASR timing attempt is excluded: a new host-side context (PID `1245711`, not visible in the container) appeared on all four GPUs and drove CUDA 0 to 100% utilization. The earlier clean-card JSON reports completed before this interference.
- Combining the byte-matched upstream serial baseline with the inner-graph unified service gives `4004.4 / 1147.7 = 3.49x` aggregate throughput for two concurrent requests. This is a two-session TTS result, not a single-session or whole-agent speedup.
- After inner predictor graph replay, batch-1 internal attribution is approximately `525 ms` outer talker forward out of `790 ms` generation. Halving outer talker cost would therefore cap the next realistic TTS gain near `1.4x`; eliminating it entirely is only a theoretical `2.5x`-class ceiling. Waveform decode is only `3-9%`, so even a free decoder is at most about a `1.1x` total improvement on these short outputs.
- The graph service/native gap is `132.6 ms` (`1147.7 - 1015.1 ms`), placing an ideal IPC/queue optimization ceiling near `1.13x` for this two-request graph workload. Shared-memory PCM transfer and fewer queue messages are worthwhile only after the production graph path is enabled.
- The UI defaults keep `tts_streaming_engine=off`, inner predictor graph disabled, exact scalar parity enabled, and one TTS worker. They also keep `defer_tts_audio_until_asr_final=true`, while the app VAD endpoint silence is `1200 ms`. For perceived first-audio latency, a confidence-gated committed-ASR release policy can save more than further ASR kernels, but it must preserve cancellation/barge-in correctness.
- Ranked next work: (1) expose fixed ASR/LLM KV block budgets and a named CUDA-0 production profile; (2) pre-capture nano-vLLM batch-1/2/4 graphs during service startup; (3) enable the parity-tested inner predictor graph and native TTS batching for compatible cohorts; (4) optimize outer talker heterogeneous-session decode with active-prefix/varlen slots, without per-position lazy graph capture; (5) reduce TTS queue/PCM-copy overhead. Codec decoder kernels and more ASR compute tuning are lower ROI.
- Implementation note: `tests/test_engine.py` is globally marked GPU/checkpoint/slow, so fixed-cache value validation belongs in a separate CPU-only `tests/test_config.py` rather than weakening that module's markers.
- nano-vLLM's `ModelRunner.allocate_kv_cache()` currently recomputes and overwrites `Config.num_kvcache_blocks` unconditionally. Propagating the value from the voice factory is insufficient unless this allocator preserves an explicit positive count.

## 2026-09-06 CUDA 0 Production Profile Implementation

- Post-profile implementation result: production ActivePrefix is now selected by `cuda0-throughput`, while varlen/paged attention remains explicit because its installer still does not replace `talker.generate`. Divergent QKV fusion and per-position outer Graph remain disabled.
- Dynamic TTS admission now groups requests by deterministic parity eligibility and tokenized prompt length. Backlogged compatible cohorts refill immediately after a native batch rather than paying another admission window. Runtime metrics expose batch sizes, queue waits, active/padded slots, refills, pending requests, and pre/during-batch cancellations.
- TTS IPC now keeps control messages on queues and transports byte payloads above `16 KiB` in the CUDA0 profile (`64 KiB` compatibility default) through shared memory. Python 3.11 and 3.13 paths both copy back to public `bytes` and unlink receiver-owned segments deterministically.
- The new full-session probe shares one resident ASR/LLM/TTS service set across each `1/2/4/8` sweep and records per-session full ASR text, LLM output, TTS inputs, canonical audio hash, p50/p95 latency, throughput, scheduler metrics, errors, and parity against the single-session baseline.
- Fake acceptance passed at every concurrency level. Real acceptance remains unmeasured: base Python lacks `flash-attn`, and physical CUDA0 returned to `2488 MiB / 100%` before rerunning with `/opt/conda/envs/nano-vllm/bin/python`.
- Post-profile implementation audit: `QwenTtsBackend` already owns mutually exclusive active-prefix, varlen paged-attention, static, and outer-graph installers plus runtime metrics. The production gap is flag/config/service propagation and scheduling policy, not a new model implementation.
- `run_real_async()` currently returns its collector on the first agent `done` even when prerecorded input feeding continues. It later closes ASR, but the collector no longer consumes `asr_final`; this explains truncated report hypotheses and invalid matched-output A/B. The timing harness needs an input-complete barrier while live WebSocket sessions should retain early `done` semantics.
- The process ASR service already multiplexes session IDs. The LLM and TTS services support request IDs, but coordinator construction resets shared LLM history and full voice-session load generation needs stateless/per-session prompt state to avoid cross-session history races.
- The graph TTS service/native gap in the fresh CUDA 0 report is about `104 ms` (`1137.0 - 1032.8 ms`). Shared memory should target PCM byte payloads only; queue-based small control messages remain simpler and preserve cancellation/error handling.
- A clean CUDA 0 window allowed final stage and residency tests. Pre-captured nano-vLLM batch-4 reached `136.1 ms` wall with `29.4-32.1 ms` TTFT. The repeated byte-matched TTS workload reached `1032.8 ms` native batch and `1137.0 ms` unified service, corresponding to `2.05x` and `2.29x` over the no-inner-graph paths; unified throughput was `324.2 codec tokens/s`.
- ASR, LLM, and TTS successfully co-resided on physical CUDA 0 with no OOM or service error. Allocated bytes at readiness were approximately `5.36 GB`, `3.12 GB`, and `4.21 GB`; final driver use was about `15.9 GB`. The committed first-turn run measured `1335.1 ms` first audio and `2512.0 ms` wall, but it completed before the rest of the prerecorded input and is not a full-file speedup control.
- Full-file ASR final text scores WER `0.0455` and CER `0.0081` against the 22-word LibriSpeech reference. Directly scoring the first committed sentence against the full two-sentence reference is invalid; its prefix WER is zero.
- Strict A/B exposed that nano-vLLM rejected temperature zero and the adapter silently changed it to stochastic `1e-5`. `SamplingParams` and `Sampler` now support request-row greedy argmax at zero temperature, and the adapter preserves zero. A post-fix matched GPU A/B remains pending because the external `2488 MiB / 100%` CUDA 0 workload returned.
- ASR and nano-vLLM now accept only automatic (`-1`) or positive fixed KV block counts. The voice app propagates `ASR_NUM_KVCACHE_BLOCKS` and `LLM_NUM_KVCACHE_BLOCKS`; nano-vLLM preserves an explicit positive count instead of overwriting it from a total-memory fraction.
- `NanoVllmStepBatchingBackend` now injects startup warmup cohorts atomically into its resident `add_request`/`step` loop. Batch sizes are positive, clipped to `max_num_seqs`, stably deduplicated, and reported separately from zeroed online statistics.
- The named `cuda0-throughput` profile selects logical CUDA 0 for all services, ASR 128 blocks, LLM 64 blocks, LLM warmup batches `1,2,4`, codec-step TTS, inner predictor graph slots 2, two TTS workers, a 10 ms admission window, and parity-gated native batching. `compat` remains default and explicit CLI/environment values win.
- Process ready messages now retain PID, CUDA allocator/device memory, and backend warmup metrics. The snapshot refuses to initialize CUDA for lightweight/CPU backends and adds no request-path RPC.
- The clean-card preflight CLI exits before importing timing/model code when CUDA 0 exceeds the configured idle threshold. A fresh check at `2026-09-06 20:05` remained blocked at `2488 MiB` and `100%` utilization for more than one minute. Driver PID `1424091` is outside this container's PID namespace; the blocked artifact is `results/cuda0_voice_agent_production_profile.json`.
- Verification after implementation: focused regression `50 passed`; final full non-GPU regression `526 passed, 47 deselected`; changed Python modules compile. No benchmark or service Python process remained. Real post-change CUDA speedup is intentionally not claimed until clean-card acceptance runs.
- Implementation mapping confirms nano-vLLM declares `num_kvcache_blocks` but its current `ModelRunner.allocate_kv_cache()` overwrites it unconditionally; honoring a positive configured value requires a small upstream-local change in `/home/nano-vllm`, not only adapter propagation.
- The resident nano-vLLM backend currently warms exactly one one-token request and leaves warmup steps in online statistics. Batch-shape pre-capture therefore needs concurrent warmup cohorts plus separate `warmup_stats`/online stats.
- The approved design and TDD implementation plan are stored in `docs/superpowers/specs/2026-09-06-cuda0-production-profile-design.md` and `docs/superpowers/plans/2026-09-06-cuda0-production-profile.md`.

## Outer KV Cohort Optimization 2026-09-08

- The first attempt at reducing outer KV overhead copied only appended tokens, but its largest-cohort selection allowed sessions with different cache positions to remain skewed. On the two-session CUDA 0 probe it produced `45` batch-1 and `30` batch-2 outer steps, despite reducing joined copies to `29,245,440` bytes; total p50 was `3882.0 ms`.
- The scheduler now selects the minimum cache position first. When several groups have that position, it selects the largest group and rotates ties. This is a catch-up policy: the shortest prefix advances until it can join the longer prefix at an identical cache position.
- The revised probe produced `15` batch-1 and `45` batch-2 outer steps, total p50 `3491.2 ms`, and joined KV copies `39,109,440` bytes. It is a `1.11x` improvement over the previous scheduler implementation but remains slower than the explicit legacy control at `2980.1 ms` (`0.85x` relative).
- The remaining bottleneck is no longer just raw KV copying. The scheduler still creates/rebinds joined cache cohorts as request membership changes, and strict exact-parity mode executes the inner predictor scalar per request (`105` scalar steps in the two-session run). A fixed slot-backed outer cache with request-to-slot remapping is the next meaningful optimization; it must preserve row ordering and exact codec parity before being enabled.

## Fixed Outer Slot Pool 2026-09-08

- Implemented a fixed-size pool per outer batch shape. The pool keeps `request_id -> physical row` mappings and binds each joined layer to the source cache currently occupying that row. Existing requests therefore retain their joined KV row when a cohort changes; only newly admitted requests reset a row's copied length.
- The pool API deliberately accepts `(scheduler request_id, cohort)` pairs. Qwen's independent batch-1 outer generators all expose local row `0`, so deriving identity from `OuterCohortState.requests[0].request_id` would alias every session and violate codec routing.
- Slot order is applied before inner predictor batching and is retained through outer decode output slicing. This preserves exact request event ownership even when a newcomer is assigned a lower physical slot than the request that was removed.
- Unit coverage passes for row reuse and copy accounting, plus the scheduler/outer/service focused suite. Full clean-GPU timing remains blocked by the external workload occupying all four devices at `2488 MiB / 100%`.
- Added runtime counters for pool binds and reused/new rows. This separates actual fixed-slot reuse from a coincidental batch-size increase in the next CUDA acceptance run.

## Strict Scheduler Audit 2026-09-08

- Before this cleanup, the production profile did not set `VOICE_TTS_REQUEST_STEP_SCHEDULER`, so the service silently selected the historical `UnifiedTtsStreamBatchScheduler` whenever a backend exposed `synthesize_stream_batch`.
- The production path is now unambiguous: ASR process service -> nano-vLLM process service -> process-isolated Qwen-TTS `RequestIdCodecScheduler`. The scheduler owns request admission, inner codec generation, outer active-prefix steps, and chunk emission in one resident TTS process.
- Explicit incompatibilities fail early. In particular, `tts_stream_batch_exact_parity=True`, `process_isolated=False`, sampling, disabled codec-step/active-prefix/CUDA Graph predictor, or competing outer engines cannot silently downgrade a strict run.
- `compat` remains a named compatibility profile rather than an implicit fallback. Its legacy batch adapter is still covered for public backward-compatible callers, but the production profile cannot enter it.

## Stable Physical Slot Correction 2026-09-09

- The first integration reused the fixed allocation but reset/reordered rows on
  membership changes, so it did not actually reduce retained-prefix copies.
- The current implementation uses an explicit `request_id -> physical slot`
  map. Retained IDs keep their row; released rows are assigned to newcomers;
  input/output ordering follows physical slots. The focused scheduler test
  verifies one retained row and one new row, and all non-GPU tests pass.
- The parity-preserving reset run remains the diagnostic reference: codec
  hashes matched the no-slot baseline for request IDs 1 through 7, but its
  two-session total p50 was `4592.4 ms` versus `3491.2 ms` for incremental
  catch-up because every membership change rematerialized the prefix. The
  stable mapping is not accepted as a speedup until a clean CUDA run confirms
  codec parity and copy reduction.

## Slot-Backed KV Write Path 2026-09-09

- The stable mapping still mirrored every newly generated KV token from each
  request cache into the joined tensor. The fixed pool now aliases each active
  `ActivePrefixLayer` row to its joined slot, so steady-state model updates are
  written once. Prefix copies occur only on admission or ownership transfer.
- Ownership is explicit across pools keyed by batch size. A request moving from
  a batch-2 pool to a batch-1 pool first restores its private backing from the
  old joined row, then aliases the new row. This avoids two pool tensors sharing
  one mutable request cache.
- Focused scheduler coverage is `10 passed`; full non-GPU coverage is `571
  passed, 47 deselected, 4 warnings`. A clean CUDA run is still required to
  validate codec SHA-256 parity and end-to-end latency; current CUDA 0 remains
  contaminated by an external `2488 MiB / 100%` workload.

## 2026-09-15 GPU Environment: The Blocker Was Thermal, Not Only A Co-Tenant

- Ten prior phases deferred timing acceptance because all four cards report
  `2488 MiB / 100%` from a PID outside this container's namespace. That
  reading is real but incomplete, and treating it as "wait for a clean card"
  has blocked Phases 29/38/49/52 indefinitely.
- `nvidia-smi -q -d PERFORMANCE` shows **`SW Thermal Slowdown : Active` on all
  four GPUs**, with SM clocks pinned at `735/795/840/765 MHz` against a
  `1410 MHz` maximum, at `81-84 C`. The cards are thermally capped to roughly
  52-60% of rated clock and will not return to a "clean" state by waiting.
- `bench/gpu_contention_probe.py` (new) measures a fixed `8192^3` FP16 matmul
  per device. Best/median TFLOPS: GPU0 `90.07/89.73`, GPU1 `90.02/89.92`,
  GPU2 `93.72/93.61`, GPU3 `89.80/87.75`. All four are within `4.2%` of each
  other and run-to-run variance after warmup is under `1%`.
- Interpretation: contention is *uniform and stable*, not spiky. A100 FP16
  tensor-core peak scaled to the throttled clock is roughly `172 TFLOPS`, so
  the measured `90` reflects a combination of throttling and the external
  tenant. Absolute latency on this host is therefore not comparable to a
  clean A100, but a *relative* A/B is valid provided arms are interleaved and
  compared by median.
- Methodology decision recorded in Phase 53: stop gating on `require_idle_cuda`
  for optimization work. Use interleaved repeated arms on one device, report
  medians, and keep exact output/hash parity as the correctness gate. Reserve
  absolute-number claims for a genuinely idle host.

## 2026-09-15 Streaming Path Audit (Read-Only, Pre-Measurement)

### ASR ingress is growing-prefix re-decode, not incremental encoder state

- `StreamingSession.feed()` appends to one growing `_buffer`, cancels the
  in-flight request, and resubmits the **entire buffer** every chunk
  (`qwen_asr_vllm/engine/streaming.py:163-174`). Audio-encoder work for a
  session of `N` chunks is therefore `O(N^2)` in audio length.
- Three policies exist: `retranscribe`, `speculate` (default), `incremental`.
  `speculate` reuses the previous `output_token_ids` as a decoder draft
  (`streaming.py:302-320`) which accelerates *text decode* only; the mel +
  conv + audio-transformer stack still reprocesses the whole prefix.
- `incremental` is the only policy that bounds encoder work: it transcribes
  `_buffer[_committed_audio_end:]` and periodically recomputes a sliding
  window (`streaming.py:227-280`, defaults `recompute_seconds=6.0`,
  `recompute_overlap_seconds=2.0`, `tail_min_seconds=0.5`). It is not the
  default anywhere.
- `commit_lag_words` defaults to `16` for `incremental` and **`0`** for
  everything else (`engine/async_engine.py:196-197`), so the default
  `speculate` path emits no `committed` events at all.
- There is **no VAD or silence endpointing in this repository**. The
  `1200 ms` figure in earlier notes belongs to the external
  `/home/voice_assistant_app`. `final` is produced only by an explicit
  client `close()` (`streaming.py:176-225`, `server.py:267-268`). The
  coordinator's RMS threshold (`async_coordinator.py:625-628`, default
  `1e-4`) is barge-in detection, not endpointing.

### The coordinator serializes more than it needs to

- LLM starts once, on the first ASR event whose kind equals `llm_trigger`
  (`agent/async_coordinator.py:218-226`); `partial` can never trigger it.
- `defer_tts_audio_until_asr_final` buffers *all* synthesized audio until
  `asr_final` arrives (`async_coordinator.py:577-620`). Coordinator default is
  `False`, but `bench/serve_voice_agent_ui.py` ships it as **`True`**, so the
  shipped UI holds every byte until the client closes the stream.
- `feed()` holds `_asr_lock` across the whole blocking transcription
  (`async_coordinator.py:127-128`), so ASR chunks cannot overlap each other.
- Final LLM remainder is only flushed after the LLM stream ends
  (`async_coordinator.py:348-350`), and `done` waits on all TTS tasks plus the
  optional ASR-final barrier (`async_coordinator.py:353-360`).

### TTS codec->waveform decode is already O(N), not O(N^2)

- `_decode_codec_chunk` decodes the slice
  `[max(0, code_start - left_context), code_end)` and trims the context
  prefix (`agent/qwen_tts_streaming.py:423-446`), so steady state decodes
  `chunk_size + left_context_size` frames (defaults `8 + 4`). Total decode
  work is linear in frame count. Earlier worry about repeated full-prefix
  decode does not apply to this path.
- `speech_tokenizer.decode()` remains **stateless** per call: there is no
  persistent convolution/transformer decoder state, so each chunk re-runs the
  neural decoder over its window. `_CodecFrameBuffer`
  (`qwen_tts_streaming.py:28-75`) only removes a repeated
  `np.stack(all_frames)` and provides a stable contiguous prefix; it does not
  reduce decoder FLOPs.
- Codec *generation* is still one full-sequence `generate` running on a
  background thread with a per-frame hook
  (`qwen_tts_streaming.py:162-168`, `agent/local_tts.py:331-365`), so the
  talker itself is not incremental across segments.

### The measured-best path is not the shipped default

- `runtime_profiles.py` `cuda0-throughput` turns on codec-step streaming,
  the inner CUDA-Graph predictor (2 slots), the active-prefix outer talker,
  2 TTS workers, a `10 ms` admission window, and greedy TTS. `compat` is the
  default profile and leaves every one of those off.
- `bench/serve_voice_agent_ui.py` defaults `--tts-streaming-engine off`,
  `--tts-cuda-graph-code-predictor False`, `--tts-stream-batch-exact-parity
  True`, `--tts-process-stream-workers 1`, and
  `--defer-tts-audio-until-asr-final True`. The shipped demo therefore runs
  the slowest available combination.

## 2026-09-15 Default Chain Baseline And The Eager-LLM Regression

### Baseline: shipped `compat` defaults, real ASR/LLM/TTS, 8.25 s LibriSpeech input

`results/streaming2026/baseline_compat_async.json`, ASR `cuda:0`, LLM `cuda:2`,
TTS `cuda:1`, greedy LLM, `tts_max_new_tokens=64`.

| event | offset (ms) |
|---|---|
| `turn_listening` | 0.8 |
| `asr_partial` (first) | 136.6 |
| `asr_committed` (first) | 392.3 |
| `llm_start` | 392.6 |
| `llm_first_chunk` | 448.1 |
| `asr_final` | 2700.6 |
| `llm_sentence_ready` (first) | 2958.0 |
| `llm_done` | 3191.6 |
| `tts_audio_ready` | 6981.4 |
| `done` | 9541.4 |

`first_audio_ms = 6981.7`, `total_ms = 9543.0`, `input_wall_ms = 2699.4`.

First-audio attribution:

| segment | ms | share of first audio |
|---|---|---|
| ASR to first committed hypothesis | 392 | 5.6% |
| LLM time to first token | 56 | 0.8% |
| LLM streaming to first TTS-ready sentence | 2510 | 36.0% |
| TTS synthesis of that sentence | 4023 | 57.6% |

So `93.6%` of perceived latency is LLM token streaming plus TTS synthesis. ASR
is not on the first-audio critical path at all in this configuration.

### The `LLM_ENFORCE_EAGER=true` workaround costs about 8.6x on LLM decode

Earlier phases adopted `LLM_ENFORCE_EAGER=true` because "nano-vLLM graph capture
failed in real timing" (`task_plan.md:130`). The *config* default is already
`false` (`/home/voice_assistant_app/src/config.py:57`), but the workaround was
baked into the documented run recipe (`README.md:600`) and into the baseline
command used here, so any operator following the README pays for it. Isolated
`bench/nano_llm_batching_probe.py` on `cuda:2`,
concurrency 1, `max_num_seqs=4`, same checkpoint:

| arm | first token | total | chunks | per-chunk |
|---|---|---|---|---|
| `--enforce-eager` | 49.6 ms | 2495.7 ms | 22 | 113.4 ms |
| CUDA graph replay | 42.0 ms | 211.5 ms | 16 | 13.2 ms |

Graph capture now succeeds with no error, so the workaround is obsolete. This is
the same effect the README already documents for ASR ("Decode was CPU-bound, not
GPU-bound"): a 0.6B decode step is launch-bound, so removing per-step Python and
kernel-launch overhead dominates any kernel tuning. `13.2 ms` versus `113.4 ms`
is `8.6x`, and the artifacts are
`results/streaming2026/llm_probe_eager.json` and `llm_probe_graph.json`.

Because LLM streaming to the first sentence is `36%` of baseline first audio,
this single configuration change is worth more than the entire outer-talker
kernel program pursued in Phases 27-52.

### ASR growing-prefix cost, quantified

The baseline emitted `21 asr_partial` events for `8.25 s` of audio at the default
`--chunk-ms 400`. Under `speculate`, each one re-runs mel + conv +
audio-transformer over the **whole** buffer, so the session transcribed roughly
`sum(0.4..8.4) ~= 92 s` of audio to ingest `8.25 s`, about `11x` the necessary
encoder work, and `input_wall_ms` was `2699.4` for `8.25 s` of input.
That is comfortably realtime, which is why it never showed up as a bug, but it
sets the concurrency ceiling and it is why ingest overlaps and slows LLM
streaming on a contended host.

## 2026-09-15 Streaming Ablation Ladder (Real ASR + nano-vLLM + Qwen-TTS)

One factor added per arm, `compat` profile, ASR `cuda:0` / LLM `cuda:2` / TTS
`cuda:1`, fixed `ASR_NUM_KVCACHE_BLOCKS=128` and `LLM_NUM_KVCACHE_BLOCKS=64`,
greedy LLM, `tts_max_new_tokens=64`, same `8.25 s` LibriSpeech input.
Artifacts: `results/streaming2026/arm_*.log`, `ladder_summary.json`.

| arm | added factor | first audio (ms) | asr>commit | llm ttft | llm>sentence | tts>audio | step | cumulative |
|---|---|---|---|---|---|---|---|---|
| A | documented recipe (eager LLM, TTS streaming off) | 7007.6 | 387.6 | 49.3 | 1973.3 | 4596.7 | - | 1.00x |
| B | LLM CUDA graph replay | 5372.3 | 419.0 | 48.8 | **76.3** | 4827.7 | 1.30x | 1.30x |
| C | TTS codec-step streaming | 2672.8 | 369.2 | 43.8 | 76.0 | **2183.4** | 2.01x | 2.62x |
| D | TTS inner CUDA graph + active-prefix outer talker | 1728.3 | 383.4 | 42.6 | 76.5 | **1225.5** | 1.55x | 4.06x |
| E | early LLM->TTS segmentation | 1716.5 | 392.8 | 58.3 | 44.3 | 1220.5 | 1.01x | 4.08x |
| F | incremental ASR ingest | 1913.5 | 467.3 | 43.8 | 43.3 | 1358.7 | 0.90x | 3.66x |

ASR hypothesis and turn-1 LLM text are identical in every arm, so the
first-audio column is output-preserving. `total_ms` is **not** comparable for
D/E/F (see the self-interruption finding below), and arm F is a regression.

Conclusions:

- **`4.08x` on first audio is available from configuration alone**, with no new
  kernel. Arms B, C and D are all existing, already-parity-tested code paths
  that the shipped defaults leave switched off.
- The two dominant factors are LLM CUDA graph replay (`1973 ms -> 76 ms` on
  LLM-to-first-sentence, `25.9x`) and TTS codec-step streaming plus the inner
  predictor graph (`4597 ms -> 1226 ms` on TTS-to-first-audio, `3.75x`).
- Early segmentation (arm E) is nearly free here because the LLM already
  reaches a sentence boundary in `76 ms` once graphs are on. It matters only
  for long first sentences.
- **Incremental ASR does not help first-audio latency** and slightly hurts it:
  ASR is off the critical path once it commits in about `390 ms`, and
  `commit_lag_words=2` delays the trigger. Its value is ingest cost and
  concurrency headroom, not latency. Do not sell it as a latency fix.
- After arm E the residual split of `1716.5 ms` is TTS `71.1%`,
  ASR-to-commit `22.9%`, LLM `6.0%`. TTS is the only remaining large target,
  which is consistent with Phases 27-52 but the ordering matters: the
  configuration wins above were `4x` and were available the whole time.

## 2026-09-15 Fast First Audio Triggers Self-Interruption

Arms D/E/F each recorded `llm_start` **twice**, `turn_interrupted` immediately
after `tts_first_chunk`, and a doubled reply
(`'...ask?...ask?'`). The cause is not TTS:

- Arms D/E/F are the first arms where `tts_audio_ready` (`1728 / 1716 / 1913 ms`)
  lands **before** `asr_final` (`2757 / 2709 / 2704 ms`). Prerecorded input is
  still being fed at that point.
- The coordinator's barge-in check treats any incoming frame above
  `_speech_rms_threshold` as an interruption while the agent is speaking
  (`agent/async_coordinator.py:625-628`). The remainder of the *same* user
  utterance therefore cancels the turn and starts a second one.

This is the real reason the shipped UI sets
`--defer-tts-audio-until-asr-final True`: holding all audio until `asr_final`
hides the self-interruption, at the cost of throwing away streaming entirely.
The barge-in policy cannot distinguish "user is still finishing the utterance
that produced this reply" from "user is interrupting the reply", so a genuine
streaming paradigm requires fixing that policy, not deferring audio.

## 2026-09-15 TTS First-Chunk Size Is The Cheapest Remaining Latency Lever

Codec-step streaming cannot emit a waveform until `first_chunk_size` codec
frames exist, so that value is a hard floor on first audio. Sweeping it against
the otherwise-identical arm E configuration:

| first_chunk_size | first audio (ms) | tts>audio (ms) | tts chunks |
|---|---|---|---|
| 8 (default) | 1716.5 | 1220.5 | 10 |
| 4 | 1180.9 | 718.7 | 15 |
| 2 | **960.8** | **461.9** | 15 |

`8 -> 2` gives `2.64x` on TTS-to-first-audio and `1.79x` on first audio, using an
existing flag (`--tts-stream-first-chunk-size`) and no new code. The decoder
already carries `left_context_size=4` frames of context, so the emitted window
is unchanged in kind, only in when it is first flushed.

Cumulative first audio from the documented recipe: `7007.6 -> 960.8 ms`,
**`7.29x`**, entirely from configuration of already-parity-tested paths.

### Where the remaining 960.8 ms goes

| stage | ms | share |
|---|---|---|
| ASR to first committed hypothesis | 404.4 | 42.1% |
| TTS to first audio | 461.9 | 48.1% |
| LLM time to first token | 49.0 | 5.1% |
| LLM to first TTS-ready sentence | 45.1 | 4.7% |

The bottleneck has moved. ASR-to-first-commit is now nearly the size of the
whole TTS stage, so further outer-talker kernel work has an Amdahl ceiling of
about `1.9x` on first audio even if TTS became free, while ASR commit policy is
now worth almost as much and has never been tuned.

## 2026-09-15 Barge-In Policy Fix And Named Profile Validation

- `AsyncVoiceAgentCoordinator` now takes `barge_in_policy` (`auto` |
  `after-asr-final` | `explicit-only`) and `barge_in_rms_threshold`. Only the
  interrupting transition is gated, so `idle -> listening` still runs under every
  policy; suppressed frames are counted in `barge_in_suppressed`.
- `build_voice_factory` was silently dropping `tts_first_sentence_immediate`, so
  the WebSocket path could never enable it. Fixed alongside the new options.
- Real validation, full streaming configuration plus
  `--barge-in-policy after-asr-final` (arm H): `turn_interrupted=0`,
  `llm_start=1`, `llm_done=1`, zero errors, single-turn reply restored, and
  `total_ms` improved from `12079.6` to `9412.1 ms` because the duplicate turn
  is gone. First audio `1069.8 ms`.
- Named-profile validation (arm I), a single `--runtime-profile low-latency`
  flag with no other tuning: first audio `1036.7 ms`, total `9229.0 ms`,
  identical ASR hypothesis and LLM reply, `14` streamed TTS chunks, zero errors.
  Against arm A that is **`6.76x` first audio**.
- Run-to-run spread across G/H/I first audio is `960.8 / 1069.8 / 1036.7 ms`,
  about `±5%`, consistent with the uniform thermal throttling measured earlier.
  Treat differences under roughly `10%` on this host as noise.

### What is left, ranked by measured headroom

1. **TTS codec generation** — `48.1%` of the remaining `~1.0 s`. This is the
   outer talker work already targeted by Phases 27-52. Amdahl ceiling on first
   audio is about `1.9x`, and only about `1.4x` for realistic halving.
2. **ASR time to first committed hypothesis** — `42.1%`, roughly `405 ms`, and
   never tuned. It is now almost as large as TTS. The levers are commit policy
   (`commit_lag_words`, `min_committed_words`, `min_committed_audio_seconds`) and
   chunk size, not kernels. This is the highest-ROI untouched area.
3. **Total drain time** — `done` is at about `9.2 s` while first audio is at
   `1.0 s`, so full-reply synthesis throughput, not latency, dominates the rest.
   That is what `cuda0-throughput` and native TTS batching address.
4. **Process-runner lifecycle** — killing the harness orphaned resident service
   processes holding about `66 GB` of GPU memory. The runners need a
   parent-death watchdog; this is an operational bug, not a performance one.

## 2026-09-15 ASR Commit Policy Sweep

All arms on the `low-latency` profile, same `8.25 s` input. **The ASR hypothesis
is byte-identical in every arm**, so the ASR columns are directly comparable;
`first_audio` and the reply are not, because a different committed prefix
triggers the LLM and therefore changes the answer.

| chunk-ms | policy | partials | ASR ingest wall | ASR RTF | asr>commit |
|---|---|---|---|---|---|
| 200 | speculate | 42 | 4347.7 ms | 0.527 | 567.3 ms |
| 400 | speculate | 21 | 2767.1 ms | 0.335 | 405.4 ms |
| **800** | **speculate** | **11** | **870.8 ms** | **0.106** | **154.5 ms** |
| 1600 | speculate | 6 | 1327.2 ms | 0.161 | 193.1 ms |
| 800 | retranscribe | 11 | 4024.7 ms | 0.488 | 357.5 ms |

- `--chunk-ms 800` is a genuine optimum: `3.18x` less ASR ingest compute than
  the `400` default and `5.0x` less than `200`. `1600` is worse than `800`,
  because each re-decode then covers more audio than it saves in call count.
- **`speculate` versus `retranscribe` at the same chunk size is `4.6x`**
  (`870.8` vs `4024.7 ms`). This corrects the earlier note in this file: the
  redundant growing-prefix work is dominated by re-decoding the **text**, not by
  the audio encoder. It matches the stage profile, where text decode is `65.2%`
  of GPU time at 10 s. Speculative drafting already removes most of it, which is
  why the encoder-focused framing understated `speculate`'s value.
- The commit gate is already at its most permissive and should stay there.
  Tightening it (`--commit-lag-words 2 --min-committed-words 3`) moved
  `asr>commit` from `154.5` to `309.7 ms` **and** degraded the answer: the LLM
  was triggered on `"Have your will, child"` and replied `"Yes, I have a child."`

### Committed-trigger buys latency with answer quality

Worth stating plainly, because the ladder hid it. With `llm_trigger=committed`
the LLM is prompted with a *fragment* of the utterance, so the reply changes with
chunk size: `400 ms` chunks gave `"Yes, I can help with that. What would you
like to ask?"`, `800 ms` chunks gave `"Sure! What can I do for you?"`, and the
tightened gate gave `"Yes, I have a child."` All are generic or wrong for the
actual sentence. Committed-trigger is a latency mechanism, not a correctness one,
and any deployment using it needs revision/replacement of the reply when
`asr_final` disagrees.

## 2026-09-15 Live-Session Latency And Whole-System Cost

### Realtime input is the honest number

`--realtime-input` feeds audio at wall-clock speed, as a live microphone does.
Same optimized profile, `chunk-ms 400`:

| metric | fed as fast as possible | realtime input |
|---|---|---|
| first audio | 1036.7 ms | **1435.3 ms** |
| asr>commit | 405.4 ms | 1004.6 ms |
| LLM ttft + to-sentence | 92.5 ms | 58.5 ms |
| TTS to first audio | 538.4 ms | 371.9 ms |

In a live session `asr>commit` is `1004.6 ms`, and almost all of it is waiting
for the user to actually speak two words. That part is physics, not engineering.
The engineering-controllable remainder is about `430 ms` (LLM plus TTS). So ASR
was never a `42%` compute bottleneck; the earlier `405 ms` figure mixed
"waiting for speech" with redundant re-decode.

### Per-turn inference cost, best configuration

One turn = `8.25 s` input audio in, `2.48 s` reply audio out, `chunk-ms 800`:

| stage | wall per turn | real-time factor | keeps up? |
|---|---|---|---|
| ASR ingest | 870.8 ms | 0.106 (9.5x realtime) | yes, comfortably |
| LLM generate | 77.4 ms, 8.6 ms/token | n/a | yes |
| TTS synthesize | 3100.2 ms | **1.25** | **no** |
| total stage occupancy | ~4.05 s | | |

TTS is `76%` of all per-turn compute and is the only stage that cannot keep up
with speech. **TTS real-time factor was above 1.0 in all seven arms
(`1.25` to `1.85`)**, so this system cannot sustain continuous conversation on
this thermally throttled host regardless of first-audio latency. Sustained
duplex operation needs TTS RTF below 1.0, which is the real target for the
outer-talker work, not first audio.

## 2026-09-15 Sub-Sentence Flushing Costs More Than It Buys

Output-matched A/B at `chunk-ms 800`, both arms measured in the same window,
identical reply `"Sure! What can I do for you?"`:

| arm | first audio | total | reply audio | TTS wall | TTS chunks |
|---|---|---|---|---|---|
| `--tts-flush-chars 12` | 830.2 ms | 4898.1 ms | 2480 ms | 4531.3 ms | 8 |
| sentence boundary only | 869.4 ms | **4231.1 ms** | **2080 ms** | **3882.0 ms** | 6 |

Character-level flushing gains `39 ms` on first audio, which is inside the `~5%`
run-to-run band on this host, while it inflates the synthesized reply by `1.19x`
and TTS compute by `1.17x`. Each sub-sentence fragment carries its own lead-in,
so total time to finish speaking gets `1.16x` worse. Removed from the
`low-latency` profile; the README's original advice to keep sentence-boundary
flushing as the default was correct.

## 2026-09-15 Measurement Variance On This Host

- `chunk-ms 400` ASR ingest wall is remarkably stable across five independent
  runs: `2752.5 / 2767.1 / 2712.4 / 2776.7 / 2767.1 ms`.
- `chunk-ms 800` is not: `870.8 / 1759.4 / 1880.8 ms`. Median-based reporting is
  therefore mandatory for the chunk-size claim, which is `~1.6x` on the median
  and `3.2x` at best, not a flat `3.2x`.
- First-audio spread on repeated identical configurations is about `±5%`
  (`960.8 / 1036.7 / 1069.8 ms`). Treat sub-`10%` differences as noise.

## 2026-09-15 The First-Audio Speedup Was Semantic Distortion, Not Acceleration

`--llm-trigger committed` was the single largest first-audio win in the ladder.
The timing report never recorded the prompt the LLM actually received, so the win
was never checked against what the agent said. It is now recorded
(`details.llm_prompt`) and added to `compare_timing_outputs`.

Three arms, `low-latency` profile, identical audio and identical ASR final
transcript (`"Have your will, child. If the boy also wills it, Montfiche
answered, feeling too ill to oppose anything very strongly just then."`, 22 words):

| arm | trigger gate | prompt the LLM saw | first audio | reply |
|---|---|---|---|---|
| `SEM_final` | `final` | all 22 words | 3194.2 ms | `"Yes, the boy wills the item, and Montfiche feels too ill to oppose it."` |
| `SEM_open` | `committed`, defaults | **`"Have"`** (1 word) | 994.3 ms | `"Yes, I can help with that. What would you like to ask?"` |
| `SEM_gated` | `committed`, 12 words / 6.0 s | 15 words | 2569.6 ms | `"If the boy also willed it, Montfichet would have felt too ill."` |

**The 3.2x first-audio speedup is entirely the agent answering a one-word
question.** `min_committed_words=1` and `min_committed_audio_seconds=0.0` are the
CLI defaults, so `SEM_open` is what the harness measured in every previous arm
that used the default trigger. Its reply is a generic "how can I help" that is
unrelated to the utterance: the LLM was asked `"Have"`. `_llm_started` then
blocks any revision, so the wrong answer is final and is what the user hears.

Gating does not rescue the idea. At 15 of 22 words the reply is fluent but wrong:
it drops the `"Have your will, child"` clause that carries the actual request and
turns a statement into a hypothetical. Latency also collapses back to
`2569.6 ms`, only `1.24x` better than `final`, because the gate has to wait for
most of the utterance anyway.

Early triggering also locks in ASR errors. `SEM_gated` was prompted with
`"Montfichet"` where the final transcript says `"Montfiche"`, so the committed
prefix is not even a stable prefix of the final text — it is a revisable
hypothesis being treated as settled input.

Conclusions:

- **`llm_trigger=committed` is not an acceleration and must not ship.** There is
  no word count that makes a truncated question safe, because completeness is
  semantic, not positional.
- The honest first-audio number for this system is **`3194.2 ms`**, not
  `960.8 ms`. Every earlier first-audio claim in this document that relied on the
  default trigger is measuring a different, easier task.
- Latency has to come from making ASR, LLM and TTS faster on the *whole*
  utterance, which makes TTS real-time factor the only remaining target.

## 2026-09-15 Both Custom Outer Talker Engines Are RTF Regressions

Isolated TTS profiling on the reply the agent actually produces
(`"Yes, the boy wills the item, and Montfiche feels too ill to oppose it."`,
55 codec frames, 4.4 s of audio), batch 1, greedy, graphed code predictor,
`cuda:1`. **Every arm produced codec SHA-256 `6c9b286b`**, so these are
output-matched comparisons, not quality trades:

| outer talker engine | generation | chunked decode | total RTF | codec frames/s |
|---|---|---|---|---|
| upstream (none) | 5824.4 ms | 291.1 ms | **1.390** | 9.44 |
| `active_prefix` | 13510.8 ms | 307.8 ms | 3.141 | 4.07 |
| `cuda_graph` | 37693.1 ms | 408.5 ms | 8.659 | 1.46 |

`active_prefix` is **2.26x slower** than doing nothing, and the CUDA-graph talker
is **6.23x slower**. The `low-latency` profile enables `active_prefix`, so the
profile as shipped more than doubles the cost of the only stage that could not
keep up with speech in the first place.

Two candidate explanations were tested and both are refuted:

- **Chunk size.** `8 / 16 / 24` measured `3.141 / 3.089 / 3.078`. Flat.
- **Static cache size.** The engines require `prompt + max_new_tokens <= max_cache_len`,
  which forces a 16384-token buffer when `max_new_tokens` is unbounded, so the
  suspicion was that attention runs over the whole backing buffer. Capping
  `max_new_tokens=64` and shrinking the cache to 1024 changed nothing:
  `active_prefix` measured `3.087` at 1024 versus `3.116` at 16384, and the
  CUDA-graph engine `8.774` versus `8.659`. The cost is intrinsic per-step
  overhead in these engines, not buffer sizing.

Codec-to-waveform decode is `291-409 ms` against `5.8-37.7 s` of generation, so
it is 1-6% of TTS and irrelevant to RTF. All TTS optimization has to attack
talker generation.

### TTS RTF below 1.0 is not reachable by configuration on this host

Sustaining conversation needs 12.5 codec frames/s. The best configuration
measured delivers **9.44**, i.e. RTF `1.39`. Removing `active_prefix` is a real
`2.26x` win and is the correct change, but it lands at `1.39`, not below `1.0`.
Closing the remaining `1.39x` cannot come from the flags that exist: decode is
already negligible, the code predictor is already graphed, and both hand-written
talker engines are slower than upstream. It needs a cheaper talker step
(quantization, a smaller talker, or kernel work), or the product has to accept
that a reply is buffered rather than streamed indefinitely.

One caveat on the absolute number: this host runs at 80-82 degrees C with other
tenants drawing ~200 W, and `findings.md` already documents thermal throttling
here. The `2.26x` ratio is measured pairwise and is trustworthy; the `1.39`
absolute value is not a clean-hardware figure.

### Warmup is worth ~10 s and must not be measured away

`--no-warmup` on the upstream path measured RTF `3.681` versus `1.429` warmed,
on identical output. Graph capture and lazy initialization cost about `10 s` on
the first generation. Resident services pay this once at startup; any benchmark
that skips warmup is measuring initialization, not steady state.

## 2026-09-15 Corrected `low-latency` Profile: TTS RTF Below 1.0

The profile now pins `llm_trigger=final` and disables both custom outer talker
engines. Verified with no TTS or trigger flags on the command line, so what is
measured is what the named profile ships:

| run | prompt | first audio | total | TTS RTF | per-chunk trend | sustains? |
|---|---|---|---|---|---|---|
| `VERIFY_profile_r1` | all 22 words | 2221.7 ms | 6556.3 ms | **0.936** | 0.97 flat | yes |
| `VERIFY_profile_r2` | all 22 words | 1755.3 ms | 5224.4 ms | **0.752** | 0.98 flat | yes |

Both produced the responsive reply `"Yes, the boy wills the item, and Montfiche
feels too ill to oppose it."` from the full transcript. **The RTF target was met
by deleting a regression, not by adding an optimization.**

Host state has to be read alongside these numbers. `asr_rtf` is a free contention
gauge because ASR work is fixed across arms:

| host state | `asr_rtf` | TTS RTF, upstream talker | TTS RTF, `active_prefix` |
|---|---|---|---|
| quiet | 0.152 - 0.207 | **0.752 - 0.936** | 0.774 |
| contended | 0.29 - 0.33 | 1.259 - 1.337 | 3.072 - 3.205 |

So RTF below 1.0 is reached on a quiet host and missed by about `1.3x` when the
external tenant is active. The `2.39x` engine ratio holds in both regimes and is
the trustworthy part; the absolute crossing of 1.0 depends on the host. The
single `0.774` reading for `active_prefix` came from a quiet-host run and is the
reason it looked good earlier: it was never compared against upstream under
matched conditions until the contention-probed pairs.

`tts_gap_growth` is the metric that made this legible. Averaging hides a stage
that is falling behind, because a reply can average under real time while each
chunk costs more than the last. Upstream holds `0.97 - 1.04` across every run;
`active_prefix` sits at `2.19 - 2.36`, meaning its per-chunk cost more than
doubles across a single 5-second reply.

## 2026-09-15 Paired Speedup Over The Naive System

The `18198.2 ms` naive figure came from an earlier session with no recorded host
state, and this host varies about `2x`, so the baseline was re-measured beside the
candidate with a contention probe before each arm. Both arms use
`--llm-trigger final` and produced the identical reply:

| pair | naive `--real-target sync` | `low-latency` first audio | first audio | total turn |
|---|---|---|---|---|
| r1 | 17353.9 ms | 3492.5 ms | **4.97x** | 1.78x |
| r2 | 19049.7 ms | 3456.9 ms | **5.51x** | 1.91x |

**About `5x` on first audio and `1.9x` on the whole turn.** The re-measured naive
values bracket the old `18198.2 ms`, so that baseline was sound; what was wrong
was comparing it against a quiet-host candidate. The withdrawn claims were
`17.6x` / `31.7x` (one-word prompt) and `8.2 - 10.4x` (mismatched host state).

These pairs landed in the busy regime, TTS RTF `1.244` and `1.290`. That matters
for what is left to win: the profile's `~9.9 s` total turn is already near the
floor RTF implies, `~2.7 s` ASR ingest plus `~6.5 s` to synthesize `5.04 s` of
reply. Overlap is essentially fully exploited, so **total-time gains from here
are bounded by TTS RTF alone**, which is why Phase 63 is the only remaining
lever and why it needs a cheaper talker step rather than more scheduling work.

## 2026-09-15 "Quiet" And "Busy" Host Are Thermal Clock States, And My Probe Cannot See Them

The quiet/busy labels used throughout these notes were inferred from `asr_rtf`,
never from a direct measurement of the environment. Querying clocks settles it.
All four cards, with none of our processes running:

| card | SM clock | max | ratio | temp | power / limit | throttle reason |
|---|---|---|---|---|---|---|
| 0 | 645 MHz | 1410 MHz | 0.46 | 83 C | 179 W / 400 W | `sw_thermal_slowdown` |
| 1 | 705 MHz | 1410 MHz | 0.50 | 82 C | 185 W / 400 W | `sw_thermal_slowdown` |
| 2 | 750 MHz | 1410 MHz | 0.53 | 82 C | 199 W / 400 W | `sw_thermal_slowdown` |
| 3 | 690 MHz | 1410 MHz | 0.49 | 84 C | **0 W** / 400 W | `sw_thermal_slowdown` |

`0x20` is NVML `SW_THERMAL_SLOWDOWN`. Power is 45% of limit, so this is cooling,
not a power cap. Card 3 draws `0 W` and is still 84 C and throttled, so the whole
chassis is heat-saturated rather than any one card being driven hot. Sampled every
6 s for a minute, clocks were pinned flat and did not recover, while cards 0-2 held
179-199 W with nothing of ours running: the neighbours' sustained load is what
keeps the node hot.

A clock ratio near `0.5` matches the observed bimodality. `asr_rtf` clusters at
`0.152 - 0.207` and `0.295 - 0.328`, about `1.9x` apart, with nothing in between
across twelve arms, and TTS RTF tracks it. So "quiet" was a window where the
neighbours' load dipped, temperature fell, and clocks rose.

### The TFLOPs probe measures the hot state by construction

`gpu_contention_probe.py` reported a flat `~90 TFLOPs` on every arm while real
workloads varied `1.9x`. The reason is that a sustained 8192-cube matmul heats the
card into thermal slowdown before the measurement finishes, so it reports the hot
steady state regardless of the state it started in. It cannot detect the cool
regime, and every probe reading collected so far happens to come from a busy-regime
arm, so those readings never validated anything.

Fixed: `query_smi()` now records `clock_sm_mhz`, `clock_max_sm_mhz`, `clock_ratio`
and decoded `throttle_reasons`, the probe's docstring states the limitation, and
`voice_agent_timing.py` records the same snapshot as `details.gpu_state` so any
future arm can be interpreted. A shorter probe run also shows the effect directly:
`tflops_best` fell to `[72.79, 76.38, 91.16, 71.55]` when the cards started at
`630-735 MHz`, versus `~90` uniformly on longer runs.

### This corrects the RTF-below-1.0 claim

The `0.752 - 0.936` TTS RTF readings came from transient cool-clock windows. The
node's steady state is thermally throttled at roughly half clock, where the same
profile measures `1.244 - 1.337`. **So in the state this node actually runs in,
TTS RTF is about `1.3` and the stage does not sustain conversation.** Removing the
`active_prefix` regression is still a real `2.39x` and still correct; what is not
established is that the corrected profile holds below 1.0 in steady state. Phase 63
is therefore required, not optional, and any future sub-1.0 claim must quote
`details.gpu_state` alongside it.

## 2026-09-15 `--realtime-input` Serialized ASR Against Speech By Construction

The pacing loop awaited `coordinator.feed()` and *then* slept for the chunk
duration, so every chunk cost `ASR compute + chunk duration`. Feeding an
`8250 ms` file took `11027.3 ms` of wall, a ratio of `1.337`: "realtime" input ran
a third slower than real time, and ASR was serialized against the audio clock
rather than overlapped with it. The instrument could not answer the question it
existed for.

Fixed by pacing against an absolute schedule, so a chunk that finishes early
waits out the remainder and a chunk that finishes late simply falls behind:

| | audio | feed wall | ratio |
|---|---|---|---|
| before | 8250.0 ms | 11027.3 ms | 1.337 |
| after | 8250.0 ms | 8271.8 ms | **1.003** |

`input_wall_ms` versus the newly recorded `input_audio_ms` is now a real signal:
exceeding it means ASR failed to keep up.

## 2026-09-15 Which Stages Actually Overlap With Speech

Live-session timeline, `--realtime-input`, corrected profile, `8250 ms` utterance
and a `5040 ms` reply (GPU at 0.57-0.72 of max clock):

| boundary | overlapped? | evidence |
|---|---|---|
| ASR / speech | **yes, fully** | partials from `121 ms`, last at `8272.3 ms`, i.e. `22 ms` after the final audio byte, with feed ratio `1.003` and zero backlog |
| LLM / speech | **no, serial** | `asr_final` at `8273.3 ms`, `llm_start` at `8273.7 ms`: the LLM begins `0.4 ms` after ASR ends and never before |
| LLM / TTS | mechanism only | `llm_sentence_ready` `8281`, `llm_done` `8529`; a one-sentence reply leaves ~`8 ms` of real overlap |
| TTS / playback | **yes** | 9 chunks streamed from `9082.2 ms` to `15263.9 ms` |

The serial ASR-to-LLM boundary is a direct consequence of `llm_trigger=final`:
starting the LLM during speech requires prompting it with an incomplete
utterance, which is exactly the distortion removed earlier today.

**That boundary is nearly free.** After the user stops speaking they wait
`6993.1 ms`, split as:

| stage | time | share of the wait |
|---|---|---|
| LLM | 255.3 ms | 3.7% |
| TTS | 6737.8 ms | **96.3%** |

So perfectly overlapping the LLM with speech would remove at most `3.7%` of the
perceived wait, at the cost of answer correctness. Across the fast-feed arms the
LLM is likewise only `1.9 - 2.7%` of the whole turn. Pipelining is already
extracted everywhere it pays; what remains is TTS generation speed, which cannot
start before the reply text exists.

Perceived latency is better than the totals suggest: first audio lands `808.9 ms`
after the user stops speaking, because ASR has no backlog to drain and the LLM is
fast. The `15266.4 ms` total is dominated by the agent still *speaking*.

## 2026-09-15 Speculative LLM Prefill Is Correct But Worth 0.5%

The proposal is to prefill the LLM on the committed ASR prefix during speech and
only sample after `asr_final`, which preserves semantics exactly because the
sampled tokens are still conditioned on the complete prompt. Measuring what it
could recover, on the live paced run:

| segment | time | share of the 6990.6 ms post-speech wait |
|---|---|---|
| prefill (TTFT 46.5 ms minus one 11.0 ms decode step) | **~35.5 ms** | **0.51%** |
| LLM decode, 19 tokens at 11.0 ms | 208.8 ms | 2.99% |
| TTS | 6743.6 ms | 96.47% |

Prefill is all that can move: decode is conditioned on the full prompt by
definition, which is the whole point of the design. So the ceiling is about
`35 ms`, or `0.5%`. The prompt is ~30 tokens, and prefilling 30 tokens on this
LLM is simply not expensive.

It also carries a correctness hazard. ASR revises its hypothesis, measured
directly: a committed prefix said `"Montfichet"` where the final transcript said
`"Montfiche"`. A speculative KV cache is therefore invalidated by ordinary ASR
behaviour and needs revision detection plus a re-prefill path. That machinery is
not worth `35 ms`.

## 2026-09-15 The Real Cross-Subsystem Defect Is Playback Starvation

RTF above 1.0 is not a throughput curiosity, it is an audible defect. Replaying
the chunk timeline against a player that starts on the first chunk and consumes
audio in real time:

| host state | first chunk | worst buffer | outcome |
|---|---|---|---|
| throttled, TTS RTF 1.34 | 160 ms of audio | **-1141.8 ms** | starves `809.7 ms` after playback starts, deficit grows every chunk |
| quiet, TTS RTF 0.94 | 160 ms of audio | +709.0 ms (min 0) | never starves, buffer grows monotonically |

In the state this node actually runs in, the agent begins speaking `808.9 ms`
after the user stops and then **stutters for the rest of the reply**, accumulating
over a second of gaps. No intra-model optimization removes this; it is a
consequence of starting playback with a 160 ms buffer while generation runs
slower than playback.

Two subsystem-level fixes, neither touching semantics:

- **Pre-roll.** Delay playback until the buffer can absorb the deficit. Shifting
  the start by `D` raises every buffer sample by `D`, so `D = 1141.8 ms` makes the
  worst sample exactly zero; with margin, ~`1200 ms`. First audio moves
  `808.9 -> ~2000 ms` after speech ends and the reply becomes gapless. Speaking
  `1.2 s` later without stuttering is clearly better than starting sooner and
  breaking up, and unlike RTF work this is available today.
- **Use the idle cards.** ASR and the LLM are done and their GPUs idle for the
  96.5% of the wait that TTS occupies, while TTS uses one card. Synthesizing
  separate sentences of a reply on separate devices is near-linear for
  multi-sentence replies. It does nothing for the single-sentence reply measured
  here, and fragment boundaries must stay at sentence level because sub-sentence
  splitting was already measured to inflate total audio by 19%.

## 2026-09-15 Playback Preroll Removes The Stutter, Paired Measurement

`tts_playback_preroll_ms` holds the opening chunks until a target buffer exists,
then flushes them in order. Paired arms, same window, same `low-latency` profile,
`--realtime-input`, only the gate differs:

| | first audio after stop | tts_rtf | total audio | worst buffer | heard |
|---|---|---|---|---|---|
| preroll off | **765.4 ms** | 1.313 | 5040.0 ms | -1056.4 ms | **starves at t+1589.7 ms** |
| preroll 1200 ms | 2443.6 ms | 1.318 | 5040.0 ms | +160.0 ms | **gapless** |

The LLM prompt and reply text are byte-identical across the two arms, and total
audio is identical at `5040.0 ms`, so the gate is a pure delivery change with no
semantic or content effect. It costs `1678.2 ms` of first-audio latency and buys
a reply that does not break up.

**The required preroll scales with reply length, so this is mitigation and not a
fix.** The deficit is proportional to the audio that has to be covered at
`RTF > 1`: here `1056.4 / 5040 = 0.21` of reply duration, so a 15 s reply needs
roughly `3.1 s` of preroll and a 30 s reply roughly `6.3 s`, which stops being
usable. A fixed preroll buys gapless short replies on a throttled host; only
getting TTS RTF below 1.0 (Phase 63) removes the problem for replies of any
length. On a quiet host at RTF 0.94 the buffer already grows monotonically and
the gate is unnecessary, which is why it defaults to off.

Two measurement notes from this work:

- `playback_worst_buffer_ms` / `playback_gapless` now derive from the chunk
  timeline in `summarize_voice_timing.py`. RTF alone does not say whether a
  listener hears a gap, and this run is the case in point: `tts_rtf` is
  essentially the same in both arms while one stutters and one does not.
- `tts_gap_growth` had to skip the preroll's flush burst. Released chunks arrive
  back to back, and those near-zero gaps put the ratio at `4004.0` before the
  fix versus `1.05` after, which is the real, flat trend.

## 2026-09-15 ASR Is Streaming On Both Sides, And The Prefix Diverges From The Final

Recorded a per-event `asr_timeline` (offset, kind, hypothesis, committed prefix)
and ran one paced arm to see exactly what a downstream stage could start on.

**Input is streaming.** Audio is fed in `400 ms` blocks
(`--chunk-ms`, default 400) one at a time, paced to the audio clock under
`--realtime-input`. `input_wall_ms 8283.9` against `input_audio_ms 8250.0`, a
ratio of `1.004`, so ASR consumes blocks as fast as a microphone produces them.

**Output is streaming.** The committed prefix grows word by word through the
utterance, `21` partials and `17` commits before `asr_final`:

| committed words | available at | lead over asr_final | still a prefix of the final? |
|---|---|---|---|
| 1 (`"Have"`) | 908.4 ms | 7377.5 ms | yes |
| 5 | 2145.2 ms | 6140.7 ms | yes |
| 10 (`"...also wills it,"`) | 4160.5 ms | **4125.4 ms** | **yes, and this is the last one** |
| 11 (`"...it, Montfichet"`) | 4514.5 ms | 3771.4 ms | **no** |
| 21 | 7737.2 ms | 548.7 ms | no |

**The committed stream never contradicts itself**: `advance_committed` accepts a
candidate only when it equals or word-extends the previous commit, otherwise it
keeps history and flags `commit_violation`. So commits are append-only by
construction, and this corrects the mechanism stated earlier — the revision does
not come from a rewritten commit.

**It comes from the final.** Each `feed` cancels the in-flight decode and
resubmits the *entire* accumulated buffer, and `close()` emits the last such
full-buffer decode as `final`. That decode is independent and is not required to
extend the committed prefix. Here it did not: the committed prefix said
`"Montfichet"` from word 11 while the final said `"Montfiche"`, so **11 of 21
committed words are not a prefix of the transcript the agent must answer.**

Consequences for prefilling the LLM on committed text:

- The lead time is real and large. The 10-word safe prefix lands `4125.4 ms`
  before `asr_final`, far more than the `~35.5 ms` prefill needs.
- Only about **half the prompt is cacheable** on this utterance, `10` of `21`
  words, roughly 14 of ~30 tokens, so the ceiling drops from `~35.5 ms` to
  `~17 ms`, about `0.25%` of the post-speech wait.
- Divergence must be detected at `asr_final` (a string prefix check is enough)
  and the cache dropped, or the agent answers text the user did not say. This is
  structural, not a tuning accident: the final is a separate re-decode.

Measured on one utterance, so the 10-of-21 split is illustrative rather than a
rate; the mechanism that permits divergence is what generalizes.

## 2026-09-15 Cross-Card TTS Is The Wrong Tool: One Card Has 1.85x Idle Headroom

Three separate reasons, in increasing order of how much they settle the question.

**1. There is nothing to split.** The reply is a single sentence and a single TTS
fragment: `llm_output` is `"Yes, the boy wills the item, and Montfiche feels too
ill to oppose it."` and `tts_inputs` has length `1` for `5040.0 ms` of audio. On
the system's actual behaviour, cross-card parallelism has exactly zero effect.

**2. It could never help first audio.** Playback is sequential, so the first
sentence still has to be synthesized on one card at one card's speed. Fragment
parallelism can only raise throughput, which means it addresses starvation and
RTF, never the latency to start speaking.

**3. A second card is not what is missing.** Batch scaling of the upstream path
on a single `cuda:1`, greedy, chunk 8 / left context 4:

| | batch 1 | batch 2 |
|---|---|---|
| utterances | 1 | 2 |
| audio produced | 4399.2 ms | 8403.6 ms (`1.91x`) |
| generation | 5833.4 ms | 6109.4 ms |
| chunked decode | 410.7 ms | 345.6 ms |
| **wall total** | **6244.1 ms** | **6455.0 ms (`1.034x`)** |
| generation_rtf | 1.326 | 0.727 |
| **total_rtf** | **1.419** | **0.768** |

Nearly twice the audio for `3.4%` more wall time: throughput gain `1.848x`, and
RTF crosses from `1.419` to **below 1.0**. A single autoregressive talker stream
at batch 1 is latency-bound on a tiny per-step matmul and leaves most of the card
idle, so a second stream is almost free. **Batching two fragments on one card
delivers the win that cross-card parallelism was supposed to buy, with no second
device and no cross-device orchestration.**

The machinery already exists and is already on: `stream_batch_exact_parity=True`
falls back to the scalar path and serializes fragments, and the `low-latency`
profile sets it to `False`, so `_group_stream_batch_texts` is live. It simply
never engages, because one fragment cannot be batched.

Two caveats:

- **Same prerequisite, unsolved.** Batching also needs two or more fragments in
  flight. It does not help a one-sentence reply either. The headroom pays for
  multi-sentence replies and, more strongly, for **concurrent sessions**: two
  users' replies can share one card for `3.4%` more wall time, which is a
  throughput result rather than a latency one.
- **Output is not bit-identical.** `codec_ids_sha256` differs between batch 1 and
  batch 2, which is why the exact-parity flag exists. Perceptual equivalence has
  to be verified rather than assumed before relying on batched output.

## 2026-09-16 The Talker Step Is 27x Off Roofline, And Quantization Is The Wrong Tool

Profiled the step directly (`bench/tts_talker_step_probe.py`) instead of reasoning
from averages. Model is `Qwen3-TTS-12Hz-1.7B-CustomVoice`, talker hidden `2048` /
`28` layers / intermediate `6144`, code predictor hidden `1024` / `5` layers, 16
codebooks per frame, 12 Hz codec so each frame is `80 ms` of audio.

**The step is nowhere near a hardware limit.** Weights touched per step are
`1.41B` params for the talker and `63M` for the code predictor, so at A100 read
bandwidth the memory roofline is `1.81 ms` + `1.29 ms` (16 codebook passes) =
`3.11 ms`. Measured `85.4 ms`:

| | per step | vs roofline |
|---|---|---|
| memory roofline | 3.11 ms | 1x |
| GPU kernel time | 57.0 ms | **18x** |
| wall | 85.4 ms | **27x** |
| audio produced | 80.0 ms | RTF 1.07 for the talker alone |

At roofline the RTF would be `0.039`. So the headroom is real and large, and RTF
`1.42` is a software result, not a hardware ceiling.

**The two halves are limited by different things**, which the averages hid:

- `talker_model_forward`, `44.7 ms/step`: eager and not graph-captured.
  `1953 cudaLaunchKernel` per step, and the GPU sits idle `28.4 ms` of every
  `85.4 ms` step (`33%`). Top kernels by count are `elementwise_kernel`,
  `unrolled_elementwise_kernel`, `vectorized_elementwise_kernel` and
  `reduce_kernel` (RMSNorm means), not GEMMs. This is an unfused eager decode
  loop, and launch plus tiny-kernel overhead is the cost.
- `code_predictor_generate`, `38.6 ms/step`: **already CUDA-graphed and working**
  — runtime metrics report `graph_captures 2`, `graph_replays 117`,
  `steady_state true`, one replay per frame covering all 15 sub-steps. Its cost is
  therefore GPU-side serialization of 15 sequential codebook sub-steps of tiny
  kernels, not launch overhead. Graphs cannot help it twice.

**This corrects the earlier recommendation.** `findings.md` previously listed
quantization first as the way to a cheaper step. Quantization reduces weight
bytes, and weight bytes are not the constraint: GPU time alone is `18x` the
bandwidth roofline. Quantization should not be expected to move RTF here.

What the diagnosis does point at, in order:

1. **Fuse and graph the upstream talker forward.** It is the one half that is
   still eager, it is `44.7 ms` of the `85.4 ms`, and launch overhead plus
   unfused elementwise work is exactly what compilation removes. Note the
   existing `--compile-step-engine` does **not** cover this path: it compiles the
   fast code predictor and the explicit outer talker engine only.
2. **Reduce the 15 sequential codebook sub-steps** for the code predictor, since
   it is already graphed and serialization is what remains.

The two hand-written outer talker engines remain unexplained regressions
(`2.26x` and `6.23x`), and chunk size and cache size were already refuted as
causes. Any new attempt has to be measured against upstream, not assumed to help.

## 2026-09-16 Synced vs Unsynced Separates The Two Halves Cleanly

Ran the same config with and without `--sync-each-step`. Without sync the
per-stage CPU timers measure how long the CPU is held inside the call; with sync
they include waiting for the device. The difference identifies which half is
CPU-bound and which is GPU-bound.

| stage, mean per step | no sync | sync | reading |
|---|---|---|---|
| `talker_model_forward` | 48.0 ms | 47.8 ms | **unchanged: CPU-bound** |
| `code_predictor_generate` | 15.8 ms | 38.9 ms | **2.5x: GPU-bound** |
| `talker_forward` (both) | 65.4 ms | 89.2 ms | |
| `generation_ms` | 4016.4 | 6084.7 | sync costs `2068.3 ms`, so it is measurement-only |

- **The talker half is launch-bound.** `48 ms` of CPU time per step that does not
  move when a device sync is added: the GPU has already finished what was issued,
  and the cost is the CPU walking `1953` kernel launches through an eager,
  unfused 28-layer decode. Graph capture or compilation should collapse most of
  this, because it removes CPU work rather than GPU work.
- **The code predictor half is GPU-bound.** It returns to the CPU in `15.8 ms`
  while the device is still busy, and costs `38.9 ms` once waited on. It is
  already CUDA-graphed, so the remaining cost is the GPU serially executing 15
  codebook sub-steps of tiny kernels. More graphing cannot help it.

Projected ceiling for fixing the talker half: with launches removed, the step
becomes GPU-serial, roughly `38.9 ms` (code predictor) plus the talker's own GPU
time, against `80 ms` of audio per frame. That points at a step near `57 ms` and
RTF around `0.7`, versus `85.4 ms` and `1.42` measured on the hot host. It does
not reach the `0.039` roofline, because after the talker is fixed the floor is the
code predictor's sequential codebook decode.

Host-state caveat, again the dominant confound: this pair measured
`generation_rtf 0.913` and `total_rtf 0.981` for a configuration that measured
`1.326` / `1.419` hours earlier. Same code, same flags. Only paired,
same-window comparisons mean anything on this node.

## 2026-09-16 torch.compile On The Talker Does Not Help, And The Code Predictor Graph Is Worth 1.9x

Tested the diagnosis directly (`bench/tts_talker_compile_probe.py`), both arms in
one process on the same weights so the result is paired against host state.

**Compiling `talker.model.forward` buys nothing.** All arms produced identical
audio, so these are output-matched:

| arm | RTF | speedup vs eager |
|---|---|---|
| eager | 2.398 | 1x |
| `torch.compile` mode `default` | 2.388 | **1.004x** |
| `torch.compile` mode `max-autotune-no-cudagraphs` | 2.447 | **0.98x** |

Inductor fusion of that module is worth nothing measurable, and autotuning cost
`265 s` and `439 s` of warmup to find that out. The most likely reason is graph
breaks: the module takes a `Cache` object and HF's decode path breaks the graph
repeatedly, so very little actually gets fused and the launch sequence survives.
**This refutes the easy version of the fix**; the launch-bound diagnosis may still
be right, but compilation of this module is not the lever that acts on it.

`mode="reduce-overhead"` cannot even be attempted: it fails with `Inplace update
to inference tensor outside InferenceMode is not allowed`, because the weights are
loaded as inference tensors and its CUDA graphs want to update them in place.

**The code predictor's CUDA graph is load-bearing.** Compiling the talker while
that graph is active fails with `Offset increment outside graph capture
encountered unexpectedly`, an RNG-offset conflict between the two mechanisms, so
the compile arms had to run with it disabled. That disabling is itself the most
useful number here: eager RTF went from `1.250` (graph on) to `2.398` (graph off),
so **the code predictor graph is worth `1.9x`** and must never be turned off.

Where this leaves the talker half, with three independent failures now recorded:

| attempt | result |
|---|---|
| `active_prefix` outer engine | 2.26x slower |
| `cuda_graph` outer talker engine | 6.23x slower |
| `torch.compile` (this work) | 1.004x, no change |

**One lead survives, and the earlier refutation of it was not tight enough.** The
buffer-size hypothesis was dismissed by shrinking `max_cache_len` from `16384` to
`1024` with no change, but this utterance needs only about `66` frames plus a
short prompt, so `1024` is still roughly 8x larger than necessary. Attention over
a padded buffer 8x too long would not show up as an improvement in that test.
A genuinely tight cache (`~128-192`) has not been measured. Given the code
predictor graph is worth `1.9x`, an equivalent win on the talker should exist, and
this is the only untested reason its graph engines regress instead.

## 2026-09-16 REVERSAL: active_prefix Is 1.30x Faster Than Upstream At A Tight Cache

The earlier conclusion that `active_prefix` is a `2.26x` regression is wrong, and
so is the refutation of the buffer-size explanation. That refutation shrank
`max_cache_len` from `16384` to `1024` and saw no change, but this reply is `55`
codec frames plus a short prompt, so `1024` is still roughly 8x larger than
needed and the padded-attention cost survives it.

At a genuinely tight cache the engine wins. Four arms, one window, `cuda:1`,
`max_new_tokens 96`, upstream arms bracketing to expose host drift:

| arm | generation | decode | total RTF | frames/s | vs upstream |
|---|---|---|---|---|---|
| upstream | 4045.7 ms | 329.4 ms | 0.994 | 13.59 | 1x |
| `cuda_graph` @ 160 | 22050.4 ms | 189.0 ms | 5.054 | 2.49 | 5.09x slower |
| **`active_prefix` @ 160** | **3158.6 ms** | **197.3 ms** | **0.763** | **17.41** | **1.30x faster** |
| upstream (repeat) | 4162.1 ms | 308.5 ms | 1.016 | 13.21 | host drift 2% |

**All four arms produced codec SHA-256 `6c9b286b6d` and 55 frames / 4.4 s of
audio, with no cache overflow**, so this is output-identical, not a quality trade.

The effect is roughly proportional to cache length, which is what a padded
attention buffer predicts: `active_prefix` measured `3.087` at `1024` and `0.763`
at `160`, a `4.05x` change for a `6.4x` change in buffer. The engine was never
slow; it was being asked to attend over a buffer 6 to 100x longer than the
utterance needed.

This puts **TTS RTF at 0.763, below 1.0 with margin**, and it is the first change
that beats upstream rather than merely matching it. The `cuda_graph` talker engine
remains a large regression even at a tight cache, so tight sizing is not a
universal fix for the hand-written engines.

Shipping this needs a sizing policy, because the engines require
`prompt + max_new_tokens <= max_cache_len` and will error otherwise. A fixed 160
works for this reply and would overflow a longer one, so the cache has to be sized
per turn from the prompt length plus an expected-frames budget, with a documented
fallback when the budget is exceeded.

## 2026-09-16 RETRACTION: The active_prefix Win Did Not Replicate

The reversal recorded above is withdrawn. A direct replication with **identical
parameters** (same script body, same model, `cuda:1`, `max_new_tokens 96`,
graphed code predictor, greedy) measured the opposite result:

| arm | tight-cache window | replication window |
|---|---|---|
| upstream | 0.994 / 1.016 | 1.486 / 1.484 |
| `active_prefix` @ 160 | **0.763** (`0.77x` upstream) | **3.120** (`2.10x` upstream) |

Both windows bracketed the engine arm with two upstream arms, and in both the
bracket was stable to 2%, so this is not host drift being mistaken for a ratio.
The **ratio itself** moved by `2.7x` between windows, which host state cannot
explain.

The cache-length sweep also fails to show the proportionality I inferred from two
points. Against a bracketed upstream of `1.485`:

| max_cache_len | total RTF | vs upstream |
|---|---|---|
| 128 | 2.063 | 1.39x slower |
| 160 | 3.120 | 2.10x slower |
| 256 | 3.174 | 2.14x slower |
| 512 | 3.148 | 2.12x slower |
| 1024 | 3.182 | 2.14x slower |

`160` through `1024` are flat, not proportional to buffer length. Only `128`
differs, and it is still slower than upstream. So padded attention is not the
explanation either, and the original refutation of the buffer hypothesis stands
after all.

What this actually establishes is that **`active_prefix` is bimodal**, which
`run_e2e_rtf2.sh` was already written to investigate. It sometimes runs near
`0.76` and sometimes near `3.15` under parameters that do not differ, so no single
measurement of it means anything and it cannot be recommended. Codec hashes were
identical (`6c9b286b6d`) in every arm of both windows, so the variance is in
timing rather than in what is computed.

Methodological correction for my own earlier reasoning: a bracketed in-window
ratio is necessary but **not sufficient** when a component is bimodal. Ratios need
replication across windows before they support a conclusion, and I recorded one
before replicating it.

Standing conclusion, unchanged: upstream remains the engine to use, and the
talker step still has no measured path below RTF 1.0 that survives replication.

## 2026-09-16 Cleanup And SOTA Regression, Semantics Verified

The talker phase changed no production code, so there was nothing to roll back
there. It did surface one real config defect: **`cuda0-throughput` still shipped
`tts_outer_active_prefix_talker_engine: True`**, the engine measured at `2.26x`
upstream and since shown to be bimodal. Both that profile and the test that
asserted the flag was `True` were locking in a measured regression. Fixed, with a
new test asserting **no** profile enables either hand-written outer talker engine.
Removed the dead compile probe and the one-off sweep drivers.

Paired regression in one window, no extra flags, naive arms bracketing:

| arm | first audio | total turn | tts_rtf | prompt == final transcript |
|---|---|---|---|---|
| naive (`compat`) | 15870.3 ms | 15872.3 ms | 2.644 | yes |
| **`low-latency`** | **3602.2 ms** | **9968.9 ms** | **1.369** | **yes** |
| naive (repeat) | 12142.8 ms | 12143.7 ms | 1.940 | yes |

**Semantics hold.** All three arms prompted the LLM with the complete 22-word
transcript and produced the identical reply, so the speedup is not paid for with a
different answer.

| | range | mean |
|---|---|---|
| first audio | 3.37x - 4.41x | **3.89x** |
| total turn | 1.22x - 1.59x | **1.41x** |

The range is wide because the two naive arms differ by `1.31x` from host drift
alone, which is the same confound as everywhere else on this node. Quote the range,
not the mean.

Also corrected a metric that would mislead: the naive arm delivers its whole reply
in **one** chunk, so `playback_gapless` reported `yes` for it. That is the absence
of streaming rather than a virtue, so the verdict now requires at least two chunks.
The streaming profile genuinely starves (`-1325.2 ms`), which is what
`--tts-playback-preroll-ms` exists to fix.

## 2026-09-16 The Proposed Cloud-Edge Pair Cannot Simulate Cloud-Edge Latency

Measured the link to the intended 4090 edge host before designing anything:

- ICMP works: **RTT 0.252 ms avg, mdev 0.016 ms**, 0% loss over 5 packets.
- SSH is refused on `22` and on `2222 / 22022 / 8022 / 2200 / 10022`.

Two consequences.

**The link is 40-200x faster than the WAN it is meant to model.** A real
edge-to-cloud hop is `10-50 ms`; this is `0.25 ms` with `0.016 ms` of jitter, so
the pair would measure a rack-local process split, not cloud-edge. Any latency
conclusion drawn from it would flatter the architecture. Simulating the intended
topology requires injected delay and jitter (`tc netem` on the edge interface),
and the injected numbers then become the thing under test.

**And the split does not attack the measured bottleneck.** ASR already runs
concurrently with speech at a ratio of `1.004`, so it contributes nothing to the
post-speech wait; moving it to the edge removes work that was already hidden.
After speech ends the wait is `96.5%` TTS and `3.65%` LLM, and TTS stays in the
cloud, so RTF `1.37` is unchanged by the topology.

What the split does change, and it is not in its favour: the reply audio now
crosses the network to reach the speaker, so **network jitter adds directly to the
playback buffer deficit**, which already measures `-1325.2 ms` on this host. The
preroll gate becomes more necessary under cloud-edge, not less.

Two findings do become more load-bearing in this topology:

- The edge must decide what to send and when, and the committed prefix was
  measured to diverge from the final transcript at word 11 of 21, so the
  divergence check has to live in the protocol rather than be assumed away.
- ASR on the 4090 must hold RTF below 1.0 to keep pace with speech, which is
  unmeasured; the A100 currently does it at `0.33`.

## 2026-09-16 Cloud-edge ASR is reachable, and it does not change the bottleneck

The 4090 edge host answers SSH on port 1212 (22 is closed). ICMP RTT is still
`0.25 ms`, so this is a rack-local split rather than a WAN. Two idle RTX 4090s
were free. The edge env is torch 2.3.1+cu118 with flash-attn, and the ASR
checkpoint is the same 0.6B used on the cloud.

A length-prefixed JSON RPC (`qwen_asr_vllm/agent/remote_asr.py`) carries PCM as
float32 and reconstructs `StreamEvent`s. The edge process is
`bench/serve_edge_asr.py`. Cloud timing takes `--asr-remote host:port`. First
bring-up hit Triton's `libcuda.so` (the box only ships `libcuda.so.1`);
`TRITON_LIBCUDA_PATH` pointing at a user-local symlink unblocked it.

Smoke: 400 ms of silence round-tripped in `66.3 ms` through an SSH local tunnel.
End-to-end, `--realtime-input`, `low-latency` profile, LLM/TTS still on this host:

| | first audio | total | asr_rtf | llm_wall | tts_rtf | prompt == final |
|---|---|---|---|---|---|---|
| local paced ASR | 9082.2 ms | 15266.4 ms | 1.003 | 255.3 ms | 1.338 | yes |
| edge 4090 ASR | 9023.4 ms | 15131.5 ms | 1.000 | 257.2 ms | 1.315 | yes |

Reply text and TTS bytes are identical to the local paced arm.
`first_audio_after_input_end` is `771 ms`. The edge ASR keeps real time. The
cloud TTS stage still starves (`buffer -1066 ms`).

So the previous optimizations still matter: they live on the cloud LLM/TTS path,
which is unchanged by where ASR runs. The split does not fix RTF or first audio,
because ASR was already hidden under speech. A real WAN would make playback
jitter worse; this link is 40-200x too fast to stand in for one.

## 2026-09-16 Edge TTS Fits On A 4090 And Drops RTF Below 1.0

Capacity, measured rather than guessed:

| | |
|---|---|
| checkpoint on disk | **4.3 GB** (`Qwen3-TTS-12Hz-1.7B-CustomVoice`) |
| resident VRAM on RTX 4090 | **5075 MiB** (~5.0 GB) after warmup + code-predictor graphs |
| ASR already on the other 4090 | 6071 MiB |
| 4090 card size | 24564 MiB |

TTS is small next to the card: one 4090 holds it with ~19 GB free, and ASR+TTS on two cards leaves more than half of each empty. They would also fit on **one** 24 GB card (~11 GB together) if needed.

Isolated 4090 synthesis of the same sentence, greedy, chunk 8 / first chunk 2, graphed code predictor: **5040 ms of audio in 3790 ms, RTF 0.752**. The A100 on this throttled host was 1.31–1.34.

End-to-end, `--realtime-input`, LLM still on this host, ASR+TTS on the 4090s, identical 22-word prompt and identical TTS bytes (`242316`):

| | first after stop | total turn | tts_rtf | playback |
|---|---|---|---|---|
| all-local paced | 809 ms | 15266 ms | 1.338 | starves (−1142 ms) |
| edge ASR, cloud TTS | 771 ms | 15131 ms | 1.315 | starves (−1066 ms) |
| **edge ASR+TTS, cloud LLM** | **498 ms** | **12141 ms** | **0.747** | **gapless (+160 ms)** |

Two effects, not one: the 4090 is not in `sw_thermal_slowdown`, so the talker keeps up with speech; and taking TTS off the A100s also cut LLM wall from 257 ms to 123 ms (less contention). The previous optimizations still apply — `final` trigger, codec-step, graphed code predictor — they just run on a cooler device.

Caveats: wav chunks are still pulled back through the SSH tunnel for the harness, so this is edge *generation* with cloud-side timestamps, not a speaker process on the 4090. LAN RTT is 0.25 ms; a WAN would add downlink jitter on those same chunks if playback stayed in the cloud, and would add nothing extra if playback is truly local to the 4090. `tc netem` is still required before calling this a WAN result. The edge env needed `transformers 4.57.3` and a cu118 `torchaudio`; SoX is missing as a binary (glibc 2.31) but 12 Hz CustomVoice imported without it.
