#!/usr/bin/env bash
# Streaming ablation ladder. Each arm adds exactly one factor so first-audio
# latency can be attributed. Greedy LLM/TTS keeps the emitted text constant.
#
# Both ASR and nano-vLLM size their KV cache from *currently free* memory, so
# back-to-back arms race a previous arm's teardown. Fixed block budgets plus a
# free-memory gate make each arm deterministic.
set -u
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
OUT=results/streaming2026
AUDIO=results/cuda0_librispeech_sample.wav

export ASR_NUM_KVCACHE_BLOCKS=128
export LLM_NUM_KVCACHE_BLOCKS=64
export LLM_TEMPERATURE=0

COMMON="--mode real --real-target async --runtime-profile compat --audio $AUDIO
        --timeout 420 --llm-trigger committed --tts-max-new-tokens 64"

wait_for_free_gpu () {
  # Every service must be able to load; require headroom on all three devices.
  for _ in $(seq 1 60); do
    busy=0
    for dev in 0 1 2; do
      free=$(nvidia-smi --id=$dev --query-gpu=memory.free --format=csv,noheader,nounits)
      if [ "$free" -lt 30000 ]; then busy=1; fi
    done
    if [ "$busy" -eq 0 ]; then return 0; fi
    sleep 5
  done
  echo "WARNING: GPUs did not return to >=30GB free"
  return 1
}

run () {
  name="$1"; shift
  eager="$1"; shift
  wait_for_free_gpu
  echo "=== ARM $name (LLM_ENFORCE_EAGER=$eager) extra: $* ==="
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
  LLM_ENFORCE_EAGER=$eager \
    $PY bench/voice_agent_timing.py $COMMON "$@" > "$OUT/arm_$name.log" 2>&1
  echo "exit=$? -> $OUT/arm_$name.log"
  sleep 10
}

TTS_GRAPH="--tts-cuda-graph-code-predictor --tts-cuda-graph-fixed-slots 2
           --tts-outer-active-prefix-talker-engine --no-tts-stream-batch-exact-parity
           --no-tts-do-sample --no-tts-subtalker-do-sample --tts-temperature 0.0"
SEGMENT="--tts-flush-chars 12 --tts-flush-after-ms 250 --tts-flush-min-chars 5
         --tts-first-sentence-immediate"

# A': re-run the eager baseline under fixed KV budgets so every arm is comparable
run A_baseline true

# B: A' + LLM CUDA graph replay
run B_llmgraph false

# C: B + TTS codec-step streaming
run C_ttsstream false --tts-streaming-engine codec-step

# D: C + TTS inner CUDA-graph predictor + active-prefix outer talker
run D_ttsgraph false --tts-streaming-engine codec-step $TTS_GRAPH

# E: D + early LLM->TTS segmentation
run E_segment false --tts-streaming-engine codec-step $TTS_GRAPH $SEGMENT

# F: E + incremental ASR ingest (bounded encoder work)
run F_asrincr false --tts-streaming-engine codec-step $TTS_GRAPH $SEGMENT \
  --asr-policy incremental --commit-lag-words 2

echo "LADDER COMPLETE"
