#!/usr/bin/env bash
# Two questions in one batch.
#
# 1. ASR commit policy. After the streaming fixes, ASR-to-first-committed is
#    ~405 ms, 42% of first audio, and has never been tuned. The commit gates
#    (commit_lag_words / min_committed_words / min_committed_audio_seconds) are
#    already at their most permissive in this harness, so the untested lever is
#    how much audio each re-decode covers: --chunk-ms.
# 2. Naive baseline. `--real-target sync` is the pre-streaming orchestration:
#    whole transcript, then LLM, then blocking per-sentence TTS.
set -u
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
OUT=results/streaming2026
AUDIO=results/cuda0_librispeech_sample.wav

export ASR_NUM_KVCACHE_BLOCKS=128
export LLM_NUM_KVCACHE_BLOCKS=64
export LLM_TEMPERATURE=0

wait_for_free_gpu () {
  for _ in $(seq 1 90); do
    busy=0
    for dev in 0 1 2; do
      free=$(nvidia-smi --id=$dev --query-gpu=memory.free --format=csv,noheader,nounits)
      if [ "$free" -lt 30000 ]; then busy=1; fi
    done
    if [ "$busy" -eq 0 ]; then return 0; fi
    sleep 5
  done
  echo "WARNING: GPUs did not free up"
  return 1
}

run () {
  name="$1"; shift
  wait_for_free_gpu
  echo "=== ARM $name : $* ==="
  LLM_ENFORCE_EAGER=${EAGER:-false} \
    $PY bench/voice_agent_timing.py --mode real --audio "$AUDIO" --timeout 480 "$@" \
    > "$OUT/arm_$name.log" 2>&1
  echo "exit=$? -> $OUT/arm_$name.log"
  sleep 10
}

# ---- Q2: naive baseline. Sync orchestration, eager LLM, no TTS streaming. ----
EAGER=true run N_naive_sync --real-target sync --runtime-profile compat \
  --llm-trigger final --tts-max-new-tokens 64

# ---- Q1: ASR chunk-size sweep on the optimized path ----
for ms in 200 800 1600; do
  run S_chunk$ms --real-target async --runtime-profile low-latency \
    --llm-trigger committed --tts-max-new-tokens 64 --chunk-ms $ms
done

# Does the speculative-draft policy help or hurt the early commit?
run S_retrans --real-target async --runtime-profile low-latency \
  --llm-trigger committed --tts-max-new-tokens 64 --chunk-ms 800 \
  --asr-policy retranscribe

# Stricter commit gate, to confirm the permissive default is right
run S_lag2 --real-target async --runtime-profile low-latency \
  --llm-trigger committed --tts-max-new-tokens 64 --chunk-ms 800 \
  --commit-lag-words 2 --min-committed-words 3

# ---- Honest live-session number: audio arrives in real time ----
run R_realtime --real-target async --runtime-profile low-latency \
  --llm-trigger committed --tts-max-new-tokens 64 --chunk-ms 400 --realtime-input

echo "ASR_SWEEP COMPLETE"
