#!/usr/bin/env bash
# Arm E is TTS-dominated. The first emitted chunk needs `first_chunk_size`
# codec frames before any waveform exists, so that value sets a floor on first
# audio. Sweep it against the otherwise-identical arm E configuration.
set -u
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
OUT=results/streaming2026
AUDIO=results/cuda0_librispeech_sample.wav

export ASR_NUM_KVCACHE_BLOCKS=128
export LLM_NUM_KVCACHE_BLOCKS=64
export LLM_TEMPERATURE=0
export LLM_ENFORCE_EAGER=false

COMMON="--mode real --real-target async --runtime-profile compat --audio $AUDIO
        --timeout 420 --llm-trigger committed --tts-max-new-tokens 64
        --tts-streaming-engine codec-step
        --tts-cuda-graph-code-predictor --tts-cuda-graph-fixed-slots 2
        --tts-outer-active-prefix-talker-engine --no-tts-stream-batch-exact-parity
        --no-tts-do-sample --no-tts-subtalker-do-sample --tts-temperature 0.0
        --tts-flush-chars 12 --tts-flush-after-ms 250 --tts-flush-min-chars 5
        --tts-first-sentence-immediate"

wait_for_free_gpu () {
  for _ in $(seq 1 60); do
    busy=0
    for dev in 0 1 2; do
      free=$(nvidia-smi --id=$dev --query-gpu=memory.free --format=csv,noheader,nounits)
      if [ "$free" -lt 30000 ]; then busy=1; fi
    done
    if [ "$busy" -eq 0 ]; then return 0; fi
    sleep 5
  done
  return 1
}

for first in 4 2; do
  wait_for_free_gpu
  echo "=== ARM G_first$first ==="
  $PY bench/voice_agent_timing.py $COMMON \
    --tts-stream-first-chunk-size $first \
    > "$OUT/arm_G_first$first.log" 2>&1
  echo "exit=$? -> $OUT/arm_G_first$first.log"
  sleep 10
done
echo "FIRSTCHUNK COMPLETE"
