set -u
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
export ASR_NUM_KVCACHE_BLOCKS=128 LLM_NUM_KVCACHE_BLOCKS=64 LLM_TEMPERATURE=0 LLM_ENFORCE_EAGER=false
OUT=results/streaming2026

wait_gpu() {
  for dev in 0 1 2; do
    while [ "$(nvidia-smi --id=$dev --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 30000 ]; do sleep 5; done
  done
}

run_arm() {
  name=$1; shift
  wait_gpu
  # llm_trigger=final is the only semantically honest setting, so every arm here
  # answers the whole utterance and the replies stay comparable.
  $PY bench/voice_agent_timing.py \
    --mode real --real-target async --runtime-profile low-latency \
    --audio results/cuda0_librispeech_sample.wav --timeout 420 \
    --llm-trigger final --tts-max-new-tokens 64 "$@" \
    > "$OUT/arm_${name}.log" 2>&1
  echo "$name exit=$?"
}

# Interleaved repeats: this host drifts, so arms must be compared pairwise.
for rep in 1 2; do
  run_arm "RTF_prefix_r${rep}"
  run_arm "RTF_upstream_r${rep}" --no-tts-outer-active-prefix-talker-engine
done

echo E2E_RTF_COMPLETE
