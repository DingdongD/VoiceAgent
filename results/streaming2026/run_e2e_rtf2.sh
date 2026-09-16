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
  # The first pair could not separate engine cost from host contention, so each
  # arm now records real achievable TFLOPs immediately before it runs.
  $PY bench/gpu_contention_probe.py --iters 10 --repeats 3 \
    --output "$OUT/probe_${name}.json" > /dev/null 2>&1
  $PY bench/voice_agent_timing.py \
    --mode real --real-target async --runtime-profile low-latency \
    --audio results/cuda0_librispeech_sample.wav --timeout 420 \
    --llm-trigger final --tts-max-new-tokens 64 "$@" \
    > "$OUT/arm_${name}.log" 2>&1
  echo "$name exit=$?"
}

for rep in 3 4 5; do
  run_arm "RTF_prefix_r${rep}"
  run_arm "RTF_upstream_r${rep}" --no-tts-outer-active-prefix-talker-engine
done

echo E2E_RTF2_COMPLETE
