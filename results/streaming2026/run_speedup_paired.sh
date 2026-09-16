set -u
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
export ASR_NUM_KVCACHE_BLOCKS=128 LLM_NUM_KVCACHE_BLOCKS=64 LLM_TEMPERATURE=0
OUT=results/streaming2026

wait_gpu() {
  for dev in 0 1 2; do
    while [ "$(nvidia-smi --id=$dev --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 25000 ]; do sleep 5; done
  done
}

run_arm() {
  name=$1; shift
  wait_gpu
  $PY bench/gpu_contention_probe.py --iters 10 --repeats 3 \
    --output "$OUT/probe_${name}.json" > /dev/null 2>&1
  $PY bench/voice_agent_timing.py \
    --mode real --audio results/cuda0_librispeech_sample.wav --timeout 600 \
    --llm-trigger final --tts-max-new-tokens 64 "$@" \
    > "$OUT/arm_${name}.log" 2>&1
  echo "$name exit=$?"
}

# The 18198.2 ms naive figure was measured in an unknown host state, and this
# host varies by ~2x, so the baseline is re-measured beside the candidate.
for rep in 1 2; do
  LLM_ENFORCE_EAGER=true run_arm "SPD_naive_r${rep}" \
    --real-target sync --runtime-profile compat
  LLM_ENFORCE_EAGER=false run_arm "SPD_profile_r${rep}" \
    --real-target async --runtime-profile low-latency
done

echo SPEEDUP_COMPLETE
