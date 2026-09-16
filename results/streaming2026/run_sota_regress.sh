set -u
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
export ASR_NUM_KVCACHE_BLOCKS=128 LLM_NUM_KVCACHE_BLOCKS=64 LLM_TEMPERATURE=0 LLM_ENFORCE_EAGER=false
OUT=results/streaming2026

# Paired in one window: the naive documented recipe against the shipped profile,
# with contention probes so host state is recorded rather than assumed. No extra
# TTS or trigger flags, so whatever this measures is what the profile ships.
run() {
  name=$1; shift
  for dev in 0 1 2; do
    while [ "$(nvidia-smi --id=$dev --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 25000 ]; do sleep 5; done
  done
  $PY bench/gpu_contention_probe.py > "$OUT/probe_sota_${name}.json" 2>&1
  $PY bench/voice_agent_timing.py --mode real --real-target async \
    --audio results/cuda0_librispeech_sample.wav --timeout 420 "$@" \
    > "$OUT/arm_SOTA_${name}.log" 2>&1
  echo "$name exit=$?"
}

run naive   --runtime-profile compat
run profile --runtime-profile low-latency --tts-max-new-tokens 64
run naive2  --runtime-profile compat

echo SOTA_REGRESS_COMPLETE
