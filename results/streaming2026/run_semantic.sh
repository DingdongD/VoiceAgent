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
  $PY bench/voice_agent_timing.py \
    --mode real --real-target async --runtime-profile low-latency \
    --audio results/cuda0_librispeech_sample.wav --timeout 420 \
    --tts-max-new-tokens 64 "$@" \
    > "$OUT/arm_${name}.log" 2>&1
  echo "$name exit=$?"
}

# Reference semantics: the LLM sees the whole utterance.
run_arm SEM_final --llm-trigger final

# Current harness/profile default: fires on the first committed word.
run_arm SEM_open --llm-trigger committed

# Same early-trigger machinery, but gated until most of the utterance exists.
run_arm SEM_gated --llm-trigger committed \
  --min-committed-words 12 --min-committed-audio-seconds 6.0

echo SEMANTIC_COMPLETE
