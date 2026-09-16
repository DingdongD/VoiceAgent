set -u
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
export ASR_NUM_KVCACHE_BLOCKS=128 LLM_NUM_KVCACHE_BLOCKS=64 LLM_TEMPERATURE=0 LLM_ENFORCE_EAGER=false
OUT=results/streaming2026

for dev in 0 1 2; do
  while [ "$(nvidia-smi --id=$dev --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 25000 ]; do sleep 5; done
done

# No TTS or trigger flags: whatever this measures is what the named profile ships.
for rep in 1 2; do
  $PY bench/voice_agent_timing.py \
    --mode real --real-target async --runtime-profile low-latency \
    --audio results/cuda0_librispeech_sample.wav --timeout 420 \
    --tts-max-new-tokens 64 \
    > "$OUT/arm_VERIFY_profile_r${rep}.log" 2>&1
  echo "verify_r${rep} exit=$?"
done

echo VERIFY_COMPLETE
