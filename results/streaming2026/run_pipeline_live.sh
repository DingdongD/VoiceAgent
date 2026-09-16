set -u
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
export ASR_NUM_KVCACHE_BLOCKS=128 LLM_NUM_KVCACHE_BLOCKS=64 LLM_TEMPERATURE=0 LLM_ENFORCE_EAGER=false
OUT=results/streaming2026

for dev in 0 1 2; do
  while [ "$(nvidia-smi --id=$dev --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 25000 ]; do sleep 5; done
done

# --realtime-input feeds the 8.25 s file at wall-clock speed, as a microphone
# does. Without it, input finishes in ~2.7 s and the ASR/LLM boundary is not
# being exercised under the timing a live session actually has.
$PY bench/voice_agent_timing.py \
  --mode real --real-target async --runtime-profile low-latency \
  --audio results/cuda0_librispeech_sample.wav --timeout 600 \
  --tts-max-new-tokens 64 --realtime-input \
  > "$OUT/arm_LIVE_final_paced.log" 2>&1
echo "live_final exit=$?"

echo PIPELINE_LIVE_COMPLETE
