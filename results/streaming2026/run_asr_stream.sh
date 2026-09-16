set -u
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
export ASR_NUM_KVCACHE_BLOCKS=128 LLM_NUM_KVCACHE_BLOCKS=64 LLM_TEMPERATURE=0 LLM_ENFORCE_EAGER=false
OUT=results/streaming2026

# Paced feed, so ASR event offsets can be read against the speech clock and the
# prefill lead time of each committed prefix is meaningful.
for dev in 0 1 2; do
  while [ "$(nvidia-smi --id=$dev --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 25000 ]; do sleep 5; done
done
$PY bench/voice_agent_timing.py \
  --mode real --real-target async --runtime-profile low-latency \
  --audio results/cuda0_librispeech_sample.wav --timeout 420 \
  --tts-max-new-tokens 64 --realtime-input \
  > "$OUT/arm_ASRSTREAM.log" 2>&1
echo "asrstream exit=$?"
echo ASRSTREAM_COMPLETE
