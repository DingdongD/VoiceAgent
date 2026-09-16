set -u
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
OUT=results/streaming2026
export ASR_NUM_KVCACHE_BLOCKS=128 LLM_NUM_KVCACHE_BLOCKS=64 LLM_TEMPERATURE=0 LLM_ENFORCE_EAGER=false
for dev in 0 1 2; do
  while [ "$(nvidia-smi --id=$dev --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 30000 ]; do sleep 5; done
done
$PY bench/voice_agent_timing.py \
  --mode real --real-target async --runtime-profile compat \
  --audio results/cuda0_librispeech_sample.wav --timeout 420 \
  --llm-trigger committed --tts-max-new-tokens 64 \
  --tts-streaming-engine codec-step \
  --tts-cuda-graph-code-predictor --tts-cuda-graph-fixed-slots 2 \
  --tts-outer-active-prefix-talker-engine --no-tts-stream-batch-exact-parity \
  --no-tts-do-sample --no-tts-subtalker-do-sample --tts-temperature 0.0 \
  --tts-flush-chars 12 --tts-flush-after-ms 250 --tts-flush-min-chars 5 \
  --tts-first-sentence-immediate --tts-stream-first-chunk-size 2 \
  --barge-in-policy after-asr-final \
  > "$OUT/arm_H_bargefix.log" 2>&1
echo "exit=$?"
echo FINAL_COMPLETE
