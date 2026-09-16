set -u
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
export ASR_NUM_KVCACHE_BLOCKS=128 LLM_NUM_KVCACHE_BLOCKS=64 LLM_TEMPERATURE=0 LLM_ENFORCE_EAGER=false
OUT=results/streaming2026

# Paired: same window, same profile, only the playback gate differs. The 1200 ms
# target comes from the measured worst-case buffer deficit of -1141.8 ms.
for preroll in 0 1200; do
  for dev in 0 1 2; do
    while [ "$(nvidia-smi --id=$dev --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 25000 ]; do sleep 5; done
  done
  $PY bench/gpu_contention_probe.py > "$OUT/probe_preroll_${preroll}.json" 2>&1
  $PY bench/voice_agent_timing.py \
    --mode real --real-target async --runtime-profile low-latency \
    --audio results/cuda0_librispeech_sample.wav --timeout 420 \
    --tts-max-new-tokens 64 --realtime-input \
    --tts-playback-preroll-ms "$preroll" \
    > "$OUT/arm_PREROLL_${preroll}.log" 2>&1
  echo "preroll_${preroll} exit=$?"
done

echo PREROLL_COMPLETE
