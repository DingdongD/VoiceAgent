set -u
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
export ASR_NUM_KVCACHE_BLOCKS=128 LLM_NUM_KVCACHE_BLOCKS=64 LLM_TEMPERATURE=0 LLM_ENFORCE_EAGER=false
wait_gpu () { for _ in $(seq 1 90); do b=0; for d in 0 1 2; do f=$(nvidia-smi --id=$d --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -lt 30000 ] && b=1; done; [ "$b" -eq 0 ] && return 0; sleep 5; done; }
BASE="--mode real --real-target async --runtime-profile low-latency
      --audio results/cuda0_librispeech_sample.wav --timeout 480
      --llm-trigger committed --tts-max-new-tokens 64 --chunk-ms 800"

# Sentence-boundary segmentation only: does dropping char-level flushing cut the
# inflated total audio without giving back first-audio latency?
wait_gpu
echo "=== ARM T_flushoff ==="
$PY bench/voice_agent_timing.py $BASE --tts-flush-chars 0 --tts-flush-after-ms 0 \
  > results/streaming2026/arm_T_flushoff.log 2>&1
echo "exit=$?"
sleep 10

# Repeat the flush-on control so both are measured in the same window.
wait_gpu
echo "=== ARM T_flushon ==="
$PY bench/voice_agent_timing.py $BASE \
  > results/streaming2026/arm_T_flushon.log 2>&1
echo "exit=$?"
echo SEGMENTATION_COMPLETE
