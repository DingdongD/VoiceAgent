set -x
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
for arm in eager graph; do
  if [ "$arm" = "eager" ]; then EAGER="--enforce-eager"; else EAGER=""; fi
  LLM_ENFORCE_EAGER=$([ "$arm" = eager ] && echo true || echo false) \
  $PY bench/nano_llm_batching_probe.py \
    --model-path /mnt/llm_data/Qwen3-0.6B \
    --device cuda:2 --concurrency 1 --max-new-tokens 32 \
    --max-num-seqs 4 $EAGER \
    --out results/streaming2026/llm_probe_$arm.json 2>&1 | tail -5
done
