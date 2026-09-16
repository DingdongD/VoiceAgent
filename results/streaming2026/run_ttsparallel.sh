set -u
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
OUT=results/streaming2026
T1="Yes, the boy wills the item, and Montfiche feels too ill to oppose it."
T2="He had no wish to argue the point, and so the matter rested there."

# Upstream path only: both custom outer engines are measured regressions.
COMMON="--device cuda:1 --greedy --chunk-size 8 --left-context-size 4 \
  --cuda-graph-code-predictor --cuda-graph-fixed-slots 2 --max-new-tokens 512"

while [ "$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 25000 ]; do sleep 5; done
$PY bench/gpu_contention_probe.py > "$OUT/probe_ttsparallel.json" 2>&1

# The load-bearing question: is one card already saturated by a single stream?
# If batch 2 costs the same wall time as batch 1, spare capacity exists on the
# card and fragment parallelism needs no second device.
$PY bench/qwen_tts_internal_profile.py \
  --texts "$T1" "$T2" $COMMON --batch-sizes 1,2 \
  --out "$OUT/ttspar_batch.json" > "$OUT/ttspar_batch.log" 2>&1
echo "batch exit=$?"

echo TTSPARALLEL_COMPLETE
