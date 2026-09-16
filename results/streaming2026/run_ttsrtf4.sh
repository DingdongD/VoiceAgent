set -u
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
OUT=results/streaming2026
TEXT="Yes, the boy wills the item, and Montfiche feels too ill to oppose it."

COMMON="--device cuda:1 --greedy --batch-sizes 1 --chunk-size 8 --left-context-size 4 \
  --cuda-graph-code-predictor --cuda-graph-fixed-slots 2 \
  --outer-active-prefix-talker-engine"

run() {
  name=$1; shift
  $PY bench/qwen_tts_internal_profile.py \
    --texts "$TEXT" $COMMON "$@" \
    --out "$OUT/ttsrtf_${name}.json" \
    > "$OUT/ttsrtf_${name}.log" 2>&1
  echo "$name exit=$?"
}

# If the active-prefix engine attends over its whole backing buffer rather than
# the active prefix, its cost scales with this number and not with the utterance.
run prefix_cache2k  --outer-graph-max-cache-len 2048
run prefix_cache4k  --outer-graph-max-cache-len 4096
run prefix_cache8k  --outer-graph-max-cache-len 8192

echo TTSRTF4_COMPLETE
