set -u
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
OUT=results/streaming2026
TEXT="Yes, the boy wills the item, and Montfiche feels too ill to oppose it."

# 64 codec frames covers this reply's 55 frames, and is what the voice agent
# benchmark already passes as --tts-max-new-tokens.
COMMON="--device cuda:1 --greedy --batch-sizes 1 --chunk-size 8 --left-context-size 4 \
  --cuda-graph-code-predictor --cuda-graph-fixed-slots 2 --max-new-tokens 64"

run() {
  name=$1; shift
  $PY bench/qwen_tts_internal_profile.py \
    --texts "$TEXT" $COMMON "$@" \
    --out "$OUT/ttsrtf_${name}.json" \
    > "$OUT/ttsrtf_${name}.log" 2>&1
  echo "$name exit=$?"
}

# Control: same generation cap, no outer engine.
run hf_mnt64            --outer-graph-max-cache-len 16384
# The static cache sized to the actual budget instead of a worst case.
run prefix_mnt64_c1024  --outer-active-prefix-talker-engine --outer-graph-max-cache-len 1024
run prefix_mnt64_c16384 --outer-active-prefix-talker-engine --outer-graph-max-cache-len 16384
run graph_mnt64_c1024   --outer-cuda-graph-talker-engine    --outer-graph-max-cache-len 1024

echo TTSRTF5_COMPLETE
