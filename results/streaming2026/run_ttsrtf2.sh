set -u
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
OUT=results/streaming2026
# The actual reply the agent produced when prompted with the full transcript.
TEXT="Yes, the boy wills the item, and Montfiche feels too ill to oppose it."

# Flags the low-latency profile turns on, expressed for the internal probe.
# 16384 matches `backing_capacity_tokens` the live TTS service reports; the
# probe's 1024 default overflows on a reply of this length.
COMMON="--device cuda:1 --greedy --batch-sizes 1 \
  --cuda-graph-code-predictor --cuda-graph-fixed-slots 2 \
  --outer-graph-max-cache-len 16384"

run() {
  name=$1; shift
  $PY bench/qwen_tts_internal_profile.py \
    --texts "$TEXT" $COMMON "$@" \
    --out "$OUT/ttsrtf_${name}.json" \
    > "$OUT/ttsrtf_${name}.log" 2>&1
  echo "$name exit=$?"
}

# Reference: no outer talker engine at all.
run hf            --chunk-size 8  --left-context-size 4
# What the low-latency profile actually runs today.
run prefix_c8     --chunk-size 8  --left-context-size 4 --outer-active-prefix-talker-engine
run prefix_c16    --chunk-size 16 --left-context-size 4 --outer-active-prefix-talker-engine
run prefix_c24    --chunk-size 24 --left-context-size 4 --outer-active-prefix-talker-engine
# The talker step itself under a CUDA graph, the lever the profile omits.
run graph_c8      --chunk-size 8  --left-context-size 4 \
                  --outer-active-prefix-talker-engine --outer-cuda-graph-talker-engine
run graph_c16     --chunk-size 16 --left-context-size 4 \
                  --outer-active-prefix-talker-engine --outer-cuda-graph-talker-engine

echo TTSRTF_COMPLETE
