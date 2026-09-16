set -u
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
OUT=results/streaming2026
TEXT="Yes, the boy wills the item, and Montfiche feels too ill to oppose it."

COMMON="--device cuda:1 --greedy \
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

# The outer engines are mutually exclusive, so the CUDA-graph talker runs alone.
run graph_c8   --batch-sizes 1 --chunk-size 8  --left-context-size 4 \
               --outer-cuda-graph-talker-engine
# Does the upstream path get cheaper per utterance when several are in flight?
run hf_batch   --batch-sizes 1,2,4 --chunk-size 8 --left-context-size 4
# Is the graphed code predictor still helping on the upstream path?
run hf_nograph_cp --batch-sizes 1 --chunk-size 8 --left-context-size 4 --no-warmup

echo TTSRTF3_COMPLETE
