set -u
PY=/opt/conda/envs/nano-vllm/bin/python
cd /home/qwen-asr-vllm
OUT=results/streaming2026
T="Yes, the boy wills the item, and Montfiche feels too ill to oppose it."
MODEL=/mnt/llm_data/voice_ckpt/qwen3_tts/Qwen3-TTS-12Hz-1.7B-CustomVoice

COMMON="--model-path $MODEL --device cuda:1 --language chinese --greedy \
  --chunk-size 8 --left-context-size 4 --batch-sizes 1 --max-new-tokens 512 \
  --cuda-graph-code-predictor --cuda-graph-fixed-slots 2"

while [ "$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 25000 ]; do sleep 5; done

# Without sync, the per-stage CPU timers measure time the CPU spent in the call,
# which for an eager loop is launch time and for a graph replay should be near
# zero. Comparing against a synchronized run separates launch cost from GPU cost
# and shows where the CPU is blocked waiting on the device.
$PY bench/qwen_tts_internal_profile.py --texts "$T" $COMMON \
  --out "$OUT/talker_nosync.json" > "$OUT/talker_nosync.log" 2>&1
echo "nosync exit=$?"

$PY bench/qwen_tts_internal_profile.py --texts "$T" $COMMON --sync-each-step \
  --out "$OUT/talker_sync.json" > "$OUT/talker_sync.log" 2>&1
echo "sync exit=$?"

echo TALKERSYNC_COMPLETE
