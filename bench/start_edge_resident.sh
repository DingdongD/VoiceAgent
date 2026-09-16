#!/usr/bin/env bash
# Run on the edge 4090 host. Keeps ASR and TTS loaded as resident TCP services.
set -euo pipefail

ROOT="${ROOT:-$HOME/cloudedge/qwen-asr-vllm}"
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-asredge}"
ASR_LOG="${ASR_LOG:-$HOME/cloudedge/edge-asr.log}"
TTS_LOG="${TTS_LOG:-$HOME/cloudedge/edge-tts.log}"
ASR_PID_FILE="${ASR_PID_FILE:-$HOME/cloudedge/edge-asr.pid}"
TTS_PID_FILE="${TTS_PID_FILE:-$HOME/cloudedge/edge-tts.pid}"
ASR_PORT="${ASR_PORT:-18765}"
TTS_PORT="${TTS_PORT:-18766}"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCHDYNAMO_DISABLE=1
export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-$HOME/.local/lib}"
export LD_LIBRARY_PATH="${TRITON_LIBCUDA_PATH}:${LD_LIBRARY_PATH:-}"
export PATH="${HOME}/.local/bin:${PATH}"
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

already_listening() {
  local port="$1"
  python - "$port" <<'PY'
import socket, sys
port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(0.4)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

start_one() {
  local name="$1" port="$2" pid_file="$3" log_file="$4"
  shift 4
  if already_listening "$port"; then
    echo "$name already listening on 127.0.0.1:$port"
    return 0
  fi
  nohup python "$@" >"$log_file" 2>&1 &
  echo $! >"$pid_file"
  echo "started $name pid=$(cat "$pid_file") log=$log_file"
}

start_one asr "$ASR_PORT" "$ASR_PID_FILE" "$ASR_LOG" \
  bench/serve_edge_asr.py --host 127.0.0.1 --port "$ASR_PORT" --device "${ASR_DEVICE:-cuda:0}"
start_one tts "$TTS_PORT" "$TTS_PID_FILE" "$TTS_LOG" \
  bench/serve_edge_tts.py --host 127.0.0.1 --port "$TTS_PORT" --device "${TTS_DEVICE:-cuda:1}"
echo "edge resident ASR :$ASR_PORT TTS :$TTS_PORT"
