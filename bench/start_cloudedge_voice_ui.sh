#!/usr/bin/env bash
# Run on the cloud host. Keeps SSH tunnels and the latency UI resident so a
# browser can use its own microphone and speaker against edge ASR/TTS.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${PYTHON:-/opt/conda/envs/nano-vllm/bin/python}"
UI_HOST="${UI_HOST:-0.0.0.0}"
UI_PORT="${UI_PORT:-8010}"
ASR_REMOTE="${ASR_REMOTE:-127.0.0.1:18765}"
TTS_REMOTE="${TTS_REMOTE:-127.0.0.1:18766}"
ASR_PORT="${ASR_REMOTE##*:}"
TTS_PORT="${TTS_REMOTE##*:}"
EDGE_HOST="${EDGE_HOST:-10.13.70.50}"
EDGE_USER="${EDGE_USER:-xuliangyu}"
EDGE_SSH_PORT="${EDGE_SSH_PORT:-1212}"
LOG_DIR="${LOG_DIR:-$ROOT/results/cloudedge}"
UI_LOG="${UI_LOG:-$LOG_DIR/voice-ui.log}"
UI_PID_FILE="${UI_PID_FILE:-$LOG_DIR/voice-ui.pid}"
TUNNEL_PID_FILE="${TUNNEL_PID_FILE:-$LOG_DIR/edge-tunnel.pid}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

port_open() {
  local host="$1" port="$2"
  "$PYTHON" - "$host" "$port" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket()
sock.settimeout(0.4)
try:
    sock.connect((host, port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

ensure_tunnels() {
  if port_open 127.0.0.1 "$ASR_PORT" && port_open 127.0.0.1 "$TTS_PORT"; then
    echo "edge tunnels already up on $ASR_REMOTE and $TTS_REMOTE"
    return 0
  fi
  echo "opening SSH tunnels to ${EDGE_USER}@${EDGE_HOST}:${EDGE_SSH_PORT}"
  SSH=(ssh -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30
       -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
       -p "$EDGE_SSH_PORT"
       -L "127.0.0.1:${ASR_PORT}:127.0.0.1:${ASR_PORT}"
       -L "127.0.0.1:${TTS_PORT}:127.0.0.1:${TTS_PORT}"
       "${EDGE_USER}@${EDGE_HOST}")
  if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null; then
    sshpass -e "${SSH[@]}"
  else
    "${SSH[@]}"
  fi
  sleep 0.5
  port_open 127.0.0.1 "$ASR_PORT"
  port_open 127.0.0.1 "$TTS_PORT"
  echo "tunnels ready"
}

already_ui() {
  if [[ -f "$UI_PID_FILE" ]] && kill -0 "$(cat "$UI_PID_FILE")" 2>/dev/null; then
    return 0
  fi
  port_open 127.0.0.1 "$UI_PORT"
}

ensure_tunnels

if already_ui; then
  echo "voice UI already resident on ${UI_HOST}:${UI_PORT}"
else
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" nohup "$PYTHON" \
    "$ROOT/bench/serve_voice_agent_ui.py" \
    --runtime-profile low-latency \
    --host "$UI_HOST" --port "$UI_PORT" \
    --asr-remote "$ASR_REMOTE" --tts-remote "$TTS_REMOTE" \
    >"$UI_LOG" 2>&1 &
  echo $! >"$UI_PID_FILE"
  echo "started voice UI pid=$(cat "$UI_PID_FILE") log=$UI_LOG"
fi

echo "open http://<cloud-host>:${UI_PORT}/agent-latency"
echo "mic and speaker stay in the browser; ASR/TTS stay on the edge; LLM stays here"
