#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
HOST="${AUDIO_FRAME_HOST:-0.0.0.0}"
PORT="${1:-${AUDIO_FRAME_PORT:-8808}}"
MODEL_ID="${AUDIO_FRAME_MODEL_ID:-openbmb/VoxCPM2}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Virtual environment not found: $VENV_PYTHON" >&2
  echo "Create one with: uv venv --python 3.12 \"$ROOT_DIR/.venv\"" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_PYTHON" -m voxcpm.http_api --host "$HOST" --port "$PORT" --model-id "$MODEL_ID"
