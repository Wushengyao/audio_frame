#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8808}"

python_supports_webui() {
  local python_bin="$1"
  "$python_bin" - >/dev/null 2>&1 <<'PY'
import importlib.util
required = ["torch", "transformers", "soundfile", "gradio"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
PY
}

if [[ -n "${AUDIO_FRAME_PYTHON:-}" ]]; then
  VENV_PYTHON="$AUDIO_FRAME_PYTHON"
else
  VENV_PYTHON=""
  candidates=("$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/../VoxCPM2/.venv/bin/python")
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]] && python_supports_webui "$candidate"; then
      VENV_PYTHON="$candidate"
      break
    fi
  done
fi

if [[ -z "$VENV_PYTHON" || ! -x "$VENV_PYTHON" ]]; then
  echo "No usable Python runtime found. Set AUDIO_FRAME_PYTHON=/path/to/python or create .venv." >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$VENV_PYTHON" "$ROOT_DIR/app.py" \
  --port "$PORT" \
  --device "${AUDIO_FRAME_DEVICE:-auto}"
