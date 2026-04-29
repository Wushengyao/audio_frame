#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${AUDIO_FRAME_MODE:-webui}"

case "$MODE" in
  api|webui)
    ;;
  *)
    echo "Unsupported AUDIO_FRAME_MODE: $MODE (expected api or webui)" >&2
    exit 2
    ;;
esac

python_supports_mode() {
  local python_bin="$1"
  local mode="$2"
  "$python_bin" - "$mode" >/dev/null 2>&1 <<'PY'
import importlib.util
import sys

mode = sys.argv[1]
required = ["torch", "transformers", "soundfile"]
if mode == "api":
    required.extend(["fastapi", "uvicorn"])
else:
    required.append("gradio")

missing = [name for name in required if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
PY
}

resolve_python() {
  if [[ -n "${AUDIO_FRAME_PYTHON:-}" ]]; then
    if [[ ! -x "$AUDIO_FRAME_PYTHON" ]]; then
      echo "AUDIO_FRAME_PYTHON is not executable: $AUDIO_FRAME_PYTHON" >&2
      return 1
    fi
    echo "$AUDIO_FRAME_PYTHON"
    return 0
  fi

  local candidates=(
    "$ROOT_DIR/.venv/bin/python"
    "$ROOT_DIR/../VoxCPM2/.venv/bin/python"
  )
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]] && python_supports_mode "$candidate" "$MODE"; then
      echo "$candidate"
      return 0
    fi
  done

  cat >&2 <<EOF
No usable Python runtime found for Audio Frame mode '$MODE'.
Set AUDIO_FRAME_PYTHON=/path/to/python, or create a local venv:
  cd "$ROOT_DIR"
  python3 -m venv .venv
  . .venv/bin/activate
  pip install -U pip
  pip install -e .
EOF
  return 1
}

VENV_PYTHON="$(resolve_python)"

export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}"
export AUDIO_FRAME_DEVICE="${AUDIO_FRAME_DEVICE:-auto}"
export AUDIO_FRAME_MODEL_ID="${AUDIO_FRAME_MODEL_ID:-openbmb/VoxCPM2}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${AUDIO_FRAME_DEVICE}" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
  export AUDIO_FRAME_NO_OPTIMIZE="${AUDIO_FRAME_NO_OPTIMIZE:-1}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${AUDIO_FRAME_TORCH_THREADS:-7}}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${AUDIO_FRAME_TORCH_THREADS:-7}}"
fi

if [[ "$MODE" == "api" ]]; then
  exec "$VENV_PYTHON" -m uvicorn voxcpm.http_api:app \
    --host "${AUDIO_FRAME_HOST:-0.0.0.0}" \
    --port "${PORT:-8808}" \
    --workers "${AUDIO_FRAME_WORKERS:-1}"
else
  exec "$VENV_PYTHON" "$ROOT_DIR/app.py" \
    --port "${PORT:-8808}" \
    --device "${AUDIO_FRAME_DEVICE}"
fi
