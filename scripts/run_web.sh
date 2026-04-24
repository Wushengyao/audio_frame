#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
PORT="${1:-8808}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Virtual environment not found: $VENV_PYTHON" >&2
  exit 1
fi

exec "$VENV_PYTHON" "$ROOT_DIR/app.py" --port "$PORT"
