#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYSTEMD_DIR="${AUDIO_FRAME_SYSTEMD_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user}"
SERVICE_PREFIX="${AUDIO_FRAME_SERVICE_PREFIX:-audio-frame}"
BASE_PORT="${AUDIO_FRAME_BASE_PORT:-8808}"
GPU_WORKERS_PER_GPU="${AUDIO_FRAME_GPU_WORKERS_PER_GPU:-2}"
GPU_NO_OPTIMIZE="${AUDIO_FRAME_GPU_NO_OPTIMIZE:-1}"
GPU_MIN_CUDA_MEMORY_GB="${AUDIO_FRAME_GPU_MIN_CUDA_MEMORY_GB:-8}"
CPU_PORT="${AUDIO_FRAME_CPU_PORT:-8812}"
CPU_WORKERS="${AUDIO_FRAME_CPU_WORKERS:-4}"
CPU_THREADS="${AUDIO_FRAME_CPU_THREADS:-7}"
ENABLE_NOW=0
INSTALL_CPU=1

usage() {
  cat <<EOF
Usage: $(basename "$0") [--enable-now] [--no-cpu]

Environment:
  AUDIO_FRAME_GPU_IDS              Space-separated physical GPU ids. Auto-detected with nvidia-smi when unset.
  AUDIO_FRAME_GPU_WORKERS_PER_GPU  GPU workers per physical GPU. Default: 2.
  AUDIO_FRAME_GPU_NO_OPTIMIZE      Disable torch.compile for GPU workers. Default: 1.
  AUDIO_FRAME_GPU_MIN_CUDA_MEMORY_GB  Minimum free VRAM gate for each GPU worker. Default: 8.
  AUDIO_FRAME_BASE_PORT            First GPU worker port. Default: 8808.
  AUDIO_FRAME_CPU_PORT             CPU fallback port. Default: 8812.
  AUDIO_FRAME_CPU_WORKERS          Uvicorn workers for CPU fallback. Default: 4.
  AUDIO_FRAME_CPU_THREADS          Torch/BLAS threads per CPU worker. Default: 7.
  AUDIO_FRAME_PYTHON               Optional Python executable to pin in generated units.
  AUDIO_FRAME_SYSTEMD_DIR          Target user unit directory. Default: ~/.config/systemd/user.
  AUDIO_FRAME_SKIP_SYSTEMCTL       Set to 1 to only write unit files.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --enable-now)
      ENABLE_NOW=1
      shift
      ;;
    --no-cpu)
      INSTALL_CPU=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

detect_gpus() {
  if [[ -n "${AUDIO_FRAME_GPU_IDS:-}" ]]; then
    printf '%s\n' "$AUDIO_FRAME_GPU_IDS"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' '
  fi
}

write_unit() {
  local name="$1"
  local port="$2"
  local device="$3"
  local cuda_visible_devices="$4"
  local workers="$5"
  local preload="$6"
  local no_optimize="$7"
  local torch_threads="${8:-}"
  local min_cuda_memory_gb="${9:-}"
  local unit_path="$SYSTEMD_DIR/$name"

  {
    cat <<EOF
[Unit]
Description=Audio Frame ${device} worker on port ${port}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${ROOT_DIR}
ExecStart=${ROOT_DIR}/scripts/run_service.sh
Restart=on-failure
RestartSec=10
Environment=PORT=${port}
Environment=AUDIO_FRAME_MODE=api
Environment=AUDIO_FRAME_DEVICE=${device}
Environment=AUDIO_FRAME_PRELOAD=${preload}
Environment=AUDIO_FRAME_WORKERS=${workers}
Environment=AUDIO_FRAME_NO_OPTIMIZE=${no_optimize}
Environment=AUDIO_FRAME_LOAD_DENOISER=0
Environment=TOKENIZERS_PARALLELISM=false
Environment=CUDA_VISIBLE_DEVICES=${cuda_visible_devices}
EOF
    if [[ -n "$min_cuda_memory_gb" ]]; then
      echo "Environment=AUDIO_FRAME_MIN_CUDA_MEMORY_GB=${min_cuda_memory_gb}"
    fi
    if [[ -n "${AUDIO_FRAME_PYTHON:-}" ]]; then
      echo "Environment=AUDIO_FRAME_PYTHON=${AUDIO_FRAME_PYTHON}"
    fi
    if [[ -n "$torch_threads" ]]; then
      cat <<EOF
Environment=AUDIO_FRAME_TORCH_THREADS=${torch_threads}
Environment=OMP_NUM_THREADS=${torch_threads}
Environment=MKL_NUM_THREADS=${torch_threads}
EOF
    fi
    cat <<EOF
TimeoutStopSec=30

[Install]
WantedBy=default.target
EOF
  } > "$unit_path"
}

mkdir -p "$SYSTEMD_DIR"

read -r -a GPU_IDS <<<"$(detect_gpus)"
SERVICE_NAMES=()
PORTS=()
port="$BASE_PORT"

if [[ "${#GPU_IDS[@]}" -gt 0 && -n "${GPU_IDS[0]}" ]]; then
  for gpu_id in "${GPU_IDS[@]}"; do
    for ((slot = 0; slot < GPU_WORKERS_PER_GPU; slot++)); do
      letter="$(printf "\\$(printf '%03o' $((97 + slot)))")"
      unit="${SERVICE_PREFIX}-gpu${gpu_id}${letter}.service"
      if [[ "${gpu_id}" == "${GPU_IDS[0]}" && "$slot" -eq 0 ]]; then
        unit="${SERVICE_PREFIX}.service"
      fi
      write_unit "$unit" "$port" "cuda" "$gpu_id" "1" "1" "$GPU_NO_OPTIMIZE" "" "$GPU_MIN_CUDA_MEMORY_GB"
      SERVICE_NAMES+=("$unit")
      PORTS+=("$port")
      port=$((port + 1))
    done
  done
else
  echo "No GPUs detected; only CPU fallback will be installed." >&2
fi

if [[ "$INSTALL_CPU" -eq 1 ]]; then
  unit="${SERVICE_PREFIX}-cpu.service"
  write_unit "$unit" "$CPU_PORT" "cpu" "" "$CPU_WORKERS" "0" "1" "$CPU_THREADS"
  SERVICE_NAMES+=("$unit")
  PORTS+=("$CPU_PORT")
fi

if [[ "${AUDIO_FRAME_SKIP_SYSTEMCTL:-0}" == "1" ]]; then
  echo "AUDIO_FRAME_SKIP_SYSTEMCTL=1; wrote units without reloading systemd."
else
  systemctl --user daemon-reload
  if [[ "$ENABLE_NOW" -eq 1 && "${#SERVICE_NAMES[@]}" -gt 0 ]]; then
    systemctl --user enable --now "${SERVICE_NAMES[@]}"
  fi
fi

echo "Installed user units in $SYSTEMD_DIR:"
printf '  %s\n' "${SERVICE_NAMES[@]}"
echo
echo "Health check after start:"
echo "  for port in ${PORTS[*]}; do curl -s http://127.0.0.1:\$port/healthz; echo; done"
