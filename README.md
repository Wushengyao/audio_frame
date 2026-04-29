# Audio Frame Deployment Guide

Audio Frame is a local VoxCPM2 text-to-speech service used by `novel_frame_2`.
This repository keeps the upstream VoxCPM model code, but the deployment entry
points here are focused on running a reliable HTTP TTS pool.

## 1. Install

Use Python 3.10+ on Linux. Python 3.12 is recommended.

```bash
cd /path/to/audio_frame
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e .
```

For CUDA machines, install a PyTorch build that matches the local NVIDIA driver
if the default wheel is not appropriate for your host.

The first request downloads `openbmb/VoxCPM2` unless the model is already cached.
The cache locations can be controlled with `HF_HUB_CACHE` and `MODELSCOPE_CACHE`.

## 2. Run One API Worker

The unified entry point is `scripts/run_service.sh`. It chooses Python in this
order:

1. `AUDIO_FRAME_PYTHON`
2. `./.venv/bin/python`
3. `../VoxCPM2/.venv/bin/python`
4. system `python3`

It skips candidates that do not have the dependencies required by the selected
mode.

GPU API worker:

```bash
cd /path/to/audio_frame
AUDIO_FRAME_MODE=api \
AUDIO_FRAME_DEVICE=cuda \
AUDIO_FRAME_PRELOAD=1 \
PORT=8808 \
scripts/run_service.sh
```

CPU API worker:

```bash
cd /path/to/audio_frame
CUDA_VISIBLE_DEVICES= \
AUDIO_FRAME_MODE=api \
AUDIO_FRAME_DEVICE=cpu \
AUDIO_FRAME_WORKERS=4 \
AUDIO_FRAME_TORCH_THREADS=7 \
PORT=8812 \
scripts/run_service.sh
```

Web UI:

```bash
cd /path/to/audio_frame
PORT=8808 scripts/run_web.sh
```

## 3. GPU + CPU Pool With systemd

The easiest path on a new machine is to generate user units from the current
checkout:

```bash
cd /path/to/audio_frame
AUDIO_FRAME_GPU_IDS="0 1" scripts/install_systemd_user.sh --enable-now
```

When `AUDIO_FRAME_GPU_IDS` is not set, the script tries to detect NVIDIA GPU
ids with `nvidia-smi`. By default it creates two API workers per GPU starting at
port `8808`, plus a CPU fallback on `8812`.

Useful installer overrides:

| Variable | Default | Notes |
| --- | --- | --- |
| `AUDIO_FRAME_GPU_IDS` | auto | Space-separated physical GPU ids, for example `"0 1"` |
| `AUDIO_FRAME_GPU_WORKERS_PER_GPU` | `2` | GPU API processes per GPU |
| `AUDIO_FRAME_GPU_NO_OPTIMIZE` | `1` | Disable `torch.compile` for GPU workers; this is the stable default for multi-process serving |
| `AUDIO_FRAME_GPU_MIN_CUDA_MEMORY_GB` | `8` | Minimum free VRAM gate per worker during preload |
| `AUDIO_FRAME_BASE_PORT` | `8808` | First GPU worker port |
| `AUDIO_FRAME_CPU_PORT` | `8812` | CPU fallback port |
| `AUDIO_FRAME_CPU_WORKERS` | `4` | Uvicorn workers for CPU fallback |
| `AUDIO_FRAME_CPU_THREADS` | `7` | Torch/BLAS threads per CPU worker |
| `AUDIO_FRAME_PYTHON` | auto | Pin a Python executable in generated units |

The generated service names match the default `novel_frame_2` pool:

| Service | Port | `CUDA_VISIBLE_DEVICES` | Purpose |
| --- | --- | --- | --- |
| `audio-frame.service` | `8808` | `0` | GPU 0 worker A |
| `audio-frame-gpu0b.service` | `8809` | `0` | GPU 0 worker B |
| `audio-frame-gpu1a.service` | `8810` | `1` | GPU 1 worker A |
| `audio-frame-gpu1b.service` | `8811` | `1` | GPU 1 worker B |
| `audio-frame-cpu.service` | `8812` | empty | CPU fallback |

Manual unit files are still straightforward if you want full control. This
example runs two workers on GPU 0, two on GPU 1, and one CPU fallback service.

GPU worker template:

```ini
[Unit]
Description=Audio Frame GPU worker %i
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/path/to/audio_frame
ExecStart=/path/to/audio_frame/scripts/run_service.sh
Restart=on-failure
RestartSec=10
Environment=AUDIO_FRAME_MODE=api
Environment=AUDIO_FRAME_DEVICE=cuda
Environment=AUDIO_FRAME_PRELOAD=1
Environment=AUDIO_FRAME_WORKERS=1
Environment=AUDIO_FRAME_NO_OPTIMIZE=1
Environment=AUDIO_FRAME_MIN_CUDA_MEMORY_GB=8
Environment=AUDIO_FRAME_LOAD_DENOISER=0
Environment=TOKENIZERS_PARALLELISM=false
Environment=PORT=8808
Environment=CUDA_VISIBLE_DEVICES=0
TimeoutStopSec=30

[Install]
WantedBy=default.target
```

Use different unit files or drop-ins for each GPU worker:

| Port | `CUDA_VISIBLE_DEVICES` | Purpose |
| --- | --- | --- |
| `8808` | `0` | GPU 0 worker A |
| `8809` | `0` | GPU 0 worker B |
| `8810` | `1` | GPU 1 worker A |
| `8811` | `1` | GPU 1 worker B |

CPU fallback unit:

```ini
[Unit]
Description=Audio Frame CPU fallback
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/path/to/audio_frame
ExecStart=/path/to/audio_frame/scripts/run_service.sh
Restart=on-failure
RestartSec=10
Environment=PORT=8812
Environment=AUDIO_FRAME_MODE=api
Environment=AUDIO_FRAME_DEVICE=cpu
Environment=AUDIO_FRAME_NO_OPTIMIZE=1
Environment=AUDIO_FRAME_PRELOAD=0
Environment=AUDIO_FRAME_WORKERS=4
Environment=AUDIO_FRAME_TORCH_THREADS=7
Environment=CUDA_VISIBLE_DEVICES=
Environment=OMP_NUM_THREADS=7
Environment=MKL_NUM_THREADS=7
TimeoutStopSec=30

[Install]
WantedBy=default.target
```

Reload and start:

```bash
systemctl --user daemon-reload
systemctl --user enable --now audio-frame.service audio-frame-gpu0b.service \
  audio-frame-gpu1a.service audio-frame-gpu1b.service audio-frame-cpu.service
```

## 4. Health Checks

```bash
for port in 8808 8809 8810 8811 8812; do
  curl -s "http://127.0.0.1:${port}/healthz"
  echo
done
```

GPU workers should report `selected_device: cuda` and, when preloaded,
`model_loaded: true`. CPU fallback should report `selected_device: cpu`.

Check GPU placement:

```bash
nvidia-smi
```

## 5. TTS API

Endpoint: `POST /api/tts`

Minimal request:

```bash
curl -s http://127.0.0.1:8808/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"测试通过。","control_instruction":"中文有声小说旁白，语气平静。"}' \
  > response.json
```

Important request fields:

| Field | Default | Description |
| --- | --- | --- |
| `text` | required | Text to synthesize |
| `control_instruction` | `""` | Style prompt prepended to text in style-control mode |
| `reference_audio_base64` | `null` | Optional WAV/MP3 reference audio |
| `prompt_text` | `""` | Optional prompt text for reference continuation |
| `cfg_value` | `2.0` | VoxCPM guidance value |
| `normalize` | `true` | Normalize text before synthesis |
| `denoise` | `false` | Denoise reference audio when reference is present |
| `inference_timesteps` | `10` | Diffusion steps |

The response contains base64 WAV audio in `audio_base64`.

## 6. novel_frame_2 Configuration

For batch-global scheduling, configure GPU endpoints plus a CPU fallback in
`novel_frame_2/novel_writer/external_services.json`:

```json
{
  "version": 1,
  "audio_frame": {
    "api_bases": [
      "http://127.0.0.1:8808",
      "http://127.0.0.1:8809",
      "http://127.0.0.1:8810",
      "http://127.0.0.1:8811"
    ],
    "workers": 4,
    "endpoints": [
      {"api_base": "http://127.0.0.1:8808", "kind": "gpu", "capacity": 1},
      {"api_base": "http://127.0.0.1:8809", "kind": "gpu", "capacity": 1},
      {"api_base": "http://127.0.0.1:8810", "kind": "gpu", "capacity": 1},
      {"api_base": "http://127.0.0.1:8811", "kind": "gpu", "capacity": 1},
      {
        "api_base": "http://127.0.0.1:8812",
        "kind": "cpu",
        "capacity": 4,
        "max_chars": 24,
        "speed": 0.15
      }
    ],
    "timeout": 0
  },
  "audiobook": {
    "backend": "audio_frame"
  }
}
```

With this shape, `novel_frame_2` sends long segments to GPU workers and uses CPU
only for short-text fill-in.

## 7. Useful Environment Variables

| Variable | Default | Notes |
| --- | --- | --- |
| `AUDIO_FRAME_MODE` | `webui` | Use `api` for HTTP API service |
| `AUDIO_FRAME_PYTHON` | auto | Explicit Python executable |
| `AUDIO_FRAME_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `AUDIO_FRAME_MODEL_ID` | `openbmb/VoxCPM2` | Hugging Face repo or local model path |
| `AUDIO_FRAME_PRELOAD` | `0` | Load model during API startup |
| `AUDIO_FRAME_WORKERS` | `1` | Uvicorn workers in API mode |
| `AUDIO_FRAME_NO_OPTIMIZE` | `1` for systemd workers | Disable `torch.compile` style optimization |
| `AUDIO_FRAME_MIN_CUDA_MEMORY_GB` | `16` globally, `8` in generated GPU units | Minimum free VRAM required before selecting CUDA |
| `AUDIO_FRAME_LOAD_DENOISER` | `0` | Load denoiser model |
| `AUDIO_FRAME_TORCH_THREADS` | `7` for CPU mode | Used for `OMP_NUM_THREADS` and `MKL_NUM_THREADS` |
| `CUDA_VISIBLE_DEVICES` | inherited | Pin one API process to one physical GPU |

## 8. Benchmark Helpers

The scripts under `scripts/benchmark*.py` are optional sizing tools. Run them
from the repository root with the Python environment activated:

```bash
python scripts/benchmark.py --gpu-only
python scripts/benchmark_multiprocess.py --runtime cuda --workers 1 2 --gpus 0 1
python scripts/benchmark_multiprocess.py --runtime cpu --workers 4 8 12 --torch-threads 7
```

Use `--offline` with `benchmark_multiprocess.py` when the model is already
cached and the machine has no network access.
