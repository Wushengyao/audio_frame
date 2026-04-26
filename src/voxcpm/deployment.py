from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

import torch


DEFAULT_MIN_SYSTEM_MEMORY_GB = 4.0
DEFAULT_MIN_CUDA_MEMORY_GB = 4.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _bytes_from_gb(value: float) -> int:
    return int(value * 1024**3)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_linux_meminfo() -> dict[str, int]:
    path = Path("/proc/meminfo")
    if not path.exists():
        return {}
    result: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            parts = raw_value.strip().split()
            if not parts:
                continue
            result[key] = int(parts[0]) * 1024
    except (OSError, ValueError):
        return {}
    return result


@dataclass(frozen=True)
class SystemMemoryStatus:
    total_bytes: int
    available_bytes: int
    required_bytes: int
    sufficient: bool

    @property
    def available_gb(self) -> float:
        return round(self.available_bytes / 1024**3, 2)


@dataclass(frozen=True)
class CudaDeviceStatus:
    index: int
    name: str
    total_bytes: int
    free_bytes: int
    required_bytes: int
    sufficient: bool

    @property
    def free_gb(self) -> float:
        return round(self.free_bytes / 1024**3, 2)


@dataclass(frozen=True)
class DeploymentPlan:
    checked_at: str
    requested_device: str
    selected_device: str
    cuda_device_index: int | None
    optimize: bool
    can_load: bool
    reason: str
    warnings: tuple[str, ...]
    system_memory: SystemMemoryStatus
    cuda_devices: tuple[CudaDeviceStatus, ...]

    @property
    def model_device(self) -> str:
        if self.selected_device == "cuda":
            return "cuda"
        return self.selected_device

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model_device"] = self.model_device
        return data


def probe_system_memory(min_system_memory_gb: float | None = None) -> SystemMemoryStatus:
    required = _bytes_from_gb(
        min_system_memory_gb
        if min_system_memory_gb is not None
        else _float_env("AUDIO_FRAME_MIN_SYSTEM_MEMORY_GB", DEFAULT_MIN_SYSTEM_MEMORY_GB)
    )
    meminfo = _read_linux_meminfo()
    total = meminfo.get("MemTotal", 0)
    available = meminfo.get("MemAvailable", total)
    if not total:
        total = available
    return SystemMemoryStatus(
        total_bytes=total,
        available_bytes=available,
        required_bytes=required,
        sufficient=available >= required if available else True,
    )


def probe_cuda_devices(min_cuda_memory_gb: float | None = None) -> tuple[CudaDeviceStatus, ...]:
    required = _bytes_from_gb(
        min_cuda_memory_gb
        if min_cuda_memory_gb is not None
        else _float_env("AUDIO_FRAME_MIN_CUDA_MEMORY_GB", DEFAULT_MIN_CUDA_MEMORY_GB)
    )
    if not torch.cuda.is_available():
        return ()

    devices = []
    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        total = int(torch.cuda.get_device_properties(index).total_memory)
        free = total
        try:
            free, total_from_runtime = torch.cuda.mem_get_info(index)
            total = int(total_from_runtime)
            free = int(free)
        except Exception:
            pass
        devices.append(
            CudaDeviceStatus(
                index=index,
                name=name,
                total_bytes=total,
                free_bytes=free,
                required_bytes=required,
                sufficient=free >= required,
            )
        )
    return tuple(devices)


def build_deployment_plan(
    *,
    requested_device: str | None = None,
    optimize_requested: bool = True,
    min_system_memory_gb: float | None = None,
    min_cuda_memory_gb: float | None = None,
    strict: bool | None = None,
) -> DeploymentPlan:
    requested = (requested_device or os.environ.get("AUDIO_FRAME_DEVICE") or "auto").strip().lower()
    strict_mode = _bool_env("AUDIO_FRAME_DEPLOY_STRICT", False) if strict is None else strict
    system_memory = probe_system_memory(min_system_memory_gb)
    cuda_devices = probe_cuda_devices(min_cuda_memory_gb)
    warnings: list[str] = []

    selected_device = "cpu"
    cuda_index: int | None = None

    if requested.startswith("cuda"):
        if not cuda_devices:
            warnings.append("CUDA was requested but is not available; falling back to CPU.")
        else:
            requested_index = 0
            if ":" in requested:
                try:
                    requested_index = int(requested.split(":", 1)[1])
                except ValueError:
                    requested_index = -1
            match = next((device for device in cuda_devices if device.index == requested_index), None)
            if match is None:
                warnings.append(f"CUDA device {requested_index} was requested but is not present; falling back to CPU.")
            elif match.sufficient:
                selected_device = "cuda"
                cuda_index = match.index
            else:
                warnings.append(
                    f"CUDA device {match.index} has {match.free_gb} GiB free, below the configured minimum; "
                    "falling back to CPU."
                )
    elif requested == "auto":
        best_cuda = max(cuda_devices, key=lambda device: device.free_bytes, default=None)
        if best_cuda and best_cuda.sufficient:
            selected_device = "cuda"
            cuda_index = best_cuda.index
        elif best_cuda:
            warnings.append(
                f"Best CUDA device has {best_cuda.free_gb} GiB free, below the configured minimum; using CPU."
            )
    elif requested == "cpu":
        selected_device = "cpu"
    else:
        warnings.append(f"Unsupported requested device '{requested}'; using auto deployment.")
        best_cuda = max(cuda_devices, key=lambda device: device.free_bytes, default=None)
        if best_cuda and best_cuda.sufficient:
            selected_device = "cuda"
            cuda_index = best_cuda.index

    if not system_memory.sufficient:
        warnings.append(
            f"System memory has {system_memory.available_gb} GiB available, below the configured minimum."
        )

    can_load = system_memory.sufficient or not strict_mode
    if strict_mode and not system_memory.sufficient:
        reason = "Insufficient system memory for model loading."
    elif selected_device == "cuda":
        reason = f"Using CUDA device {cuda_index} with sufficient free VRAM."
    else:
        reason = "Using CPU deployment."

    return DeploymentPlan(
        checked_at=_utc_now(),
        requested_device=requested,
        selected_device=selected_device,
        cuda_device_index=cuda_index,
        optimize=bool(optimize_requested and selected_device == "cuda"),
        can_load=can_load,
        reason=reason,
        warnings=tuple(warnings),
        system_memory=system_memory,
        cuda_devices=cuda_devices,
    )


def apply_deployment_plan(plan: DeploymentPlan) -> None:
    if plan.selected_device == "cuda" and plan.cuda_device_index is not None:
        torch.cuda.set_device(plan.cuda_device_index)
