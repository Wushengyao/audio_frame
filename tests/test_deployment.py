from __future__ import annotations

from types import SimpleNamespace

from voxcpm import deployment


def _patch_memory(monkeypatch, *, available_gb: float = 12.0, total_gb: float = 16.0) -> None:
    monkeypatch.setattr(
        deployment,
        "_read_linux_meminfo",
        lambda: {
            "MemTotal": int(total_gb * 1024**3),
            "MemAvailable": int(available_gb * 1024**3),
        },
    )


def _patch_cuda(monkeypatch, free_gb_by_index: list[float], total_gb_by_index: list[float] | None = None) -> None:
    totals = total_gb_by_index or [12.0] * len(free_gb_by_index)
    monkeypatch.setattr(deployment.torch.cuda, "is_available", lambda: bool(free_gb_by_index))
    monkeypatch.setattr(deployment.torch.cuda, "device_count", lambda: len(free_gb_by_index))
    monkeypatch.setattr(deployment.torch.cuda, "get_device_name", lambda index: f"GPU {index}")
    monkeypatch.setattr(
        deployment.torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(total_memory=int(totals[index] * 1024**3)),
    )
    monkeypatch.setattr(
        deployment.torch.cuda,
        "mem_get_info",
        lambda index: (int(free_gb_by_index[index] * 1024**3), int(totals[index] * 1024**3)),
    )


def test_auto_deployment_selects_best_sufficient_cuda(monkeypatch):
    _patch_memory(monkeypatch)
    _patch_cuda(monkeypatch, [3.0, 8.0])

    plan = deployment.build_deployment_plan(
        requested_device="auto",
        optimize_requested=True,
        min_cuda_memory_gb=4.0,
    )

    assert plan.can_load is True
    assert plan.selected_device == "cuda"
    assert plan.cuda_device_index == 1
    assert plan.optimize is True


def test_auto_deployment_prefers_idle_cuda_over_larger_busy_card(monkeypatch):
    _patch_memory(monkeypatch)
    _patch_cuda(monkeypatch, [23.0, 28.0], total_gb_by_index=[24.0, 32.0])

    plan = deployment.build_deployment_plan(
        requested_device="auto",
        optimize_requested=True,
        min_cuda_memory_gb=16.0,
    )

    assert plan.selected_device == "cuda"
    assert plan.cuda_device_index == 0
    assert "1.0 GiB already used" in plan.reason


def test_auto_deployment_falls_back_to_cpu_when_vram_is_low(monkeypatch):
    _patch_memory(monkeypatch)
    _patch_cuda(monkeypatch, [2.0])

    plan = deployment.build_deployment_plan(
        requested_device="auto",
        optimize_requested=True,
        min_cuda_memory_gb=4.0,
    )

    assert plan.can_load is True
    assert plan.selected_device == "cpu"
    assert plan.optimize is False
    assert "below the configured minimum" in plan.warnings[0]


def test_strict_deployment_blocks_low_system_memory(monkeypatch):
    _patch_memory(monkeypatch, available_gb=1.0, total_gb=8.0)
    _patch_cuda(monkeypatch, [])

    plan = deployment.build_deployment_plan(
        requested_device="cpu",
        min_system_memory_gb=4.0,
        strict=True,
    )

    assert plan.can_load is False
    assert plan.selected_device == "cpu"
    assert "Insufficient system memory" in plan.reason
