"""Unit tests for GPU hygiene and VRAM tracking."""

from __future__ import annotations

from narezka.utils.gpu import cleanup_vram, get_cuda_device_info, get_gpu_memory_info, track_vram


def test_cleanup_vram_does_not_crash() -> None:
    info = cleanup_vram()
    assert isinstance(info, dict)
    assert "allocated_mb" in info


def test_get_gpu_memory_info() -> None:
    mem = get_gpu_memory_info()
    assert "allocated_mb" in mem
    assert "reserved_mb" in mem
    assert "peak_mb" in mem


def test_track_vram_context_manager() -> None:
    with track_vram("test_operation") as metrics:
        assert metrics["stage"] == "test_operation"
        # Perform some dummy work
        _ = [i * 2 for i in range(1000)]

    assert "peak_mb" in metrics
    assert "end_mb" in metrics


def test_get_cuda_device_info() -> None:
    info = get_cuda_device_info()
    assert "available" in info
