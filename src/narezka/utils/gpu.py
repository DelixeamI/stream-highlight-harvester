"""GPU hygiene utilities and VRAM tracking."""

from __future__ import annotations

import gc
import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


def is_torch_cuda_available() -> bool:
    """Check if PyTorch with CUDA support is installed and operational."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def get_cuda_device_info(device_idx: int = 0) -> dict[str, Any]:
    """Retrieve detailed information about the target CUDA device."""
    if not is_torch_cuda_available():
        return {
            "available": False,
            "error": "PyTorch with CUDA is not available",
        }

    import torch

    device_count = torch.cuda.device_count()
    if device_idx >= device_count:
        return {
            "available": False,
            "error": f"Invalid device index {device_idx}, available devices: {device_count}",
        }

    props = torch.cuda.get_device_properties(device_idx)
    cap_major, cap_minor = props.major, props.minor
    compute_capability = f"{cap_major}.{cap_minor}"
    is_blackwell = cap_major >= 12

    total_memory_mb = props.total_memory / (1024 * 1024)

    return {
        "available": True,
        "device_index": device_idx,
        "name": props.name,
        "compute_capability": compute_capability,
        "is_blackwell": is_blackwell,
        "total_memory_mb": round(total_memory_mb, 1),
        "multi_processor_count": props.multi_processor_count,
    }


def get_gpu_memory_info(device_idx: int = 0) -> dict[str, float]:
    """Get current allocated, reserved, and peak VRAM in megabytes."""
    if not is_torch_cuda_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}

    import torch

    allocated = torch.cuda.memory_allocated(device_idx) / (1024 * 1024)
    reserved = torch.cuda.memory_reserved(device_idx) / (1024 * 1024)
    peak = torch.cuda.max_memory_allocated(device_idx) / (1024 * 1024)

    return {
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "peak_mb": round(peak, 2),
    }


def cleanup_vram() -> dict[str, float]:
    """Strict GPU hygiene: invoke garbage collector and empty CUDA caches."""
    gc.collect()

    if is_torch_cuda_available():
        import torch

        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        return get_gpu_memory_info()

    return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}


@contextmanager
def track_vram(stage_name: str) -> Generator[dict[str, Any], None, None]:
    """Context manager to measure peak VRAM and ensure cleanup after block execution."""
    metrics: dict[str, Any] = {"stage": stage_name, "start_mb": 0.0, "peak_mb": 0.0, "end_mb": 0.0}

    if is_torch_cuda_available():
        import torch

        torch.cuda.reset_peak_memory_stats()
        metrics["start_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)

    try:
        yield metrics
    finally:
        if is_torch_cuda_available():
            import torch

            metrics["peak_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

        cleanup_vram()

        if is_torch_cuda_available():
            import torch

            metrics["end_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)

        logger.debug(
            "VRAM tracking for '%s': start=%.1f MB, peak=%.1f MB, end=%.1f MB",
            stage_name,
            metrics["start_mb"],
            metrics["peak_mb"],
            metrics["end_mb"],
        )
