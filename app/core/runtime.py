from __future__ import annotations

import gc

import torch


def _is_cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(preferred: str | None = None) -> str:
    normalized = str(preferred or "auto").strip().lower()
    if normalized == "auto":
        return "cuda" if _is_cuda_available() else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda"):
        return normalized if _is_cuda_available() else "cpu"
    raise ValueError(f"Unsupported device preference: {preferred!r}")


def resolve_dtype(device: str) -> torch.dtype:
    return torch.float16 if str(device).startswith("cuda") else torch.float32


def resolve_attn_implementation(device: str) -> str:
    if str(device).startswith("cuda"):
        return "sdpa"
    return "eager"


def resolve_generation_cache_implementation(preferred: str | None = None, *, device: str) -> str:
    normalized = str(preferred or "dynamic").strip().lower()
    allowed = {
        "dynamic",
        "static",
        "offloaded",
        "offloaded_static",
        "quantized",
        "sliding_window",
        "hybrid",
        "mamba",
    }
    if normalized == "auto":
        return "static" if str(device).startswith("cuda") else "dynamic"
    if normalized not in allowed:
        return "dynamic"
    if not str(device).startswith("cuda") and normalized in {"offloaded", "offloaded_static"}:
        return "dynamic"
    return normalized


def get_gpu_total_memory_gb() -> float | None:
    """Return total GPU memory in GB, or None if no CUDA device available."""
    if not _is_cuda_available():
        return None
    try:
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return None


def release_memory() -> None:
    gc.collect()
    if _is_cuda_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
