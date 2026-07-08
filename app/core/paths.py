from __future__ import annotations

import sys
from pathlib import Path


def _is_compiled_runtime() -> bool:
    return bool(getattr(sys, "frozen", False) or globals().get("__compiled__", False))


def get_app_root() -> Path:
    if _is_compiled_runtime():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def get_bundle_root() -> Path:
    if _is_compiled_runtime():
        return get_app_root()
    return Path(__file__).resolve().parents[2]


def get_runtime_dir(*parts: str) -> Path:
    return get_app_root().joinpath(*parts)


def get_bundle_file(*parts: str) -> Path:
    return get_bundle_root().joinpath(*parts)


def get_vendor_dir(*parts: str) -> Path:
    """Third-party source libraries (index-tts, Qwen3-ASR-GGUF, etc)."""
    return get_app_root() / "vendor" / Path(*parts)


def get_checkpoints_dir(*parts: str) -> Path:
    """Model weight files downloaded from HuggingFace/ModelScope."""
    return get_app_root() / "checkpoints" / Path(*parts)


def get_bin_dir(*parts: str) -> Path:
    """Pre-compiled binary tools (transcribe.exe, ffmpeg, etc)."""
    return get_app_root() / "bin" / Path(*parts)
