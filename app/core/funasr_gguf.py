from __future__ import annotations

import importlib
import os
import re
import shutil
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from app.core.logging import get_logger
from app.core.paths import get_app_root, get_bin_dir

BASE_DIR = get_app_root()
VENDOR_FUNASR_GGUF_DIR = BASE_DIR / "vendor" / "Fun-ASR-GGUF"
RUNTIME_BIN_DIR = get_bin_dir("fun-asr-gguf")
BASE_BIN_DIR = get_bin_dir()
EXPECTED_LLAMA_CPP_CUDA_BUILD = "b7798"
_CUDA_BUILD_RE = re.compile(r"llama-(b\d+)-bin-win-cuda-", re.IGNORECASE)
logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class QuantizedASRArtifacts:
    model_dir: Path
    encoder_onnx_path: Path
    ctc_onnx_path: Path
    decoder_gguf_path: Path
    tokens_path: Path


def _resolve_path(raw: str | None) -> Path | None:
    value = str(raw or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def _candidate_model_dirs() -> tuple[Path, ...]:
    env_dir = _resolve_path(
        os.getenv("FUNASR_GGUF_MODEL_DIR") or os.getenv("FUNASR_GGUF_EXPORT_DIR")
    )
    candidates: list[Path] = []
    if env_dir is not None:
        candidates.append(env_dir)
    candidates.extend(
        [
            BASE_DIR / "checkpoints" / "Fun-ASR-Nano-2512-gguf",
            VENDOR_FUNASR_GGUF_DIR / "model",
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).casefold()
        if key in seen:
            continue
        deduped.append(candidate)
        seen.add(key)
    return tuple(deduped)


def resolve_quantized_artifacts() -> QuantizedASRArtifacts | None:
    for model_dir in _candidate_model_dirs():
        encoder = model_dir / "Fun-ASR-Nano-Encoder-Adaptor.int4.onnx"
        ctc = model_dir / "Fun-ASR-Nano-CTC.int4.onnx"
        decoder = model_dir / "Fun-ASR-Nano-Decoder.q5_k.gguf"
        tokens = model_dir / "tokens.txt"
        if encoder.exists() and ctc.exists() and decoder.exists() and tokens.exists():
            return QuantizedASRArtifacts(
                model_dir=model_dir,
                encoder_onnx_path=encoder,
                ctc_onnx_path=ctc,
                decoder_gguf_path=decoder,
                tokens_path=tokens,
            )
    return None


def ensure_vendor_path() -> Path:
    vendor_dir = VENDOR_FUNASR_GGUF_DIR.resolve()
    vendor_dir_str = str(vendor_dir)
    if vendor_dir_str not in sys.path:
        sys.path.insert(0, vendor_dir_str)
    return vendor_dir


def _cuda_runtime_archives() -> tuple[Path, ...]:
    candidates: list[Path] = []
    for pattern in (
        "llama-b*-bin-win-cuda-*-x64.zip",
        "cudart-llama-bin-win-cuda-*-x64.zip",
    ):
        candidates.extend(sorted(BASE_BIN_DIR.glob(pattern)))
    return tuple(candidates)


def _runtime_has_cuda_backend(runtime_dir: Path) -> bool:
    required = (
        runtime_dir / "llama.dll",
        runtime_dir / "ggml.dll",
        runtime_dir / "ggml-base.dll",
        runtime_dir / "ggml-cuda.dll",
        runtime_dir / "cudart64_12.dll",
    )
    return all(path.exists() for path in required)


def _restore_vulkan_backend(runtime_dir: Path) -> None:
    disabled_path = runtime_dir / "ggml-vulkan.dll.disabled"
    restored_path = runtime_dir / "ggml-vulkan.dll"
    if disabled_path.exists() and not restored_path.exists():
        disabled_path.replace(restored_path)


def _disable_conflicting_runtime_backends(runtime_dir: Path) -> None:
    vulkan_dll = runtime_dir / "ggml-vulkan.dll"
    if not vulkan_dll.exists():
        return
    disabled_path = runtime_dir / "ggml-vulkan.dll.disabled"
    if disabled_path.exists():
        vulkan_dll.unlink()
        return
    vulkan_dll.replace(disabled_path)


def _cuda_build_from_archives(archives: tuple[Path, ...]) -> str | None:
    for archive_path in archives:
        match = _CUDA_BUILD_RE.search(archive_path.name)
        if match:
            return match.group(1).lower()
    return None


def _allow_untested_cuda_runtime() -> bool:
    raw = str(os.getenv("FUNASR_GGUF_ALLOW_UNTESTED_CUDA") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _cuda_runtime_is_supported(archives: tuple[Path, ...]) -> bool:
    build = _cuda_build_from_archives(archives)
    if build is None:
        return False
    if build == EXPECTED_LLAMA_CPP_CUDA_BUILD.casefold():
        return True
    if _allow_untested_cuda_runtime():
        logger.warning(
            "检测到未验证的 llama.cpp CUDA 运行时 {}，已按环境变量要求继续启用。",
            build,
        )
        return True
    logger.warning(
        "检测到未验证的 llama.cpp CUDA 运行时 {}，当前 Fun-ASR 绑定预期为 {}；将回退到 vendor 自带运行时。",
        build,
        EXPECTED_LLAMA_CPP_CUDA_BUILD,
    )
    return False


def _extract_runtime_archive(archive_path: Path, target_dir: Path) -> None:
    shutil.unpack_archive(str(archive_path), str(target_dir))


def _install_cuda_runtime(target_dir: Path) -> bool:
    archives = _cuda_runtime_archives()
    if not archives:
        return False
    if not _cuda_runtime_is_supported(archives):
        return False
    for archive_path in archives:
        _extract_runtime_archive(archive_path, target_dir)
    if _runtime_has_cuda_backend(target_dir):
        _disable_conflicting_runtime_backends(target_dir)
        logger.info("已安装 llama.cpp CUDA 运行时到 {}", target_dir)
        return True
    return False


def ensure_runtime_binaries() -> Path:
    source_dir = VENDOR_FUNASR_GGUF_DIR / "fun_asr_gguf" / "inference" / "bin"
    target_dir = RUNTIME_BIN_DIR.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    archives = _cuda_runtime_archives()
    if archives and _runtime_has_cuda_backend(target_dir) and _cuda_runtime_is_supported(archives):
        _disable_conflicting_runtime_backends(target_dir)
        return target_dir
    if _install_cuda_runtime(target_dir):
        return target_dir
    _restore_vulkan_backend(target_dir)
    if source_dir.exists():
        for source_path in source_dir.iterdir():
            if not source_path.is_file():
                continue
            target_path = target_dir / source_path.name
            if target_path.exists():
                try:
                    if target_path.stat().st_size == source_path.stat().st_size:
                        continue
                except OSError:
                    pass
            shutil.copy2(source_path, target_path)
        logger.info("已回退到 vendor 自带 llama.cpp 运行时: {}", source_dir)
    return target_dir


def prepare_runtime_environment() -> None:
    # Torch/FunASR and llama.cpp may load different OpenMP runtimes in one process.
    # Allow coexistence so the desktop app can switch to the GGUF backend without aborting.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("FUNASR_GGUF_BIN_DIR", str(ensure_runtime_binaries()))


@cache
def load_funasr_gguf_runtime() -> tuple[type[object], type[object]]:
    prepare_runtime_environment()
    ensure_vendor_path()
    module = importlib.import_module("fun_asr_gguf")
    return module.FunASREngine, module.ASREngineConfig
