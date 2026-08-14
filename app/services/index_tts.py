from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import shutil
import subprocess
import threading
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from app.core import get_indextts_vendor_loader
from app.core.config import (
    get_settings,
    resolve_index_tts_model_dir,
    resolve_tts_device,
)
from app.core.index_tts_files import get_index_tts_missing_files
from app.core.logging import get_logger
from app.core.runtime import release_memory
from app.core.audio import resample_waveform
from app.services.translation import detect_text_language
logger = get_logger(__name__)
_DEFAULT_FIRST_CHUNK_WARMUP_TEXT = "你好，欢迎使用。"
_JAPANESE_FIRST_CHUNK_WARMUP_TEXT = "こんにちは。"
_INDEX_TTS_V25_LANGUAGE_TOKENS: dict[str, str] = {
    "zh": "ZH",
    "en": "EN",
    "ja": "JA",
    "es": "ES",
    "ar": "AR",
}
_DEFAULT_INDEX_TTS_LANGUAGE = "zh"
_INDEX_TTS_V25_MAX_MEL_TOKENS = 1500


def _resolve_tts_language_token(text: str | None) -> tuple[str, str]:
    language = detect_text_language(str(text or "").strip()) or _DEFAULT_INDEX_TTS_LANGUAGE
    normalized = language if language in _INDEX_TTS_V25_LANGUAGE_TOKENS else _DEFAULT_INDEX_TTS_LANGUAGE
    return normalized, _INDEX_TTS_V25_LANGUAGE_TOKENS[normalized]


def _sanitize_text(text: str) -> str:
    """Remove/replace characters that confuse IndexTTS's SentencePiece tokenizer."""
    text = text.replace("\n", " ").strip()
    if not text:
        return text
    if not text.endswith((".", "!", "?", "。", "！", "？")):
        text += "."
    return text


def _split_stream_text(text: str, max_chars: int = 24) -> list[str]:
    parts = re.split(r'(?<=[，。！？、；：,\.!\?;:\n])', text)
    chunks: list[str] = []
    for part in parts:
        if not part:
            continue
        if len(part) <= max_chars:
            chunks.append(part)
            continue
        current = ""
        for ch in part:
            if current and len(current) >= max_chars:
                chunks.append(current)
                current = ch
            else:
                current += ch
        if current:
            chunks.append(current)
    return chunks or [text]


def _select_warmup_text(model_dir: Path) -> str:
    configured = os.getenv("INDEX_TTS_WARMUP_TEXT")
    if configured and configured.strip():
        return configured.strip()
    settings = get_settings()
    hints = " ".join(
        (
            settings.index_tts_model_repo,
            model_dir.name,
            str(model_dir),
        )
    ).lower()
    if "japanese" in hints or re.search(r"(^|[-_/])ja($|[-_/])", hints):
        return _JAPANESE_FIRST_CHUNK_WARMUP_TEXT
    return _DEFAULT_FIRST_CHUNK_WARMUP_TEXT


def _normalize_stream_audio_item(item: Any) -> np.ndarray | None:
    if item is None or isinstance(item, tuple):
        return None
    if isinstance(item, torch.Tensor):
        tensor = item.detach()
        wav = tensor.numpy() if tensor.device.type == "cpu" else tensor.cpu().numpy()
    elif isinstance(item, np.ndarray):
        wav = item
    else:
        wav = np.array(item)
    if wav.ndim == 1:
        wav = wav[:, None]
    elif wav.ndim == 2 and wav.shape[0] < wav.shape[1]:
        wav = wav.T
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / 32768.0
    elif wav.dtype != np.float32:
        wav = wav.astype(np.float32)
    elif np.max(np.abs(wav)) > 2.0:
        wav = wav.astype(np.float32, copy=False) / 32768.0
    return wav


def _assert_index_tts_model_files_ready(model_dir: Path) -> None:
    missing_files = get_index_tts_missing_files(model_dir, local_only=True)
    if not missing_files:
        return
    raise FileNotFoundError(
        "当前 IndexTTS-2.5 模型目录不完整，缺少以下文件："
        f"{', '.join(missing_files)}。"
        "请先在“模型下载”中手动下载对应模型，程序不会自动下载。"
    )


def _normalize_transformers_aux_configs(model_dir: Path) -> None:
    """Patch known upstream config quirks that trigger noisy Transformers 5.x warnings."""
    w2v_config_path = model_dir / "hf_cache" / "w2v-bert-2.0" / "config.json"
    if not w2v_config_path.exists():
        return
    try:
        config = json.loads(w2v_config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("读取语义前端配置失败，跳过兼容修正: {}", exc)
        return

    token_ids = [
        int(value)
        for key in ("pad_token_id", "bos_token_id", "eos_token_id")
        if isinstance((value := config.get(key)), int)
    ]
    minimum_vocab_size = max(token_ids, default=-1) + 1
    current_vocab_size = config.get("vocab_size")
    if isinstance(current_vocab_size, int) and current_vocab_size >= minimum_vocab_size:
        return

    config["vocab_size"] = minimum_vocab_size
    w2v_config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "已修正语义前端配置中的 vocab_size 以兼容新版 Transformers: {} -> {}",
        current_vocab_size,
        minimum_vocab_size,
    )


@cache
def _load_indextts2_class() -> type[Any]:
    """按需加载 IndexTTS 2.5，避免模块导入阶段就触发重型依赖初始化。"""
    module = get_indextts_vendor_loader().import_module("indextts.infer_v2_5")
    return module.IndexTTS2


@cache
def _load_torchaudio() -> Any:
    """按需加载 torchaudio，避免模块导入阶段触发额外初始化。"""
    return importlib.import_module("torchaudio")


def _activate_msvc_environment(vcvars_bat: Path) -> bool:
    try:
        result = subprocess.run(
            [
                "cmd",
                "/d",
                "/s",
                "/c",
                f'call "{vcvars_bat}" >nul 2>&1 && set',
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=True,
        )
    except Exception as exc:
        logger.debug("激活 MSVC 环境失败: {}", exc)
        return False

    important_keys = {
        "CC", "CXX", "INCLUDE", "LIB", "LIBPATH", "PATH",
        "UCRTVersion", "UniversalCRTSdkDir", "VCToolsInstallDir",
        "VisualStudioVersion", "VSINSTALLDIR", "VCINSTALLDIR",
        "WindowsSdkDir", "WindowsSDKVersion",
    }
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in important_keys and value:
            os.environ[key] = value
    return True


def _find_windows_msvc() -> tuple[str | None, str | None]:
    env_compiler = _resolve_msvc_from_environment()
    if env_compiler is not None:
        return env_compiler, "CC 环境变量"
    path_compiler = shutil.which("cl")
    if path_compiler:
        return path_compiler, "PATH"

    candidates = _collect_msvc_candidates()
    resolved = _resolve_existing_msvc_candidate(candidates)
    if resolved is not None:
        os.environ.setdefault("CC", str(resolved))
        os.environ.setdefault("CXX", str(resolved))
        return str(resolved), "Visual Studio Build Tools"
    return None, None


def _resolve_msvc_from_environment() -> str | None:
    env_cl = os.environ.get("CC")
    if env_cl and Path(env_cl).exists():
        return env_cl
    return None


def _collect_msvc_candidates() -> list[Path]:
    candidates: list[Path] = []
    vc_tools_dir = os.environ.get("VCToolsInstallDir")
    if vc_tools_dir:
        candidates.extend(Path(vc_tools_dir).glob(r"bin/Hostx64/x64/cl.exe"))
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    candidates.extend(_collect_msvc_candidates_from_vswhere(program_files_x86))
    if not candidates:
        candidates.extend(_collect_msvc_candidates_from_standard_roots(program_files_x86))
    return candidates


def _collect_msvc_candidates_from_vswhere(program_files_x86: str) -> list[Path]:
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    candidates: list[Path] = []
    if vswhere.exists():
        try:
            query = subprocess.run(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=True,
            )
            install_dir = query.stdout.strip()
            if install_dir:
                candidates.extend(_collect_install_path_msvc_candidates(Path(install_dir)))
        except Exception as exc:
            logger.debug("通过 vswhere 查询 MSVC 失败: {}", exc)
    return candidates


def _collect_msvc_candidates_from_standard_roots(program_files_x86: str) -> list[Path]:
    candidates: list[Path] = []
    vs_roots = [
        Path(program_files_x86) / "Microsoft Visual Studio",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft Visual Studio",
    ]
    editions = ("BuildTools", "Community", "Professional", "Enterprise")
    years = ("2022", "2019", "2017")
    for root in vs_roots:
        for year in years:
            for edition in editions:
                candidates.extend(_collect_install_path_msvc_candidates(root / year / edition))
    return candidates


def _collect_install_path_msvc_candidates(install_path: Path) -> list[Path]:
    vcvars64 = install_path / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if vcvars64.exists():
        _activate_msvc_environment(vcvars64)
    return sorted(
        install_path.glob(r"VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe"),
        reverse=True,
    )


def _resolve_existing_msvc_candidate(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _configure_c_compiler() -> tuple[bool, str]:
    env_cc = os.environ.get("CC")
    if env_cc:
        env_cc_path = shutil.which(env_cc) or (env_cc if Path(env_cc).exists() else None)
        if env_cc_path:
            os.environ["CC"] = env_cc_path
            return True, f"CC={Path(env_cc_path).name}"

    if os.name == "nt":
        msvc_path, source = _find_windows_msvc()
        if msvc_path:
            os.environ["CC"] = msvc_path
            os.environ.setdefault("CXX", msvc_path)
            return True, f"{Path(msvc_path).name} ({source})"

    for cc in ("gcc", "clang"):
        found = shutil.which(cc)
        if found:
            os.environ.setdefault("CC", found)
            return True, f"{cc} (PATH)"
    return False, "未检测到可用编译器"


class IndexTTSService:

    """Lightweight wrapper around indextts.infer_v2_5.IndexTTS2 for text-to-speech."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir or resolve_index_tts_model_dir()
        self.device = resolve_tts_device()
        self._use_torch_compile = get_settings().experimental_torch_compile_enabled
        self._use_accel_gpt = get_settings().experimental_accel_gpt_enabled
        self._warmup_text = _select_warmup_text(self.model_dir)
        self._tts: Any = None
        self._model_dir_str = str(self.model_dir)
        self._lock = threading.RLock()
        self._state_changed = threading.Condition(self._lock)
        self._active_operation_count = 0
        self._unload_in_progress = False

    # ------------------------------------------------------------------ #
    #  load / unload
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        with self._lock:
            while self._unload_in_progress:
                self._state_changed.wait()
            if self._tts is not None:
                return
            if not self.model_dir.exists():
                raise FileNotFoundError(
                    "IndexTTS-2.5 model directory not found. Set INDEX_TTS_MODEL_DIR or place the model under "
                    f"{resolve_index_tts_model_dir()}."
                )
            config_path = self.model_dir / "config.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"IndexTTS-2.5 config not found: {config_path}")
            _assert_index_tts_model_files_ready(self.model_dir)
            _normalize_transformers_aux_configs(self.model_dir)

            use_accel = self._use_accel_gpt
            if use_accel:
                logger.info("实验性极速模式已启用，将尝试使用更快的语音生成路径。")
            else:
                logger.info("实验性极速模式未启用，使用常规推理路径。")

            use_torch_compile = self._use_torch_compile
            if use_torch_compile:
                compiler_available, compiler_desc = _configure_c_compiler()
                if compiler_available:
                    logger.info("实验性进阶模式已启用，检测到可用运行环境: {}", compiler_desc)
                else:
                    logger.warning(
                        "实验性进阶模式已开启，但当前环境暂不满足要求，已回退到常规推理。"
                    )
                    use_torch_compile = False
            else:
                logger.info("实验性进阶模式未启用，使用常规推理路径。")
            if not use_torch_compile and self._use_torch_compile:
                logger.info("IndexTTS 将跳过 torch.compile / inductor autotune，直接使用常规推理。")

            logger.info(
                "正在加载 IndexTTS-2.5 模型 (设备={} 模式=多语言自动识别)",
                self.device,
            )
            index_tts_cls = _load_indextts2_class()
            use_bf16 = bool(
                str(self.device).startswith("cuda")
                and torch.cuda.is_available()
                and torch.cuda.is_bf16_supported()
            )
            self._tts = index_tts_cls(
                cfg_path=str(config_path),
                model_dir=self._model_dir_str,
                use_bf16=use_bf16,
                device=self.device if str(self.device).startswith("cuda") else None,
                use_cuda_kernel=True,
                use_accel=use_accel,
                use_torch_compile=use_torch_compile,
            )
            logger.info("IndexTTS-2.5 加载完成")

    def warmup_speaker(self, reference_audio_path: str, emo_vector: list[float] | None = None) -> None:
        """后台预热参考音色与首段生成链，降低首次出声延迟。"""
        try:
            with self._use_tts(required=False) as tts:
                if tts is None:
                    return
                self._warmup_reference_features(tts, reference_audio_path, emo_vector)
                self._warmup_first_stream_chunk(tts, reference_audio_path, emo_vector)
        finally:
            release_memory()

    def _warmup_reference_features(self, tts: Any, reference_audio_path: str, emo_vector: list[float] | None) -> None:
        logger.info("正在预热 TTS 参考音频特征: {}", reference_audio_path)
        try:
            _, language_token = _resolve_tts_language_token(".")
            gen = tts.infer(
                spk_audio_prompt=reference_audio_path,
                text=".",
                output_path="",
                lang=language_token,
                verbose=False,
                emo_vector=emo_vector,
                stream_return=True,
            )
            self._consume_warmup_stream(gen, stop_after_first_audio=False)
            logger.info("TTS 参考音色特征预热完成")
        except Exception as exc:
            logger.warning("TTS 参考音频预热失败: {}", exc)

    def _warmup_first_stream_chunk(self, tts: Any, reference_audio_path: str, emo_vector: list[float] | None) -> None:
        logger.info("正在预热 TTS 首段生成链路: {}", self._warmup_text)
        warmup_stream = None
        try:
            detected_language, language_token = _resolve_tts_language_token(self._warmup_text)
            warmup_stream = tts.infer(
                spk_audio_prompt=reference_audio_path,
                text=self._warmup_text,
                output_path="",
                lang=language_token,
                verbose=False,
                emo_vector=emo_vector,
                stream_return=True,
                max_text_tokens_per_segment=80,
                max_mel_tokens=_INDEX_TTS_V25_MAX_MEL_TOKENS,
                more_segment_before=80,
            )
            if self._consume_warmup_stream(warmup_stream, stop_after_first_audio=True):
                logger.info("TTS 首段生成链路预热完成 (lang={})", detected_language)
        except Exception as exc:
            logger.warning("TTS 首段生成链路预热失败: {}", exc)
        finally:
            self._close_stream_if_possible(warmup_stream)

    @staticmethod
    def _consume_warmup_stream(stream: Any, *, stop_after_first_audio: bool) -> bool:
        saw_audio = False
        for item in stream:
            if item is None or isinstance(item, tuple):
                break
            saw_audio = True
            if stop_after_first_audio:
                break
        return saw_audio

    @staticmethod
    def _close_stream_if_possible(stream: Any) -> None:
        if stream is None:
            return
        close_fn = getattr(stream, "close", None)
        if callable(close_fn):
            close_fn()


    def unload(self) -> None:
        with self._lock:
            while self._unload_in_progress:
                self._state_changed.wait()
            if self._tts is None:
                return
            self._unload_in_progress = True
            tts = self._tts
            self._tts = None
            while self._active_operation_count > 0:
                self._state_changed.wait()
        try:
            for attr in ("gpt", "bigvgan", "s2mel", "semantic_model", "semantic_codec", "campplus_model", "qwen_emo"):
                obj = getattr(tts, attr, None)
                if obj is not None and hasattr(obj, "cpu"):
                    obj.cpu()
                    del obj
            del tts
            release_memory()
            logger.info("IndexTTS-2.5 已卸载")
        finally:
            with self._lock:
                self._unload_in_progress = False
                self._state_changed.notify_all()

    # ------------------------------------------------------------------ #
    #  synthesize
    # ------------------------------------------------------------------ #

    def synthesize(
        self,
        text: str,
        reference_audio_path: str,
        output_path: str | None = None,
        *,
        emo_vector: list[float] | None = None,
        emo_alpha: float = 1.0,
        duration_factor: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        text = _sanitize_text(text)
        if not text:
            raise ValueError("TTS 文本为空。")
        detected_language, language_token = _resolve_tts_language_token(text)

        logger.info(
            "IndexTTS-2.5 推理: text={!r}, ref={}, lang={}, emo_vector={}, emo_alpha={}, duration_factor={}",
            text[:80], reference_audio_path, detected_language, emo_vector, emo_alpha, duration_factor,
        )

        try:
            with self._use_tts(error_message="IndexTTS-2.5 服务尚未加载，请先调用 .load()。") as tts:
                result = tts.infer(
                    spk_audio_prompt=reference_audio_path,
                    text=text,
                    output_path=output_path or "",
                    lang=language_token,
                    verbose=False,
                    emo_vector=emo_vector,
                    emo_alpha=emo_alpha,
                    duration_factor=max(0.5, min(2.0, float(duration_factor))),
                    max_mel_tokens=_INDEX_TTS_V25_MAX_MEL_TOKENS,
                    max_text_tokens_per_segment=80,
                )

            # V2 infer returns (sampling_rate, wav_data) or output_path
            if isinstance(result, str):
                # output_path was provided — result is the path
                wav, sample_rate = sf.read(result, dtype="float32")
            elif isinstance(result, tuple) and len(result) == 2:
                sample_rate, wav = result
            else:
                raise RuntimeError(f"Unexpected IndexTTS-2.5 infer result type: {type(result)}")

            if not isinstance(wav, np.ndarray):
                wav = np.array(wav)
            # IndexTTS2 returns int16; convert to float32 for downstream mixing
            if wav.dtype == np.int16:
                wav = wav.astype(np.float32) / 32768.0
            elif wav.dtype != np.float32:
                wav = wav.astype(np.float32)

            # Ensure shape is (samples, channels)
            if wav.ndim == 2 and wav.shape[0] < wav.shape[1]:
                wav = wav.T
            elif wav.ndim == 1:
                wav = wav[:, None]
            return wav, int(sample_rate)
        finally:
            release_memory()

    def synthesize_stream(
        self,
        text: str,
        reference_audio_path: str,
        *,
        emo_vector: list[float] | None = None,
        emo_alpha: float = 1.0,
        duration_factor: float = 1.0,
    ):
        """Streaming synthesis: yields (audio_chunk_np, sample_rate) per segment."""
        text = _sanitize_text(text)
        if not text:
            raise ValueError("TTS 文本为空。")
        detected_language, language_token = _resolve_tts_language_token(text)

        text_chunks = _split_stream_text(text)
        logger.info(
            "IndexTTS-2.5 流式推理: text={!r}, ref={}, lang={}, chunks={}, duration_factor={}",
            text[:80], reference_audio_path, detected_language, len(text_chunks), duration_factor,
        )

        sample_rate = 22050
        try:
            with self._use_tts() as tts:
                for chunk_index, text_chunk in enumerate(text_chunks, start=1):
                    logger.debug("IndexTTS-2.5 流式分块 {}/{}: {!r}", chunk_index, len(text_chunks), text_chunk)
                    gen = tts.infer(
                        spk_audio_prompt=reference_audio_path,
                        text=text_chunk,
                        output_path="",
                        lang=language_token,
                        verbose=False,
                        emo_vector=emo_vector,
                        emo_alpha=emo_alpha,
                        stream_return=True,
                        duration_factor=max(0.5, min(2.0, float(duration_factor))),
                        max_mel_tokens=_INDEX_TTS_V25_MAX_MEL_TOKENS,
                        max_text_tokens_per_segment=80,
                        more_segment_before=80,
                    )

                    for item in gen:
                        wav = _normalize_stream_audio_item(item)
                        if wav is None:
                            break
                        yield wav, sample_rate
        finally:
            release_memory()

    @contextmanager
    def _use_tts(
        self,
        *,
        required: bool = True,
        error_message: str = "IndexTTS-2.5 service is not loaded. Call .load() first.",
    ):
        tts: Any | None = None
        with self._lock:
            while self._unload_in_progress or self._active_operation_count > 0:
                self._state_changed.wait()
            tts = self._tts
            if tts is None:
                if required:
                    raise RuntimeError(error_message)
            else:
                self._active_operation_count += 1
        if tts is None:
            yield None
            return
        try:
            yield tts
        finally:
            with self._lock:
                self._active_operation_count -= 1
                self._state_changed.notify_all()
