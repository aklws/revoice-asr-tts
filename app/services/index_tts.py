from __future__ import annotations

import asyncio
import importlib
import os
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

from app.core.config import DEFAULT_INDEX_TTS_MODEL_DIR, resolve_tts_device
from app.core.logging import get_logger
from app.core.runtime import release_memory
from app.core.audio import resample_waveform

logger = get_logger(__name__)


def _sanitize_text(text: str) -> str:
    """Remove/replace characters that confuse IndexTTS's SentencePiece tokenizer."""
    text = text.replace("\n", " ").strip()
    if not text:
        return text
    if not text.endswith((".", "!", "?", "。", "！", "？")):
        text += "."
    return text


@cache
def _load_indextts2_class() -> type[Any]:
    """按需加载 IndexTTS2，避免模块导入阶段就触发重型依赖初始化。"""
    module = importlib.import_module("indextts.infer_v2")
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
    candidates: list[Path] = []

    env_cl = os.environ.get("CC")
    if env_cl and Path(env_cl).exists():
        return env_cl, "CC 环境变量"

    found = shutil.which("cl")
    if found:
        return found, "PATH"

    vc_tools_dir = os.environ.get("VCToolsInstallDir")
    if vc_tools_dir:
        candidates.extend(Path(vc_tools_dir).glob(r"bin/Hostx64/x64/cl.exe"))

    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
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
                install_path = Path(install_dir)
                vcvars64 = install_path / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if vcvars64.exists():
                    _activate_msvc_environment(vcvars64)
                candidates.extend(
                    sorted(
                        install_path.glob(r"VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe"),
                        reverse=True,
                    )
                )
        except Exception as exc:
            logger.debug("通过 vswhere 查询 MSVC 失败: {}", exc)

    if not candidates:
        vs_roots = [
            Path(program_files_x86) / "Microsoft Visual Studio",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft Visual Studio",
        ]
        editions = ("BuildTools", "Community", "Professional", "Enterprise")
        years = ("2022", "2019", "2017")
        for root in vs_roots:
            for year in years:
                for edition in editions:
                    install_path = root / year / edition
                    vcvars64 = install_path / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                    if vcvars64.exists():
                        _activate_msvc_environment(vcvars64)
                    candidates.extend(
                        sorted(
                            install_path.glob(r"VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe"),
                            reverse=True,
                        )
                    )

    for candidate in candidates:
        if candidate.exists():
            os.environ.setdefault("CC", str(candidate))
            os.environ.setdefault("CXX", str(candidate))
            return str(candidate), "Visual Studio Build Tools"
    return None, None


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

    """Lightweight wrapper around indextts.infer_v2.IndexTTS2 for text-to-speech."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir or DEFAULT_INDEX_TTS_MODEL_DIR
        self.device = resolve_tts_device()
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
                    "IndexTTS-2 model directory not found. Set INDEX_TTS_MODEL_DIR or place the model under "
                    f"{DEFAULT_INDEX_TTS_MODEL_DIR}."
                )
            config_path = self.model_dir / "config.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"IndexTTS-2 config not found: {config_path}")

            accel_available, compiler_desc = _configure_c_compiler()
            use_accel = False
            if accel_available:
                logger.info("检测到 C 编译器 ({}), 但已主动关闭 IndexTTS Accel GPT 以降低显存占用。", compiler_desc)
            else:
                logger.info("未检测到可用 C 编译器或 MSVC 环境，IndexTTS Accel GPT 保持关闭。")

            logger.info("正在加载 IndexTTS-2 模型 (设备={})", self.device)
            index_tts_cls = _load_indextts2_class()
            self._tts = index_tts_cls(
                cfg_path=str(config_path),
                model_dir=self._model_dir_str,
                use_fp16=True,
                device=self.device if str(self.device).startswith("cuda") else None,
                use_cuda_kernel=True,
                use_accel=use_accel,
                use_torch_compile=True,
            )
            logger.info("IndexTTS-2 加载完成")

    def warmup_speaker(self, reference_audio_path: str, emo_vector: list[float] | None = None) -> None:
        """预热并缓存说话人参考音频的特征，避免首次推理时产生 2~3 秒延迟。"""
        try:
            with self._use_tts(required=False) as tts:
                if tts is None:
                    return
                logger.info("正在预热 TTS 参考音频特征: {}", reference_audio_path)
                try:
                    # 调用一次空文本推理来强制 TTS 提取并缓存特征
                    gen = tts.infer(
                        spk_audio_prompt=reference_audio_path,
                        text=".",
                        output_path="",
                        verbose=False,
                        emo_vector=emo_vector,
                        stream_return=True,
                    )
                    for _ in gen:
                        pass
                    logger.info("TTS 参考音频特征预热完成")
                except Exception as e:
                    logger.warning("TTS 参考音频预热失败: {}", e)
        finally:
            release_memory()


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
            logger.info("IndexTTS-2 已卸载")
        finally:
            with self._lock:
                self._unload_in_progress = False
                self._state_changed.notify_all()

    def extract_voiceprints_batch(self, audio_list: list[np.ndarray], sample_rate: int) -> np.ndarray:
        """批量提取 CAM++ 声纹向量，返回 shape=(N, 192) 的 float32 数组。"""
        with self._use_tts() as tts:
            if not audio_list:
                return np.empty((0, 192), dtype=np.float32)

            torchaudio = _load_torchaudio()
            feature_tensors: list[torch.Tensor] = []
            max_frame_count = 0

            for audio_np in audio_list:
                if sample_rate != 16000:
                    audio_np = resample_waveform(audio_np, sample_rate, 16000)

                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=1)

                audio_tensor = torch.from_numpy(audio_np.astype(np.float32, copy=False)).unsqueeze(0)
                feat = torchaudio.compliance.kaldi.fbank(
                    audio_tensor,
                    num_mel_bins=80,
                    dither=0,
                    sample_frequency=16000,
                )
                feat = feat - feat.mean(dim=0, keepdim=True)
                feature_tensors.append(feat)
                max_frame_count = max(max_frame_count, int(feat.shape[0]))

            if not feature_tensors or max_frame_count <= 0:
                return np.empty((0, 192), dtype=np.float32)

            batch_size = len(feature_tensors)
            feature_batch = torch.zeros((batch_size, max_frame_count, 80), dtype=torch.float32)
            for index, feat in enumerate(feature_tensors):
                frame_count = int(feat.shape[0])
                feature_batch[index, :frame_count, :] = feat

            with torch.no_grad():
                if hasattr(tts, "_compute_campplus_style"):
                    style = tts._compute_campplus_style(feature_batch)
                else:
                    style = tts.campplus_model(feature_batch.to(tts.device))
                return style.cpu().numpy().astype(np.float32, copy=False)
            
    def extract_voiceprint(self, audio_np: np.ndarray, sample_rate: int) -> list[float]:
        """Extract a 192-dimensional voiceprint embedding using CAM++."""
        embeddings = self.extract_voiceprints_batch([audio_np], sample_rate)
        if embeddings.shape[0] == 0:
            raise RuntimeError("未能提取声纹向量。")
        return embeddings[0].tolist()

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
    ) -> tuple[np.ndarray, int]:
        text = _sanitize_text(text)
        if not text:
            raise ValueError("TTS 文本为空。")

        logger.info(
            "IndexTTS-2 推理: text={!r}, ref={}, emo_vector={}, emo_alpha={}",
            text[:80], reference_audio_path, emo_vector, emo_alpha,
        )

        try:
            with self._use_tts(error_message="IndexTTS-2 服务尚未加载，请先调用 .load()。") as tts:
                result = tts.infer(
                    spk_audio_prompt=reference_audio_path,
                    text=text,
                    output_path=output_path or "",
                    verbose=False,
                    emo_vector=emo_vector,
                    emo_alpha=emo_alpha,
                    max_text_tokens_per_segment=120,
                )

            # V2 infer returns (sampling_rate, wav_data) or output_path
            if isinstance(result, str):
                # output_path was provided — result is the path
                wav, sample_rate = sf.read(result, dtype="float32")
            elif isinstance(result, tuple) and len(result) == 2:
                sample_rate, wav = result
            else:
                raise RuntimeError(f"Unexpected IndexTTS-2 infer result type: {type(result)}")

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
    ):
        """Streaming synthesis: yields (audio_chunk_np, sample_rate) per segment."""
        text = _sanitize_text(text)
        if not text:
            raise ValueError("TTS 文本为空。")

        text_chunks = [text]
        logger.info(
            "IndexTTS-2 流式推理: text={!r}, ref={}, chunks={}",
            text[:80], reference_audio_path, len(text_chunks),
        )

        sample_rate = 22050
        try:
            with self._use_tts() as tts:
                for chunk_index, text_chunk in enumerate(text_chunks, start=1):
                    logger.debug("IndexTTS-2 流式分块 {}/{}: {!r}", chunk_index, len(text_chunks), text_chunk)
                    gen = tts.infer(
                        spk_audio_prompt=reference_audio_path,
                        text=text_chunk,
                        output_path="",
                        verbose=False,
                        emo_vector=emo_vector,
                        emo_alpha=emo_alpha,
                        stream_return=True,
                        max_text_tokens_per_segment=120,
                    )

                    for item in gen:
                        if item is None:
                            break
                        if isinstance(item, tuple):
                            break  # (sampling_rate, wav) — end of stream
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
                        # IndexTTS yields float32 clamped to int16 range; normalize to [-1, 1]
                        if wav.dtype == np.int16:
                            wav = wav.astype(np.float32) / 32768.0
                        elif wav.dtype != np.float32:
                            wav = wav.astype(np.float32)
                        elif np.max(np.abs(wav)) > 2.0:
                            wav = wav.astype(np.float32, copy=False) / 32768.0
                        yield wav, sample_rate
        finally:
            release_memory()

    @contextmanager
    def _use_tts(
        self,
        *,
        required: bool = True,
        error_message: str = "IndexTTS-2 service is not loaded. Call .load() first.",
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
