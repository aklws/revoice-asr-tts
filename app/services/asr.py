from __future__ import annotations

import threading
from functools import cache
import importlib
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import soundfile as sf
import torch

from app.core.audio import resample_waveform
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.modelscope_compat import snapshot_download as modelscope_snapshot_download
from app.core.runtime import release_memory
from app.models.asr import ASRRequest
from app.services.translation import detect_text_language

logger = get_logger(__name__)


@cache
def _load_funasr_automodel_class() -> type[object]:
    module = importlib.import_module("funasr")
    return module.AutoModel


@dataclass(frozen=True, slots=True)
class ASRResult:
    text: str
    language: str | None = None


class ASRService:
    """Wrap FunASR ASR backends and keep the model hot for live microphone chunks."""

    def __init__(self, _model_dir: Path | None = None) -> None:
        settings = get_settings()
        self._loaded = False
        self.engine = None
        self._lock = threading.RLock()
        self.model_dir = (_model_dir or settings.asr_model_dir).expanduser().resolve()
        self.model_repo = settings.asr_model_repo
        self.device = self._normalize_device(settings.asr_device)

    @staticmethod
    def _normalize_device(raw: str | None) -> str:
        value = str(raw or "cpu").strip().lower()
        if value in {"cuda", "gpu"}:
            return "cuda:0"
        return value or "cpu"

    @staticmethod
    def _normalize_language(raw: str | None) -> str | None:
        if raw is None:
            return None
        normalized = raw.strip().casefold()
        if not normalized:
            return None
        mapping = {
            "auto": None,
            "chinese": "中文",
            "zh": "中文",
            "zh-cn": "中文",
            "中文": "中文",
            "english": "英文",
            "en": "英文",
            "英文": "英文",
            "japanese": "日文",
            "ja": "日文",
            "jp": "日文",
            "日语": "日文",
            "日文": "日文",
        }
        return mapping.get(normalized, raw.strip())

    @staticmethod
    def _build_hotwords(context: str | None) -> list[str] | None:
        if context is None:
            return None
        phrases = [
            part.strip()
            for part in re.split(r"[\r\n,，。！？；;]+", context)
            if part and part.strip()
        ]
        if not phrases:
            return None
        # Use only the most recent short phrases to bias recognition without bloating the prompt.
        hotwords = [phrase[:64] for phrase in phrases[-4:]]
        return hotwords or None

    @staticmethod
    def _extract_text(result: object) -> str:
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                return str(first.get("text") or "").strip()
        if isinstance(result, dict):
            return str(result.get("text") or "").strip()
        return str(result or "").strip()

    @staticmethod
    def _extract_language(result: object) -> str | None:
        candidates: list[object] = []
        if isinstance(result, list) and result:
            candidates.append(result[0])
        candidates.append(result)
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            for key in ("language", "lang", "source_language", "detected_language"):
                value = candidate.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    @staticmethod
    def _normalize_waveform(waveform: np.ndarray) -> np.ndarray:
        normalized = np.asarray(waveform, dtype=np.float32)
        if normalized.ndim > 1:
            normalized = normalized.mean(axis=1, dtype=np.float32)
        return np.ascontiguousarray(normalized, dtype=np.float32)

    def _ensure_model_available(self) -> str:
        config_path = self.model_dir / "config.yaml"
        if config_path.exists():
            return str(self.model_dir)
        logger.info(
            "FunASR 模型目录缺失，正在下载到 checkpoints: repo={} target={}",
            self.model_repo,
            self.model_dir,
        )
        self.model_dir.parent.mkdir(parents=True, exist_ok=True)
        modelscope_snapshot_download(self.model_repo, local_dir=str(self.model_dir))
        if not config_path.exists():
            raise RuntimeError(f"FunASR 模型下载后仍缺少 config.yaml: {self.model_dir}")
        return str(self.model_dir)

    def _load_default_backend(self, model_target: str) -> None:
        AutoModel = _load_funasr_automodel_class()
        model_kwargs = {
            "model": model_target,
            "hub": "ms",
            "device": self.device,
            "vad_model": "fsmn-vad",
            "vad_kwargs": {"max_single_segment_time": 30_000},
            "disable_update": True,
        }
        remote_code_file = (Path(model_target) / "model.py").resolve()
        if remote_code_file.exists():
            model_kwargs["trust_remote_code"] = True
            model_kwargs["remote_code"] = str(remote_code_file)
            logger.info(
                "正在初始化 FunASR AutoModel (model={} device={} remote_code={}) ...",
                model_target,
                self.device,
                remote_code_file,
            )
        else:
            logger.info(
                "正在初始化 FunASR AutoModel (model={} device={} remote_code=disabled) ...",
                model_target,
                self.device,
            )
        self.engine = AutoModel(**model_kwargs)

    def _generate(self, waveform: np.ndarray, *, language: str | None, hotwords: list[str] | None, warmup: bool = False):
        if self.engine is None:
            raise RuntimeError("ASR 引擎未初始化。")
        normalized_hotwords = hotwords or []
        waveform_tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
        return self.engine.generate(
            input=[waveform_tensor],
            cache={},
            batch_size=1,
            hotwords=normalized_hotwords,
            language=language,
            itn=True,
        )

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return

            model_target = self._ensure_model_available()
            self._load_default_backend(model_target)

            logger.info("正在预热 ASR 引擎 ...")
            warmup_wav = np.zeros(16000, dtype=np.float32)
            self._generate(warmup_wav, language="中文", hotwords=None, warmup=True)
            self._loaded = True
            logger.info("ASR 引擎预热完成")

    def unload(self) -> None:
        with self._lock:
            if self.engine is not None:
                self.engine = None
            self._loaded = False
        release_memory()
        logger.info("ASR 引擎已卸载")

    def transcribe(self, request: ASRRequest) -> ASRResult:
        with self._lock:
            if not self._loaded or self.engine is None:
                raise RuntimeError("ASRService is not loaded")

            lang = self._normalize_language(request.language)
            context = request.context or ""
            hotwords = self._build_hotwords(context)

            if request.audio_ndarray is not None:
                waveform, sr = request.audio_ndarray
                if sr != 16000:
                    waveform = resample_waveform(waveform, sr, 16000)
                waveform = self._normalize_waveform(waveform)
                result = self._generate(waveform, language=lang, hotwords=hotwords)
                text = self._extract_text(result)
                detected_language = self._extract_language(result)
                if request.return_language and detected_language is None and lang is None:
                    detected_language = detect_text_language(text)
                return ASRResult(text=text, language=detected_language or request.language or lang)

            waveform, sr = sf.read(request.audio_path, dtype="float32")
            if sr != 16000:
                waveform = resample_waveform(waveform, sr, 16000)
            waveform = self._normalize_waveform(waveform)
            result = self._generate(waveform, language=lang, hotwords=hotwords)
            text = self._extract_text(result)
            detected_language = self._extract_language(result)
            if request.return_language and detected_language is None and lang is None:
                detected_language = detect_text_language(text)
            return ASRResult(text=text, language=detected_language or request.language or lang)
