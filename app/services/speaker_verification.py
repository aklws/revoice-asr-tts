from __future__ import annotations

import importlib
import threading
from functools import cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.core.audio import resample_waveform
from app.core.config import (
    CHECKPOINTS_DIR,
    DEFAULT_SPEAKER_MODEL_DIR,
    get_settings,
)
from app.core.logging import get_logger
from app.core.modelscope_compat import snapshot_download as modelscope_snapshot_download

logger = get_logger(__name__)
StatusCallback = Callable[[str], None]
_EXISTENCE_MARKERS = ("configuration.json", "README.md", "config.yaml")


@cache
def _load_modelscope_pipeline() -> Any:
    module = importlib.import_module("modelscope.pipelines")
    return module.pipeline


@cache
def _load_modelscope_tasks() -> Any:
    module = importlib.import_module("modelscope.utils.constant")
    return module.Tasks


@cache
def _register_modelscope_funasr_pipeline() -> None:
    importlib.import_module("modelscope.pipelines.audio.funasr_pipeline")


@cache
def _register_modelscope_funasr_model() -> None:
    importlib.import_module("modelscope.models.audio.funasr.model")


def _has_local_model_files(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    for marker in _EXISTENCE_MARKERS:
        if (model_dir / marker).exists():
            return True
    return any(model_dir.iterdir())


def _resolve_checkpoint_model_dir(model_dir: Path) -> Path:
    try:
        resolved = model_dir.expanduser().resolve()
    except OSError:
        resolved = model_dir.expanduser()

    checkpoints_dir = CHECKPOINTS_DIR.expanduser().resolve()
    try:
        resolved.relative_to(checkpoints_dir)
        return resolved
    except ValueError:
        fallback_dir = (checkpoints_dir / resolved.name).resolve()
        logger.warning(
            "说话人模型目录 {} 不在 checkpoints 下，已收敛到 {}",
            resolved,
            fallback_dir,
        )
        return fallback_dir


class SpeakerVerificationService:
    """CAMPPlus speaker verification service backed by ModelScope/FunASR."""

    def __init__(
        self,
        model_dir: Path | None = None,
        *,
        model_id: str | None = None,
        model_revision: str | None = None,
    ) -> None:
        settings = get_settings()
        requested_model_dir = model_dir or settings.speaker_model_dir or DEFAULT_SPEAKER_MODEL_DIR
        self.model_dir = _resolve_checkpoint_model_dir(requested_model_dir)
        self.model_id = model_id or settings.speaker_model_repo
        self.model_revision = model_revision or settings.speaker_model_revision
        self.device = self._normalize_modelscope_device(settings.asr_device)
        self._pipeline: Any = None
        self._lock = threading.RLock()

    @staticmethod
    def _normalize_modelscope_device(raw: str | None) -> str:
        normalized = str(raw or "cpu").strip().lower()
        if normalized.startswith("cuda") or normalized == "gpu":
            return "gpu"
        return "cpu"

    @staticmethod
    def _normalize_waveform(audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        waveform = np.asarray(audio_np, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1, dtype=np.float32)
        if sample_rate != 16000:
            waveform = resample_waveform(waveform, sample_rate, 16000)
        return np.ascontiguousarray(waveform, dtype=np.float32)

    @staticmethod
    def _extract_embedding_from_result(result: object) -> np.ndarray:
        if not isinstance(result, dict):
            raise RuntimeError(f"说话人模型返回格式异常: {type(result)}")
        embs = result.get("embs")
        if embs is None:
            raise RuntimeError("说话人模型未返回 embedding。")
        emb_array = np.asarray(embs, dtype=np.float32)
        if emb_array.ndim == 2 and emb_array.shape[0] >= 1:
            return np.ascontiguousarray(emb_array[0], dtype=np.float32)
        if emb_array.ndim == 1 and emb_array.size > 0:
            return np.ascontiguousarray(emb_array, dtype=np.float32)
        raise RuntimeError(f"说话人 embedding 形状异常: {emb_array.shape}")

    def load(self, *, status_callback: StatusCallback | None = None) -> None:
        with self._lock:
            if self._pipeline is not None:
                return
            self._ensure_model_available(status_callback=status_callback)
            _register_modelscope_funasr_pipeline()
            _register_modelscope_funasr_model()
            pipeline = _load_modelscope_pipeline()
            Tasks = _load_modelscope_tasks()
            model_target = str(self.model_dir)

            logger.info("正在加载 CAMPPlus 说话人识别模型 (设备={})", self.device)
            if status_callback is not None:
                status_callback("正在加载说话人识别模型...")
            try:
                self._pipeline = pipeline(
                    task=Tasks.speaker_verification,
                    model=model_target,
                    model_revision=self.model_revision,
                    device=self.device,
                )
            except TypeError:
                self._pipeline = pipeline(
                    task=Tasks.speaker_verification,
                    model=model_target,
                    model_revision=self.model_revision,
                )
            logger.info("CAMPPlus 说话人识别模型加载完成")

    def unload(self) -> None:
        with self._lock:
            if self._pipeline is None:
                return
            self._pipeline = None
        logger.info("CAMPPlus 说话人识别模型已卸载")

    def extract_embedding(self, audio_np: np.ndarray, sample_rate: int) -> list[float]:
        embeddings = self.extract_embeddings_batch([audio_np], sample_rate)
        if embeddings.shape[0] == 0:
            raise RuntimeError("未能提取说话人 embedding。")
        return embeddings[0].tolist()

    def extract_embeddings_batch(
        self,
        audio_list: list[np.ndarray],
        sample_rate: int,
    ) -> np.ndarray:
        with self._lock:
            if self._pipeline is None:
                raise RuntimeError("SpeakerVerificationService 尚未加载。")
            if not audio_list:
                return np.empty((0, 192), dtype=np.float32)

            embeddings: list[np.ndarray] = []
            for audio_np in audio_list:
                normalized = self._normalize_waveform(audio_np, sample_rate)
                if normalized.size == 0:
                    continue
                result = self._pipeline([normalized], output_emb=True)
                embeddings.append(self._extract_embedding_from_result(result))

            if not embeddings:
                return np.empty((0, 192), dtype=np.float32)
            return np.stack(embeddings).astype(np.float32, copy=False)

    def _ensure_model_available(self, *, status_callback: StatusCallback | None = None) -> None:
        if _has_local_model_files(self.model_dir):
            return

        self.model_dir.parent.mkdir(parents=True, exist_ok=True)
        if status_callback is not None:
            status_callback(f"正在下载说话人识别模型到 {self.model_dir} ...")
        modelscope_snapshot_download(
            self.model_id,
            revision=self.model_revision,
            local_dir=str(self.model_dir),
        )
