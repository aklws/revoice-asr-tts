from __future__ import annotations

import faulthandler
import importlib
import tempfile
import threading
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from app.core.audio import resample_waveform
from app.core.config import CHECKPOINTS_DIR, DEFAULT_EMOTION_MODEL_DIR, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
StatusCallback = Callable[[str], None]
EMOTION2VEC_MODEL_ID = "iic/emotion2vec_plus_base"
EMOTION2VEC_SAMPLE_RATE = 16_000
_EXISTENCE_MARKERS = ("configuration.json", "README.md", "model.pt", "config.yaml")
_ZH_TO_EN = {
    "生气": "angry",
    "厌恶": "disgusted",
    "恐惧": "fearful",
    "开心": "happy",
    "中立": "neutral",
    "其他": "other",
    "难过": "sad",
    "吃惊": "surprised",
    "未知": "unknown",
}
_IMPORT_PROBE_TRACE_HANDLE = None


@dataclass(frozen=True, slots=True)
class EmotionPrediction:
    label_en: str
    label_display: str
    confidence: float
    labels: tuple[str, ...]
    scores: tuple[float, ...]
    raw: object


@cache
def _load_modelscope_pipeline() -> Any:
    module = importlib.import_module("modelscope.pipelines")
    return module.pipeline


@cache
def _register_modelscope_funasr_pipeline() -> None:
    # PyInstaller may ship an empty ModelScope AST index, so import the module
    # directly to ensure the decorator-based pipeline registration runs.
    importlib.import_module("modelscope.pipelines.audio.funasr_pipeline")


@cache
def _register_modelscope_funasr_model() -> None:
    # FunASRPipeline resolves the backend model from the MODELS registry.
    # Import the registration module explicitly so PyInstaller builds do not
    # depend on ModelScope's prebuilt AST index.
    importlib.import_module("modelscope.models.audio.funasr.model")


@cache
def _load_modelscope_tasks() -> Any:
    module = importlib.import_module("modelscope.utils.constant")
    return module.Tasks


@cache
def _load_modelscope_snapshot_download() -> Any:
    module = importlib.import_module("modelscope.hub.snapshot_download")
    return module.snapshot_download


@cache
def _load_funasr_automodel() -> Any:
    module = importlib.import_module("funasr")
    return module.AutoModel


def _normalize_label(raw_label: str) -> tuple[str, str]:
    normalized = raw_label.strip()
    if not normalized:
        return "unknown", "未知"
    if "/" in normalized:
        display_label, english_label = normalized.split("/", 1)
        english = english_label.strip().casefold() or "unknown"
        display = display_label.strip() or english
        return english, display
    english = _ZH_TO_EN.get(normalized, normalized.casefold())
    display = normalized
    if english == "unknown" and not display:
        display = "未知"
    return english, display


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
            "emotion2vec 模型目录 {} 不在 checkpoints 下，已收敛到 {}",
            resolved,
            fallback_dir,
        )
        return fallback_dir


def _enable_import_probe_faulthandler() -> None:
    global _IMPORT_PROBE_TRACE_HANDLE
    if _IMPORT_PROBE_TRACE_HANDLE is not None:
        return
    try:
        trace_path = Path(tempfile.gettempdir()) / "revoice-emotion-import-crash.log"
        _IMPORT_PROBE_TRACE_HANDLE = trace_path.open("a", encoding="utf-8")
        faulthandler.enable(file=_IMPORT_PROBE_TRACE_HANDLE, all_threads=True)
        logger.info("已启用情感识别崩溃跟踪日志: {}", trace_path)
    except Exception as exc:
        logger.warning("启用情感识别崩溃跟踪失败: {}", exc)


class EmotionRecognitionService:
    """Lazy emotion2vec wrapper for utterance-level microphone emotion inference."""

    def __init__(self, model_dir: Path | None = None, *, model_id: str = EMOTION2VEC_MODEL_ID) -> None:
        settings = get_settings()
        requested_model_dir = model_dir or settings.emotion_model_dir or DEFAULT_EMOTION_MODEL_DIR
        self.model_dir = _resolve_checkpoint_model_dir(requested_model_dir)
        self.model_id = model_id
        self.device = settings.emotion_device
        self._pipeline: Any = None
        self._backend = "modelscope"
        self._lock = threading.RLock()

    def load(self, *, status_callback: StatusCallback | None = None) -> None:
        with self._lock:
            if self._pipeline is not None:
                return
            self._ensure_model_available(status_callback=status_callback)
            self._log_debug_import_state()
            _register_modelscope_funasr_pipeline()
            _register_modelscope_funasr_model()
            pipeline = _load_modelscope_pipeline()
            Tasks = _load_modelscope_tasks()
            model_target = str(self.model_dir)

            logger.info("正在加载 emotion2vec+ 模型 (设备={})", self.device)
            if status_callback is not None:
                status_callback("正在加载实验性情感识别模型...")

            try:
                self._pipeline = pipeline(
                    task=Tasks.emotion_recognition,
                    model=model_target,
                    device=self.device,
                )
            except TypeError:
                # Older ModelScope releases may not accept an explicit device argument here.
                self._pipeline = pipeline(
                    task=Tasks.emotion_recognition,
                    model=model_target,
                )
            except KeyError as exc:
                if "funasr-pipeline" not in str(exc):
                    logger.exception("emotion2vec+ pipeline 初始化失败: model_dir={} device={}", model_target, self.device)
                    raise
                logger.warning("ModelScope 未注册 funasr-pipeline，回退到 FunASR AutoModel: {}", exc)
                if status_callback is not None:
                    status_callback("ModelScope 注册缺失，正在回退到 FunASR 情感识别...")
                AutoModel = _load_funasr_automodel()
                self._pipeline = AutoModel(
                    model=model_target,
                    device=self.device,
                    disable_update=True,
                )
                self._backend = "funasr"
            except Exception:
                logger.exception("emotion2vec+ pipeline 初始化失败: model_dir={} device={}", model_target, self.device)
                raise
            else:
                self._backend = "modelscope"
            logger.info("emotion2vec+ 加载完成")

    def unload(self) -> None:
        with self._lock:
            if self._pipeline is None:
                return
            self._pipeline = None
        logger.info("emotion2vec+ 已卸载")

    def predict_waveform(self, audio_np: np.ndarray, sample_rate: int) -> EmotionPrediction:
        with self._lock:
            if self._pipeline is None:
                raise RuntimeError("EmotionRecognitionService 尚未加载。")
            waveform = np.asarray(audio_np, dtype=np.float32)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1, dtype=np.float32)
            if waveform.size == 0:
                raise ValueError("音频为空，无法识别情感。")
            if sample_rate != EMOTION2VEC_SAMPLE_RATE:
                waveform = resample_waveform(waveform, sample_rate, EMOTION2VEC_SAMPLE_RATE)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                temp_path = Path(handle.name)
            try:
                sf.write(temp_path, waveform, EMOTION2VEC_SAMPLE_RATE)
                if self._backend == "funasr":
                    raw_result = self._pipeline.generate(
                        str(temp_path),
                        granularity="utterance",
                        extract_embedding=False,
                    )
                else:
                    raw_result = self._pipeline(
                        str(temp_path),
                        granularity="utterance",
                        extract_embedding=False,
                    )
            finally:
                temp_path.unlink(missing_ok=True)

            return self._parse_prediction(raw_result)

    def _ensure_model_available(self, *, status_callback: StatusCallback | None = None) -> None:
        if _has_local_model_files(self.model_dir):
            return

        snapshot_download = _load_modelscope_snapshot_download()
        self.model_dir.parent.mkdir(parents=True, exist_ok=True)
        if status_callback is not None:
            status_callback(f"正在下载实验性情感识别模型到 {self.model_dir} ...")
        logger.info("emotion2vec+ 未找到，开始下载到 {}", self.model_dir)
        snapshot_download(self.model_id, local_dir=str(self.model_dir))
        if not _has_local_model_files(self.model_dir):
            raise RuntimeError(f"emotion2vec+ 下载后仍未在本地目录找到模型文件: {self.model_dir}")

    def _log_debug_import_state(self) -> None:
        _enable_import_probe_faulthandler()
        modules_to_probe = (
            "modelscope",
            "modelscope.pipelines",
            "datasets",
            "pandas",
            "pandas._libs",
            "pandas._libs.lib",
        )
        for module_name in modules_to_probe:
            try:
                logger.info("情感识别依赖探测开始: {}", module_name)
                module = importlib.import_module(module_name)
                module_file = str(getattr(module, "__file__", "<built-in>"))
                logger.info("情感识别依赖探测成功: {} -> {}", module_name, module_file)
            except Exception:
                logger.exception("情感识别依赖探测失败: {}", module_name)
                raise

    @staticmethod
    def _parse_prediction(raw_result: object) -> EmotionPrediction:
        payload = raw_result
        if isinstance(payload, list) and payload:
            payload = payload[0]
        if not isinstance(payload, dict):
            raise RuntimeError(f"无法解析 emotion2vec 输出: {type(raw_result)}")

        raw_labels = payload.get("labels")
        raw_scores = payload.get("scores")
        if not isinstance(raw_labels, list) or not isinstance(raw_scores, list):
            raise RuntimeError(f"emotion2vec 输出缺少 labels/scores: {payload}")

        ranked: list[tuple[float, str, str]] = []
        display_labels: list[str] = []
        numeric_scores: list[float] = []
        for raw_label, raw_score in zip(raw_labels, raw_scores, strict=False):
            if not isinstance(raw_label, str):
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            english_label, display_label = _normalize_label(raw_label)
            display_labels.append(display_label)
            numeric_scores.append(score)
            if english_label == "unknown" and score <= 0.0:
                continue
            ranked.append((score, english_label, display_label))

        if not ranked:
            return EmotionPrediction(
                label_en="unknown",
                label_display="未知",
                confidence=0.0,
                labels=tuple(display_labels),
                scores=tuple(numeric_scores),
                raw=raw_result,
            )

        confidence, label_en, label_display = max(ranked, key=lambda item: item[0])
        return EmotionPrediction(
            label_en=label_en,
            label_display=label_display,
            confidence=float(confidence),
            labels=tuple(display_labels),
            scores=tuple(numeric_scores),
            raw=raw_result,
        )
