from __future__ import annotations

import importlib
import threading
from functools import cache
from typing import Callable

import numpy as np

from app.core.audio import resample_waveform
from app.core.logging import get_logger
from app.core.runtime import release_memory

logger = get_logger(__name__)
StatusCallback = Callable[[str], None]


@cache
def _load_funasr_automodel_class() -> type[object]:
    module = importlib.import_module("funasr")
    return module.AutoModel


class StreamingVADService:
    """Wrap FunASR FSMN-VAD streaming inference for live microphone segmentation."""

    def __init__(
        self,
        *,
        model_name: str = "fsmn-vad",
        device: str = "cpu",
        chunk_size_ms: int = 200,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.chunk_size_ms = max(1, int(chunk_size_ms))
        self._lock = threading.RLock()
        self._model = None
        self._cache: dict[str, object] = {}

    @staticmethod
    def _normalize_waveform(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        normalized = np.asarray(waveform, dtype=np.float32)
        if normalized.ndim > 1:
            normalized = normalized.mean(axis=1, dtype=np.float32)
        if sample_rate != 16000:
            normalized = resample_waveform(normalized, sample_rate, 16000)
        return np.ascontiguousarray(normalized, dtype=np.float32)

    @staticmethod
    def _extract_segments(result: object) -> list[tuple[int, int]]:
        if not isinstance(result, list):
            return []
        segments: list[tuple[int, int]] = []
        for item in result:
            if not isinstance(item, dict):
                continue
            values = item.get("value")
            if not isinstance(values, list):
                continue
            for pair in values:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                start, end = pair
                try:
                    start_ms = int(start)
                    end_ms = int(end)
                except Exception:
                    continue
                segments.append((start_ms, end_ms))
        return segments

    def load(self, *, status_callback: StatusCallback | None = None) -> None:
        with self._lock:
            if self._model is not None:
                return
            AutoModel = _load_funasr_automodel_class()
            if status_callback is not None:
                status_callback("正在加载 FunASR 流式 VAD ...")
            logger.info(
                "正在初始化 FunASR 流式 VAD (model={} device={} chunk={}ms) ...",
                self.model_name,
                self.device,
                self.chunk_size_ms,
            )
            self._model = AutoModel(
                model=self.model_name,
                hub="ms",
                device=self.device,
                disable_update=True,
            )
            self._cache = {}
            logger.info("FunASR 流式 VAD 加载完成")

    def reset(self) -> None:
        with self._lock:
            self._cache = {}

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._cache = {}
        release_memory()
        logger.info("FunASR 流式 VAD 已卸载")

    def process_chunk(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        is_final: bool = False,
    ) -> list[tuple[int, int]]:
        with self._lock:
            if self._model is None:
                raise RuntimeError("StreamingVADService 尚未加载。")
            normalized = self._normalize_waveform(waveform, sample_rate)
            if normalized.size == 0:
                if not is_final:
                    return []
            result = self._model.generate(
                input=normalized,
                cache=self._cache,
                is_final=is_final,
                chunk_size=self.chunk_size_ms,
            )
            return self._extract_segments(result)
