from __future__ import annotations

import os
import sys
import threading
from functools import cache
import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from app.core.audio import resample_waveform
from app.core.logging import get_logger
from app.core.paths import get_bin_dir
from app.core.runtime import release_memory
from app.models.asr import ASRRequest

logger = get_logger(__name__)


@cache
def _load_qwen_asr_symbols() -> tuple[type[object], type[object]]:
    asr_module = importlib.import_module("qwen_asr_gguf.inference.asr")
    schema_module = importlib.import_module("qwen_asr_gguf.inference.schema")
    return asr_module.QwenASREngine, schema_module.ASREngineConfig


@dataclass(frozen=True, slots=True)
class ASRResult:
    text: str
    language: str | None = None


class ASRService:
    """Calls QwenASREngine directly to keep model loaded and avoid 1.2s initialization delay per chunk."""

    def __init__(self, _model_dir: Path | None = None) -> None:
        self._loaded = False
        self.engine = None
        self._lock = threading.RLock()
        # Make sure the qwen_asr_gguf package is in sys.path
        qwen_bin_dir = str(get_bin_dir("Qwen3-ASR-Transcribe"))
        if qwen_bin_dir not in sys.path:
            sys.path.insert(0, qwen_bin_dir)

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return

            # 解决由于多次加载 OpenMP 运行时库导致的崩溃问题 (常见于 PyTorch 与 GGUF/ONNX 混用时)
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            QwenASREngine, ASREngineConfig = _load_qwen_asr_symbols()

            logger.info("正在初始化 QwenASREngine ...")
            model_dir = get_bin_dir("Qwen3-ASR-Transcribe", "model")
            if not model_dir.exists():
                raise FileNotFoundError(f"ASR model directory not found: {model_dir}")
            config = ASREngineConfig(
                model_dir=str(model_dir),
                use_dml=True,
                n_ctx=2048,
                chunk_size=40.0,
                memory_num=1,
                verbose=False,
            )
            self.engine = QwenASREngine(config)
            # 预热
            logger.info("正在预热 ASR 引擎...")
            warmup_wav = np.zeros(16000, dtype=np.float32)
            self.engine.asr(warmup_wav, context="", language="Chinese")
            self._loaded = True
            logger.info("ASR 引擎预热完成")

    def unload(self) -> None:
        with self._lock:
            if self.engine is not None:
                self.engine.shutdown()
                self.engine = None
            self._loaded = False
        release_memory()
        logger.info("ASR 引擎已卸载")

    def transcribe(self, request: ASRRequest) -> ASRResult:
        with self._lock:
            if not self._loaded or self.engine is None:
                raise RuntimeError("ASRService is not loaded")

            lang = request.language or "Chinese"
            context = request.context or ""

            if request.audio_ndarray is not None:
                waveform, sr = request.audio_ndarray
                # The engine expects 16000 Hz, workers.py already resamples if needed
                res = self.engine.asr(
                    audio=waveform,
                    context=context,
                    language=lang,
                )
                return ASRResult(text=res.text, language=lang)

            waveform, sr = sf.read(request.audio_path, dtype="float32")
            if sr != 16000:
                waveform = resample_waveform(waveform, sr, 16000)
            res = self.engine.asr(
                audio=waveform,
                context=context,
                language=lang,
            )
            return ASRResult(text=res.text, language=lang)
