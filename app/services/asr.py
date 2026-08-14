from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
import re
from contextlib import contextmanager

import numpy as np

from app.core.audio import resample_waveform
from app.core.config import get_settings
from app.core.funasr_gguf import load_funasr_gguf_runtime, resolve_quantized_artifacts
from app.core.logging import get_logger
from app.core.runtime import release_memory
from app.models.asr import ASRRequest
from app.services.translation import detect_text_language

logger = get_logger(__name__)
_RICH_TRANSCRIPTION_TAG_RE = re.compile(r"<\|[^|<>]+\|>")


@dataclass(frozen=True, slots=True)
class ASRResult:
    text: str
    language: str | None = None


@dataclass(slots=True)
class _StreamingSession:
    stream: object
    waveform: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))


class ASRService:
    """Wrap the quantized Fun-ASR GGUF backend for live microphone chunks."""

    def __init__(self, _model_dir=None) -> None:
        settings = get_settings()
        self._loaded = False
        self.engine = None
        self._lock = threading.RLock()
        self.model_repo = settings.asr_model_repo
        self.quantized_artifacts = resolve_quantized_artifacts()
        self.model_dir = self.quantized_artifacts.model_dir if self.quantized_artifacts is not None else settings.asr_model_dir
        self.backend_name = "funasr_gguf"

    @staticmethod
    def _normalize_language(raw: str | None) -> str | None:
        if raw is None:
            return None
        normalized = raw.strip().casefold()
        if not normalized:
            return None
        mapping = {
            "auto": None,
            "automatic": None,
            "detect": None,
            "auto-detect": None,
            "自动": None,
            "自动识别": None,
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
        hotwords = [phrase[:64] for phrase in phrases[-4:]]
        return hotwords or None

    @staticmethod
    def _clean_transcript_text(text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        cleaned = _RICH_TRANSCRIPTION_TAG_RE.sub("", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip()

    @staticmethod
    def _normalize_waveform(waveform: np.ndarray) -> np.ndarray:
        normalized = np.asarray(waveform, dtype=np.float32)
        if normalized.ndim > 1:
            normalized = normalized.mean(axis=1, dtype=np.float32)
        return np.ascontiguousarray(normalized, dtype=np.float32)

    def _build_request_options(
        self,
        *,
        language: str | None,
        context: str | None,
        enable_ctc: bool,
        enable_prompt_context: bool,
    ) -> tuple[str | None, list[str] | None, str | None]:
        lang = self._normalize_language(language)
        normalized_context = str(context or "").strip()
        hotwords = self._build_hotwords(normalized_context) if (enable_ctc and enable_prompt_context) else None
        effective_context = normalized_context if enable_prompt_context else None
        if self.backend_name == "funasr_gguf":
            # The current GGUF/llama.cpp backend may crash in native code during
            # long-running live sessions when contextual biasing is enabled.
            # Disable both raw prompt context and hotwords for stability.
            hotwords = None
            effective_context = None
        return lang, hotwords, effective_context

    @staticmethod
    def _resolve_result_language(
        *,
        text: str,
        return_language: bool,
        request_language: str | None,
        normalized_language: str | None,
    ) -> str | None:
        detected_language = None
        if return_language and normalized_language is None:
            detected_language = detect_text_language(text)
        return detected_language or request_language or normalized_language

    def _normalize_request_waveform(self, audio_ndarray: tuple[np.ndarray, int]) -> np.ndarray:
        waveform, sample_rate = audio_ndarray
        if sample_rate != 16000:
            waveform = resample_waveform(waveform, sample_rate, 16000)
        return self._normalize_waveform(waveform)

    def _decode_stream_result(
        self,
        stream: object,
        *,
        language: str | None,
        hotwords: list[str] | None,
        context: str | None,
        enable_ctc: bool,
    ) -> str:
        if hasattr(self.engine, "update_hotwords"):
            self.engine.update_hotwords(hotwords or [])
        with self._engine_ctc_mode(enable_ctc):
            result = self.engine.decode_stream(
                stream,
                language=language,
                context=context,
                verbose=False,
            )
        return self._clean_transcript_text(getattr(result, "text", ""))

    def _build_asr_result(
        self,
        *,
        text: str,
        return_language: bool,
        request_language: str | None,
        normalized_language: str | None,
    ) -> ASRResult:
        return ASRResult(
            text=text,
            language=self._resolve_result_language(
                text=text,
                return_language=return_language,
                request_language=request_language,
                normalized_language=normalized_language,
            ),
        )

    @staticmethod
    def _normalize_gguf_onnx_provider(raw: str | None) -> str:
        normalized = str(raw or "").strip().lower()
        mapping = {
            "cpu": "CPU",
            "cuda": "CUDA",
            "gpu": "CUDA",
            "dml": "DML",
            "directml": "DML",
            "trt": "TensorRT",
            "tensorrt": "TensorRT",
        }
        return mapping.get(normalized, "CPU")

    def _resolve_gguf_onnx_provider(self) -> str:
        configured = str(os.getenv("FUNASR_GGUF_ONNX_PROVIDER") or "").strip()
        if configured:
            return self._normalize_gguf_onnx_provider(configured)
        return self._normalize_gguf_onnx_provider(get_settings().asr_device)

    @staticmethod
    def _resolve_gguf_llm_use_gpu() -> bool:
        raw = str(os.getenv("FUNASR_GGUF_LLM_USE_GPU") or "").strip().lower()
        if raw in {"0", "false", "no", "off"}:
            return False
        if raw in {"1", "true", "yes", "on"}:
            return True
        return True

    def _load_quantized_backend(self) -> None:
        if self.quantized_artifacts is None:
            raise RuntimeError("量化 ASR 产物不存在。")
        FunASREngine, ASREngineConfig = load_funasr_gguf_runtime()
        config = ASREngineConfig(
            encoder_onnx_path=str(self.quantized_artifacts.encoder_onnx_path),
            ctc_onnx_path=str(self.quantized_artifacts.ctc_onnx_path),
            decoder_gguf_path=str(self.quantized_artifacts.decoder_gguf_path),
            tokens_path=str(self.quantized_artifacts.tokens_path),
            enable_ctc=True,
            onnx_provider=self._resolve_gguf_onnx_provider(),
            llm_use_gpu=self._resolve_gguf_llm_use_gpu(),
            verbose=False,
        )
        logger.info(
            "正在初始化 FunASR GGUF 引擎 (model_dir={} onnx_provider={} llm_use_gpu={}) ...",
            self.quantized_artifacts.model_dir,
            config.onnx_provider,
            config.llm_use_gpu,
        )
        self.engine = FunASREngine(config)
        self.backend_name = "funasr_gguf"

    @contextmanager
    def _engine_ctc_mode(self, enabled: bool):
        if self.engine is None or not hasattr(self.engine, "models"):
            yield
            return
        models = getattr(self.engine, "models", None)
        config = getattr(models, "config", None)
        if config is None or not hasattr(config, "enable_ctc"):
            yield
            return
        previous = bool(config.enable_ctc)
        config.enable_ctc = bool(enabled)
        try:
            yield
        finally:
            config.enable_ctc = previous

    def _generate_quantized(
        self,
        waveform: np.ndarray,
        *,
        language: str | None,
        context: str | None,
        hotwords: list[str] | None,
    ) -> str:
        if self.engine is None:
            raise RuntimeError("ASR 引擎未初始化。")
        normalized_hotwords = hotwords or []
        if hasattr(self.engine, "update_hotwords"):
            self.engine.update_hotwords(normalized_hotwords)
        stream = self.engine.create_stream()
        stream.accept_waveform(16000, np.asarray(waveform, dtype=np.float32))
        result = self.engine.decode_stream(
            stream,
            language=language,
            context=context or None,
            verbose=False,
        )
        return self._clean_transcript_text(getattr(result, "text", ""))

    def create_streaming_session(self) -> object:
        with self._lock:
            if not self._loaded or self.engine is None:
                raise RuntimeError("ASRService is not loaded")
            if not hasattr(self.engine, "create_stream"):
                raise RuntimeError("当前 ASR 后端不支持流式会话。")
            return _StreamingSession(stream=self.engine.create_stream())

    def reset_streaming_session(self, session: object) -> None:
        with self._lock:
            if not isinstance(session, _StreamingSession):
                return
            if self.engine is None or not hasattr(self.engine, "create_stream"):
                session.waveform = np.empty(0, dtype=np.float32)
                return
            session.stream = self.engine.create_stream()
            session.waveform = np.empty(0, dtype=np.float32)

    def transcribe_stream_chunk(self, session: object, request: ASRRequest) -> ASRResult:
        with self._lock:
            if not self._loaded or self.engine is None:
                raise RuntimeError("ASRService is not loaded")
            if not isinstance(session, _StreamingSession):
                raise RuntimeError("无效的流式会话对象。")
            if request.audio_ndarray is None:
                raise RuntimeError("流式识别需要音频块。")

            chunk = self._normalize_request_waveform(request.audio_ndarray)
            if chunk.size == 0:
                return ASRResult(text="", language=request.language)

            if session.waveform.size == 0:
                session.waveform = chunk
            else:
                session.waveform = np.concatenate((session.waveform, chunk)).astype(np.float32, copy=False)
            session.stream.accept_waveform(16000, session.waveform)

            enable_ctc = bool(request.enable_ctc)
            lang, hotwords, effective_context = self._build_request_options(
                language=request.language,
                context=request.context,
                enable_ctc=enable_ctc,
                enable_prompt_context=bool(request.enable_prompt_context),
            )
            text = self._decode_stream_result(
                session.stream,
                language=lang,
                hotwords=hotwords,
                context=effective_context,
                enable_ctc=enable_ctc,
            )
            return self._build_asr_result(
                text=text,
                return_language=bool(request.return_language),
                request_language=request.language,
                normalized_language=lang,
            )

    def transcribe_stream_state(
        self,
        session: object,
        *,
        language: str | None,
        return_language: bool,
        context: str | None,
        enable_ctc: bool,
        enable_prompt_context: bool,
    ) -> ASRResult:
        with self._lock:
            if not self._loaded or self.engine is None:
                raise RuntimeError("ASRService is not loaded")
            if not isinstance(session, _StreamingSession):
                raise RuntimeError("无效的流式会话对象。")
            if session.waveform.size == 0:
                return ASRResult(text="", language=language)

            lang, hotwords, effective_context = self._build_request_options(
                language=language,
                context=context,
                enable_ctc=enable_ctc,
                enable_prompt_context=enable_prompt_context,
            )
            text = self._decode_stream_result(
                session.stream,
                language=lang,
                hotwords=hotwords,
                context=effective_context,
                enable_ctc=enable_ctc,
            )
            return self._build_asr_result(
                text=text,
                return_language=return_language,
                request_language=language,
                normalized_language=lang,
            )

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return

            self._load_quantized_backend()

            logger.info("正在预热 ASR 引擎 ...")
            warmup_wav = np.zeros(16000, dtype=np.float32)
            self._generate_quantized(
                warmup_wav,
                language=None,
                context=None,
                hotwords=None,
            )
            self._loaded = True
            logger.info("ASR 引擎预热完成 (backend={})", self.backend_name)

    def unload(self) -> None:
        with self._lock:
            if self.engine is not None:
                if self.backend_name == "funasr_gguf" and hasattr(self.engine, "cleanup"):
                    self.engine.cleanup()
                self.engine = None
            self._loaded = False
        release_memory()
        logger.info("ASR 引擎已卸载")

    def transcribe(self, request: ASRRequest) -> ASRResult:
        with self._lock:
            if not self._loaded or self.engine is None:
                raise RuntimeError("ASRService is not loaded")

            enable_ctc = bool(request.enable_ctc)
            lang, hotwords, effective_context = self._build_request_options(
                language=request.language,
                context=request.context,
                enable_ctc=enable_ctc,
                enable_prompt_context=bool(request.enable_prompt_context),
            )

            if request.audio_ndarray is not None:
                waveform = self._normalize_request_waveform(request.audio_ndarray)
                with self._engine_ctc_mode(enable_ctc):
                    text = self._generate_quantized(
                        waveform,
                        language=lang,
                        context=effective_context,
                        hotwords=hotwords,
                    )
                return self._build_asr_result(
                    text=text,
                    return_language=bool(request.return_language),
                    request_language=request.language,
                    normalized_language=lang,
                )

            if hasattr(self.engine, "update_hotwords"):
                self.engine.update_hotwords(hotwords or [])
            with self._engine_ctc_mode(enable_ctc):
                result = self.engine.transcribe(
                    request.audio_path,
                    language=lang,
                    context=effective_context,
                    verbose=False,
                    segment_size=60.0,
                    overlap=4.0,
                    srt=False,
                    temperature=0.4,
                )
            text = self._clean_transcript_text(getattr(result, "text", ""))
            return self._build_asr_result(
                text=text,
                return_language=bool(request.return_language),
                request_language=request.language,
                normalized_language=lang,
            )
