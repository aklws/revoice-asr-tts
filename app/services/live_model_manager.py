from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger
from app.services.asr import ASRService
from app.services.emotion_recognition import EmotionRecognitionService
from app.services.index_tts import IndexTTSService
from app.services.meanvc2 import MeanVC2Service

try:
    from app.services.speaker_verification import SpeakerVerificationService
except ModuleNotFoundError:
    SpeakerVerificationService = None  # type: ignore[assignment]


logger = get_logger(__name__)
StatusCallback = Callable[[str], None]
SettingsProvider = Callable[[], Any]


@dataclass(slots=True)
class _ActivityStats:
    sample_rate: int = 16000
    total_samples: int = 0
    speech_start_sample: int | None = None
    silence_samples: int = 0
    noise_floor: float = 0.0025


class StreamingActivityDetector:
    """Lightweight streaming speech activity detector for live segmentation."""

    def __init__(
        self,
        *,
        chunk_size_ms: int = 200,
        preroll_ms: int = 120,
        silence_ms: int = 420,
        max_segment_ms: int = 12_000,
    ) -> None:
        self._chunk_size_ms = max(int(chunk_size_ms), 40)
        self._preroll_ms = max(int(preroll_ms), 0)
        self._silence_ms = max(int(silence_ms), self._chunk_size_ms)
        self._max_segment_ms = max(int(max_segment_ms), self._chunk_size_ms * 2)
        self._stats = _ActivityStats()

    def reset(self) -> None:
        self._stats = _ActivityStats(sample_rate=self._stats.sample_rate)

    @staticmethod
    def _to_ms(sample_index: int, sample_rate: int) -> int:
        return int(round(max(sample_index, 0) * 1000.0 / max(sample_rate, 1)))

    def process_chunk(
        self,
        samples: Any,
        sample_rate: int,
        *,
        is_final: bool = False,
    ) -> list[tuple[int, int]]:
        import numpy as np

        waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sample_rate <= 0:
            return []
        if self._stats.sample_rate != sample_rate:
            self._stats = _ActivityStats(sample_rate=sample_rate)

        if waveform.size == 0:
            if is_final and self._stats.speech_start_sample is not None:
                end_ms = self._to_ms(self._stats.total_samples, sample_rate)
                self._stats.speech_start_sample = None
                self._stats.silence_samples = 0
                return [(-1, end_ms)]
            return []

        stats = self._stats
        start_total = stats.total_samples
        stats.total_samples += int(waveform.shape[0])

        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float32), dtype=np.float32)))
        peak = float(np.max(np.abs(waveform), initial=0.0))
        if stats.speech_start_sample is None:
            stats.noise_floor = 0.92 * stats.noise_floor + 0.08 * min(rms, max(stats.noise_floor * 1.5, 0.01))

        speech_threshold = max(stats.noise_floor * 3.0, 0.008)
        silence_threshold = max(stats.noise_floor * 1.6, speech_threshold * 0.45, 0.0035)
        chunk_active = rms >= speech_threshold or peak >= speech_threshold * 2.2

        preroll_samples = int(sample_rate * self._preroll_ms / 1000)
        silence_samples_limit = int(sample_rate * self._silence_ms / 1000)
        max_segment_samples = int(sample_rate * self._max_segment_ms / 1000)
        events: list[tuple[int, int]] = []

        if stats.speech_start_sample is None:
            if chunk_active:
                stats.speech_start_sample = max(0, start_total - preroll_samples)
                stats.silence_samples = 0
                events.append((self._to_ms(stats.speech_start_sample, sample_rate), -1))
        else:
            if chunk_active or rms >= silence_threshold:
                stats.silence_samples = 0
            else:
                stats.silence_samples += int(waveform.shape[0])

            speech_samples = stats.total_samples - stats.speech_start_sample
            should_close = stats.silence_samples >= silence_samples_limit or speech_samples >= max_segment_samples
            if should_close:
                end_sample = stats.total_samples - max(stats.silence_samples - int(sample_rate * 0.08), 0)
                end_sample = max(end_sample, stats.speech_start_sample)
                events.append((-1, self._to_ms(end_sample, sample_rate)))
                stats.speech_start_sample = None
                stats.silence_samples = 0

        if is_final and stats.speech_start_sample is not None:
            events.append((-1, self._to_ms(stats.total_samples, sample_rate)))
            stats.speech_start_sample = None
            stats.silence_samples = 0
        return events


class LiveModelManager:
    def __init__(self, *, settings_provider: SettingsProvider, status_callback: StatusCallback | None = None) -> None:
        self._settings_provider = settings_provider
        self._status_callback = status_callback
        self._settings = settings_provider()
        self._lock = threading.RLock()

        self.asr_service: ASRService | None = None
        self.vad_service: StreamingActivityDetector | None = None
        self.index_tts_service: IndexTTSService | None = None
        self.meanvc2_service: MeanVC2Service | None = None
        self.speaker_service: SpeakerVerificationService | None = None
        self.emotion_service: EmotionRecognitionService | None = None

        self._asr_last_used_at = time.monotonic()
        self._tts_last_used_at = time.monotonic()
        self._tts_warm_reference_audio_path: str | None = None
        self._idle_unload_deadlines: dict[str, float | None] = {"asr": None, "tts": None}
        self._busy = False
        self._idle_scheduler_stop = threading.Event()
        self._idle_scheduler_wake = threading.Event()
        self._idle_scheduler_thread = threading.Thread(
            target=self._idle_unload_scheduler_loop,
            daemon=True,
            name="live-model-idle-unload",
        )
        self._idle_scheduler_thread.start()

    def refresh_settings(self) -> Any:
        self._settings = self._settings_provider()
        return self._settings

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        if not self._busy:
            self._idle_scheduler_wake.set()

    def _emit_status(self, message: str) -> None:
        if self._status_callback is not None:
            self._status_callback(message)

    def _get_idle_timeout(self, model_name: str) -> float:
        if model_name == "asr":
            return float(self._settings.live_asr_idle_unload_sec)
        return float(self._settings.live_tts_idle_unload_sec)

    def _set_idle_deadline(self, model_name: str, deadline: float | None) -> None:
        with self._lock:
            self._idle_unload_deadlines[model_name] = deadline
        self._idle_scheduler_wake.set()

    def clear_idle_deadline(self, model_name: str) -> None:
        self._set_idle_deadline(model_name, None)

    def defer_idle_unload(self, model_name: str, *, timeout: float | None = None) -> None:
        idle_timeout = self._get_idle_timeout(model_name) if timeout is None else timeout
        if idle_timeout <= 0:
            self.clear_idle_deadline(model_name)
            return
        self._set_idle_deadline(model_name, time.monotonic() + idle_timeout)

    def mark_model_used(self, model_name: str) -> None:
        now = time.monotonic()
        with self._lock:
            if model_name == "asr":
                self._asr_last_used_at = now
            else:
                self._tts_last_used_at = now
        self.defer_idle_unload(model_name)

    def _idle_unload_scheduler_loop(self) -> None:
        while not self._idle_scheduler_stop.is_set():
            wait_timeout: float | None = None
            due_models: list[str] = []
            now = time.monotonic()
            with self._lock:
                deadlines = {
                    model_name: deadline
                    for model_name, deadline in self._idle_unload_deadlines.items()
                    if deadline is not None
                }
            if deadlines:
                next_deadline = min(deadlines.values())
                if next_deadline > now:
                    wait_timeout = max(0.0, next_deadline - now)
                else:
                    due_models = [model_name for model_name, deadline in deadlines.items() if deadline <= now]

            if not due_models:
                triggered = self._idle_scheduler_wake.wait(wait_timeout)
                self._idle_scheduler_wake.clear()
                if self._idle_scheduler_stop.is_set():
                    return
                if triggered:
                    continue
                now = time.monotonic()
                with self._lock:
                    due_models = [
                        model_name
                        for model_name, deadline in self._idle_unload_deadlines.items()
                        if deadline is not None and deadline <= now
                    ]
                if not due_models:
                    continue

            for model_name in due_models:
                timeout = self._get_idle_timeout(model_name)
                with self._lock:
                    deadline = self._idle_unload_deadlines.get(model_name)
                    if deadline is None or deadline > time.monotonic():
                        continue
                    if self._busy:
                        self._idle_unload_deadlines[model_name] = time.monotonic() + max(timeout, 0.1)
                        self._idle_scheduler_wake.set()
                        continue
                    if model_name == "asr":
                        service_loaded = self.asr_service is not None
                        last_used_at = self._asr_last_used_at
                    else:
                        service_loaded = self.index_tts_service is not None or self.meanvc2_service is not None
                        last_used_at = self._tts_last_used_at
                    if not service_loaded:
                        self._idle_unload_deadlines[model_name] = None
                        continue
                    if timeout > 0 and (time.monotonic() - last_used_at) < timeout:
                        self._idle_unload_deadlines[model_name] = last_used_at + timeout
                        self._idle_scheduler_wake.set()
                        continue
                    self._idle_unload_deadlines[model_name] = None

                if model_name == "asr":
                    logger.info("ASR 模型空闲超时，已自动卸载以节省内存。")
                    self.unload_asr()
                else:
                    logger.info("TTS 模型空闲超时，已自动卸载以节省内存。")
                    self.unload_tts()

    def ensure_asr_loaded(self, *, emit_status: bool = False) -> ASRService:
        if self.asr_service is None:
            with self._lock:
                if self.asr_service is None:
                    self.refresh_settings()
                    if emit_status:
                        self._emit_status("正在初始化语音识别引擎...")
                    asr_service = ASRService()
                    asr_service.load()
                    self.asr_service = asr_service
        self.mark_model_used("asr")
        return self.asr_service

    def ensure_vad_loaded(self, *, emit_status: bool = False, chunk_size_ms: int = 200) -> StreamingActivityDetector:
        if self.vad_service is None:
            with self._lock:
                if self.vad_service is None:
                    if emit_status:
                        self._emit_status("正在初始化流式语音分段器...")
                    self.vad_service = StreamingActivityDetector(chunk_size_ms=chunk_size_ms)
        return self.vad_service

    def ensure_tts_loaded(
        self,
        *,
        reference_audio_path: str | None,
        emit_status: bool = False,
        warmup_reference: bool = False,
    ) -> IndexTTSService:
        if self.index_tts_service is None:
            with self._lock:
                if self.index_tts_service is None:
                    self.refresh_settings()
                    if emit_status:
                        self._emit_status("正在初始化 IndexTTS-2.5 多语言语音合成引擎...")
                    tts_service = IndexTTSService(self._settings.index_tts_model_dir)
                    tts_service.load()
                    self.index_tts_service = tts_service
                    self._tts_warm_reference_audio_path = None
        self.mark_model_used("tts")
        if warmup_reference:
            self.warmup_tts_reference_if_needed(
                reference_audio_path=reference_audio_path,
                emit_status=emit_status,
            )
        return self.index_tts_service

    def ensure_meanvc2_loaded(
        self,
        *,
        reference_audio_path: str | None,
        emit_status: bool = False,
    ) -> MeanVC2Service:
        normalized_reference = str(reference_audio_path or "").strip()
        if not normalized_reference:
            raise ValueError("MeanVC2 需要参考音频。")
        if self.meanvc2_service is None:
            with self._lock:
                if self.meanvc2_service is None:
                    self.refresh_settings()
                    if emit_status:
                        self._emit_status("正在初始化 MeanVC2 实时变声引擎...")
                    meanvc2_service = MeanVC2Service(device=self._settings.tts_device)
                    meanvc2_service.load(normalized_reference)
                    self.meanvc2_service = meanvc2_service
        elif self.meanvc2_service.reference_audio_path != normalized_reference:
            if emit_status:
                self._emit_status("正在切换 MeanVC2 参考音色...")
            self.meanvc2_service.warmup_speaker(normalized_reference)
        self.mark_model_used("tts")
        return self.meanvc2_service

    def ensure_emotion_loaded(self, *, emit_status: bool = False) -> EmotionRecognitionService:
        if self.emotion_service is None:
            with self._lock:
                if self.emotion_service is None:
                    self.refresh_settings()
                    if emit_status:
                        self._emit_status("正在初始化实验性情感识别...")
                    emotion_service = EmotionRecognitionService(self._settings.emotion_model_dir)
                    emotion_service.load(status_callback=self._status_callback if emit_status else None)
                    self.emotion_service = emotion_service
        return self.emotion_service

    def ensure_speaker_loaded(self, *, emit_status: bool = False) -> SpeakerVerificationService | None:
        if SpeakerVerificationService is None:
            logger.warning("未找到 app.services.speaker_verification，声纹识别功能已禁用。")
            if emit_status:
                self._emit_status("未找到说话人识别模块，已跳过声纹识别功能。")
            return None
        if self.speaker_service is None:
            with self._lock:
                if self.speaker_service is None:
                    self.refresh_settings()
                    if emit_status:
                        self._emit_status("正在初始化说话人识别模型...")
                    speaker_service = SpeakerVerificationService(self._settings.speaker_model_dir)
                    speaker_service.load(status_callback=self._status_callback if emit_status else None)
                    self.speaker_service = speaker_service
        return self.speaker_service

    def warmup_tts_reference_if_needed(
        self,
        *,
        reference_audio_path: str | None,
        emit_status: bool = False,
        force: bool = False,
    ) -> None:
        normalized_path = str(reference_audio_path or "").strip()
        with self._lock:
            if self.index_tts_service is None or not normalized_path:
                return
            if not force and self._tts_warm_reference_audio_path == normalized_path:
                return
            self.mark_model_used("tts")
            if emit_status:
                self._emit_status("正在准备参考音色特征（运行时预热）...")
            self.index_tts_service.warmup_speaker(normalized_path)
            self._tts_warm_reference_audio_path = normalized_path
            self.mark_model_used("tts")

    def switch_tts_model(self) -> None:
        self.refresh_settings()
        self.unload_tts()

    def unload_asr(self) -> None:
        with self._lock:
            if self.asr_service is not None:
                self.asr_service.unload()
                self.asr_service = None
            self.vad_service = None
            self._idle_unload_deadlines["asr"] = None

    def unload_vad(self) -> None:
        with self._lock:
            self.vad_service = None

    def unload_tts(self) -> None:
        with self._lock:
            if self.index_tts_service is not None:
                self.index_tts_service.unload()
                self.index_tts_service = None
            if self.meanvc2_service is not None:
                self.meanvc2_service.unload()
                self.meanvc2_service = None
            self._tts_warm_reference_audio_path = None
            self._idle_unload_deadlines["tts"] = None

    def unload_speaker(self) -> None:
        with self._lock:
            if self.speaker_service is not None:
                self.speaker_service.unload()
                self.speaker_service = None

    def unload_emotion(self) -> None:
        with self._lock:
            if self.emotion_service is not None:
                self.emotion_service.unload()
                self.emotion_service = None

    def shutdown(self) -> None:
        self._idle_scheduler_stop.set()
        self._idle_scheduler_wake.set()
        self._idle_scheduler_thread.join(timeout=1.0)
        self.unload_asr()
        self.unload_vad()
        self.unload_tts()
        self.unload_speaker()
        self.unload_emotion()
