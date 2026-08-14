from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from enum import StrEnum
from queue import Empty, Full, Queue
from uuid import uuid4

import numpy as np
import sounddevice as sd
import soundfile as sf
import soxr
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtMultimedia import QAudioDevice, QAudioFormat, QMediaDevices

from app import get_settings
from app.core.audio import resample_waveform
from app.core.logging import get_logger
from app.core.process_audio import PROCESS_AUDIO_SAMPLE_RATE, ProcessAudioStream
from app.core.wasapi_render import WasapiRenderClient
from app.models.asr import ASRRequest
from app.services.emotion_recognition import EmotionPrediction
from app.services.live_model_manager import LiveModelManager, StreamingActivityDetector
from app.services.translation import TranslationService, detect_text_language


logger = get_logger(__name__)
ASR_SAMPLE_RATE = 16_000
OUTPUT_LANGUAGE_LABELS = {"zh": "中文", "en": "英文", "ja": "日语"}
MIC_CHANNELS = 1
MIC_VAD_CHUNK_MS = 200
MIC_MIN_SEGMENT_SEC = 0.2
MIC_ASR_CONTEXT_MAX_CHARS = 80
MIC_VAD_HISTORY_SEC = 1.2
VOICEPRINT_WINDOW_SEC = 0.8
VOICEPRINT_HOP_SEC = 0.24
VOICEPRINT_PAD_SEC = 0.08
VOICEPRINT_MIN_KEEP_SEC = 0.35
VOICEPRINT_PEAK_RELAX = 0.12
VOICEPRINT_NEAR_PEAK_MARGIN = 0.06
EMOTION_MIN_SEGMENT_SEC = 0.6
EMOTION_MIN_CONFIDENCE = 0.45
_PLAYBACK_SENTINEL = object()
QUEUE_WAIT_LOG_THRESHOLD_MS = 20.0
QUEUE_RESIDENCE_LOG_THRESHOLD_MS = 80.0
AUDIO_CACHE_DIR = os.path.join(tempfile.gettempdir(), "pywork_audio_cache")
_ASR_SILENCE_TOKEN_RE = re.compile(r"(?i)(?:^|\s)(?:/?sil|<sil>|\[sil\])(?=\s|$)")


def _debug_report_program_listen_worker(hypothesis_id: str, message: str, data: dict[str, object]) -> None:
    env_path = os.path.join(".dbg", "program-listen-failure.env")
    url = "http://127.0.0.1:7777/event"
    session_id = "program-listen-failure"
    try:
        if os.path.exists(env_path):
            with open(env_path, encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if line.startswith("DEBUG_SERVER_URL="):
                        url = line.split("=", 1)[1].strip() or url
                    elif line.startswith("DEBUG_SESSION_ID="):
                        session_id = line.split("=", 1)[1].strip() or session_id
        payload = {
            "sessionId": session_id,
            "runId": os.environ.get("TRAE_DEBUG_RUN_ID", "pre-fix"),
            "hypothesisId": hypothesis_id,
            "location": "ui/workers.py",
            "msg": f"[DEBUG] {message}",
            "data": data,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=0.25):
            pass
    except Exception:
        pass


@dataclass(slots=True)
class _PlaybackDispatch:
    audio_np: np.ndarray
    sample_rate: int
    done: threading.Event
    error: BaseException | None = None


@dataclass(slots=True)
class _QueuedSegment:
    waveform: np.ndarray
    segment_index: int
    enqueued_at: float
    model_switch_revision: int
    source_kind: str
    speaker_label: str


@dataclass(slots=True)
class _QueuedPlaybackChunk:
    audio_np: np.ndarray
    sample_rate: int
    chunk_index: int
    enqueued_at: float


@dataclass(slots=True)
class _QueuedSynthesisTask:
    voice_output_model: str
    segment_label: str
    message_id: str | None
    switch_revision: int | None
    asr_latency_ms: float = 0.0
    trans_latency_ms: float = 0.0
    transcript: str | None = None
    waveform: np.ndarray | None = None
    emo_vector: list[float] | None = None
    emo_alpha: float | None = None


@dataclass(slots=True)
class _PlaybackSessionStart:
    label: str
    started_at: float


@dataclass(slots=True)
class _PlaybackSessionEnd:
    label: str
    started_at: float
    done: threading.Event
    error: BaseException | None = None


class _DevicePlaybackEndpoint:
    def __init__(self, *, device: str | None, label: str) -> None:
        self.device = device
        self.label = label
        self._queue: Queue[object] = Queue()
        self._resampler: soxr.ResampleStream | None = None
        self._resampler_in_rate: int | None = None
        self._resampler_out_rate: int | None = None
        self._resampler_channels: int | None = None
        self._renderer: WasapiRenderClient | None = None
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"playback-{label}",
        )
        self._thread.start()

    def _reset_resampler(self) -> None:
        self._resampler = None
        self._resampler_in_rate = None
        self._resampler_out_rate = None
        self._resampler_channels = None

    def submit(self, audio_np: np.ndarray, sample_rate: int) -> _PlaybackDispatch:
        dispatch = _PlaybackDispatch(
            audio_np=audio_np,
            sample_rate=sample_rate,
            done=threading.Event(),
        )
        self._queue.put(dispatch)
        return dispatch

    def close(self) -> None:
        self._queue.put(_PLAYBACK_SENTINEL)
        self._thread.join()

    def _run(self) -> None:
        try:
            while True:
                queued = self._queue.get()
                if queued is _PLAYBACK_SENTINEL:
                    return
                if not isinstance(queued, _PlaybackDispatch):
                    continue

                try:
                    if self._renderer is None:
                        self._renderer = WasapiRenderClient(
                            endpoint_id=self.device,
                            label=self.label,
                        )
                    prepared_audio, channels = LiveSpeechWorker._prepare_output_audio_for_channels(
                        queued.audio_np,
                        self._renderer.channels,
                    )
                    prepared_rate = self._renderer.sample_rate
                    if queued.sample_rate != prepared_rate:
                        if (
                            self._resampler is None
                            or self._resampler_in_rate != int(queued.sample_rate)
                            or self._resampler_out_rate != int(prepared_rate)
                            or self._resampler_channels != channels
                        ):
                            self._resampler = soxr.ResampleStream(
                                float(queued.sample_rate),
                                float(prepared_rate),
                                channels,
                                dtype="float32",
                                quality="HQ",
                            )
                            self._resampler_in_rate = int(queued.sample_rate)
                            self._resampler_out_rate = int(prepared_rate)
                            self._resampler_channels = channels
                        prepared_audio = self._resampler.resample_chunk(prepared_audio, last=False)
                    else:
                        self._reset_resampler()
                    self._renderer.write(prepared_audio)
                except BaseException as exc:
                    queued.error = exc
                    if self._renderer is not None:
                        try:
                            self._renderer.close()
                        except Exception:
                            pass
                        self._renderer = None
                    self._reset_resampler()
                finally:
                    queued.done.set()
        finally:
            if self._renderer is not None:
                try:
                    self._renderer.close()
                except Exception:
                    pass
                self._renderer = None
            self._reset_resampler()


class LiveWorkerState(StrEnum):
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"


def normalize_asr_transcript(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return normalized
    cleaned = _ASR_SILENCE_TOKEN_RE.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", cleaned).strip()
    if not normalized:
        logger.info("ASR 静音标记已清理: {}", text)
        return ""
    for prefix_len in range(min(6, len(normalized) // 2), 1, -1):
        prefix = normalized[:prefix_len]
        if normalized.startswith(prefix * 2):
            deduped = normalized[prefix_len:]
            logger.info("ASR 句首重复已修正: {} -> {}", normalized, deduped)
            return deduped
    return normalized


def merge_asr_segments(previous: str | None, current: str) -> str:
    merged = current.strip()
    if not previous or not merged:
        return merged
    previous_clean = previous.strip()
    max_overlap = min(len(previous_clean), len(merged))
    for overlap_len in range(max_overlap, 1, -1):
        suffix = previous_clean[-overlap_len:]
        prefix = merged[:overlap_len]
        if suffix == prefix:
            deduped = merged[overlap_len:].lstrip()
            if not deduped:
                logger.info("ASR 段间整句重复已跳过: {}", merged)
                return ""
            logger.info("ASR 段间重叠已修正: overlap={} | current={} -> {}", suffix, merged, deduped)
            return deduped
    return merged


def canonicalize_asr_text(text: str) -> str:
    if not text:
        return ""
    translation_table = str.maketrans("", "", " \t\r\n,.;:!?，。；：！？、\"'“”‘’()（）[]【】{}<>《》-—")
    return text.strip().lower().translate(translation_table)


def build_asr_context(transcript_parts: list[str]) -> str | None:
    if not transcript_parts:
        return None
    latest = transcript_parts[-1].strip()
    if not latest:
        return None
    if len(latest) <= MIC_ASR_CONTEXT_MAX_CHARS:
        return latest
    return latest[-MIC_ASR_CONTEXT_MAX_CHARS:].lstrip()


def resample_audio_output(audio_np: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if source_sample_rate == target_sample_rate or audio_np.size == 0:
        return audio_np.astype(np.float32, copy=False)
    if audio_np.ndim == 1:
        return resample_waveform(audio_np, source_sample_rate, target_sample_rate)

    channel_count = int(audio_np.shape[1])
    first_channel = resample_waveform(audio_np[:, 0], source_sample_rate, target_sample_rate)
    output = np.empty((first_channel.shape[0], channel_count), dtype=np.float32)
    output[:, 0] = first_channel
    for channel_index in range(1, channel_count):
        output[:, channel_index] = resample_waveform(
            audio_np[:, channel_index], source_sample_rate, target_sample_rate
        )
    return output


def speech_rate_to_duration_factor(speech_rate: float) -> float:
    if speech_rate <= 0:
        raise ValueError("语速倍率必须大于 0。")
    return max(0.5, min(2.0, 1.0 / float(speech_rate)))


def _audio_device_key(device: QAudioDevice) -> str:
    return bytes(device.id()).hex()


def _resolve_audio_device(device_key: str | None, *, output: bool) -> QAudioDevice:
    devices = list(QMediaDevices.audioOutputs() if output else QMediaDevices.audioInputs())
    default_device = QMediaDevices.defaultAudioOutput() if output else QMediaDevices.defaultAudioInput()

    if device_key:
        for device in devices:
            if _audio_device_key(device) == device_key:
                return device

    if not default_device.isNull():
        return default_device
    if devices:
        return devices[0]
    raise RuntimeError("未找到可用的音频设备。")


def _build_supported_audio_format(
    device: QAudioDevice,
    sample_rates: list[int],
    channel_counts: list[int],
    sample_formats: list[QAudioFormat.SampleFormat],
) -> QAudioFormat:
    supported_sample_formats = set(device.supportedSampleFormats())
    min_rate = int(device.minimumSampleRate() or 0)
    max_rate = int(device.maximumSampleRate() or 0)
    max_channels = int(device.maximumChannelCount() or 0)

    for channels in channel_counts:
        if channels <= 0:
            continue
        if max_channels > 0 and channels > max_channels:
            continue
        for sample_rate in sample_rates:
            if sample_rate <= 0:
                continue
            if min_rate > 0 and sample_rate < min_rate:
                continue
            if max_rate > 0 and sample_rate > max_rate:
                continue
            for sample_format in sample_formats:
                if supported_sample_formats and sample_format not in supported_sample_formats:
                    continue
                audio_format = QAudioFormat()
                audio_format.setSampleRate(sample_rate)
                audio_format.setChannelCount(channels)
                audio_format.setSampleFormat(sample_format)
                if device.isFormatSupported(audio_format):
                    return audio_format

    preferred_format = device.preferredFormat()
    if preferred_format.isValid() and device.isFormatSupported(preferred_format):
        return preferred_format

    raise RuntimeError(f"设备不支持可用音频格式: {device.description()}")


def _pcm_bytes_to_float32(data: object, audio_format: QAudioFormat) -> np.ndarray:
    if not data:
        return np.empty(0, dtype=np.float32)

    try:
        buffer = memoryview(data)
    except TypeError:
        buffer = memoryview(bytes(data))

    sample_format = audio_format.sampleFormat()
    if sample_format == QAudioFormat.SampleFormat.Int16:
        samples = np.frombuffer(buffer, dtype="<i2").astype(np.float32)
        np.multiply(samples, 1.0 / 32768.0, out=samples)
    elif sample_format == QAudioFormat.SampleFormat.Int32:
        samples = np.frombuffer(buffer, dtype="<i4").astype(np.float32)
        np.multiply(samples, 1.0 / 2147483648.0, out=samples)
    elif sample_format == QAudioFormat.SampleFormat.UInt8:
        samples = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        np.subtract(samples, 128.0, out=samples)
        np.multiply(samples, 1.0 / 128.0, out=samples)
    elif sample_format == QAudioFormat.SampleFormat.Float:
        samples = np.frombuffer(buffer, dtype="<f4").astype(np.float32, copy=False)
    else:
        raise RuntimeError(f"不支持的录音采样格式: {sample_format}")

    channel_count = max(1, int(audio_format.channelCount()))
    usable_size = samples.size - (samples.size % channel_count)
    if usable_size <= 0:
        return np.empty(0, dtype=np.float32)
    samples = samples[:usable_size]
    if channel_count > 1:
        samples = samples.reshape(-1, channel_count).mean(axis=1, dtype=np.float32)
    return samples.astype(np.float32, copy=False)


class LiveSpeechWorker(QObject):
    ready = Signal()
    finished = Signal(str)
    error = Signal(str)
    status = Signal(str)
    chat_message = Signal(object)
    perf_metrics = Signal(object)
    runtime_state = Signal(str)
    emotion_state = Signal(object)
    waveform = Signal(object)
    transcript = Signal(str)
    voiceprint_ready = Signal(object)
    voiceprint_error = Signal(str)

    def __init__(self, *, initial_reference_audio_path: str) -> None:
        super().__init__()
        self.reference_audio_path = initial_reference_audio_path
        self.language: str | None = "auto"
        self.instruction: str | None = None
        self.max_new_tokens = 2048
        self.max_record_sec = 20.0
        self.input_device: str | None = None
        self.input_device_label: str | None = None
        self.output_device: str | None = None
        self.output_device_label: str | None = None
        self.monitor_output_device: str | None = None
        self.monitor_output_device_label: str | None = None
        self.output_volume: float = 1.0
        self.monitor_output_volume: float = 1.0
        self.speech_rate: float = 1.0
        self._playback_controls_lock = threading.Lock()
        self._voice_output_model_lock = threading.Lock()
        self._segment_flush_requested = threading.Event()
        self._voice_output_model_revision = 0
        self._settings = get_settings()
        self.voice_output_model: str = str(self._settings.voice_output_model or "meanvc2").strip().lower() or "meanvc2"
        self._pending_voice_output_model: str | None = None
        self.index_emo_vector: list[float] | None = None
        self.index_emo_alpha: float = 1.0
        self.output_language: str = self._settings.output_voice_language
        self.translation_enabled: bool = self._settings.translation_enabled
        self.translation_target_language: str = self._settings.translation_target_language
        self.translation_node_id: str | None = self._settings.translation_node_id
        self.translation_service = TranslationService()
        self._loopback_thread: threading.Thread | None = None
        self._loopback_stop = threading.Event()
        self._loopback_lock = threading.Lock()
        self._loopback_active = False
        self._models_ready = False
        self._busy = False
        self._runtime_state = LiveWorkerState.INITIALIZING
        self._stop_event = threading.Event()
        self.auto_emotion_enabled = False
        self.auto_emotion_strength = 0.75
        self.user_voiceprint: np.ndarray | None = None
        self.voiceprint_threshold: float = 0.65
        self._last_status_text: str | None = None
        self._last_transcript_text: str | None = None
        self._last_emotion_payload: object = None
        self._last_tts_transcript_key: str | None = None
        self._last_asr_latency_ms: float = 0.0
        self._last_trans_latency_ms: float = 0.0
        self._last_tts_latency_ms: float = 0.0
        self._model_manager = LiveModelManager(
            settings_provider=get_settings,
            status_callback=self._emit_status,
        )
        self._playback_queue: Queue[object] = Queue(maxsize=6)
        self._playback_thread_lock = threading.Lock()
        self._synthesis_lock = threading.Lock()
        self._inline_text_priority = threading.Condition()
        self._pending_inline_text_requests = 0
        self._playback_thread: threading.Thread | None = None
        self._playback_worker_error: BaseException | None = None
        self._start_playback_worker_locked()

    def _reset_emit_cache(self) -> None:
        self._last_status_text = None
        self._last_transcript_text = None
        self._last_emotion_payload = None
        self._last_tts_transcript_key = None
        self._last_asr_latency_ms = 0.0
        self._last_trans_latency_ms = 0.0
        self._last_tts_latency_ms = 0.0

    def _emit_status(self, message: str) -> None:
        if self._last_status_text == message:
            return
        self._last_status_text = message
        self.status.emit(message)

    def _fail_inline_text_request(self, *, message_id: str | None, message: str) -> None:
        self._emit_status(message)
        if message_id:
            self.chat_message.emit(
                {
                    "message_id": message_id,
                    "input_mode": "text",
                    "translated_text": f"发送失败：{message}",
                    "state": "error",
                }
            )

    def _begin_inline_text_priority(self) -> None:
        with self._inline_text_priority:
            self._pending_inline_text_requests += 1
            self._inline_text_priority.notify_all()

    def _end_inline_text_priority(self) -> None:
        with self._inline_text_priority:
            self._pending_inline_text_requests = max(0, self._pending_inline_text_requests - 1)
            self._inline_text_priority.notify_all()

    def _wait_for_inline_text_priority(self) -> None:
        with self._inline_text_priority:
            while self._pending_inline_text_requests > 0 and not self._stop_event.is_set():
                self._inline_text_priority.wait(timeout=0.05)

    @Slot(object)
    def enqueue_text_request(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        if str(payload.get("input_mode") or "").strip() != "text":
            return
        message_id = str(payload.get("ui_message_id") or "").strip() or None
        if not self._models_ready:
            self._fail_inline_text_request(message_id=message_id, message="语音合成引擎尚未完成初始化，请稍后再试。")
            return
        if not self._busy or self._runtime_state != LiveWorkerState.RUNNING:
            self._fail_inline_text_request(message_id=message_id, message="当前没有正在运行的实时监听任务。")
            return

        def _inline_text_task() -> None:
            transcript = str(payload.get("input_text") or "").strip()
            if not transcript:
                self._fail_inline_text_request(message_id=message_id, message="文本模式下输入文本为空。")
                return
            try:
                self.reference_audio_path = str(payload.get("reference_audio_path") or self.reference_audio_path)
                with self._voice_output_model_lock:
                    self.voice_output_model = self._normalize_voice_output_model(
                        payload.get("voice_output_model") or self.voice_output_model or "meanvc2"
                    )
                self.translation_enabled = bool(payload.get("translation_enabled", self.translation_enabled))
                self.translation_target_language = str(
                    payload.get("translation_target_language") or self.translation_target_language or "zh"
                ).strip() or "zh"
                raw_translation_node_id = payload.get("translation_node_id")
                if isinstance(raw_translation_node_id, str):
                    self.translation_node_id = raw_translation_node_id.strip() or None
                elif raw_translation_node_id is None:
                    self.translation_node_id = None
                raw_instruction = payload.get("instruction")
                self.instruction = str(raw_instruction).strip() if isinstance(raw_instruction, str) and raw_instruction else None
                self.index_emo_vector = payload.get("index_emo_vector", self.index_emo_vector)
                self.index_emo_alpha = float(payload.get("index_emo_alpha", self.index_emo_alpha))
                self._apply_pending_voice_output_model_switch()
                if self.voice_output_model == "meanvc2":
                    self._fail_inline_text_request(message_id=message_id, message="MeanVC2 不支持文本模式，请切换到 Index-TTS。")
                    return
                translated_text: str | None = None
                if self.translation_enabled:
                    tts_text = self._prepare_tts_text(transcript)
                    translated_text = tts_text.strip() or None
                else:
                    tts_text = transcript
                if translated_text is not None:
                    self._emit_chat_record(
                        input_mode="text",
                        source_text=transcript,
                        message_id=message_id,
                        source_language=detect_text_language(transcript),
                        translated_text=translated_text,
                        state="pending",
                    )
                self._begin_inline_text_priority()
                self._emit_status("文字请求已加入当前实时链路，等待 Index-TTS 合成...")
                try:
                    with self._synthesis_lock:
                        self._synthesize(tts_text, "文本", progressive_transcript=True, message_id=message_id)
                finally:
                    self._end_inline_text_priority()
                self._emit_perf_metrics(
                    asr_latency=0.0,
                    trans_latency=self._last_trans_latency_ms,
                    tts_latency=self._last_tts_latency_ms,
                    total_latency=self._last_trans_latency_ms + self._last_tts_latency_ms,
                )
                self._emit_chat_record(
                    input_mode="text",
                    source_text=transcript,
                    message_id=message_id,
                    source_language=detect_text_language(transcript),
                    translated_text=translated_text,
                    state="ready",
                )
            except Exception as exc:
                logger.exception("插入文字合成失败")
                self._fail_inline_text_request(message_id=message_id, message=str(exc))

        threading.Thread(target=_inline_text_task, daemon=True, name="live-inline-text").start()

    def _emit_perf_metrics(self, *, asr_latency: float, trans_latency: float, tts_latency: float, total_latency: float) -> None:
        self.perf_metrics.emit(
            {
                "asr_latency": max(0.0, float(asr_latency)),
                "trans_latency": max(0.0, float(trans_latency)),
                "tts_latency": max(0.0, float(tts_latency)),
                "total_latency": max(0.0, float(total_latency)),
            }
        )

    def _emit_transcript(self, transcript: str) -> None:
        if self._last_transcript_text == transcript:
            return
        self._last_transcript_text = transcript
        self.transcript.emit(transcript)

    def _emit_emotion_state(self, payload: object) -> None:
        if self._last_emotion_payload == payload:
            return
        self._last_emotion_payload = payload
        self.emotion_state.emit(payload)

    def _set_runtime_state(self, state: LiveWorkerState) -> None:
        if self._runtime_state == state:
            return
        self._runtime_state = state
        self.runtime_state.emit(state.value)

    def _start_playback_worker_locked(self) -> None:
        self._playback_queue = Queue(maxsize=6)
        self._playback_worker_error = None
        self._playback_thread = threading.Thread(
            target=self._playback_worker,
            daemon=True,
            name="tts-playback",
        )
        self._playback_thread.start()

    def _ensure_playback_worker(self) -> None:
        with self._playback_thread_lock:
            if self._playback_thread is not None and self._playback_thread.is_alive():
                return
            if self._playback_worker_error is not None:
                logger.warning("共享播放线程已退出，正在重建: {}", self._playback_worker_error)
            else:
                logger.warning("共享播放线程未运行，正在重建。")
            self._start_playback_worker_locked()

    def _prepare_tts_text(self, text: str, *, source_language: str | None = None) -> str:
        stripped = text.strip()
        self._last_trans_latency_ms = 0.0
        if not stripped:
            return stripped
        if not self.translation_enabled:
            return stripped
        target_language = str(self.translation_target_language or self.output_language or "zh").strip() or "zh"
        source_guess = source_language or detect_text_language(stripped)
        if source_guess == target_language:
            target_label = OUTPUT_LANGUAGE_LABELS.get(target_language, target_language)
            self._emit_status(f"检测到输入语言与输出语言同为{target_label}，已跳过翻译。")
            return stripped
        translate_started_at = time.monotonic()
        result = self.translation_service.translate(
            stripped,
            target_language=target_language,
            source_language=source_guess,
            node_id=self.translation_node_id,
        )
        self._last_trans_latency_ms = (time.monotonic() - translate_started_at) * 1000.0
        preview = result.text if len(result.text) <= 120 else f"{result.text[:117]}..."
        target_label = OUTPUT_LANGUAGE_LABELS.get(target_language, target_language)
        self._emit_status(f"已通过翻译节点 {result.node_name} 转为{target_label}: {preview}")
        return result.text

    def _emit_chat_record(
        self,
        *,
        input_mode: str,
        source_text: str,
        message_id: str | None = None,
        source_language: str | None = None,
        translated_text: str | None = None,
        speaker_label: str | None = None,
        state: str = "ready",
    ) -> None:
        source_display = str(source_text or "").strip()
        if not source_display:
            return
        translated_display = str(translated_text or "").strip()
        if translated_display and canonicalize_asr_text(translated_display) == canonicalize_asr_text(source_display):
            translated_display = ""
        self.chat_message.emit(
            {
                "message_id": str(message_id or "").strip() or None,
                "created_at": time.strftime("%H:%M:%S"),
                "input_mode": input_mode,
                "source_text": source_display,
                "speaker_label": str(speaker_label or "").strip(),
                "translated_text": translated_display or None,
                "source_language": str(source_language or "").strip() or None,
                "target_language": self.translation_target_language if translated_display else None,
                "state": state,
            }
        )

    @staticmethod
    def _fail_pending_playback_sessions(playback_queue: Queue[object], error: BaseException) -> None:
        pending_session_ends: list[_PlaybackSessionEnd] = []
        while True:
            try:
                queued = playback_queue.get_nowait()
            except Empty:
                break
            if isinstance(queued, _PlaybackSessionEnd):
                pending_session_ends.append(queued)
        for session_end in pending_session_ends:
            session_end.error = error
            session_end.done.set()

    def _put_playback_item(self, playback_queue: Queue[object], item: object, segment_label: str) -> None:
        while True:
            try:
                playback_queue.put(item, timeout=2.0)
                return
            except Full:
                thread_alive = self._playback_thread is not None and self._playback_thread.is_alive()
                logger.warning(
                    "{} 播放队列写入等待中: playback_thread_alive={} queue_backlog={}",
                    segment_label,
                    thread_alive,
                    self._safe_queue_size(playback_queue),
                )
                if not thread_alive:
                    raise self._playback_worker_error or RuntimeError("共享播放线程已退出。")

    @staticmethod
    def _safe_queue_size(queue: Queue[object]) -> int:
        try:
            return int(queue.qsize())
        except Exception:
            return -1

    @staticmethod
    def _audio_duration_ms(audio_np: np.ndarray, sample_rate: int) -> float:
        if sample_rate <= 0 or audio_np.size == 0:
            return 0.0
        sample_count = int(audio_np.shape[0]) if audio_np.ndim >= 1 else 0
        return sample_count * 1000.0 / float(sample_rate)

    def _defer_idle_unload(self, model_name: str, *, timeout: float | None = None) -> None:
        self._model_manager.defer_idle_unload(model_name, timeout=timeout)

    def _voice_output_model_label(self) -> str:
        return "MeanVC2" if self.voice_output_model == "meanvc2" else "Index-TTS"

    @staticmethod
    def _normalize_voice_output_model(model_name: str | None) -> str:
        normalized = str(model_name or "").strip().lower()
        if normalized not in {"meanvc2", "indextts"}:
            return "meanvc2"
        return normalized

    @staticmethod
    def _voice_output_model_label_for(model_name: str) -> str:
        return "MeanVC2" if model_name == "meanvc2" else "Index-TTS"

    def _current_voice_output_model_revision(self) -> int:
        with self._voice_output_model_lock:
            return self._voice_output_model_revision

    def _queue_voice_output_model_switch(self, model_name: str) -> bool:
        normalized = self._normalize_voice_output_model(model_name)
        with self._voice_output_model_lock:
            if normalized == self.voice_output_model:
                return False
            self.voice_output_model = normalized
            self._pending_voice_output_model = None
            self._voice_output_model_revision += 1
            self._segment_flush_requested.set()
        dropped_chunks = 0
        if self._playback_queue is not None:
            dropped_chunks = self._drop_pending_playback_chunks(self._playback_queue)
        if dropped_chunks > 0:
            logger.info("语音输出模型切换时已主动丢弃 {} 个待播放音频块。", dropped_chunks)
        return True

    def _apply_pending_voice_output_model_switch(self) -> bool:
        with self._voice_output_model_lock:
            pending_model = self._pending_voice_output_model
            if not pending_model or pending_model == self.voice_output_model:
                self._pending_voice_output_model = None
                return False
            self.voice_output_model = pending_model
            self._pending_voice_output_model = None
        self._model_manager.switch_tts_model()
        self._emit_status(f"语音输出模型已切换为 {self._voice_output_model_label()}，当前语音段已终止。")
        return True

    def _is_segment_switch_interrupted(self, revision: int) -> bool:
        return revision != self._current_voice_output_model_revision()

    @staticmethod
    def _map_detected_emotion_to_vector(label_en: str) -> list[float] | None:
        base_vectors: dict[str, list[float] | None] = {
            "happy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "angry": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "sad": [0.0, 0.0, 0.85, 0.0, 0.0, 0.35, 0.0, 0.0],
            "fearful": [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0],
            "disgusted": [0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
            "surprised": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 0.0],
            "neutral": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "other": None,
            "unknown": None,
        }
        vector = base_vectors.get(label_en)
        return list(vector) if vector is not None else None

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: list[float] | np.ndarray) -> float:
        emb_a = np.asarray(vec_a, dtype=np.float32)
        emb_b = np.asarray(vec_b, dtype=np.float32)
        norm_a = float(np.linalg.norm(emb_a))
        norm_b = float(np.linalg.norm(emb_b))
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))

    def _filter_waveform_by_voiceprint(self, waveform: np.ndarray) -> tuple[np.ndarray | None, str]:
        speaker_service = self._model_manager.speaker_service
        if self.user_voiceprint is None or self.user_voiceprint.size == 0 or speaker_service is None:
            return waveform, ""

        sample_count = int(waveform.shape[0])
        if sample_count <= 0:
            return None, "声纹过滤后无有效语音。"

        window_size = max(1, int(ASR_SAMPLE_RATE * VOICEPRINT_WINDOW_SEC))
        hop_size = max(1, int(ASR_SAMPLE_RATE * VOICEPRINT_HOP_SEC))
        pad_size = max(0, int(ASR_SAMPLE_RATE * VOICEPRINT_PAD_SEC))
        min_keep_samples = max(1, int(ASR_SAMPLE_RATE * VOICEPRINT_MIN_KEEP_SEC))

        candidate_ranges: list[tuple[int, int]] = []

        def add_candidate_range(start: int, end: int) -> None:
            chunk = waveform[start:end]
            if chunk.size <= 0:
                return
            candidate_ranges.append((start, end))

        if sample_count <= window_size:
            add_candidate_range(0, sample_count)
        else:
            starts = list(range(0, sample_count - window_size + 1, hop_size))
            last_start = sample_count - window_size
            if not starts or starts[-1] != last_start:
                starts.append(last_start)
            for start in starts:
                add_candidate_range(start, start + window_size)

        if not candidate_ranges:
            return None, "声纹过滤后无可用语音窗口。"

        candidate_chunks = [waveform[start:end] for start, end in candidate_ranges]
        embeddings = speaker_service.extract_embeddings_batch(candidate_chunks, ASR_SAMPLE_RATE)
        windows: list[tuple[int, int, float]] = []
        for (start, end), embedding in zip(candidate_ranges, embeddings, strict=False):
            similarity = self._cosine_similarity(self.user_voiceprint, embedding)
            windows.append((start, end, similarity))

        peak_similarity = max(score for _, _, score in windows)
        min_peak_required = max(0.42, self.voiceprint_threshold - VOICEPRINT_PEAK_RELAX)
        if peak_similarity < min_peak_required:
            return None, f"声纹峰值不足 ({peak_similarity:.2f})，已忽略当前片段。"

        keep_threshold = min(
            self.voiceprint_threshold,
            max(min_peak_required, peak_similarity - VOICEPRINT_NEAR_PEAK_MARGIN),
        )
        kept_windows = [(start, end, score) for start, end, score in windows if score >= keep_threshold]
        if not kept_windows:
            return None, "声纹过滤后没有找到目标语音区间。"

        peak_window_index = max(range(len(windows)), key=lambda idx: windows[idx][2])
        peak_start, peak_end, _ = windows[peak_window_index]

        clusters: list[list[tuple[int, int, float]]] = []
        for start, end, score in kept_windows:
            if not clusters or start > clusters[-1][-1][1] + hop_size:
                clusters.append([(start, end, score)])
            else:
                clusters[-1].append((start, end, score))

        selected_cluster: list[tuple[int, int, float]] | None = None
        for cluster in clusters:
            cluster_start = cluster[0][0]
            cluster_end = cluster[-1][1]
            if cluster_start <= peak_end and cluster_end >= peak_start:
                selected_cluster = cluster
                break
        if selected_cluster is None:
            selected_cluster = max(clusters, key=lambda cluster: max(score for _, _, score in cluster))

        crop_start = max(0, selected_cluster[0][0] - pad_size)
        crop_end = min(sample_count, selected_cluster[-1][1] + pad_size)
        if crop_end <= crop_start:
            return None, "声纹过滤后目标语音区间无效。"

        filtered_waveform = waveform[crop_start:crop_end].astype(np.float32, copy=False)
        kept_samples = int(filtered_waveform.shape[0])
        if kept_samples < min_keep_samples:
            return None, "声纹保留语音过短，已忽略当前片段。"

        kept_sec = kept_samples / ASR_SAMPLE_RATE
        original_sec = sample_count / ASR_SAMPLE_RATE
        return filtered_waveform, (
            f"声纹过滤保留连续语音 {kept_sec:.2f}s / {original_sec:.2f}s，峰值相似度 {peak_similarity:.2f}。"
        )

    def _mark_model_used(self, model_name: str) -> None:
        self._model_manager.mark_model_used(model_name)

    def _ensure_asr_loaded(self, *, emit_status: bool = False):
        return self._model_manager.ensure_asr_loaded(emit_status=emit_status)

    def _ensure_vad_loaded(self, *, emit_status: bool = False):
        return self._model_manager.ensure_vad_loaded(
            emit_status=emit_status,
            chunk_size_ms=MIC_VAD_CHUNK_MS,
        )

    def _ensure_tts_loaded(self, *, emit_status: bool = False, warmup_reference: bool = False):
        return self._model_manager.ensure_tts_loaded(
            reference_audio_path=self.reference_audio_path,
            emit_status=emit_status,
            warmup_reference=warmup_reference,
        )

    def _ensure_meanvc2_loaded(self, *, emit_status: bool = False):
        return self._model_manager.ensure_meanvc2_loaded(
            reference_audio_path=self.reference_audio_path,
            emit_status=emit_status,
        )

    def _ensure_emotion_loaded(self, *, emit_status: bool = False):
        return self._model_manager.ensure_emotion_loaded(emit_status=emit_status)

    def _ensure_speaker_loaded(self, *, emit_status: bool = False):
        return self._model_manager.ensure_speaker_loaded(emit_status=emit_status)

    def _detect_segment_emotion(self, waveform: np.ndarray) -> tuple[list[float] | None, float, EmotionPrediction | None]:
        if not self.auto_emotion_enabled:
            return self.index_emo_vector, self.index_emo_alpha, None
        duration_sec = float(waveform.shape[0]) / float(ASR_SAMPLE_RATE)
        if duration_sec < EMOTION_MIN_SEGMENT_SEC:
            self._emit_emotion_state(
                {"state": "idle", "summary": "当前片段过短，沿用手动情感。"}
            )
            return self.index_emo_vector, self.index_emo_alpha, None

        emotion_service = self._ensure_emotion_loaded(emit_status=True)
        if emotion_service is None:
            return self.index_emo_vector, self.index_emo_alpha, None

        prediction = emotion_service.predict_waveform(waveform, ASR_SAMPLE_RATE)
        summary = f"{prediction.label_display}/{prediction.label_en} {prediction.confidence:.2f}"
        detected_vector = self._map_detected_emotion_to_vector(prediction.label_en)
        if detected_vector is None or prediction.confidence < EMOTION_MIN_CONFIDENCE:
            self._emit_emotion_state(
                {
                    "state": "ready",
                    "summary": f"当前识别：{summary}，已回退手动情感。",
                    "label": prediction.label_en,
                    "display_label": prediction.label_display,
                    "confidence": prediction.confidence,
                    "applied_vector": self.index_emo_vector,
                    "applied_alpha": self.index_emo_alpha,
                    "vector_source": "manual_fallback",
                }
            )
            self._emit_status(
                f"实验性情感识别结果 {summary}，置信度不足，沿用手动情感。"
            )
            return self.index_emo_vector, self.index_emo_alpha, prediction

        self._emit_emotion_state(
            {
                "state": "ready",
                "summary": f"当前识别：{summary}",
                "label": prediction.label_en,
                "display_label": prediction.label_display,
                "confidence": prediction.confidence,
                "applied_vector": detected_vector,
                "applied_alpha": self.auto_emotion_strength,
                "vector_source": "detected",
            }
        )
        self._emit_status(
            f"识别到当前情绪 {summary}，已应用实验性情感控制。"
        )
        return detected_vector, self.auto_emotion_strength, prediction

    def _warmup_tts_reference_if_needed(self, *, emit_status: bool = False, force: bool = False) -> None:
        self._model_manager.warmup_tts_reference_if_needed(
            reference_audio_path=self.reference_audio_path,
            emit_status=emit_status,
            force=force,
        )

    def _preload_reference_models(self, *, input_mode: str, voice_output_model: str) -> None:
        normalized_input_mode = str(input_mode or "microphone").strip().lower() or "microphone"
        normalized_voice_output_model = self._normalize_voice_output_model(voice_output_model)
        has_reference_audio = bool(str(self.reference_audio_path or "").strip())

        if has_reference_audio:
            if normalized_voice_output_model == "meanvc2":
                self._ensure_meanvc2_loaded(emit_status=True)
                self._emit_status("MeanVC2 参考音色已加载完成。")
            else:
                self._ensure_tts_loaded(emit_status=True, warmup_reference=True)
                self._emit_status("Index-TTS 参考音色与首句链路已预热完成。")

        if normalized_input_mode != "text":
            self._ensure_vad_loaded(emit_status=True)
            self._ensure_asr_loaded(emit_status=True)

    @Slot(str)
    def update_reference_audio_path(self, path: str) -> None:
        normalized = str(path or "").strip()
        if normalized == self.reference_audio_path:
            return
        self.reference_audio_path = normalized
        if not normalized:
            return
        try:
            if self.voice_output_model == "meanvc2":
                self._ensure_meanvc2_loaded(emit_status=True)
                self._emit_status("MeanVC2 参考音色已加载完成，可开始变声。")
            elif self._model_manager.index_tts_service is not None:
                self._warmup_tts_reference_if_needed(emit_status=True, force=True)
                self._emit_status("Index-TTS 参考音色已更新并预热完成。")
        except Exception as exc:
            logger.warning("更新参考音频失败: {}", exc)

    @Slot(float)
    def update_output_volume(self, value: float) -> None:
        with self._playback_controls_lock:
            self.output_volume = max(0.0, float(value))

    @Slot(float)
    def update_monitor_output_volume(self, value: float) -> None:
        with self._playback_controls_lock:
            self.monitor_output_volume = max(0.0, float(value))

    @Slot(float)
    def update_speech_rate(self, value: float) -> None:
        with self._playback_controls_lock:
            self.speech_rate = max(0.5, min(1.5, float(value)))

    @Slot(object)
    def update_translation_settings(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        self.translation_enabled = bool(payload.get("translation_enabled", self.translation_enabled))
        raw_target_language = payload.get("translation_target_language")
        if isinstance(raw_target_language, str):
            self.translation_target_language = raw_target_language.strip() or self.translation_target_language
        raw_node_id = payload.get("translation_node_id")
        if raw_node_id is None:
            self.translation_node_id = None
        elif isinstance(raw_node_id, str):
            self.translation_node_id = raw_node_id.strip() or None

    def _snapshot_playback_controls(self) -> tuple[float, float, float]:
        with self._playback_controls_lock:
            return self.output_volume, self.monitor_output_volume, self.speech_rate

    @Slot(str, bool)
    def replay_cached_audio(self, message_id: str, to_virtual_device: bool) -> None:
        normalized_message_id = str(message_id or "").strip()
        if not normalized_message_id:
            self.error.emit("重播失败：缺少消息 ID。")
            return

        def _replay_task() -> None:
            cache_path = self._resolve_cached_audio_path(normalized_message_id)
            if cache_path is None:
                self.error.emit(f"未找到该消息的缓存音频: {normalized_message_id}")
                return

            label = "重放到虚拟设备" if to_virtual_device else "重播"
            target_device = self.output_device if to_virtual_device else (self.monitor_output_device or self.output_device)
            output_volume, monitor_output_volume, _ = self._snapshot_playback_controls()
            target_volume = output_volume if to_virtual_device else (
                monitor_output_volume if self.monitor_output_device else output_volume
            )

            try:
                audio_np, sample_rate = sf.read(cache_path, dtype="float32", always_2d=False)
                if not isinstance(audio_np, np.ndarray) or audio_np.size == 0:
                    raise RuntimeError("缓存音频为空。")
                if abs(target_volume - 1.0) >= 1e-6:
                    audio_np = np.ascontiguousarray(audio_np.astype(np.float32, copy=False) * target_volume)
                    np.clip(audio_np, -1.0, 1.0, out=audio_np)
                endpoint = _DevicePlaybackEndpoint(
                    device=target_device,
                    label="replay-main" if to_virtual_device else "replay-monitor",
                )
                try:
                    dispatch = endpoint.submit(audio_np, int(sample_rate))
                    replay_error = self._await_playback_dispatch(dispatch)
                    if replay_error is not None:
                        raise replay_error
                finally:
                    endpoint.close()
                self._emit_status(f"已完成{label}。")
            except Exception as exc:
                logger.exception("重播缓存音频失败: message_id={}", normalized_message_id)
                self.error.emit(f"{label}失败: {exc}")

        threading.Thread(target=_replay_task, daemon=True, name="replay-cached").start()

    @Slot()
    def switch_tts_model(self) -> None:
        self._settings = get_settings()
        normalized_model = self._normalize_voice_output_model(self._settings.voice_output_model)
        with self._voice_output_model_lock:
            self.voice_output_model = normalized_model
            self._pending_voice_output_model = None
        self._segment_flush_requested.clear()
        self.output_language = self._settings.output_voice_language
        self.translation_enabled = self._settings.translation_enabled
        self.translation_target_language = self._settings.translation_target_language
        self.translation_node_id = self._settings.translation_node_id
        self._model_manager.switch_tts_model()
        self._emit_status("TTS 配置已更新，当前已就绪；下次合成时会按新设置重新加载模型。")

    @Slot(str)
    def update_voice_output_model(self, model_name: str) -> None:
        normalized_model = self._normalize_voice_output_model(model_name)
        if not self._queue_voice_output_model_switch(normalized_model):
            return
        self._emit_status(
            f"语音输出模型已切换为 {self._voice_output_model_label_for(normalized_model)}，当前语音段会立即终止。"
        )

    def _resolve_input_stream_settings(self) -> tuple[int | None, int]:
        device_index: int | None = None
        device_name = (self.input_device_label or "").strip()

        if device_name:
            normalized_target = device_name.casefold()
            for index, device in enumerate(sd.query_devices()):
                if int(device.get("max_input_channels") or 0) <= 0:
                    continue
                current_name = str(device.get("name") or "").strip()
                normalized_name = current_name.casefold()
                if normalized_target == normalized_name or normalized_target in normalized_name:
                    device_index = index
                    break

        default_rate = 0
        if device_index is not None:
            device_info = sd.query_devices(device_index, "input")
            default_rate = int(round(float(device_info.get("default_samplerate") or 0.0)))
        else:
            try:
                default_input_index = sd.default.device[0]
            except Exception:
                default_input_index = None
            if isinstance(default_input_index, int) and default_input_index >= 0:
                device_index = default_input_index
                device_info = sd.query_devices(device_index, "input")
                default_rate = int(round(float(device_info.get("default_samplerate") or 0.0)))

        candidate_rates: list[int] = []
        for rate in (default_rate, 48_000, 44_100, 32_000, ASR_SAMPLE_RATE):
            if rate > 0 and rate not in candidate_rates:
                candidate_rates.append(rate)

        last_error: Exception | None = None
        for candidate in candidate_rates:
            try:
                sd.check_input_settings(device=device_index, channels=MIC_CHANNELS, samplerate=candidate, dtype="float32")
                return device_index, candidate
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise RuntimeError(
                f"当前录音设备不支持可用采样率: {self.input_device_label or '系统默认输入'} | {last_error}"
            ) from last_error
        raise RuntimeError("未找到可用的录音设备。")

    @Slot(object)
    def preload_models(self, payload: object = None) -> None:
        self._set_runtime_state(LiveWorkerState.INITIALIZING)
        self._settings = get_settings()
        reference_audio_path: str | None = None
        input_mode = "microphone"
        voice_output_model = str(self._settings.voice_output_model or "indextts").strip().lower() or "indextts"
        if isinstance(payload, dict):
            raw_reference_audio_path = payload.get("reference_audio_path")
            if isinstance(raw_reference_audio_path, str):
                reference_audio_path = raw_reference_audio_path.strip() or None
            raw_input_mode = payload.get("input_mode")
            if isinstance(raw_input_mode, str) and raw_input_mode.strip():
                input_mode = raw_input_mode.strip()
            raw_voice_output_model = payload.get("voice_output_model")
            if isinstance(raw_voice_output_model, str) and raw_voice_output_model.strip():
                voice_output_model = raw_voice_output_model.strip().lower()
        elif isinstance(payload, str):
            reference_audio_path = payload.strip() or None

        if reference_audio_path:
            self.reference_audio_path = reference_audio_path
        self.voice_output_model = voice_output_model

        try:
            self._reset_emit_cache()
            did_preload_models = False
            if self.reference_audio_path:
                self._preload_reference_models(
                    input_mode=input_mode,
                    voice_output_model=voice_output_model,
                )
                did_preload_models = True
            self._models_ready = True
            if input_mode == "text":
                self._emit_status(
                    "文本模式引擎已就绪，可以开始合成。"
                    if self.reference_audio_path
                    else "文本模式已就绪，请先选择参考音频。"
                )
            elif did_preload_models and input_mode == "program":
                self._emit_status("程序监听引擎已就绪，可以开始转写。")
            elif did_preload_models:
                if voice_output_model == "meanvc2":
                    self._emit_status("语音引擎已就绪，可以开始变声。")
                else:
                    self._emit_status("语音引擎已就绪，可以开始合成。")
            self._set_runtime_state(LiveWorkerState.READY)
            self.ready.emit()
        except Exception as exc:
            logger.exception("预加载模型失败")
            self.error.emit(str(exc))

    @Slot(object)
    def extract_voiceprint(self, payload: object) -> None:
        try:
            if not isinstance(payload, dict):
                raise RuntimeError("声纹提取参数无效。")
            raw_audio = payload.get("audio")
            raw_sample_rate = payload.get("sample_rate")
            sample_rate = int(raw_sample_rate) if raw_sample_rate is not None else 0
            waveform = np.asarray(raw_audio, dtype=np.float32).reshape(-1)
            if sample_rate <= 0:
                raise RuntimeError("声纹提取采样率无效。")
            if waveform.size == 0:
                raise RuntimeError("录制的声纹音频为空。")

            self._emit_status("正在提取说话人 embedding...")
            speaker_service = self._ensure_speaker_loaded(emit_status=True)
            if speaker_service is None:
                raise RuntimeError("说话人识别模型未就绪，无法提取声纹。")
            voiceprint = speaker_service.extract_embedding(waveform, sample_rate)
            self._emit_status("声纹提取完成。")
            self.voiceprint_ready.emit(voiceprint)
        except Exception as exc:
            logger.exception("提取声纹失败")
            self.voiceprint_error.emit(str(exc))

    def _record_microphone_segments(self, segment_queue: Queue[object], *, allow_empty: bool = False) -> None:
        input_device, capture_sample_rate = self._resolve_input_stream_settings()
        blocksize = max(1, int(capture_sample_rate * MIC_VAD_CHUNK_MS / 1000))
        emitted_segments = 0
        segment_queue_peak = 0
        segment_queue_put_wait_ms_max = 0.0
        finished = False
        capture_buffer = np.empty(0, dtype=np.float32)
        buffer_offset_samples = 0
        stream_total_samples = 0
        last_emitted_range: tuple[int, int] | None = None
        active_start_ms: int | None = None
        history_samples = max(1, int(capture_sample_rate * MIC_VAD_HISTORY_SEC))

        vad_service = StreamingActivityDetector(chunk_size_ms=MIC_VAD_CHUNK_MS)
        vad_service.reset()

        self._emit_status(
            f"开始录音，请说话... 当前采集采样率 {capture_sample_rate} Hz，"
        )
        start_time = time.monotonic()

        def trim_capture_buffer() -> None:
            nonlocal capture_buffer, buffer_offset_samples
            if capture_buffer.size == 0:
                return
            if active_start_ms is not None:
                keep_from_abs = max(0, int(round(active_start_ms * capture_sample_rate / 1000.0)) - history_samples)
            else:
                keep_from_abs = max(0, stream_total_samples - history_samples)
            if keep_from_abs <= buffer_offset_samples:
                return
            drop_samples = keep_from_abs - buffer_offset_samples
            if drop_samples <= 0:
                return
            if drop_samples >= int(capture_buffer.shape[0]):
                capture_buffer = np.empty(0, dtype=np.float32)
                buffer_offset_samples = keep_from_abs
                return
            capture_buffer = np.ascontiguousarray(capture_buffer[drop_samples:], dtype=np.float32)
            buffer_offset_samples = keep_from_abs

        def emit_segment(start_ms: int, end_ms: int) -> None:
            nonlocal emitted_segments, segment_queue_peak, segment_queue_put_wait_ms_max, last_emitted_range
            if end_ms <= start_ms:
                return
            start_abs = max(0, int(round(start_ms * capture_sample_rate / 1000.0)))
            end_abs = max(start_abs, int(round(end_ms * capture_sample_rate / 1000.0)))
            original_start_abs = start_abs
            if last_emitted_range is not None:
                last_end_abs = last_emitted_range[1]
                if end_abs <= last_end_abs:
                    logger.debug(
                        "流式 VAD 片段已被上一段完全覆盖，跳过: start_ms={} end_ms={} last_end_ms={:.1f}",
                        start_ms,
                        end_ms,
                        last_end_abs * 1000.0 / capture_sample_rate,
                    )
                    return
                if start_abs < last_end_abs:
                    start_abs = last_end_abs

            if start_abs < buffer_offset_samples:
                logger.debug(
                    "流式 VAD 片段起点早于当前缓冲区，已裁剪: start_ms={} end_ms={} offset_ms={:.1f}",
                    start_ms,
                    end_ms,
                    buffer_offset_samples * 1000.0 / capture_sample_rate,
                )
                start_abs = buffer_offset_samples

            segment_range = (start_abs, end_abs)
            if last_emitted_range == segment_range:
                return
            start_rel = start_abs - buffer_offset_samples
            end_rel = end_abs - buffer_offset_samples
            if start_rel < 0 or end_rel > int(capture_buffer.shape[0]) or end_rel <= start_rel:
                logger.warning(
                    "流式 VAD 片段越界，已跳过: start_ms={} end_ms={} offset_samples={} buffer_samples={}",
                    start_ms,
                    end_ms,
                    buffer_offset_samples,
                    int(capture_buffer.shape[0]),
                )
                return
            if start_abs != original_start_abs:
                logger.info(
                    "流式 VAD 片段已按已消费音频裁剪: original_start_ms={:.1f} adjusted_start_ms={:.1f} end_ms={:.1f}",
                    original_start_abs * 1000.0 / capture_sample_rate,
                    start_abs * 1000.0 / capture_sample_rate,
                    end_abs * 1000.0 / capture_sample_rate,
                )
            waveform = capture_buffer[start_rel:end_rel].astype(np.float32, copy=False)
            duration = waveform.shape[0] / capture_sample_rate
            if duration < MIC_MIN_SEGMENT_SEC:
                return
            if capture_sample_rate != ASR_SAMPLE_RATE:
                waveform = resample_waveform(waveform, capture_sample_rate, ASR_SAMPLE_RATE)
            emitted_segments += 1
            last_emitted_range = segment_range
            queued_segment = _QueuedSegment(
                waveform=waveform,
                segment_index=emitted_segments,
                enqueued_at=time.monotonic(),
                model_switch_revision=self._current_voice_output_model_revision(),
                source_kind="microphone",
                speaker_label="麦克风",
            )
            put_started_at = time.monotonic()
            segment_queue.put(queued_segment)
            put_wait_ms = (time.monotonic() - put_started_at) * 1000.0
            segment_queue_put_wait_ms_max = max(segment_queue_put_wait_ms_max, put_wait_ms)
            queue_size = self._safe_queue_size(segment_queue)
            if queue_size > segment_queue_peak:
                segment_queue_peak = queue_size
                logger.info("录音分段队列积压新高: backlog={}/4", queue_size)
            if put_wait_ms >= QUEUE_WAIT_LOG_THRESHOLD_MS:
                logger.warning(
                    "录音分段队列入队等待偏高: segment={} wait_ms={:.1f} backlog={}",
                    emitted_segments,
                    put_wait_ms,
                    queue_size,
                )
            self._emit_status(
                f"流式分段器已检测到第 {emitted_segments} 段语音 "
                f"({start_ms}ms - {end_ms}ms)。"
            )

        def handle_vad_events(events: list[tuple[int, int]]) -> None:
            nonlocal active_start_ms
            for start_ms, end_ms in events:
                if start_ms >= 0:
                    active_start_ms = start_ms
                resolved_start_ms = start_ms if start_ms >= 0 else active_start_ms
                if end_ms >= 0:
                    if resolved_start_ms is None:
                        logger.warning("流式 VAD 返回了结束点但缺少开始点: end_ms={}", end_ms)
                    else:
                        emit_segment(resolved_start_ms, end_ms)
                    active_start_ms = None

        def flush_active_segment_if_requested() -> None:
            nonlocal active_start_ms
            if not self._segment_flush_requested.is_set():
                return
            self._segment_flush_requested.clear()
            if active_start_ms is None:
                return
            current_end_ms = int(round(stream_total_samples * 1000.0 / max(capture_sample_rate, 1)))
            if current_end_ms <= active_start_ms:
                return
            self._emit_status("检测到语音模型切换请求，已保持麦克风开启并切分当前语音片段。")
            emit_segment(active_start_ms, current_end_ms)
            active_start_ms = None
            vad_service.reset()
            trim_capture_buffer()

        try:
            with sd.InputStream(
                samplerate=capture_sample_rate,
                channels=MIC_CHANNELS,
                dtype="float32",
                device=input_device,
                blocksize=blocksize,
            ) as stream:
                while not finished:
                    if self._stop_event.is_set():
                        finished = True
                        break
                    block, overflowed = stream.read(blocksize)
                    if overflowed:
                        logger.warning("麦克风输入发生溢出，当前块可能不完整。")
                    samples = np.asarray(block, dtype=np.float32).reshape(-1)
                    if samples.size:
                        capture_buffer = np.concatenate((capture_buffer, samples)).astype(np.float32, copy=False)
                        stream_total_samples += int(samples.shape[0])
                        events = vad_service.process_chunk(
                            samples,
                            capture_sample_rate,
                            is_final=False,
                        )
                        if events:
                            handle_vad_events(events)
                        flush_active_segment_if_requested()
                        trim_capture_buffer()

            final_events = vad_service.process_chunk(
                np.empty(0, dtype=np.float32),
                capture_sample_rate,
                is_final=True,
            )
            if final_events:
                handle_vad_events(final_events)

            if self._stop_event.is_set():
                self._emit_status("麦克风任务已停止。")
                segment_queue.put(None)
                return
            if emitted_segments == 0 and not allow_empty:
                raise RuntimeError("没有检测到有效语音输入，请重试。")
            self._emit_status(f"录音阶段结束，共捕获 {emitted_segments} 段语音。")
            segment_queue.put(None)
        except Exception as exc:
            segment_queue.put(exc)
            segment_queue.put(None)
        finally:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            logger.info(
                "录音分段队列统计: segments={} peak_backlog={} max_put_wait_ms={:.1f} session_ms={:.1f}",
                emitted_segments,
                segment_queue_peak,
                segment_queue_put_wait_ms_max,
                elapsed_ms,
            )

    def _record_program_segments(
        self,
        segment_queue: Queue[object],
        *,
        program_pid: int,
        program_label: str | None,
    ) -> None:
        capture_sample_rate = PROCESS_AUDIO_SAMPLE_RATE
        emitted_segments = 0
        segment_queue_peak = 0
        segment_queue_put_wait_ms_max = 0.0
        capture_buffer = np.empty(0, dtype=np.float32)
        buffer_offset_samples = 0
        stream_total_samples = 0
        read_chunk_count = 0
        empty_read_count = 0
        empty_read_reported = False
        first_chunk_reported = False
        last_emitted_range: tuple[int, int] | None = None
        active_start_ms: int | None = None
        history_samples = max(1, int(capture_sample_rate * MIC_VAD_HISTORY_SEC))
        speaker_label = str(program_label or "").strip() or f"程序 PID {program_pid}"

        vad_service = StreamingActivityDetector(chunk_size_ms=MIC_VAD_CHUNK_MS)
        vad_service.reset()
        # #region debug-point B:program-recorder-entry
        _debug_report_program_listen_worker(
            "B",
            "Program recorder thread entered.",
            {
                "program_pid": program_pid,
                "program_label": speaker_label,
                "stop_event_set": self._stop_event.is_set(),
            },
        )
        # #endregion
        self._emit_status(f"开始监听程序音频: {speaker_label}，当前采集采样率 {capture_sample_rate} Hz。")
        start_time = time.monotonic()

        def trim_capture_buffer() -> None:
            nonlocal capture_buffer, buffer_offset_samples
            if capture_buffer.size == 0:
                return
            if active_start_ms is not None:
                keep_from_abs = max(0, int(round(active_start_ms * capture_sample_rate / 1000.0)) - history_samples)
            else:
                keep_from_abs = max(0, stream_total_samples - history_samples)
            if keep_from_abs <= buffer_offset_samples:
                return
            drop_samples = keep_from_abs - buffer_offset_samples
            if drop_samples <= 0:
                return
            if drop_samples >= int(capture_buffer.shape[0]):
                capture_buffer = np.empty(0, dtype=np.float32)
                buffer_offset_samples = keep_from_abs
                return
            capture_buffer = np.ascontiguousarray(capture_buffer[drop_samples:], dtype=np.float32)
            buffer_offset_samples = keep_from_abs

        def emit_segment(start_ms: int, end_ms: int) -> None:
            nonlocal emitted_segments, segment_queue_peak, segment_queue_put_wait_ms_max, last_emitted_range
            if end_ms <= start_ms:
                return
            start_abs = max(0, int(round(start_ms * capture_sample_rate / 1000.0)))
            end_abs = max(start_abs, int(round(end_ms * capture_sample_rate / 1000.0)))
            if last_emitted_range is not None:
                last_end_abs = last_emitted_range[1]
                if end_abs <= last_end_abs:
                    return
                if start_abs < last_end_abs:
                    start_abs = last_end_abs
            if start_abs < buffer_offset_samples:
                start_abs = buffer_offset_samples

            start_rel = start_abs - buffer_offset_samples
            end_rel = end_abs - buffer_offset_samples
            if start_rel < 0 or end_rel > int(capture_buffer.shape[0]) or end_rel <= start_rel:
                return
            waveform = capture_buffer[start_rel:end_rel].astype(np.float32, copy=False)
            duration = waveform.shape[0] / capture_sample_rate
            if duration < MIC_MIN_SEGMENT_SEC:
                return
            if capture_sample_rate != ASR_SAMPLE_RATE:
                waveform = resample_waveform(waveform, capture_sample_rate, ASR_SAMPLE_RATE)
            emitted_segments += 1
            last_emitted_range = (start_abs, end_abs)
            queued_segment = _QueuedSegment(
                waveform=waveform,
                segment_index=emitted_segments,
                enqueued_at=time.monotonic(),
                model_switch_revision=self._current_voice_output_model_revision(),
                source_kind="program",
                speaker_label=speaker_label,
            )
            put_started_at = time.monotonic()
            segment_queue.put(queued_segment)
            put_wait_ms = (time.monotonic() - put_started_at) * 1000.0
            segment_queue_put_wait_ms_max = max(segment_queue_put_wait_ms_max, put_wait_ms)
            queue_size = self._safe_queue_size(segment_queue)
            if queue_size > segment_queue_peak:
                segment_queue_peak = queue_size
            # #region debug-point C:program-segment-emitted
            _debug_report_program_listen_worker(
                "C",
                "Program recorder emitted a VAD segment.",
                {
                    "program_pid": program_pid,
                    "segment_index": emitted_segments,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_sec": round(duration, 3),
                    "queue_size": queue_size,
                },
            )
            # #endregion
            self._emit_status(
                f"程序分段器已检测到第 {emitted_segments} 段音频 "
                f"({start_ms}ms - {end_ms}ms，来源：{speaker_label})。"
            )

        def handle_vad_events(events: list[tuple[int, int]]) -> None:
            nonlocal active_start_ms
            for start_ms, end_ms in events:
                if start_ms >= 0:
                    active_start_ms = start_ms
                resolved_start_ms = start_ms if start_ms >= 0 else active_start_ms
                if end_ms >= 0:
                    if resolved_start_ms is not None:
                        emit_segment(resolved_start_ms, end_ms)
                    active_start_ms = None

        stream = ProcessAudioStream(program_pid)
        try:
            # #region debug-point B:program-stream-start
            _debug_report_program_listen_worker(
                "B",
                "Starting ProcessAudioStream for program recorder.",
                {"program_pid": program_pid, "program_label": speaker_label},
            )
            # #endregion
            stream.start()
            # #region debug-point B:program-stream-started
            _debug_report_program_listen_worker(
                "B",
                "ProcessAudioStream started.",
                {"program_pid": program_pid, "program_label": speaker_label},
            )
            # #endregion
            while not self._stop_event.is_set():
                samples = stream.read_mono_float32(timeout=0.2)
                if samples is None or samples.size == 0:
                    empty_read_count += 1
                    if not empty_read_reported and empty_read_count >= 10:
                        empty_read_reported = True
                        # #region debug-point B:program-stream-empty-reads
                        _debug_report_program_listen_worker(
                            "B",
                            "Program recorder observed repeated empty reads.",
                            {
                                "program_pid": program_pid,
                                "empty_read_count": empty_read_count,
                                "stream_total_samples": stream_total_samples,
                            },
                        )
                        # #endregion
                    continue
                read_chunk_count += 1
                capture_buffer = np.concatenate((capture_buffer, samples)).astype(np.float32, copy=False)
                stream_total_samples += int(samples.shape[0])
                if not first_chunk_reported:
                    first_chunk_reported = True
                    # #region debug-point B:program-stream-first-chunk
                    _debug_report_program_listen_worker(
                        "B",
                        "Program recorder received first non-empty chunk.",
                        {
                            "program_pid": program_pid,
                            "chunk_samples": int(samples.shape[0]),
                            "stream_total_samples": stream_total_samples,
                        },
                    )
                    # #endregion
                events = vad_service.process_chunk(samples, capture_sample_rate, is_final=False)
                if events:
                    handle_vad_events(events)
                trim_capture_buffer()

            final_events = vad_service.process_chunk(np.empty(0, dtype=np.float32), capture_sample_rate, is_final=True)
            if final_events:
                handle_vad_events(final_events)
            segment_queue.put(None)
        except Exception as exc:
            # #region debug-point D:program-recorder-exception
            _debug_report_program_listen_worker(
                "D",
                "Program recorder raised an exception.",
                {
                    "program_pid": program_pid,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "read_chunk_count": read_chunk_count,
                    "stream_total_samples": stream_total_samples,
                },
            )
            # #endregion
            segment_queue.put(exc)
            segment_queue.put(None)
        finally:
            try:
                stream.close()
            except Exception:
                pass
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            # #region debug-point C:program-recorder-finally
            _debug_report_program_listen_worker(
                "C",
                "Program recorder finished.",
                {
                    "program_pid": program_pid,
                    "emitted_segments": emitted_segments,
                    "read_chunk_count": read_chunk_count,
                    "empty_read_count": empty_read_count,
                    "stream_total_samples": stream_total_samples,
                    "stop_event_set": self._stop_event.is_set(),
                    "elapsed_ms": round(elapsed_ms, 1),
                },
            )
            # #endregion
            logger.info(
                "程序监听分段队列统计: pid={} segments={} peak_backlog={} max_put_wait_ms={:.1f} session_ms={:.1f}",
                program_pid,
                emitted_segments,
                segment_queue_peak,
                segment_queue_put_wait_ms_max,
                elapsed_ms,
            )

    @staticmethod
    def _prepare_output_audio_for_channels(
        audio_np: np.ndarray, target_channels: int
    ) -> tuple[np.ndarray, int]:
        direct = np.ascontiguousarray(audio_np.astype(np.float32, copy=False))
        np.clip(direct, -1.0, 1.0, out=direct)
        if direct.ndim == 1:
            direct = direct.reshape(-1, 1)
        source_channels = int(direct.shape[1])
        if source_channels == target_channels:
            return direct, target_channels
        if target_channels <= 1:
            mono = np.mean(direct, axis=1, dtype=np.float32)
            return np.ascontiguousarray(mono.reshape(-1, 1), dtype=np.float32), 1
        if source_channels == 1:
            duplicated = np.repeat(direct, target_channels, axis=1)
            return np.ascontiguousarray(duplicated, dtype=np.float32), target_channels
        if source_channels > target_channels:
            return np.ascontiguousarray(direct[:, :target_channels], dtype=np.float32), target_channels
        extra = np.repeat(direct[:, -1:], target_channels - source_channels, axis=1)
        return np.ascontiguousarray(np.concatenate((direct, extra), axis=1), dtype=np.float32), target_channels

    @staticmethod
    def _ensure_audio_cache_dir() -> str:
        os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
        return AUDIO_CACHE_DIR

    def _cache_audio_file(
        self,
        message_id: str,
        audio_np: np.ndarray,
        sample_rate: int,
        *,
        prefer_ogg: bool = True,
    ) -> str | None:
        normalized_message_id = str(message_id or "").strip()
        if not normalized_message_id or sample_rate <= 0 or audio_np.size == 0:
            return None
        cache_dir = self._ensure_audio_cache_dir()
        audio_out = np.ascontiguousarray(audio_np.astype(np.float32, copy=False))
        ogg_path = os.path.join(cache_dir, f"{normalized_message_id}.ogg")
        wav_path = os.path.join(cache_dir, f"{normalized_message_id}.wav")
        if prefer_ogg:
            try:
                sf.write(ogg_path, audio_out, sample_rate, format="OGG", subtype="VORBIS")
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                return ogg_path
            except Exception as exc:
                logger.warning("写入 OGG 音频缓存失败，已回退 WAV: {}", exc)
        try:
            sf.write(wav_path, audio_out, sample_rate, format="WAV", subtype="PCM_16")
            return wav_path
        except Exception as exc:
            logger.warning("写入 WAV 音频缓存失败: {}", exc)
            return None

    def _resolve_cached_audio_path(self, message_id: str) -> str | None:
        normalized_message_id = str(message_id or "").strip()
        if not normalized_message_id:
            return None
        cache_dir = self._ensure_audio_cache_dir()
        for ext in (".ogg", ".wav"):
            candidate = os.path.join(cache_dir, f"{normalized_message_id}{ext}")
            if os.path.exists(candidate):
                return candidate
        return None

    @staticmethod
    def _await_playback_dispatch(dispatch: _PlaybackDispatch) -> BaseException | None:
        dispatch.done.wait()
        return dispatch.error

    def _play_audio_chunk(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        primary_endpoint: _DevicePlaybackEndpoint,
        monitor_endpoint: _DevicePlaybackEndpoint | None,
        output_volume: float,
        monitor_output_volume: float,
    ) -> float:
        playback_started_at = time.monotonic()
        primary_audio = audio_np
        if abs(output_volume - 1.0) >= 1e-6:
            primary_audio = np.ascontiguousarray(audio_np.astype(np.float32, copy=False) * output_volume)
            np.clip(primary_audio, -1.0, 1.0, out=primary_audio)
        primary_dispatch = primary_endpoint.submit(primary_audio, sample_rate)
        monitor_audio = audio_np
        if abs(monitor_output_volume - 1.0) >= 1e-6:
            monitor_audio = np.ascontiguousarray(audio_np.astype(np.float32, copy=False) * monitor_output_volume)
            np.clip(monitor_audio, -1.0, 1.0, out=monitor_audio)
        monitor_dispatch = None if monitor_endpoint is None else monitor_endpoint.submit(monitor_audio, sample_rate)

        primary_error = self._await_playback_dispatch(primary_dispatch)
        monitor_error = None if monitor_dispatch is None else self._await_playback_dispatch(monitor_dispatch)
        if monitor_error:
            logger.warning("耳返设备播放失败，已自动忽略: {}", monitor_error)
        if primary_error:
            raise primary_error
        return (time.monotonic() - playback_started_at) * 1000.0

    @staticmethod
    def _finalize_playback_session(
        session_end: _PlaybackSessionEnd | None,
        *,
        chunk_count: int,
        queue_wait_ms_max: float,
        play_ms_max: float,
        total_audio_ms: float,
    ) -> None:
        if session_end is None:
            return
        logger.info(
            "播放队列统计: chunks={} max_queue_wait_ms={:.1f} max_play_ms={:.1f} total_audio_ms={:.1f} total_ms={:.1f}",
            chunk_count,
            queue_wait_ms_max,
            play_ms_max,
            total_audio_ms,
            (time.monotonic() - session_end.started_at) * 1000.0,
        )
        session_end.done.set()

    @staticmethod
    def _drain_playback_session_queue(
        playback_queue: Queue[object],
    ) -> tuple[int, _PlaybackSessionEnd | None, bool]:
        discarded = 0
        session_end: _PlaybackSessionEnd | None = None
        sentinel_found = False
        while True:
            try:
                queued = playback_queue.get_nowait()
            except Empty:
                break
            if queued is _PLAYBACK_SENTINEL:
                sentinel_found = True
                break
            if isinstance(queued, _QueuedPlaybackChunk):
                discarded += 1
                continue
            if isinstance(queued, _PlaybackSessionEnd):
                session_end = queued
                break
        return discarded, session_end, sentinel_found

    @staticmethod
    def _drop_pending_playback_chunks(playback_queue: Queue[object]) -> int:
        preserved: list[object] = []
        dropped = 0
        while True:
            try:
                queued = playback_queue.get_nowait()
            except Empty:
                break
            if isinstance(queued, _QueuedPlaybackChunk):
                dropped += 1
                continue
            preserved.append(queued)
        for item in preserved:
            playback_queue.put_nowait(item)
        return dropped

    def _playback_worker(self) -> None:
        primary_endpoint: _DevicePlaybackEndpoint | None = None
        monitor_endpoint: _DevicePlaybackEndpoint | None = None
        primary_device: str | None | object = object()
        monitor_device: str | None = None
        session_end: _PlaybackSessionEnd | None = None
        session_error: BaseException | None = None
        playback_chunk_count = 0
        playback_queue_residence_ms_max = 0.0
        playback_elapsed_ms_max = 0.0
        playback_total_audio_ms = 0.0
        try:
            while True:
                queued = self._playback_queue.get()
                if queued is _PLAYBACK_SENTINEL:
                    return
                if isinstance(queued, _PlaybackSessionStart):
                    target_primary_device = self.output_device
                    target_monitor_device = (
                        self.monitor_output_device
                        if self.monitor_output_device is not None and self.monitor_output_device != self.output_device
                        else None
                    )
                    if primary_endpoint is None or primary_device != target_primary_device:
                        if primary_endpoint is not None:
                            primary_endpoint.close()
                        primary_endpoint = _DevicePlaybackEndpoint(
                            device=target_primary_device,
                            label="primary",
                        )
                        primary_device = target_primary_device
                    if target_monitor_device != monitor_device:
                        if monitor_endpoint is not None:
                            monitor_endpoint.close()
                            monitor_endpoint = None
                        if target_monitor_device is not None:
                            monitor_endpoint = _DevicePlaybackEndpoint(
                                device=target_monitor_device,
                                label="monitor",
                            )
                        monitor_device = target_monitor_device
                    session_end = None
                    session_error = None
                    playback_chunk_count = 0
                    playback_queue_residence_ms_max = 0.0
                    playback_elapsed_ms_max = 0.0
                    playback_total_audio_ms = 0.0
                    continue
                if isinstance(queued, _PlaybackSessionEnd):
                    session_end = queued
                    session_end.error = session_error
                    self._finalize_playback_session(
                        session_end,
                        chunk_count=playback_chunk_count,
                        queue_wait_ms_max=playback_queue_residence_ms_max,
                        play_ms_max=playback_elapsed_ms_max,
                        total_audio_ms=playback_total_audio_ms,
                    )
                    session_end = None
                    session_error = None
                    playback_chunk_count = 0
                    playback_queue_residence_ms_max = 0.0
                    playback_elapsed_ms_max = 0.0
                    playback_total_audio_ms = 0.0
                    continue
                if self._stop_event.is_set():
                    current_discarded = 1 if isinstance(queued, _QueuedPlaybackChunk) else 0
                    discarded, drained_session_end, sentinel_found = self._drain_playback_session_queue(self._playback_queue)
                    total_discarded = current_discarded + discarded
                    if total_discarded > 0:
                        logger.info("停止请求已收到，已丢弃 {} 个待播放音频块。", total_discarded)
                    if drained_session_end is not None:
                        session_end = drained_session_end
                        session_end.error = session_error
                        self._finalize_playback_session(
                            session_end,
                            chunk_count=playback_chunk_count,
                            queue_wait_ms_max=playback_queue_residence_ms_max,
                            play_ms_max=playback_elapsed_ms_max,
                            total_audio_ms=playback_total_audio_ms,
                        )
                        session_end = None
                        session_error = None
                        playback_chunk_count = 0
                        playback_queue_residence_ms_max = 0.0
                        playback_elapsed_ms_max = 0.0
                        playback_total_audio_ms = 0.0
                    if sentinel_found:
                        return
                    continue
                if not isinstance(queued, _QueuedPlaybackChunk):
                    continue
                if primary_endpoint is None:
                    primary_endpoint = _DevicePlaybackEndpoint(
                        device=self.output_device,
                        label="primary",
                    )
                    primary_device = self.output_device
                playback_chunk_count += 1
                queue_residence_ms = (time.monotonic() - queued.enqueued_at) * 1000.0
                playback_queue_residence_ms_max = max(playback_queue_residence_ms_max, queue_residence_ms)
                if queue_residence_ms >= QUEUE_RESIDENCE_LOG_THRESHOLD_MS:
                    logger.info(
                        "播放队列驻留偏高: chunk={} queue_wait_ms={:.1f} backlog={}",
                        queued.chunk_index,
                        queue_residence_ms,
                        self._safe_queue_size(self._playback_queue),
                    )
                try:
                    output_volume, monitor_output_volume, _ = self._snapshot_playback_controls()
                    playback_elapsed_ms = self._play_audio_chunk(
                        queued.audio_np,
                        queued.sample_rate,
                        primary_endpoint,
                        monitor_endpoint,
                        output_volume,
                        monitor_output_volume,
                    )
                    chunk_duration_ms = self._audio_duration_ms(queued.audio_np, queued.sample_rate)
                    playback_elapsed_ms_max = max(playback_elapsed_ms_max, playback_elapsed_ms)
                    playback_total_audio_ms += chunk_duration_ms
                    if playback_elapsed_ms >= max(chunk_duration_ms * 1.15, QUEUE_RESIDENCE_LOG_THRESHOLD_MS):
                        logger.info(
                            "播放块耗时偏高: chunk={} queue_wait_ms={:.1f} play_ms={:.1f} audio_ms={:.1f}",
                            queued.chunk_index,
                            queue_residence_ms,
                            playback_elapsed_ms,
                            chunk_duration_ms,
                        )
                except BaseException as exc:
                    session_error = exc
                    logger.exception("播放块处理失败")
        except BaseException as exc:
            self._playback_worker_error = exc
            logger.exception("共享播放线程异常退出")
            self._fail_pending_playback_sessions(self._playback_queue, exc)
        finally:
            if session_end is not None:
                session_end.error = session_error
                self._finalize_playback_session(
                    session_end,
                    chunk_count=playback_chunk_count,
                    queue_wait_ms_max=playback_queue_residence_ms_max,
                    play_ms_max=playback_elapsed_ms_max,
                    total_audio_ms=playback_total_audio_ms,
                )
            if monitor_endpoint is not None:
                monitor_endpoint.close()
            if primary_endpoint is not None:
                primary_endpoint.close()

    @Slot()
    def stop_live(self) -> None:
        if self._busy:
            self._emit_status("正在停止录音...")
            self._set_runtime_state(LiveWorkerState.STOPPING)
            self._stop_event.set()
            self._segment_flush_requested.set()

    @Slot(object)
    def run_once(self, payload: dict[str, object]) -> None:
        if self._busy:
            self._emit_status("当前已有变声任务在执行，请稍候。")
            return
        if not self._models_ready:
            self.error.emit("语音合成引擎尚未完成初始化，请稍后再试。")
            return

        self._busy = True
        self._model_manager.set_busy(True)
        self._set_runtime_state(LiveWorkerState.RUNNING)
        self._stop_event.clear()
        self._segment_flush_requested.clear()
        self._reset_emit_cache()
        try:
            input_mode = str(payload.get("input_mode") or "microphone")
            input_text = str(payload.get("input_text") or "").strip()
            ui_message_id = str(payload.get("ui_message_id") or "").strip() or None
            self.reference_audio_path = str(payload["reference_audio_path"])
            with self._voice_output_model_lock:
                self.voice_output_model = self._normalize_voice_output_model(
                    payload.get("voice_output_model") or self._settings.voice_output_model or "meanvc2"
                )
                self._pending_voice_output_model = None
            self.language = str(payload.get("language") or "auto")
            self.output_language = str(payload.get("output_language") or self._settings.output_voice_language or "zh")
            self.translation_enabled = bool(payload.get("translation_enabled", self._settings.translation_enabled))
            self.translation_target_language = str(
                payload.get("translation_target_language") or self._settings.translation_target_language or "zh"
            ).strip() or "zh"
            raw_translation_node_id = payload.get("translation_node_id")
            if isinstance(raw_translation_node_id, str):
                self.translation_node_id = raw_translation_node_id.strip() or None
            else:
                self.translation_node_id = self._settings.translation_node_id
            raw_instruction = payload.get("instruction")
            self.instruction = str(raw_instruction).strip() if isinstance(raw_instruction, str) and raw_instruction else None
            self.index_emo_vector = payload.get("index_emo_vector")
            self.index_emo_alpha = float(payload.get("index_emo_alpha", 1.0))
            self.auto_emotion_enabled = bool(payload.get("auto_emotion_enabled"))
            self.auto_emotion_strength = float(payload.get("auto_emotion_strength", 0.75))
            raw_user_voiceprint = payload.get("user_voiceprint")
            if isinstance(raw_user_voiceprint, np.ndarray):
                self.user_voiceprint = raw_user_voiceprint.astype(np.float32, copy=False)
            elif isinstance(raw_user_voiceprint, (list, tuple)) and raw_user_voiceprint:
                self.user_voiceprint = np.asarray(raw_user_voiceprint, dtype=np.float32)
            else:
                self.user_voiceprint = None
            self.voiceprint_threshold = float(payload.get("voiceprint_threshold", 0.65))
            self.max_new_tokens = int(payload.get("max_new_tokens") or 2048)
            self.input_device = payload.get("input_device") if isinstance(payload.get("input_device"), str) else None
            self.input_device_label = (
                str(payload.get("input_device_label")).strip()
                if isinstance(payload.get("input_device_label"), str) and str(payload.get("input_device_label")).strip()
                else None
            )
            self.output_device = payload.get("output_device") if isinstance(payload.get("output_device"), str) else None
            self.output_device_label = (
                str(payload.get("output_device_label")).strip()
                if isinstance(payload.get("output_device_label"), str) and str(payload.get("output_device_label")).strip()
                else None
            )
            self.monitor_output_device = (
                payload.get("monitor_output_device") if isinstance(payload.get("monitor_output_device"), str) else None
            )
            self.monitor_output_device_label = (
                str(payload.get("monitor_output_device_label")).strip()
                if isinstance(payload.get("monitor_output_device_label"), str)
                and str(payload.get("monitor_output_device_label")).strip()
                else None
            )
            raw_program_pid = payload.get("program_pid")
            program_pid = int(raw_program_pid) if isinstance(raw_program_pid, int) or str(raw_program_pid or "").isdigit() else 0
            self.program_label = (
                str(payload.get("program_label")).strip()
                if isinstance(payload.get("program_label"), str) and str(payload.get("program_label")).strip()
                else None
            )
            # #region debug-point A:worker-run-once-payload
            _debug_report_program_listen_worker(
                "A",
                "Worker received live payload.",
                {
                    "input_mode": input_mode,
                    "program_pid": program_pid,
                    "program_label": self.program_label,
                    "reference_audio_present": bool(self.reference_audio_path),
                    "stop_event_set": self._stop_event.is_set(),
                    "models_ready": self._models_ready,
                },
            )
            # #endregion
            self.update_output_volume(float(payload.get("output_volume", self.output_volume)))
            self.update_monitor_output_volume(float(payload.get("monitor_output_volume", self.monitor_output_volume)))
            self.update_speech_rate(float(payload.get("speech_rate") or 1.0))

            self._emit_status(f"使用 {self._voice_output_model_label()} 引擎...")
            if self.auto_emotion_enabled and input_mode != "text":
                self._emit_emotion_state({"state": "busy", "summary": "实验性情感识别已开启，等待语音片段..."})
            else:
                self._emit_emotion_state({"state": "idle", "summary": "实验性情感识别未启用。"})
            self._emit_status(
                f"当前设备: 录音={self.input_device_label or '系统默认输入'} / "
                f"播放={self.output_device_label or '系统默认输出'} / "
                f"耳返={self.monitor_output_device_label or '关闭耳返'}"
            )
            _, _, speech_rate = self._snapshot_playback_controls()
            if self.voice_output_model == "indextts" and abs(speech_rate - 1.0) >= 1e-6:
                self._emit_status(f"当前 Index-TTS 原生语速: {speech_rate:.2f}x")
            if self.instruction:
                self._emit_status(f"当前语气控制: {self.instruction}")
            if self.index_emo_vector:
                self._emit_status(
                    f"情感向量: [{', '.join(f'{v:.2f}' for v in self.index_emo_vector)}] 强度={self.index_emo_alpha:.2f}"
                )

            if input_mode == "text":
                transcript = input_text
                if not transcript:
                    raise RuntimeError("文本模式下输入文本为空。")
                self._apply_pending_voice_output_model_switch()
                if self.voice_output_model == "meanvc2":
                    raise RuntimeError("MeanVC2 不支持文本模式，请切换到麦克风输入或改用 Index-TTS。")
                self._ensure_tts_loaded()
                self._emit_status("文本模式：跳过 ASR，直接开始流式 TTS。")
                translated_text: str | None = None
                if self.translation_enabled:
                    tts_text = self._prepare_tts_text(transcript)
                    translated_text = tts_text.strip() or None
                else:
                    tts_text = transcript
                if translated_text is not None:
                    self._emit_chat_record(
                        input_mode="text",
                        source_text=transcript,
                        message_id=ui_message_id,
                        source_language=detect_text_language(transcript),
                        translated_text=translated_text,
                        state="pending",
                    )
                self._synthesize(tts_text, "文本", progressive_transcript=True, message_id=ui_message_id)
                self._emit_perf_metrics(
                    asr_latency=0.0,
                    trans_latency=self._last_trans_latency_ms,
                    tts_latency=self._last_tts_latency_ms,
                    total_latency=self._last_trans_latency_ms + self._last_tts_latency_ms,
                )
                self._emit_chat_record(
                    input_mode="text",
                    source_text=transcript,
                    message_id=ui_message_id,
                    source_language=detect_text_language(transcript),
                    translated_text=translated_text,
                    state="ready",
                )
            else:
                asr_service = self._ensure_asr_loaded()
                self._ensure_vad_loaded()
                segment_queue: Queue[object] = Queue(maxsize=8)
                synthesis_queue: Queue[object] | None = None
                synthesis_thread: threading.Thread | None = None
                synthesis_error: list[BaseException] = []
                recorder_threads: list[threading.Thread] = []
                synth_elapsed_ms_max = 0.0
                synth_metrics_lock = threading.Lock()

                def record_synth_elapsed(elapsed_ms: float) -> None:
                    nonlocal synth_elapsed_ms_max
                    with synth_metrics_lock:
                        synth_elapsed_ms_max = max(synth_elapsed_ms_max, elapsed_ms)

                def get_synth_error() -> BaseException | None:
                    with synth_metrics_lock:
                        return synthesis_error[0] if synthesis_error else None

                def synthesis_worker() -> None:
                    assert synthesis_queue is not None
                    while True:
                        queued_task = synthesis_queue.get()
                        if queued_task is None:
                            break
                        if not isinstance(queued_task, _QueuedSynthesisTask):
                            continue
                        started_at = time.monotonic()
                        try:
                            self._wait_for_inline_text_priority()
                            with self._synthesis_lock:
                                if queued_task.voice_output_model == "meanvc2":
                                    if queued_task.waveform is None:
                                        raise RuntimeError("缺少 MeanVC2 需要的语音波形。")
                                    self._synthesize_meanvc2(
                                        queued_task.waveform,
                                        queued_task.segment_label,
                                        message_id=queued_task.message_id,
                                        switch_revision=queued_task.switch_revision,
                                    )
                                else:
                                    if queued_task.transcript is None:
                                        raise RuntimeError("缺少 Index-TTS 需要的文本内容。")
                                    self._synthesize(
                                        queued_task.transcript,
                                        queued_task.segment_label,
                                        emo_vector=queued_task.emo_vector,
                                        emo_alpha=queued_task.emo_alpha,
                                        message_id=queued_task.message_id,
                                        switch_revision=queued_task.switch_revision,
                                    )
                            self._emit_perf_metrics(
                                asr_latency=queued_task.asr_latency_ms,
                                trans_latency=queued_task.trans_latency_ms,
                                tts_latency=self._last_tts_latency_ms,
                                total_latency=queued_task.asr_latency_ms + queued_task.trans_latency_ms + self._last_tts_latency_ms,
                            )
                        except BaseException as exc:
                            with synth_metrics_lock:
                                if not synthesis_error:
                                    synthesis_error.append(exc)
                            self._stop_event.set()
                            self._segment_flush_requested.set()
                            logger.exception("麦克风合成任务失败")
                            break
                        finally:
                            record_synth_elapsed((time.monotonic() - started_at) * 1000.0)
                if input_mode in {"microphone", "dual"}:
                    synthesis_queue = Queue()
                    synthesis_thread = threading.Thread(
                        target=synthesis_worker,
                        daemon=True,
                        name="live-mic-synthesis",
                    )
                    synthesis_thread.start()
                    recorder_threads.append(
                        threading.Thread(
                            target=self._record_microphone_segments,
                            args=(segment_queue,),
                            kwargs={"allow_empty": input_mode == "dual"},
                            daemon=True,
                        )
                    )
                if input_mode in {"program", "dual"}:
                    if program_pid <= 0:
                        raise RuntimeError("程序监听缺少有效的目标进程。")
                    recorder_threads.append(
                        threading.Thread(
                            target=self._record_program_segments,
                            args=(segment_queue,),
                            kwargs={"program_pid": program_pid, "program_label": self.program_label},
                            daemon=True,
                        )
                    )
                if not recorder_threads:
                    raise RuntimeError("当前没有可用的监听来源。")
                # #region debug-point E:worker-recorder-start
                _debug_report_program_listen_worker(
                    "E",
                    "Worker is about to start recorder threads.",
                    {
                        "input_mode": input_mode,
                        "program_pid": program_pid,
                        "recorder_count": len(recorder_threads),
                        "stop_event_set": self._stop_event.is_set(),
                    },
                )
                # #endregion
                for recorder_thread in recorder_threads:
                    recorder_thread.start()

                transcript_parts: list[str] = []
                source_context_parts: dict[str, list[str]] = {"microphone": [], "program": []}
                latest_transcript = ""
                asr_segment_index = 0
                segment_queue_residence_ms_max = 0.0
                asr_elapsed_ms_max = 0.0
                finished_recorders = 0

                while True:
                    pending_synth_error = get_synth_error()
                    if pending_synth_error is not None:
                        raise pending_synth_error
                    if self._stop_event.is_set():
                        break
                    try:
                        queued = segment_queue.get(timeout=0.1)
                    except Empty:
                        pending_synth_error = get_synth_error()
                        if pending_synth_error is not None:
                            raise pending_synth_error
                        if not any(thread.is_alive() for thread in recorder_threads):
                            continue
                        continue

                    if queued is None:
                        finished_recorders += 1
                        if finished_recorders >= len(recorder_threads):
                            break
                        continue
                    if isinstance(queued, Exception):
                        raise queued
                    queue_residence_ms = 0.0
                    queued_segment_index: int | None = None
                    queued_segment_revision = self._current_voice_output_model_revision()
                    source_kind = "microphone"
                    speaker_label = "麦克风"
                    if isinstance(queued, _QueuedSegment):
                        queue_residence_ms = (time.monotonic() - queued.enqueued_at) * 1000.0
                        queued_segment_index = queued.segment_index
                        queued_segment_revision = queued.model_switch_revision
                        source_kind = queued.source_kind
                        speaker_label = queued.speaker_label
                        waveform = queued.waveform
                    else:
                        waveform = queued
                    if not isinstance(waveform, np.ndarray):
                        continue
                    if self._is_segment_switch_interrupted(queued_segment_revision):
                        self._emit_status(
                            f"第 {queued_segment_index if queued_segment_index is not None else '?'} 段旧模型语音已取消，立即切换到新模型。"
                        )
                        continue
                    if queue_residence_ms > segment_queue_residence_ms_max:
                        segment_queue_residence_ms_max = queue_residence_ms
                    if queue_residence_ms >= QUEUE_RESIDENCE_LOG_THRESHOLD_MS:
                        logger.info(
                            "录音分段队列驻留偏高: segment={} queue_wait_ms={:.1f} backlog={}",
                            queued_segment_index if queued_segment_index is not None else "?",
                            queue_residence_ms,
                            self._safe_queue_size(segment_queue),
                        )
                    if self._stop_event.is_set():
                        break

                    self._apply_pending_voice_output_model_switch()
                    segment_revision = self._current_voice_output_model_revision()

                    # --- Voiceprint Filter Check ---
                    if source_kind == "microphone" and self.user_voiceprint is not None and self.user_voiceprint.size > 0:
                        self._emit_status("正在执行声纹过滤...")
                        try:
                            self._ensure_speaker_loaded(emit_status=False)
                            filtered_waveform, filter_message = self._filter_waveform_by_voiceprint(waveform)
                            if filter_message:
                                self._emit_status(filter_message)
                            if filtered_waveform is None:
                                continue
                            waveform = filtered_waveform
                        except Exception as e:
                            logger.error("声纹过滤失败: {}", e)

                    asr_segment_index += 1
                    segment_emo_vector = self.index_emo_vector
                    segment_emo_alpha = self.index_emo_alpha
                    if source_kind == "microphone" and self.auto_emotion_enabled:
                        try:
                            segment_emo_vector, segment_emo_alpha, _prediction = self._detect_segment_emotion(waveform)
                        except Exception as exc:
                            logger.warning("实验性情感识别失败，已回退到手动情感: {}", exc)
                            self._emit_status(f"实验性情感识别失败，已回退手动情感: {exc}")
                            self._emit_emotion_state(
                                {"state": "error", "summary": "实验性情感识别失败，已回退手动情感。"}
                            )
                            self.auto_emotion_enabled = False
                    self._emit_status(f"正在识别第 {asr_segment_index} 段语音（{speaker_label}）...")
                    asr_context = build_asr_context(source_context_parts.get(source_kind, []))
                    self._mark_model_used("asr")
                    asr_started_at = time.monotonic()
                    result = asr_service.transcribe(
                        ASRRequest(
                            audio_ndarray=(waveform, ASR_SAMPLE_RATE),
                            language=None if (self.language or "auto").strip().lower() in {"", "auto"} else self.language,
                            return_language=True,
                            context=asr_context,
                        )
                    )
                    asr_elapsed_ms = (time.monotonic() - asr_started_at) * 1000.0
                    self._last_asr_latency_ms = asr_elapsed_ms
                    asr_elapsed_ms_max = max(asr_elapsed_ms_max, asr_elapsed_ms)
                    if self._is_segment_switch_interrupted(segment_revision):
                        self._emit_status(
                            f"第 {asr_segment_index} 段已因语音模型切换而终止，后续语音将使用 {self._voice_output_model_label()}。"
                        )
                        continue

                    transcript = normalize_asr_transcript(result.text)
                    previous_parts = source_context_parts.get(source_kind, [])
                    previous_transcript = previous_parts[-1] if previous_parts else None
                    transcript = merge_asr_segments(previous_transcript, transcript)
                    if not transcript:
                        if previous_parts:
                            previous_parts.clear()
                            self._last_tts_transcript_key = None
                            self._emit_status("FunASR VAD 未检测到新语音，已重置 ASR 上下文。")
                        self._emit_status(f"第 {asr_segment_index} 段识别为空，已跳过（{speaker_label}）。")
                        continue
                    transcript_key = canonicalize_asr_text(transcript)
                    if not transcript_key:
                        logger.info("ASR 结果无有效文本，跳过 TTS: {}", transcript)
                        self._emit_status(f"第 {asr_segment_index} 段无有效文本，已跳过（{speaker_label}）。")
                        continue
                    if source_kind == "microphone" and transcript_key == self._last_tts_transcript_key:
                        logger.info("ASR 连续重复结果已跳过，不再发送 TTS: {}", transcript)
                        self._emit_status(f"第 {asr_segment_index} 段与上一句重复，已跳过。")
                        continue

                    transcript_parts.append(transcript)
                    source_context_parts.setdefault(source_kind, []).append(transcript)
                    latest_transcript = transcript
                    if source_kind == "microphone":
                        self._last_tts_transcript_key = transcript_key
                    self._emit_transcript(latest_transcript)
                    self._emit_status(f"第 {asr_segment_index} 段识别结果（{speaker_label}）: {transcript}")
                    segment_pipeline_started_at = time.monotonic()
                    translated_text: str | None = None
                    message_id = f"chat-{uuid4().hex[:8]}"
                    if self._is_segment_switch_interrupted(segment_revision):
                        self._emit_status(
                            f"第 {asr_segment_index} 段已因语音模型切换而终止，后续语音将使用 {self._voice_output_model_label()}。"
                        )
                        continue
                    if source_kind == "program":
                        if message_id and waveform.size > 0 and not self._stop_event.is_set():
                            try:
                                self._cache_audio_file(message_id, waveform, ASR_SAMPLE_RATE, prefer_ogg=True)
                            except Exception as exc:
                                logger.warning("写入程序监听音频缓存失败: message_id={} error={}", message_id, exc)
                        if self.translation_enabled:
                            translated_candidate = self._prepare_tts_text(transcript, source_language=result.language)
                            translated_text = translated_candidate.strip() or None
                        self._emit_chat_record(
                            input_mode=input_mode,
                            source_text=transcript,
                            message_id=message_id,
                            source_language=result.language,
                            translated_text=translated_text,
                            speaker_label=speaker_label,
                        )
                        self._emit_perf_metrics(
                            asr_latency=asr_elapsed_ms,
                            trans_latency=self._last_trans_latency_ms,
                            tts_latency=0.0,
                            total_latency=asr_elapsed_ms + self._last_trans_latency_ms,
                        )
                    elif self.voice_output_model == "meanvc2":
                        if self.translation_enabled:
                            translated_candidate = self._prepare_tts_text(transcript, source_language=result.language)
                            translated_text = translated_candidate.strip() or None
                        self._emit_chat_record(
                            input_mode=input_mode,
                            source_text=transcript,
                            message_id=message_id,
                            source_language=result.language,
                            translated_text=translated_text,
                            speaker_label="麦克风",
                        )
                        if synthesis_queue is not None:
                            synthesis_queue.put(
                                _QueuedSynthesisTask(
                                    voice_output_model="meanvc2",
                                    segment_label=f"第 {asr_segment_index} 段",
                                    message_id=message_id,
                                    switch_revision=segment_revision,
                                    asr_latency_ms=asr_elapsed_ms,
                                    trans_latency_ms=self._last_trans_latency_ms,
                                    waveform=np.ascontiguousarray(waveform.astype(np.float32, copy=False)),
                                )
                            )
                        else:
                            self._synthesize_meanvc2(
                                waveform,
                                f"第 {asr_segment_index} 段",
                                message_id=message_id,
                                switch_revision=segment_revision,
                            )
                            self._emit_perf_metrics(
                                asr_latency=asr_elapsed_ms,
                                trans_latency=self._last_trans_latency_ms,
                                tts_latency=self._last_tts_latency_ms,
                                total_latency=asr_elapsed_ms + self._last_trans_latency_ms + self._last_tts_latency_ms,
                            )
                    else:
                        if self.translation_enabled:
                            tts_text = self._prepare_tts_text(transcript, source_language=result.language)
                            translated_text = tts_text.strip() or None
                        else:
                            tts_text = transcript
                        self._emit_chat_record(
                            input_mode=input_mode,
                            source_text=transcript,
                            message_id=message_id,
                            source_language=result.language,
                            translated_text=translated_text,
                            speaker_label="麦克风",
                        )
                        if synthesis_queue is not None:
                            synthesis_queue.put(
                                _QueuedSynthesisTask(
                                    voice_output_model="indextts",
                                    segment_label=f"第 {asr_segment_index} 段",
                                    message_id=message_id,
                                    switch_revision=segment_revision,
                                    asr_latency_ms=asr_elapsed_ms,
                                    trans_latency_ms=self._last_trans_latency_ms,
                                    transcript=tts_text,
                                    emo_vector=list(segment_emo_vector) if segment_emo_vector is not None else None,
                                    emo_alpha=segment_emo_alpha,
                                )
                            )
                        else:
                            self._synthesize(
                                tts_text, f"第 {asr_segment_index} 段",
                                emo_vector=segment_emo_vector,
                                emo_alpha=segment_emo_alpha,
                                message_id=message_id,
                                switch_revision=segment_revision,
                            )
                            self._emit_perf_metrics(
                                asr_latency=asr_elapsed_ms,
                                trans_latency=self._last_trans_latency_ms,
                                tts_latency=self._last_tts_latency_ms,
                                total_latency=asr_elapsed_ms + self._last_trans_latency_ms + self._last_tts_latency_ms,
                            )
                    segment_pipeline_elapsed_ms = (time.monotonic() - segment_pipeline_started_at) * 1000.0
                    logger.info(
                        "语音段处理统计: segment={} queue_wait_ms={:.1f} asr_ms={:.1f} synth_ms={:.1f}",
                        asr_segment_index,
                        queue_residence_ms,
                        asr_elapsed_ms,
                        segment_pipeline_elapsed_ms,
                    )

                for recorder_thread in recorder_threads:
                    recorder_thread.join(timeout=1.0)
                if synthesis_queue is not None:
                    synthesis_queue.put(None)
                if synthesis_thread is not None:
                    synthesis_thread.join(timeout=5.0)
                pending_synth_error = get_synth_error()
                if pending_synth_error is not None:
                    raise pending_synth_error
                logger.info(
                    "录音消费统计: segments={} max_queue_wait_ms={:.1f} max_asr_ms={:.1f} max_synth_ms={:.1f}",
                    asr_segment_index,
                    segment_queue_residence_ms_max,
                    asr_elapsed_ms_max,
                    synth_elapsed_ms_max,
                )
                if self._stop_event.is_set():
                    transcript = latest_transcript or "\n".join(transcript_parts).strip()
                    self._emit_status("监听任务已停止。")
                    self.finished.emit(transcript)
                    return
                final_transcript = "\n".join(transcript_parts).strip()
                if not final_transcript:
                    # #region debug-point C:no-final-transcript
                    _debug_report_program_listen_worker(
                        "C",
                        "Worker finished capture without final transcript.",
                        {
                            "input_mode": input_mode,
                            "program_pid": program_pid,
                            "segments_seen_by_asr": asr_segment_index,
                            "finished_recorders": finished_recorders,
                            "stop_event_set": self._stop_event.is_set(),
                        },
                    )
                    # #endregion
                    raise RuntimeError("没有检测到可用的语音内容，请重试。")
                transcript = latest_transcript or final_transcript

            self.finished.emit(transcript)
        except Exception as exc:
            # #region debug-point D:worker-run-once-exception
            _debug_report_program_listen_worker(
                "D",
                "Worker run_once raised an exception.",
                {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            # #endregion
            logger.exception("Live 页面执行失败")
            self.error.emit(str(exc))
        finally:
            self._busy = False
            self._model_manager.set_busy(False)
            if self._models_ready:
                self._set_runtime_state(LiveWorkerState.READY)
            if self._model_manager.asr_service is not None:
                self._defer_idle_unload("asr")
            if self._model_manager.index_tts_service is not None or self._model_manager.meanvc2_service is not None:
                self._defer_idle_unload("tts")

    def _synthesize(
        self,
        transcript: str,
        segment_label: str,
        *,
        progressive_transcript: bool = False,
        emo_vector: list[float] | None = None,
        emo_alpha: float | None = None,
        message_id: str | None = None,
        switch_revision: int | None = None,
    ) -> int:
        tts_service = self._ensure_tts_loaded(warmup_reference=False)
        self._emit_status(f"{segment_label}进入 Index-TTS 流式合成...")
        self._last_tts_latency_ms = 0.0
        active_emo_vector = self.index_emo_vector if emo_vector is None else emo_vector
        active_emo_alpha = self.index_emo_alpha if emo_alpha is None else emo_alpha
        sample_rate = 22050
        chunk_count = 0
        playback_queue_peak = 0
        playback_queue_put_wait_ms_max = 0.0
        total_audio_ms = 0.0
        synth_started_at = time.monotonic()
        first_chunk_latency_ms: float | None = None
        previous_chunk_at: float | None = None
        max_chunk_gap_ms = 0.0
        audio_chunks: list[np.ndarray] = []
        self._ensure_playback_worker()
        playback_queue = self._playback_queue
        playback_thread = self._playback_thread
        playback_session_end = _PlaybackSessionEnd(
            label=segment_label,
            started_at=synth_started_at,
            done=threading.Event(),
        )
        self._put_playback_item(
            playback_queue,
            _PlaybackSessionStart(
                label=segment_label,
                started_at=synth_started_at,
            ),
            segment_label,
        )

        try:
            _, _, speech_rate = self._snapshot_playback_controls()
            duration_factor = speech_rate_to_duration_factor(speech_rate)
            for chunk_np, sr in tts_service.synthesize_stream(
                text=transcript,
                reference_audio_path=self.reference_audio_path,
                emo_vector=active_emo_vector,
                emo_alpha=active_emo_alpha,
                duration_factor=duration_factor,
            ):
                if self._stop_event.is_set():
                    break
                if switch_revision is not None and self._is_segment_switch_interrupted(switch_revision):
                    self._emit_status(f"{segment_label}已终止旧模型输出，立即切到 {self._voice_output_model_label()}。")
                    break
                self._mark_model_used("tts")
                sample_rate = sr
                audio_chunks.append(np.ascontiguousarray(chunk_np.astype(np.float32, copy=False)))
                if self._stop_event.is_set():
                    break
                if switch_revision is not None and self._is_segment_switch_interrupted(switch_revision):
                    self._emit_status(f"{segment_label}已终止旧模型输出，立即切到 {self._voice_output_model_label()}。")
                    break
                now = time.monotonic()
                chunk_count += 1
                if first_chunk_latency_ms is None:
                    first_chunk_latency_ms = (now - synth_started_at) * 1000.0
                if previous_chunk_at is not None:
                    max_chunk_gap_ms = max(max_chunk_gap_ms, (now - previous_chunk_at) * 1000.0)
                previous_chunk_at = now
                total_audio_ms += self._audio_duration_ms(chunk_np, sample_rate)
                self.waveform.emit(chunk_np)
                queued_chunk = _QueuedPlaybackChunk(
                    audio_np=chunk_np,
                    sample_rate=sample_rate,
                    chunk_index=chunk_count,
                    enqueued_at=time.monotonic(),
                )
                put_started_at = time.monotonic()
                self._put_playback_item(playback_queue, queued_chunk, segment_label)
                put_wait_ms = (time.monotonic() - put_started_at) * 1000.0
                playback_queue_put_wait_ms_max = max(playback_queue_put_wait_ms_max, put_wait_ms)
                queue_size = self._safe_queue_size(playback_queue)
                if queue_size > playback_queue_peak:
                    playback_queue_peak = queue_size
                    logger.info("播放队列积压新高: backlog={}/6", queue_size)
                if put_wait_ms >= QUEUE_WAIT_LOG_THRESHOLD_MS:
                    logger.warning(
                        "{} 播放队列入队等待偏高: chunk={} wait_ms={:.1f} backlog={}",
                        segment_label,
                        chunk_count,
                        put_wait_ms,
                        queue_size,
                    )
        finally:
            interrupted = switch_revision is not None and self._is_segment_switch_interrupted(switch_revision)
            if self._stop_event.is_set() or interrupted:
                dropped_chunks = self._drop_pending_playback_chunks(playback_queue)
                if dropped_chunks > 0:
                    reason = "停止" if self._stop_event.is_set() else "模型切换"
                    logger.info("{} {}时已主动丢弃 {} 个待播放音频块。", segment_label, reason, dropped_chunks)
            self._put_playback_item(playback_queue, playback_session_end, segment_label)
            wait_timeout = 0.3 if (self._stop_event.is_set() or interrupted) else 2.0
            max_stop_wait_sec = 1.0
            stop_wait_started_at = time.monotonic()
            while not playback_session_end.done.wait(timeout=wait_timeout):
                thread_alive = playback_thread.is_alive() if playback_thread is not None else False
                logger.warning(
                    "{} 播放收尾等待中: playback_thread_alive={} queue_backlog={}",
                    segment_label,
                    thread_alive,
                    self._safe_queue_size(playback_queue),
                )
                if not thread_alive:
                    playback_session_end.error = self._playback_worker_error or RuntimeError("共享播放线程已退出。")
                    playback_session_end.done.set()
                    break
                if (self._stop_event.is_set() or interrupted) and (time.monotonic() - stop_wait_started_at) >= max_stop_wait_sec:
                    logger.info("{} 在停止/切模型请求下跳过播放收尾等待，立即结束当前任务。", segment_label)
                    break
            elapsed_ms = (time.monotonic() - synth_started_at) * 1000.0
            logger.info(
                "{} TTS 统计: chunks={} first_chunk_ms={} max_chunk_gap_ms={:.1f} total_audio_ms={:.1f} "
                "playback_peak={} max_put_wait_ms={:.1f} total_ms={:.1f}",
                segment_label,
                chunk_count,
                f"{first_chunk_latency_ms:.1f}" if first_chunk_latency_ms is not None else "n/a",
                max_chunk_gap_ms,
                total_audio_ms,
                playback_queue_peak,
                playback_queue_put_wait_ms_max,
                elapsed_ms,
            )
            self._last_tts_latency_ms = first_chunk_latency_ms or 0.0

        if playback_session_end.error is not None:
            raise playback_session_end.error
        if (
            message_id
            and audio_chunks
            and not self._stop_event.is_set()
            and (switch_revision is None or not self._is_segment_switch_interrupted(switch_revision))
        ):
            try:
                self._cache_audio_file(message_id, np.concatenate(audio_chunks, axis=0), sample_rate, prefer_ogg=True)
            except Exception as exc:
                logger.warning("写入 TTS 音频缓存失败: message_id={} error={}", message_id, exc)
        if progressive_transcript:
            self._emit_transcript(transcript)
        return sample_rate

    def _synthesize_meanvc2(
        self,
        waveform: np.ndarray,
        segment_label: str,
        *,
        message_id: str | None = None,
        switch_revision: int | None = None,
    ) -> int:
        meanvc2_service = self._ensure_meanvc2_loaded(emit_status=True)
        self._emit_status(f"{segment_label}进入 MeanVC2 实时变声...")
        self._last_tts_latency_ms = 0.0
        synth_started_at = time.monotonic()
        converted_audio, sample_rate = meanvc2_service.convert_waveform(
            waveform,
            ASR_SAMPLE_RATE,
            reference_audio_path=self.reference_audio_path,
        )
        self._mark_model_used("tts")
        if switch_revision is not None and self._is_segment_switch_interrupted(switch_revision):
            self._emit_status(f"{segment_label}已终止旧模型输出，立即切到 {self._voice_output_model_label()}。")
            return sample_rate
        if converted_audio.size == 0:
            logger.warning("{} MeanVC2 未生成可播放音频。", segment_label)
            return sample_rate
        self.waveform.emit(converted_audio)

        self._ensure_playback_worker()
        playback_queue = self._playback_queue
        playback_thread = self._playback_thread
        playback_session_end = _PlaybackSessionEnd(
            label=segment_label,
            started_at=synth_started_at,
            done=threading.Event(),
        )
        self._put_playback_item(
            playback_queue,
            _PlaybackSessionStart(
                label=segment_label,
                started_at=synth_started_at,
            ),
            segment_label,
        )
        self._put_playback_item(
            playback_queue,
            _QueuedPlaybackChunk(
                audio_np=converted_audio,
                sample_rate=sample_rate,
                chunk_index=1,
                enqueued_at=time.monotonic(),
            ),
            segment_label,
        )
        self._put_playback_item(playback_queue, playback_session_end, segment_label)

        wait_timeout = 0.3 if self._stop_event.is_set() else 2.0
        while not playback_session_end.done.wait(timeout=wait_timeout):
            thread_alive = playback_thread.is_alive() if playback_thread is not None else False
            logger.warning(
                "{} MeanVC2 播放收尾等待中: playback_thread_alive={} queue_backlog={}",
                segment_label,
                thread_alive,
                self._safe_queue_size(playback_queue),
            )
            if not thread_alive:
                playback_session_end.error = self._playback_worker_error or RuntimeError("共享播放线程已退出。")
                playback_session_end.done.set()
                break

        elapsed_ms = (time.monotonic() - synth_started_at) * 1000.0
        logger.info(
            "{} MeanVC2 统计: audio_ms={:.1f} total_ms={:.1f}",
            segment_label,
            self._audio_duration_ms(converted_audio, sample_rate),
            elapsed_ms,
        )
        self._last_tts_latency_ms = elapsed_ms
        if playback_session_end.error is not None:
            raise playback_session_end.error
        if message_id and converted_audio.size > 0 and not self._stop_event.is_set():
            self._cache_audio_file(message_id, converted_audio, sample_rate, prefer_ogg=True)
        return sample_rate

    @Slot()
    def shutdown(self) -> None:
        self._stop_event.set()
        self._segment_flush_requested.set()
        self._stop_system_loopback()
        with self._playback_thread_lock:
            if self._playback_thread is not None and self._playback_thread.is_alive():
                self._playback_queue.put(_PLAYBACK_SENTINEL)
                self._playback_thread.join(timeout=1.0)
        self._model_manager.shutdown()
        self._models_ready = False

    # ------------------------------------------------------------------ #
    #  system audio loopback
    # ------------------------------------------------------------------ #

    def toggle_system_loopback(self, enabled: bool) -> None:
        if enabled:
            self._start_system_loopback()
        else:
            self._stop_system_loopback()

    def _start_system_loopback(self) -> None:
        with self._loopback_lock:
            self._loopback_active = False
        self._emit_status("Qt 音频后端暂不支持系统音频回采，已自动禁用该功能。")

    def _stop_system_loopback(self) -> None:
        with self._loopback_lock:
            self._loopback_active = False
        self._loopback_stop.set()
        self._loopback_thread = None

    def _run_loopback(self) -> None:
        with self._loopback_lock:
            self._loopback_active = False

    @Slot()
    def unload_asr(self) -> None:
        self._unload_asr()

    @Slot()
    def unload_emotion(self) -> None:
        self._unload_emotion()

    @Slot()
    def reload_asr(self) -> None:
        if self._model_manager.asr_service is not None:
            return
        try:
            self._ensure_asr_loaded(emit_status=True)
            self._emit_status("ASR 模型已重新加载。")
        except Exception as exc:
            logger.exception("重新加载 ASR 模型失败")
            self.error.emit(str(exc))

    def _unload_asr(self) -> None:
        self._model_manager.unload_asr()

    def _unload_vad(self) -> None:
        self._model_manager.unload_vad()

    def _unload_tts(self) -> None:
        self._model_manager.unload_tts()

    def _unload_speaker(self) -> None:
        self._model_manager.unload_speaker()

    def _unload_emotion(self) -> None:
        self._model_manager.unload_emotion()

class ModelDownloadWorker(QThread):
    progress = Signal(object)
    status = Signal(str)
    finished = Signal(str, bool, str)

    def __init__(self, target_repo: str, parent=None) -> None:
        super().__init__(parent)
        self.target_repo = str(target_repo or "").strip()
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            from app.core.model_setup import ensure_downloadable_model
            if not self.target_repo:
                raise ValueError("模型仓库不能为空。")

            self.status.emit(f"正在下载 {self.target_repo} ...")
            self.progress.emit(
                {
                    "value": 0,
                    "maximum": 100,
                    "indeterminate": True,
                    "detail": f"正在准备下载 {self.target_repo} ...",
                }
            )

            def _on_status(message: str) -> None:
                if self._stop_event.is_set():
                    raise InterruptedError("下载已取消")
                self.status.emit(message)

            def _on_progress(payload: dict[str, object]) -> None:
                if self._stop_event.is_set():
                    raise InterruptedError("下载已取消")
                self.progress.emit(dict(payload))

            ensure_downloadable_model(
                self.target_repo,
                status_callback=_on_status,
                progress_callback=_on_progress,
            )
            self.progress.emit(
                {
                    "value": 100,
                    "maximum": 100,
                    "indeterminate": False,
                    "detail": f"{self.target_repo} 下载完成。",
                }
            )
            self.finished.emit(self.target_repo, True, f"{self.target_repo} 下载完成。")
        except InterruptedError:
            self.finished.emit(self.target_repo, False, "下载已取消。")
        except Exception as exc:
            self.finished.emit(self.target_repo, False, str(exc))

    def stop(self) -> None:
        self._stop_event.set()
