from __future__ import annotations

import threading
import time
from queue import Empty, Queue

import librosa
import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, Signal
from PySide6.QtMultimedia import QAudioDevice, QAudioFormat, QMediaDevices

from app import ASRService, get_settings
from app.core.audio import resample_waveform
from app.core.logging import get_logger
from app.models.asr import ASRRequest
from app.services.emotion_recognition import EmotionPrediction, EmotionRecognitionService
from app.services.index_tts import IndexTTSService


logger = get_logger(__name__)
ASR_SAMPLE_RATE = 16_000
MIC_CHANNELS = 1
MIC_BLOCK_SEC = 0.25
MIC_SILENCE_THRESHOLD = 0.012
MIC_SEGMENT_SILENCE_SEC = 0.22
MIC_SHORT_SEGMENT_SEC = 1.0
MIC_SHORT_SEGMENT_SILENCE_SEC = 0.38
MIC_SESSION_END_SILENCE_SEC = 0.9
MIC_PREROLL_SEC = 0.5
MIC_MIN_SEGMENT_SEC = 0.2
MIC_ASR_TRIM_BLOCK_SEC = 0.03
MIC_ASR_TRIM_PAD_SEC = 0.08
MIC_MIN_VOICED_SEC = 0.16
MIC_MIN_VOICED_RATIO = 0.22
MIC_ASR_CONTEXT_MAX_CHARS = 80
VOICEPRINT_WINDOW_SEC = 0.8
VOICEPRINT_HOP_SEC = 0.24
VOICEPRINT_PAD_SEC = 0.08
VOICEPRINT_MIN_KEEP_SEC = 0.35
VOICEPRINT_PEAK_RELAX = 0.12
VOICEPRINT_NEAR_PEAK_MARGIN = 0.06
VOICEPRINT_TRIM_BLOCK_SEC = 0.03
VOICEPRINT_TRIM_PAD_SEC = 0.09
EMOTION_MIN_SEGMENT_SEC = 0.6
EMOTION_MIN_CONFIDENCE = 0.45
_PLAYBACK_SENTINEL = object()


def normalize_asr_transcript(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return normalized
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


def trim_waveform_for_asr(waveform: np.ndarray) -> tuple[np.ndarray | None, str]:
    if waveform.size <= 0:
        return None, "当前片段为空，已跳过。"

    block_size = max(1, int(ASR_SAMPLE_RATE * MIC_ASR_TRIM_BLOCK_SEC))
    pad_size = max(0, int(ASR_SAMPLE_RATE * MIC_ASR_TRIM_PAD_SEC))
    energy_threshold = MIC_SILENCE_THRESHOLD

    if waveform.shape[0] <= block_size:
        rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
        if rms < energy_threshold:
            return None, "当前片段能量过低，已按静音跳过。"
        return waveform.astype(np.float32, copy=False), ""

    frame_ranges: list[tuple[int, int]] = []
    voiced_flags: list[bool] = []
    voiced_samples = 0
    for frame_start in range(0, int(waveform.shape[0]), block_size):
        frame_end = min(int(waveform.shape[0]), frame_start + block_size)
        frame = waveform[frame_start:frame_end]
        rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
        is_voiced = rms >= energy_threshold
        frame_ranges.append((frame_start, frame_end))
        voiced_flags.append(is_voiced)
        if is_voiced:
            voiced_samples += frame_end - frame_start

    try:
        first_voiced = voiced_flags.index(True)
        last_voiced = len(voiced_flags) - 1 - voiced_flags[::-1].index(True)
    except ValueError:
        return None, "当前片段未检测到有效语音，已跳过。"

    trimmed_start = max(0, frame_ranges[first_voiced][0] - pad_size)
    trimmed_end = min(int(waveform.shape[0]), frame_ranges[last_voiced][1] + pad_size)
    if trimmed_end <= trimmed_start:
        return None, "当前片段裁剪后无有效语音，已跳过。"

    trimmed = waveform[trimmed_start:trimmed_end].astype(np.float32, copy=False)
    voiced_sec = voiced_samples / ASR_SAMPLE_RATE
    voiced_ratio = voiced_samples / max(1, int(trimmed.shape[0]))
    if voiced_sec < MIC_MIN_VOICED_SEC or voiced_ratio < MIC_MIN_VOICED_RATIO:
        return None, "当前片段有效语音过少，已按静音跳过。"
    return trimmed, ""


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


def adjust_audio_speed(audio_np: np.ndarray, speed: float) -> np.ndarray:
    if audio_np.size == 0 or abs(speed - 1.0) < 1e-6:
        return audio_np.astype(np.float32, copy=False)
    if speed <= 0:
        raise ValueError("语速倍率必须大于 0。")
    if audio_np.ndim == 1:
        stretched = librosa.effects.time_stretch(audio_np.astype(np.float32, copy=False), rate=speed)
        return stretched.astype(np.float32, copy=False)

    channel_count = int(audio_np.shape[1])
    stretched_channels: list[np.ndarray] = []
    min_length = 0
    for channel_index in range(channel_count):
        stretched = librosa.effects.time_stretch(audio_np[:, channel_index].astype(np.float32, copy=False), rate=speed)
        stretched_f32 = stretched.astype(np.float32, copy=False)
        stretched_channels.append(stretched_f32)
        current_length = int(stretched_f32.shape[0])
        min_length = current_length if channel_index == 0 else min(min_length, current_length)

    if min_length <= 0:
        return np.empty((0, channel_count), dtype=np.float32)
    output = np.empty((min_length, channel_count), dtype=np.float32)
    for channel_index, channel in enumerate(stretched_channels):
        output[:, channel_index] = channel[:min_length]
    return output


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
    emotion_state = Signal(object)
    waveform = Signal(object)
    transcript = Signal(str)

    def __init__(self, *, initial_reference_audio_path: str) -> None:
        super().__init__()
        self.reference_audio_path = initial_reference_audio_path
        self.language: str | None = "Chinese"
        self.instruction: str | None = None
        self.max_new_tokens = 2048
        self.max_record_sec = 20.0
        self.input_device: str | None = None
        self.input_device_label: str | None = None
        self.output_device: int | None = None
        self.output_device_label: str | None = None
        self.monitor_output_device: int | None = None
        self.monitor_output_device_label: str | None = None
        self.speech_rate: float = 1.0
        self._settings = get_settings()
        self.index_emo_vector: list[float] | None = None
        self.index_emo_alpha: float = 1.0
        self._loopback_thread: threading.Thread | None = None
        self._loopback_stop = threading.Event()
        self._loopback_lock = threading.Lock()
        self._model_lock = threading.RLock()
        self._loopback_active = False
        self.asr_service: ASRService | None = None
        self.index_tts_service: IndexTTSService | None = None
        self._asr_last_used_at = time.monotonic()
        self._tts_last_used_at = time.monotonic()
        self._asr_idle_generation = 0
        self._tts_idle_generation = 0
        self._tts_warm_reference_audio_path: str | None = None
        self._models_ready = False
        self._busy = False
        self._stop_event = threading.Event()
        self.emotion_service: EmotionRecognitionService | None = None
        self.auto_emotion_enabled = False
        self.auto_emotion_strength = 0.75
        self.user_voiceprint: np.ndarray | None = None
        self.voiceprint_threshold: float = 0.65
        self._last_status_text: str | None = None
        self._last_transcript_text: str | None = None
        self._last_emotion_payload: object = None
        self._last_tts_transcript_key: str | None = None

    def _reset_emit_cache(self) -> None:
        self._last_status_text = None
        self._last_transcript_text = None
        self._last_emotion_payload = None
        self._last_tts_transcript_key = None

    def _emit_status(self, message: str) -> None:
        if self._last_status_text == message:
            return
        self._last_status_text = message
        self.status.emit(message)

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
        if self.user_voiceprint is None or self.user_voiceprint.size == 0 or self.index_tts_service is None:
            return waveform, ""

        sample_count = int(waveform.shape[0])
        if sample_count <= 0:
            return None, "声纹过滤后无有效语音。"

        window_size = max(1, int(ASR_SAMPLE_RATE * VOICEPRINT_WINDOW_SEC))
        hop_size = max(1, int(ASR_SAMPLE_RATE * VOICEPRINT_HOP_SEC))
        pad_size = max(0, int(ASR_SAMPLE_RATE * VOICEPRINT_PAD_SEC))
        min_keep_samples = max(1, int(ASR_SAMPLE_RATE * VOICEPRINT_MIN_KEEP_SEC))
        energy_gate = MIC_SILENCE_THRESHOLD * 1.15
        trim_block_size = max(1, int(ASR_SAMPLE_RATE * VOICEPRINT_TRIM_BLOCK_SEC))
        trim_pad_size = max(0, int(ASR_SAMPLE_RATE * VOICEPRINT_TRIM_PAD_SEC))

        candidate_ranges: list[tuple[int, int]] = []

        def add_candidate_range(start: int, end: int) -> None:
            chunk = waveform[start:end]
            if chunk.size <= 0:
                return
            rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
            if rms < energy_gate:
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
        embeddings = self.index_tts_service.extract_voiceprints_batch(candidate_chunks, ASR_SAMPLE_RATE)
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

        cropped_waveform = waveform[crop_start:crop_end].astype(np.float32, copy=False)
        if cropped_waveform.shape[0] < min_keep_samples:
            return None, "声纹保留语音过短，已忽略当前片段。"

        def trim_by_vad(audio: np.ndarray) -> tuple[int, int]:
            if audio.size <= trim_block_size:
                return 0, int(audio.shape[0])
            frame_ranges: list[tuple[int, int]] = []
            voiced_flags: list[bool] = []
            for frame_start in range(0, int(audio.shape[0]), trim_block_size):
                frame_end = min(int(audio.shape[0]), frame_start + trim_block_size)
                frame = audio[frame_start:frame_end]
                rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
                frame_ranges.append((frame_start, frame_end))
                voiced_flags.append(rms >= MIC_SILENCE_THRESHOLD)
            try:
                first_voiced = voiced_flags.index(True)
                last_voiced = len(voiced_flags) - 1 - voiced_flags[::-1].index(True)
            except ValueError:
                return 0, int(audio.shape[0])
            start = max(0, frame_ranges[first_voiced][0] - trim_pad_size)
            end = min(int(audio.shape[0]), frame_ranges[last_voiced][1] + trim_pad_size)
            return start, end

        trimmed_start, trimmed_end = trim_by_vad(cropped_waveform)
        filtered_waveform = cropped_waveform[trimmed_start:trimmed_end].astype(np.float32, copy=False)
        kept_samples = int(filtered_waveform.shape[0])
        if kept_samples < min_keep_samples:
            return None, "首尾裁剪后目标语音过短，已忽略当前片段。"

        kept_sec = kept_samples / ASR_SAMPLE_RATE
        original_sec = sample_count / ASR_SAMPLE_RATE
        return filtered_waveform, (
            f"声纹过滤保留连续语音 {kept_sec:.2f}s / {original_sec:.2f}s，峰值相似度 {peak_similarity:.2f}。"
        )

    def _mark_model_used(self, model_name: str) -> None:
        now = time.monotonic()
        with self._model_lock:
            if model_name == "asr":
                self._asr_last_used_at = now
                self._asr_idle_generation += 1
            else:
                self._tts_last_used_at = now
                self._tts_idle_generation += 1

    def _schedule_idle_unload(self, model_name: str) -> None:
        if model_name == "asr":
            timeout = float(self._settings.live_asr_idle_unload_sec)
            generation_attr = "_asr_idle_generation"
            last_used_attr = "_asr_last_used_at"
        else:
            timeout = float(self._settings.live_tts_idle_unload_sec)
            generation_attr = "_tts_idle_generation"
            last_used_attr = "_tts_last_used_at"
        if timeout <= 0:
            return

        with self._model_lock:
            generation = getattr(self, generation_attr)

        def unload_when_idle() -> None:
            time.sleep(timeout)
            with self._model_lock:
                if getattr(self, generation_attr) != generation or self._busy:
                    return
                last_used_at = float(getattr(self, last_used_attr))
                if (time.monotonic() - last_used_at) < timeout:
                    return
            if model_name == "asr":
                logger.info("ASR 模型空闲超时，已自动卸载以节省内存。")
                self._unload_asr()
            else:
                logger.info("TTS 模型空闲超时，已自动卸载以节省内存。")
                self._unload_tts()

        threading.Thread(target=unload_when_idle, daemon=True, name=f"idle-unload-{model_name}").start()

    def _ensure_asr_loaded(self, *, emit_status: bool = False) -> None:
        if self.asr_service is None:
            with self._model_lock:
                if self.asr_service is None:
                    if emit_status:
                        self._emit_status("正在初始化语音识别引擎...")
                    asr_service = ASRService()
                    asr_service.load()
                    self.asr_service = asr_service
        self._mark_model_used("asr")

    def _ensure_tts_loaded(self, *, emit_status: bool = False, warmup_reference: bool = True) -> None:
        if self.index_tts_service is None:
            with self._model_lock:
                if self.index_tts_service is None:
                    if emit_status:
                        self._emit_status("正在初始化语音合成引擎...")
                    tts_service = IndexTTSService(self._settings.index_tts_model_dir)
                    tts_service.load()
                    self.index_tts_service = tts_service
                    self._tts_warm_reference_audio_path = None
        self._mark_model_used("tts")
        if warmup_reference:
            self._warmup_tts_reference_if_needed(emit_status=emit_status)

    def _ensure_emotion_loaded(self, *, emit_status: bool = False) -> None:
        if self.emotion_service is None:
            with self._model_lock:
                if self.emotion_service is None:
                    if emit_status:
                        self._emit_status("正在初始化实验性情感识别...")
                    emotion_service = EmotionRecognitionService(self._settings.emotion_model_dir)
                    emotion_service.load(status_callback=self._emit_status if emit_status else None)
                    self.emotion_service = emotion_service

    def _detect_segment_emotion(self, waveform: np.ndarray) -> tuple[list[float] | None, float, EmotionPrediction | None]:
        if not self.auto_emotion_enabled:
            return self.index_emo_vector, self.index_emo_alpha, None
        duration_sec = float(waveform.shape[0]) / float(ASR_SAMPLE_RATE)
        if duration_sec < EMOTION_MIN_SEGMENT_SEC:
            self._emit_emotion_state(
                {"state": "idle", "summary": "当前片段过短，沿用手动情感。"}
            )
            return self.index_emo_vector, self.index_emo_alpha, None

        self._ensure_emotion_loaded(emit_status=True)
        if self.emotion_service is None:
            return self.index_emo_vector, self.index_emo_alpha, None

        prediction = self.emotion_service.predict_waveform(waveform, ASR_SAMPLE_RATE)
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
        with self._model_lock:
            if self.index_tts_service is None or not self.reference_audio_path:
                return
            if not force and self._tts_warm_reference_audio_path == self.reference_audio_path:
                return
            # Mark the model as active before warmup so the idle-unload timer
            # cannot reclaim it mid-initialization.
            self._mark_model_used("tts")
            if emit_status:
                self._emit_status("正在准备参考音色特征（运行时预热）...")
            self.index_tts_service.warmup_speaker(self.reference_audio_path)
            self._tts_warm_reference_audio_path = self.reference_audio_path
            self._mark_model_used("tts")

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

    def preload_models(self, payload: object = None) -> None:
        reference_audio_path: str | None = None
        input_mode = "microphone"
        if isinstance(payload, dict):
            raw_reference_audio_path = payload.get("reference_audio_path")
            if isinstance(raw_reference_audio_path, str):
                reference_audio_path = raw_reference_audio_path.strip() or None
            raw_input_mode = payload.get("input_mode")
            if isinstance(raw_input_mode, str) and raw_input_mode.strip():
                input_mode = raw_input_mode.strip()
        elif isinstance(payload, str):
            reference_audio_path = payload.strip() or None

        if self._models_ready:
            if reference_audio_path and reference_audio_path != self.reference_audio_path:
                self.reference_audio_path = reference_audio_path
                self._ensure_tts_loaded(emit_status=True, warmup_reference=False)
                self._warmup_tts_reference_if_needed(emit_status=True, force=True)
                self._schedule_idle_unload("tts")
            self.ready.emit()
            return

        if reference_audio_path:
            self.reference_audio_path = reference_audio_path

        try:
            self._reset_emit_cache()
            if input_mode != "text":
                self._ensure_asr_loaded(emit_status=True)
            self._ensure_tts_loaded(emit_status=True, warmup_reference=True)
            self._models_ready = True
            if input_mode == "text":
                self._emit_status("文本模式引擎已就绪，可以开始合成。")
            else:
                self._emit_status("语音引擎已就绪，可以开始变声。")
            self.ready.emit()
            if self.asr_service is not None:
                self._schedule_idle_unload("asr")
            self._schedule_idle_unload("tts")
        except Exception as exc:
            logger.exception("预加载模型失败")
            self.error.emit(str(exc))

    def _record_microphone_segments(self, segment_queue: Queue[object]) -> None:
        input_device, capture_sample_rate = self._resolve_input_stream_settings()
        blocksize = int(capture_sample_rate * MIC_BLOCK_SEC)
        preroll_blocks = max(1, int(MIC_PREROLL_SEC / MIC_BLOCK_SEC))
        preroll: list[np.ndarray] = []
        current_segment: list[np.ndarray] = []
        speech_started = False
        emitted_segments = 0
        last_voice_time = time.monotonic()
        finished = False

        self._emit_status(f"开始录音，请说话... 当前采集采样率 {capture_sample_rate} Hz")
        start_time = time.monotonic()

        def flush_current_segment(*, discard: bool = False) -> None:
            nonlocal current_segment, speech_started, emitted_segments
            if not current_segment:
                speech_started = False
                return
            waveform = np.concatenate(current_segment, axis=0).astype(np.float32, copy=False)
            duration = waveform.shape[0] / capture_sample_rate
            current_segment = []
            speech_started = False
            if discard:
                self._emit_status("已停止麦克风采集，丢弃未完成语音段。")
                return
            if duration < MIC_MIN_SEGMENT_SEC:
                return
            if capture_sample_rate != ASR_SAMPLE_RATE:
                waveform = resample_waveform(waveform, capture_sample_rate, ASR_SAMPLE_RATE)
            emitted_segments += 1
            segment_queue.put(waveform)
            self._emit_status(f"检测到第 {emitted_segments} 段语音...")

        def process_block(block: np.ndarray) -> None:
            nonlocal speech_started, last_voice_time, finished, preroll, current_segment
            rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
            now = time.monotonic()

            if not speech_started:
                preroll.append(block)
                if len(preroll) > preroll_blocks:
                    preroll = preroll[-preroll_blocks:]
                if rms >= MIC_SILENCE_THRESHOLD:
                    speech_started = True
                    last_voice_time = now
                    current_segment = list(preroll)
                    preroll.clear()
                    current_segment.append(block)
                    self._emit_status("检测到语音，持续录制直到明显停顿...")
            else:
                current_segment.append(block)
                if rms >= MIC_SILENCE_THRESHOLD:
                    last_voice_time = now

            active_segment_duration = (
                sum(chunk.shape[0] for chunk in current_segment) / capture_sample_rate if current_segment else 0.0
            )

            if speech_started and (now - last_voice_time) >= MIC_SESSION_END_SILENCE_SEC:
                flush_current_segment()
            elif speech_started and active_segment_duration >= self.max_record_sec:
                self._emit_status(
                    f"单段语音已达到 {self.max_record_sec:.0f} 秒，已强制提交当前整段识别。"
                )
                flush_current_segment()
            
            if self._stop_event.is_set():
                if speech_started:
                    flush_current_segment(discard=True)
                finished = True

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
                        if speech_started:
                            flush_current_segment(discard=True)
                        finished = True
                        break
                    block, overflowed = stream.read(blocksize)
                    if overflowed:
                        logger.warning("麦克风输入发生溢出，当前块可能不完整。")
                    samples = np.asarray(block, dtype=np.float32).reshape(-1)
                    if samples.size:
                        process_block(samples)

            if current_segment and not self._stop_event.is_set():
                flush_current_segment()
            if self._stop_event.is_set():
                self._emit_status("麦克风任务已停止。")
                segment_queue.put(None)
                return
            if emitted_segments == 0:
                raise RuntimeError("没有检测到有效语音输入，请重试。")
            self._emit_status(f"录音阶段结束，共捕获 {emitted_segments} 段语音。")
            segment_queue.put(None)
        except Exception as exc:
            segment_queue.put(exc)
            segment_queue.put(None)

    @staticmethod
    def _resolve_output_audio(
        audio_np: np.ndarray, sample_rate: int, device: int | None
    ) -> tuple[np.ndarray, int, int]:
        channels = 1 if audio_np.ndim == 1 else int(audio_np.shape[1])
        candidates: list[int] = []
        try:
            device_info = sd.query_devices(device, "output")
        except Exception:
            device_info = None

        default_sample_rate = 0.0
        device_name = "系统默认输出" if device is None else f"输出设备 {device}"
        if isinstance(device_info, dict):
            default_sample_rate = float(device_info.get("default_samplerate") or 0.0)
            device_name = str(device_info.get("name") or device_name)
        default_rate_candidate = int(round(default_sample_rate)) if default_sample_rate > 0 else 0
        if default_rate_candidate > 48_000:
            default_rate_candidate = 0

        for rate in (
            int(sample_rate),
            48_000,
            44_100,
            32_000,
            24_000,
            default_rate_candidate,
        ):
            if rate > 0 and rate not in candidates:
                candidates.append(rate)

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                sd.check_output_settings(device=device, channels=channels, samplerate=candidate, dtype="float32")
                if candidate == sample_rate:
                    return audio_np.astype(np.float32, copy=False), candidate, channels
                converted = resample_audio_output(audio_np, sample_rate, candidate)
                logger.warning("播放设备不支持采样率 {} Hz，已自动重采样到 {} Hz 后输出: {}", sample_rate, candidate, device_name)
                return converted, candidate, channels
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise RuntimeError(f"当前播放设备不支持可用采样率: {device_name} | {last_error}") from last_error
        return audio_np.astype(np.float32, copy=False), sample_rate, channels

    @staticmethod
    def _play_audio_to_device(audio_np: np.ndarray, sample_rate: int, device: int | None) -> None:
        prepared_audio, prepared_rate, channels = LiveSpeechWorker._resolve_output_audio(audio_np, sample_rate, device)
        with sd.OutputStream(samplerate=prepared_rate, channels=channels, dtype="float32", device=device) as stream:
            stream.write(prepared_audio)

    def _play_audio_chunk(self, audio_np: np.ndarray, sample_rate: int) -> None:
        primary_device = self.output_device
        monitor_device = self.monitor_output_device

        if monitor_device is None or monitor_device == primary_device:
            self._play_audio_to_device(audio_np, sample_rate, primary_device)
            return

        primary_error: list[BaseException] = []
        monitor_error: list[BaseException] = []

        def play_primary() -> None:
            try:
                self._play_audio_to_device(audio_np, sample_rate, primary_device)
            except BaseException as exc:
                primary_error.append(exc)

        def play_monitor() -> None:
            try:
                self._play_audio_to_device(audio_np, sample_rate, monitor_device)
            except BaseException as exc:
                monitor_error.append(exc)

        primary_thread = threading.Thread(
            target=play_primary, daemon=True,
        )
        monitor_thread = threading.Thread(
            target=play_monitor, daemon=True,
        )
        primary_thread.start()
        monitor_thread.start()
        primary_thread.join()
        monitor_thread.join()
        if monitor_error:
            logger.warning("耳返设备播放失败，已自动忽略: {}", monitor_error[0])
        if primary_error:
            raise primary_error[0]

    def _playback_worker(
        self,
        playback_queue: Queue[object],
        playback_errors: Queue[BaseException],
    ) -> None:
        try:
            while True:
                queued = playback_queue.get()
                if queued is _PLAYBACK_SENTINEL:
                    return
                audio_np, sample_rate = queued
                self._play_audio_chunk(audio_np, sample_rate)
        except BaseException as exc:
            playback_errors.put(exc)

    def stop_live(self) -> None:
        if self._busy:
            self._emit_status("正在停止录音...")
            self._stop_event.set()

    def run_once(self, payload: dict[str, object]) -> None:
        if self._busy:
            self._emit_status("当前已有变声任务在执行，请稍候。")
            return
        if not self._models_ready:
            self.error.emit("语音合成引擎尚未完成初始化，请稍后再试。")
            return

        self._busy = True
        self._stop_event.clear()
        self._reset_emit_cache()
        try:
            input_mode = str(payload.get("input_mode") or "microphone")
            input_text = str(payload.get("input_text") or "").strip()
            self.reference_audio_path = str(payload["reference_audio_path"])
            self.language = str(payload.get("language") or "Chinese")
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
            self.output_device = payload.get("output_device") if isinstance(payload.get("output_device"), int) else None
            self.output_device_label = (
                str(payload.get("output_device_label")).strip()
                if isinstance(payload.get("output_device_label"), str) and str(payload.get("output_device_label")).strip()
                else None
            )
            self.monitor_output_device = (
                payload.get("monitor_output_device") if isinstance(payload.get("monitor_output_device"), int) else None
            )
            self.monitor_output_device_label = (
                str(payload.get("monitor_output_device_label")).strip()
                if isinstance(payload.get("monitor_output_device_label"), str)
                and str(payload.get("monitor_output_device_label")).strip()
                else None
            )
            self.speech_rate = float(payload.get("speech_rate") or 1.0)

            self._emit_status("使用 Index-TTS 引擎...")
            if self.auto_emotion_enabled and input_mode != "text":
                self._emit_emotion_state({"state": "busy", "summary": "实验性情感识别已开启，等待语音片段..."})
            else:
                self._emit_emotion_state({"state": "idle", "summary": "实验性情感识别未启用。"})
            self._emit_status(
                f"当前设备: 录音={self.input_device_label or '系统默认输入'} / "
                f"播放={self.output_device_label or '系统默认输出'} / "
                f"耳返={self.monitor_output_device_label or '关闭耳返'}"
            )
            if abs(self.speech_rate - 1.0) >= 1e-6:
                self._emit_status(f"当前语速倍率: {self.speech_rate:.2f}x")
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
                self._ensure_tts_loaded()
                self._emit_status("文本模式：跳过 ASR，直接开始流式 TTS。")
                self._synthesize(transcript, "文本", progressive_transcript=True)
            else:
                self._ensure_asr_loaded()
                segment_queue: Queue[object] = Queue(maxsize=4)
                recorder_thread = threading.Thread(target=self._record_microphone_segments, args=(segment_queue,), daemon=True)
                recorder_thread.start()

                transcript_parts: list[str] = []
                latest_transcript = ""
                asr_segment_index = 0
                last_asr_activity_at: float | None = None

                while True:
                    if self._stop_event.is_set():
                        break
                    try:
                        queued = segment_queue.get(timeout=0.1)
                    except Empty:
                        if not recorder_thread.is_alive():
                            continue
                        continue

                    if queued is None:
                        break
                    if isinstance(queued, Exception):
                        raise queued
                    waveform = queued
                    if not isinstance(waveform, np.ndarray):
                        continue
                    if self._stop_event.is_set():
                        break

                    now = time.monotonic()
                    if (
                        transcript_parts
                        and last_asr_activity_at is not None
                        and (now - last_asr_activity_at) >= MIC_SESSION_END_SILENCE_SEC
                    ):
                        transcript_parts.clear()
                        self._last_tts_transcript_key = None
                        self._emit_status("静默较久，已重置 ASR 上下文，避免重复补全上一句。")

                    # --- Voiceprint Filter Check ---
                    if self.user_voiceprint is not None and self.user_voiceprint.size > 0 and self.index_tts_service is not None:
                        self._emit_status("正在执行声纹过滤...")
                        try:
                            filtered_waveform, filter_message = self._filter_waveform_by_voiceprint(waveform)
                            if filter_message:
                                self._emit_status(filter_message)
                            if filtered_waveform is None:
                                continue
                            waveform = filtered_waveform
                        except Exception as e:
                            logger.error("声纹过滤失败: {}", e)

                    trimmed_waveform, trim_message = trim_waveform_for_asr(waveform)
                    if trim_message:
                        self._emit_status(trim_message)
                    if trimmed_waveform is None:
                        continue
                    waveform = trimmed_waveform
                    last_asr_activity_at = now

                    asr_segment_index += 1
                    segment_emo_vector = self.index_emo_vector
                    segment_emo_alpha = self.index_emo_alpha
                    if self.auto_emotion_enabled:
                        try:
                            segment_emo_vector, segment_emo_alpha, _prediction = self._detect_segment_emotion(waveform)
                        except Exception as exc:
                            logger.warning("实验性情感识别失败，已回退到手动情感: {}", exc)
                            self._emit_status(f"实验性情感识别失败，已回退手动情感: {exc}")
                            self._emit_emotion_state(
                                {"state": "error", "summary": "实验性情感识别失败，已回退手动情感。"}
                            )
                            self.auto_emotion_enabled = False
                    self._emit_status(f"正在识别第 {asr_segment_index} 段语音...")
                    asr_context = build_asr_context(transcript_parts)
                    self._mark_model_used("asr")
                    result = self.asr_service.transcribe(
                        ASRRequest(
                            audio_ndarray=(waveform, ASR_SAMPLE_RATE),
                            language=self.language or "Chinese",
                            return_language=True,
                            context=asr_context,
                        )
                    )

                    transcript = normalize_asr_transcript(result.text)
                    previous_transcript = transcript_parts[-1] if transcript_parts else None
                    transcript = merge_asr_segments(previous_transcript, transcript)
                    if not transcript:
                        self._emit_status(f"第 {asr_segment_index} 段识别为空，已跳过。")
                        continue
                    transcript_key = canonicalize_asr_text(transcript)
                    if transcript_key and transcript_key == self._last_tts_transcript_key:
                        logger.info("ASR 连续重复结果已跳过，不再发送 TTS: {}", transcript)
                        self._emit_status(f"第 {asr_segment_index} 段与上一句重复，已跳过。")
                        continue

                    transcript_parts.append(transcript)
                    latest_transcript = transcript
                    self._last_tts_transcript_key = transcript_key
                    self._emit_transcript(latest_transcript)
                    self._emit_status(f"第 {asr_segment_index} 段识别结果: {transcript}")
                    self._synthesize(
                        transcript, f"第 {asr_segment_index} 段",
                        emo_vector=segment_emo_vector,
                        emo_alpha=segment_emo_alpha,
                    )

                recorder_thread.join(timeout=1.0)
                if self._stop_event.is_set():
                    transcript = latest_transcript or "\n".join(transcript_parts).strip()
                    self._emit_status("麦克风任务已停止。")
                    self.finished.emit(transcript)
                    return
                final_transcript = "\n".join(transcript_parts).strip()
                if not final_transcript:
                    raise RuntimeError("ASR 结果为空，无法继续 TTS。")
                transcript = latest_transcript or final_transcript

            self.finished.emit(transcript)
        except Exception as exc:
            logger.exception("Live 页面执行失败")
            self.error.emit(str(exc))
        finally:
            self._busy = False
            if self.asr_service is not None:
                self._schedule_idle_unload("asr")
            if self.index_tts_service is not None:
                self._schedule_idle_unload("tts")

    def _synthesize(
        self,
        transcript: str,
        segment_label: str,
        *,
        progressive_transcript: bool = False,
        emo_vector: list[float] | None = None,
        emo_alpha: float | None = None,
    ) -> int:
        self._ensure_tts_loaded(warmup_reference=True)
        self._emit_status(f"{segment_label}进入 Index-TTS 流式合成...")
        active_emo_vector = self.index_emo_vector if emo_vector is None else emo_vector
        active_emo_alpha = self.index_emo_alpha if emo_alpha is None else emo_alpha
        sample_rate = 22050
        playback_queue: Queue[object] = Queue(maxsize=6)
        playback_errors: Queue[BaseException] = Queue()
        playback_thread = threading.Thread(
            target=self._playback_worker,
            args=(playback_queue, playback_errors),
            daemon=True,
            name="tts-playback",
        )
        playback_thread.start()

        try:
            for chunk_np, sr in self.index_tts_service.synthesize_stream(
                text=transcript,
                reference_audio_path=self.reference_audio_path,
                emo_vector=active_emo_vector,
                emo_alpha=active_emo_alpha,
            ):
                self._mark_model_used("tts")
                if not playback_errors.empty():
                    raise playback_errors.get()
                sample_rate = sr
                if abs(self.speech_rate - 1.0) >= 1e-6:
                    chunk_np = adjust_audio_speed(chunk_np, self.speech_rate)
                self.waveform.emit(chunk_np)
                playback_queue.put((chunk_np, sample_rate))
        finally:
            playback_queue.put(_PLAYBACK_SENTINEL)
            playback_thread.join()

        if not playback_errors.empty():
            raise playback_errors.get()
        if progressive_transcript:
            self._emit_transcript(transcript)
        return sample_rate

    def shutdown(self) -> None:
        self._stop_system_loopback()
        self._unload_asr()
        self._unload_tts()
        self._unload_emotion()
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

    def unload_asr(self) -> None:
        self._unload_asr()

    def unload_emotion(self) -> None:
        self._unload_emotion()

    def reload_asr(self) -> None:
        if self.asr_service is not None:
            return
        try:
            self._ensure_asr_loaded(emit_status=True)
            self._emit_status("ASR 模型已重新加载。")
        except Exception as exc:
            logger.exception("重新加载 ASR 模型失败")
            self.error.emit(str(exc))

    def _unload_asr(self) -> None:
        with self._model_lock:
            if self.asr_service is not None:
                self.asr_service.unload()
                self.asr_service = None
            self._asr_idle_generation += 1

    def _unload_tts(self) -> None:
        with self._model_lock:
            if self.index_tts_service is not None:
                self.index_tts_service.unload()
                self.index_tts_service = None
            self._tts_warm_reference_audio_path = None
            self._tts_idle_generation += 1

    def _unload_emotion(self) -> None:
        with self._model_lock:
            if self.emotion_service is not None:
                self.emotion_service.unload()
                self.emotion_service = None
