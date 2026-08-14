from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

PROCESS_AUDIO_SAMPLE_RATE = 48_000
PROCESS_AUDIO_CHANNELS = 2


@dataclass(frozen=True, slots=True)
class RunningAudioProgram:
    pid: int
    name: str
    window_title: str
    display_label: str


def _build_program_display_label(name: str, window_title: str) -> str:
    normalized_name = str(name or "").strip() or "未知程序"
    normalized_title = str(window_title or "").strip()
    if normalized_title and normalized_title.casefold() not in {"thumbnailwindow", normalized_name.casefold()}:
        return f"{normalized_name} - {normalized_title}"
    return normalized_name


def is_process_audio_capture_supported() -> bool:
    try:
        from process_audio_capture import ProcessAudioCapture as _EnumerationCapture
        from proctap import ProcessAudioCapture as _RealtimeCapture
    except Exception as exc:
        logger.warning("进程音频监听依赖不可用: {}", exc)
        return False

    try:
        if not _EnumerationCapture.is_supported():
            return False
        _ = _RealtimeCapture
        return True
    except Exception as exc:
        logger.warning("检测进程音频监听能力失败: {}", exc)
        return False


def list_running_audio_programs() -> tuple[RunningAudioProgram, ...]:
    from process_audio_capture import ProcessAudioCapture

    sessions = ProcessAudioCapture.enumerate_audio_processes()
    by_pid: dict[int, RunningAudioProgram] = {}
    duplicate_counts: dict[str, int] = {}

    for session in sessions:
        pid = int(getattr(session, "pid", 0) or 0)
        if pid <= 0:
            continue
        name = str(getattr(session, "name", "") or "").strip() or "未知程序"
        window_title = str(getattr(session, "window_title", "") or "").strip()
        display_label = _build_program_display_label(name, window_title)
        existing = by_pid.get(pid)
        if existing is None or (window_title and not existing.window_title):
            by_pid[pid] = RunningAudioProgram(
                pid=pid,
                name=name,
                window_title=window_title,
                display_label=display_label,
            )
        duplicate_counts[display_label] = duplicate_counts.get(display_label, 0) + 1

    normalized: list[RunningAudioProgram] = []
    for program in by_pid.values():
        label = program.display_label
        if duplicate_counts.get(label, 0) > 1:
            label = f"{label} (PID {program.pid})"
        normalized.append(
            RunningAudioProgram(
                pid=program.pid,
                name=program.name,
                window_title=program.window_title,
                display_label=label,
            )
        )

    normalized.sort(key=lambda item: (item.name.casefold(), item.window_title.casefold(), item.pid))
    return tuple(normalized)


class ProcessAudioStream:
    def __init__(self, pid: int) -> None:
        from proctap import ProcessAudioCapture

        self.pid = int(pid)
        self.sample_rate = PROCESS_AUDIO_SAMPLE_RATE
        self.channels = PROCESS_AUDIO_CHANNELS
        self._capture = ProcessAudioCapture(self.pid)

    def start(self) -> None:
        self._capture.start()

    def stop(self) -> None:
        self._capture.stop()

    def close(self) -> None:
        try:
            self.stop()
        finally:
            close = getattr(self._capture, "close", None)
            if callable(close):
                close()

    def read_mono_float32(self, *, timeout: float = 0.5) -> np.ndarray | None:
        chunk = self._capture.read(timeout=timeout)
        if not chunk:
            return None
        pcm = np.frombuffer(chunk, dtype=np.float32)
        if pcm.size < self.channels:
            return None
        frame_count = pcm.size // self.channels
        if frame_count <= 0:
            return None
        trimmed = pcm[: frame_count * self.channels].reshape(frame_count, self.channels)
        mono = trimmed.mean(axis=1, dtype=np.float32)
        return np.ascontiguousarray(mono, dtype=np.float32)
