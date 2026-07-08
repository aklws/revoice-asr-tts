from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget


WAVEFORM_THEME_COLORS: dict[str, dict[str, str]] = {
    "dark": {
        "background": "#0e1012",
        "midline": "#3b4148",
        "placeholder": "#7b8088",
        "fill": "#b6bcc4",
        "stroke": "#d9dde2",
    },
    "light": {
        "background": "#f5f8fd",
        "midline": "#bfd0e6",
        "placeholder": "#7e8da3",
        "fill": "#87aeea",
        "stroke": "#2e6cdf",
    },
}


class WaveformWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._samples = np.zeros(0, dtype=np.float32)
        self._max_samples = 48_000 * 20
        self._theme = "dark"
        self._background_cache = QPixmap()
        self._waveform_cache_valid = False
        self._cached_upper_path = QPainterPath()
        self._cached_lower_path = QPainterPath()
        self._cached_fill_path = QPainterPath()
        self._update_pending = False
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(33)
        self._update_timer.timeout.connect(self._flush_update)
        self.setMinimumHeight(180)

    def clear_waveform(self) -> None:
        self._samples = np.zeros(0, dtype=np.float32)
        self._invalidate_waveform_cache()
        self._schedule_update(force=True)

    def set_theme(self, theme_name: str) -> None:
        resolved_theme = theme_name if theme_name in WAVEFORM_THEME_COLORS else "dark"
        if self._theme != resolved_theme:
            self._theme = resolved_theme
            self._invalidate_background_cache()
        self._schedule_update(force=True)

    def append_audio_chunk(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        if audio.ndim == 2:
            mono = audio.mean(axis=1, dtype=np.float32)
        else:
            mono = audio.astype(np.float32, copy=False)
        # Remove DC offset so the visualization stays centered on the zero line.
        mono = mono - np.mean(mono, dtype=np.float32)
        mono = np.clip(mono, -1.0, 1.0)
        if self._samples.size == 0:
            self._samples = mono[-self._max_samples :]
        else:
            self._samples = np.concatenate([self._samples, mono])[-self._max_samples :]
        self._invalidate_waveform_cache()
        self._schedule_update()

    def _schedule_update(self, force: bool = False) -> None:
        self._update_pending = True
        if force:
            self._update_timer.stop()
            self._flush_update()
            return
        if not self._update_timer.isActive():
            self._update_timer.start()

    def _flush_update(self) -> None:
        if not self._update_pending:
            return
        self._update_pending = False
        self.update()

    def _invalidate_background_cache(self) -> None:
        self._background_cache = QPixmap()

    def _invalidate_waveform_cache(self) -> None:
        self._waveform_cache_valid = False
        self._cached_upper_path = QPainterPath()
        self._cached_lower_path = QPainterPath()
        self._cached_fill_path = QPainterPath()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        self._invalidate_background_cache()
        self._invalidate_waveform_cache()
        super().resizeEvent(event)

    def _ensure_background_cache(self) -> None:
        rect = self.rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        if (
            not self._background_cache.isNull()
            and self._background_cache.size() == rect.size()
        ):
            return

        colors = WAVEFORM_THEME_COLORS.get(self._theme, WAVEFORM_THEME_COLORS["dark"])
        background = QPixmap(rect.size())
        background.fill(QColor(colors["background"]))

        painter = QPainter(background)
        painter.setRenderHint(QPainter.Antialiasing, True)
        center_y = rect.center().y()
        painter.setPen(QPen(QColor(colors["midline"]), 1, Qt.DashLine))
        painter.drawLine(rect.left() + 12, center_y, rect.right() - 12, center_y)
        painter.end()
        self._background_cache = background

    def _ensure_waveform_cache(self) -> None:
        if self._waveform_cache_valid:
            return

        self._cached_upper_path = QPainterPath()
        self._cached_lower_path = QPainterPath()
        self._cached_fill_path = QPainterPath()

        if self._samples.size == 0:
            self._waveform_cache_valid = True
            return

        rect = self.rect()
        waveform_width = max(1, rect.width() - 24)
        start_x = float(rect.left() + 12)
        center_y = rect.center().y()
        upper, lower = self._build_envelope(self._samples, waveform_width)
        if upper.size == 0:
            self._waveform_cache_valid = True
            return

        magnitude = np.maximum(np.abs(upper), np.abs(lower)).astype(np.float32, copy=False)
        amplitude = max(float(np.max(magnitude)), 1e-3)
        scale = (rect.height() * 0.42) / amplitude
        step = (waveform_width - 1) / max(1, len(upper) - 1)

        upper_path = QPainterPath()
        upper_path.moveTo(QPointF(start_x, center_y - float(magnitude[0]) * scale))
        for index, sample in enumerate(magnitude[1:], start=1):
            x = start_x + index * step
            y = center_y - float(sample) * scale
            upper_path.lineTo(QPointF(x, y))

        lower_path = QPainterPath()
        lower_path.moveTo(QPointF(start_x + (len(magnitude) - 1) * step, center_y + float(magnitude[-1]) * scale))
        for reverse_index, sample in enumerate(magnitude[-2::-1], start=1):
            x = start_x + (len(magnitude) - 1 - reverse_index) * step
            y = center_y + float(sample) * scale
            lower_path.lineTo(QPointF(x, y))

        fill_path = QPainterPath(upper_path)
        fill_path.connectPath(lower_path)
        fill_path.closeSubpath()

        self._cached_upper_path = upper_path
        self._cached_lower_path = lower_path
        self._cached_fill_path = fill_path
        self._waveform_cache_valid = True

    @staticmethod
    def _build_envelope(samples: np.ndarray, columns: int) -> tuple[np.ndarray, np.ndarray]:
        if samples.size == 0:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty

        columns = max(1, min(columns, samples.size))
        if samples.size <= columns:
            values = samples.astype(np.float32, copy=False)
            return values, values

        trimmed_size = (samples.size // columns) * columns
        if trimmed_size <= 0:
            values = samples[:columns].astype(np.float32, copy=False)
            return values, values

        trimmed = samples[-trimmed_size:].reshape(columns, -1)
        upper = trimmed.max(axis=1).astype(np.float32, copy=False)
        lower = trimmed.min(axis=1).astype(np.float32, copy=False)
        return upper, lower

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        self._ensure_background_cache()
        self._ensure_waveform_cache()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect = self.rect()
        colors = WAVEFORM_THEME_COLORS.get(self._theme, WAVEFORM_THEME_COLORS["dark"])
        if self._background_cache.isNull():
            painter.fillRect(rect, QColor(colors["background"]))
        else:
            painter.drawPixmap(rect.topLeft(), self._background_cache)

        center_y = rect.center().y()

        if self._samples.size == 0:
            painter.setPen(QColor(colors["placeholder"]))
            painter.drawText(rect, Qt.AlignCenter, "等待 TTS 生成波形...")
            return

        if self._cached_upper_path.isEmpty() or self._cached_lower_path.isEmpty() or self._cached_fill_path.isEmpty():
            return

        painter.fillPath(self._cached_fill_path, QColor(colors["fill"]))
        painter.setPen(QPen(QColor(colors["stroke"]), 1.2))
        painter.drawPath(self._cached_upper_path)
        painter.drawPath(self._cached_lower_path)
