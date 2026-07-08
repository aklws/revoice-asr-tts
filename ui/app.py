from __future__ import annotations

import importlib
import multiprocessing
import sys
import threading
from functools import cache

from PySide6.QtCore import QObject, QLineF, QPointF, QRectF, QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QIcon, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QMessageBox, QProgressBar, QVBoxLayout

from app.core.logging import get_logger, setup_logging
from app.core.paths import get_bundle_file

from ui.main_window import APP_DISPLAY_NAME, MainWindow
from ui.theme import DEFAULT_THEME, get_theme_stylesheet
from ui.window_chrome import apply_window_chrome_theme

logger = get_logger(__name__)
APP_ICON_PATH = get_bundle_file("ui", "assets", "revoice_asr_tts.png")
PROGRESS_BAR_SCALE = 1000


@cache
def _load_ensure_models():
    module = importlib.import_module("app.core.model_setup")
    return module.ensure_models


@cache
def _get_startup_progress_dialog() -> StartupProgressDialog:
    dialog = StartupProgressDialog()
    if APP_ICON_PATH.exists():
        dialog.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    return dialog


@cache
def _get_startup_error_dialog() -> QMessageBox:
    dialog = QMessageBox()
    dialog.setIcon(QMessageBox.Critical)
    dialog.setWindowTitle("启动失败")
    dialog.setStandardButtons(QMessageBox.Ok)
    dialog.setModal(True)
    if APP_ICON_PATH.exists():
        dialog.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    apply_window_chrome_theme(dialog, DEFAULT_THEME)
    return dialog


class ModelPreparationWorker(QObject):
    status = Signal(str)
    progress = Signal(object)
    finished = Signal()
    failed = Signal(str)

    @Slot()
    def run(self) -> None:
        try:
            ensure_models = _load_ensure_models()
            ensure_models(
                status_callback=self.status.emit,
                progress_callback=self.progress.emit,
            )
            self.finished.emit()
        except Exception as exc:
            logger.exception("启动阶段的模型检查失败")
            self.failed.emit(str(exc))


class StartupProgressDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_DISPLAY_NAME} 启动中")
        self.setModal(True)
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setFixedSize(960, 540)
        self.setObjectName("StartupProgressDialog")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            """
QDialog#StartupProgressDialog {
    background: transparent;
}
QLabel#SplashTitle {
    color: #111827;
    font-size: 42px;
    font-weight: 800;
    letter-spacing: 0.5px;
}
QLabel#SplashSubtitle {
    color: #202939;
    font-size: 18px;
    font-weight: 500;
}
QLabel#SplashStatus {
    color: #374151;
    font-size: 16px;
    font-weight: 600;
}
QLabel#SplashDetail {
    color: #6b7280;
    font-size: 12px;
}
QLabel#SplashPhase {
    color: #374151;
    font-size: 14px;
    font-weight: 600;
}
QLabel#SplashSubPhase {
    color: #4b5563;
    font-size: 12px;
    font-weight: 600;
}
QProgressBar#SplashProgress {
    background: rgba(17, 24, 39, 0.08);
    border: none;
    border-radius: 3px;
}
QProgressBar#SplashProgress::chunk {
    border-radius: 3px;
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 #8ab8ff,
        stop: 1 #2563eb
    );
}
QProgressBar#SplashSubProgress {
    background: rgba(17, 24, 39, 0.06);
    border: none;
    border-radius: 2px;
}
QProgressBar#SplashSubProgress::chunk {
    border-radius: 2px;
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 #b8cfff,
        stop: 1 #5b8cff
    );
}
"""
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(72, 48, 72, 44)
        layout.setSpacing(10)
        layout.addStretch(6)

        title = QLabel("语音变声器")
        title.setObjectName("SplashTitle")
        title.setAlignment(Qt.AlignHCenter)

        subtitle = QLabel("新一代AI变声器")
        subtitle.setObjectName("SplashSubtitle")
        subtitle.setAlignment(Qt.AlignHCenter)

        self.status_label = QLabel("正在准备模型与运行环境")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignHCenter)
        self.status_label.setObjectName("SplashStatus")

        self.detail_label = QLabel("首次启动可能需要下载文件，请稍候。")
        self.detail_label.setWordWrap(True)
        self.detail_label.setAlignment(Qt.AlignHCenter)
        self.detail_label.setObjectName("SplashDetail")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(12)
        layout.addWidget(self.status_label)
        layout.addWidget(self.detail_label)
        layout.addStretch(5)

        self.phase_label = QLabel("准备进入变声器")
        self.phase_label.setObjectName("SplashPhase")
        self.phase_label.setAlignment(Qt.AlignHCenter)

        self.progress = QProgressBar()
        self.progress.setObjectName("SplashProgress")
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setRange(0, 0)

        self.sub_phase_label = QLabel("当前子任务")
        self.sub_phase_label.setObjectName("SplashSubPhase")
        self.sub_phase_label.setAlignment(Qt.AlignHCenter)

        self.sub_progress = QProgressBar()
        self.sub_progress.setObjectName("SplashSubProgress")
        self.sub_progress.setTextVisible(False)
        self.sub_progress.setFixedHeight(4)
        self.sub_progress.setRange(0, 0)

        layout.addWidget(self.phase_label)
        layout.addSpacing(6)
        layout.addWidget(self.progress)
        layout.addSpacing(10)
        layout.addWidget(self.sub_phase_label)
        layout.addSpacing(4)
        layout.addWidget(self.sub_progress)
        layout.addStretch(1)

    def reset_state(self) -> None:
        self.status_label.setText("正在准备模型与运行环境")
        self.detail_label.setText("首次启动可能需要下载文件，请稍候。")
        self.phase_label.setText("准备进入变声器")
        self.progress.setRange(0, 0)
        self.progress.setValue(0)
        self.sub_phase_label.setText("当前子任务")
        self.sub_progress.setRange(0, 0)
        self.sub_progress.setValue(0)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect()

        background = QLinearGradient(rect.topLeft(), rect.topRight())
        background.setColorAt(0.0, QColor("#efe8dc"))
        background.setColorAt(0.28, QColor("#f5f2ea"))
        background.setColorAt(0.62, QColor("#f3f3f1"))
        background.setColorAt(1.0, QColor("#e3e7ee"))
        painter.fillRect(rect, background)

        overlay = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        overlay.setColorAt(0.0, QColor(255, 255, 255, 130))
        overlay.setColorAt(1.0, QColor(255, 255, 255, 35))
        painter.fillRect(rect, overlay)

        self._draw_rings(painter, rect)
        self._draw_waveform(painter, rect)
        super().paintEvent(event)

    def _draw_rings(self, painter: QPainter, rect: QRectF) -> None:
        center = QPointF(rect.center().x(), rect.center().y() - 26)
        ring_pen = QPen(QColor(99, 102, 112, 26))
        ring_pen.setWidthF(1.0)
        painter.setPen(ring_pen)
        for radius in (54, 82, 110, 138, 166):
            painter.drawEllipse(center, radius, radius)

    def _draw_waveform(self, painter: QPainter, rect: QRectF) -> None:
        center_x = rect.center().x()
        center_y = rect.center().y() - 10
        bar_width = 2.0
        bar_gap = 4.0
        wave_heights = [
            1, 1, 2, 3, 4, 6, 8, 11, 14, 18, 24, 31, 40, 52, 38, 26, 18, 12, 9, 7,
            5, 4, 3, 2, 2, 1, 1,
        ]
        wave_pen = QPen(QColor(148, 163, 184, 58))
        wave_pen.setWidthF(bar_width)
        wave_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(wave_pen)

        for index, height in enumerate(wave_heights, start=1):
            offset = index * bar_gap
            left_x = center_x - offset
            right_x = center_x + offset
            half_height = height * 2.2
            painter.drawLine(QLineF(left_x, center_y - half_height, left_x, center_y + half_height))
            painter.drawLine(QLineF(right_x, center_y - half_height, right_x, center_y + half_height))

    @Slot(str)
    def set_status(self, message: str) -> None:
        normalized = message.strip() or "正在准备模型与运行环境"
        self.status_label.setText(normalized)
        if self.progress.maximum() <= 0:
            self.phase_label.setText(normalized)

    @Slot(object)
    def set_progress(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        detail = str(payload.get("detail", "")).strip()
        if detail:
            self.detail_label.setText(detail)

        indeterminate = bool(payload.get("indeterminate", False))
        value = int(payload.get("value", 0) or 0)
        maximum = int(payload.get("maximum", 0) or 0)
        phase_prefix = self.status_label.text().strip() or "准备进入变声器"
        sub_label = str(payload.get("sub_label", "")).strip()
        sub_value = int(payload.get("sub_value", 0) or 0)
        sub_maximum = int(payload.get("sub_maximum", 0) or 0)
        sub_indeterminate = bool(payload.get("sub_indeterminate", False))

        if indeterminate or maximum <= 0:
            self.progress.setRange(0, 0)
            self.phase_label.setText(phase_prefix)
        else:
            value = max(0, min(value, maximum))
            scaled_value = int((value / maximum) * PROGRESS_BAR_SCALE) if maximum else 0
            scaled_value = max(0, min(scaled_value, PROGRESS_BAR_SCALE))
            self.progress.setRange(0, PROGRESS_BAR_SCALE)
            self.progress.setValue(scaled_value)
            percent = int((value / maximum) * 100) if maximum else 0
            self.phase_label.setText(f"{phase_prefix} {percent}%")

        if sub_label:
            self.sub_phase_label.setText(sub_label)
        if sub_indeterminate or sub_maximum <= 0:
            self.sub_progress.setRange(0, 0)
            return

        sub_value = max(0, min(sub_value, sub_maximum))
        scaled_sub_value = int((sub_value / sub_maximum) * PROGRESS_BAR_SCALE) if sub_maximum else 0
        scaled_sub_value = max(0, min(scaled_sub_value, PROGRESS_BAR_SCALE))
        self.sub_progress.setRange(0, PROGRESS_BAR_SCALE)
        self.sub_progress.setValue(scaled_sub_value)


def _install_exception_hooks() -> None:
    def handle_exception(exc_type, exc_value, exc_traceback) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.opt(exception=(exc_type, exc_value, exc_traceback)).error("UI 主线程未捕获异常")

    def handle_thread_exception(args: threading.ExceptHookArgs) -> None:
        if issubclass(args.exc_type, KeyboardInterrupt):
            return
        logger.opt(exception=(args.exc_type, args.exc_value, args.exc_traceback)).error(
            "UI Python 线程未捕获异常: {}",
            getattr(args.thread, "name", "unknown"),
        )

    sys.excepthook = handle_exception
    threading.excepthook = handle_thread_exception


def _configure_application(app: QApplication) -> None:
    app.setApplicationName(APP_DISPLAY_NAME)
    app.setApplicationDisplayName(APP_DISPLAY_NAME)
    if APP_ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    else:
        logger.warning("应用图标不存在: {}", APP_ICON_PATH)
    app.setStyle("Fusion")
    app.setStyleSheet(get_theme_stylesheet(DEFAULT_THEME))
    app.setFont(QFont("Microsoft YaHei UI", 10))


def _run_startup_model_check(app: QApplication) -> str | None:
    dialog = _get_startup_progress_dialog()
    dialog.reset_state()

    worker = ModelPreparationWorker()
    thread = QThread(app)
    worker.moveToThread(thread)

    error_message: dict[str, str] = {}

    worker.status.connect(dialog.set_status)
    worker.progress.connect(dialog.set_progress)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(dialog.accept)

    def on_failed(message: str) -> None:
        error_message["message"] = message
        dialog.reject()

    worker.failed.connect(on_failed)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)

    thread.start()
    dialog.exec()
    thread.wait()
    thread.deleteLater()
    return error_message.get("message")


def main() -> int:
    setup_logging()
    _install_exception_hooks()
    logger.info("{} UI 应用启动中", APP_DISPLAY_NAME)

    app = QApplication(sys.argv)
    _configure_application(app)

    error_message = _run_startup_model_check(app)
    if error_message:
        dialog = _get_startup_error_dialog()
        dialog.setText(error_message)
        dialog.setInformativeText("")
        dialog.exec()
        logger.error("启动阶段终止: {}", error_message)
        return 1

    window = MainWindow()
    if APP_ICON_PATH.exists():
        window.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    window.show()

    exit_code = app.exec()
    logger.info("{} UI 应用退出，exit_code={}", APP_DISPLAY_NAME, exit_code)
    return exit_code


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
