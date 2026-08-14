from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
import zipfile
from html import escape
from pathlib import Path
from uuid import uuid4

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, QSize, QSignalBlocker, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtGui import QColor, QIcon, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSizePolicy,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QStyle,
    QScrollArea,
    QSpacerItem,
)

from app import get_settings
from app.core.config import (
    DEFAULT_TTS_MAX_NEW_TOKENS,
    set_output_voice_language,
    set_voice_output_model,
    set_supported_index_tts_model_repos,
    set_translation_enabled,
    set_translation_target_language,
    set_translation_node_id,
    set_translation_nodes,
    set_experimental_accel_gpt_enabled,
    set_experimental_torch_compile_enabled,
)
from app.core.index_tts_files import get_index_tts_missing_files
from app.core.logging import get_logger
from app.core.model_setup import list_downloadable_model_catalog
from app.core.process_audio import is_process_audio_capture_supported, list_running_audio_programs
from app.core.runtime import (
    get_cpu_utilization,
    get_gpu_memory_used_gb,
    get_gpu_total_memory_gb,
    get_gpu_utilization,
    get_memory_utilization,
)
from app.core.version import get_app_display_name, get_project_version
from app.core.win_audio import list_windows_audio_endpoints
from app.services.translation import TranslationService, detect_text_language
from ui.icon_registry import render_svg_template_icon
from ui.theme import DEFAULT_THEME, THEME_OPTIONS, get_theme_palette, get_theme_stylesheet
from ui.window_chrome import apply_window_chrome_theme
from ui.waveform_widget import WaveformWidget
from ui.workers import AUDIO_CACHE_DIR, LiveSpeechWorker, LiveWorkerState, ModelDownloadWorker


def _debug_report_program_listen_ui(hypothesis_id: str, message: str, data: dict[str, object]) -> None:
    env_path = Path(".dbg/program-listen-failure.env")
    url = "http://127.0.0.1:7777/event"
    session_id = "program-listen-failure"
    try:
        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                if raw_line.startswith("DEBUG_SERVER_URL="):
                    url = raw_line.split("=", 1)[1].strip() or url
                elif raw_line.startswith("DEBUG_SESSION_ID="):
                    session_id = raw_line.split("=", 1)[1].strip() or session_id
        payload = {
            "sessionId": session_id,
            "runId": os.environ.get("TRAE_DEBUG_RUN_ID", "pre-fix"),
            "hypothesisId": hypothesis_id,
            "location": "ui/main_window.py",
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


class ChatMessageWidget(QWidget):
    replay_clicked = Signal(str)
    broadcast_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.message_id = ""
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 12)
        
        self.timestamp_label = QLabel()
        self.timestamp_label.setTextFormat(Qt.PlainText)
        self.timestamp_label.setFixedWidth(68)
        self.timestamp_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.timestamp_label.setStyleSheet("font-size:10px;padding-top:16px;")
        
        self.bubble_frame = QFrame()
        self.bubble_frame.setObjectName("BubbleFrame")
        self.bubble_layout = QVBoxLayout(self.bubble_frame)
        self.bubble_layout.setContentsMargins(20, 14, 20, 14)
        self.bubble_layout.setSpacing(6)
        
        self.action_layout = QHBoxLayout()
        self.action_layout.setContentsMargins(0, 0, 0, 0)
        self.action_layout.setSpacing(12)
        
        self.replay_button = QPushButton()
        self.replay_button.setCursor(Qt.PointingHandCursor)
        self.replay_button.setFlat(True)
        self.replay_button.clicked.connect(self._on_replay_clicked)
        
        self.broadcast_button = QPushButton()
        self.broadcast_button.setCursor(Qt.PointingHandCursor)
        self.broadcast_button.setFlat(True)
        self.broadcast_button.clicked.connect(self._on_broadcast_clicked)
        
        self.action_layout.addWidget(self.replay_button)
        self.action_layout.addWidget(self.broadcast_button)
        
        self.speaker_label = QLabel()
        self.speaker_label.setTextFormat(Qt.PlainText)
        self.primary_label = QLabel()
        self.primary_label.setTextFormat(Qt.RichText)
        self.primary_label.setWordWrap(True)
        self.primary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.primary_label.setStyleSheet("font-size:13px;font-weight:600;line-height:1.65;")
        
        self.secondary_label = QLabel()
        self.secondary_label.setTextFormat(Qt.RichText)
        self.secondary_label.setWordWrap(True)
        self.secondary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        self.bubble_layout.addLayout(self.action_layout)
        self.bubble_layout.addWidget(self.speaker_label)
        self.bubble_layout.addWidget(self.primary_label)
        self.bubble_layout.addWidget(self.secondary_label)
        
    def _on_replay_clicked(self):
        self.replay_clicked.emit(self.message_id)

    def _on_broadcast_clicked(self):
        self.broadcast_clicked.emit(self.message_id)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.bubble_frame.setMaximumWidth(int(self.width() * 0.75))
        
    def update_view_model(self, view_model: dict):
        self.message_id = view_model.get('message_id', "")
        self.timestamp_label.setText(view_model['timestamp'])
        self.bubble_frame.setStyleSheet(f"""
            QFrame#BubbleFrame {{
                background-color: {view_model['card_bg']};
                border: 2px solid {view_model['card_border']};
                border-radius: 24px;
            }}
        """)
        
        self.replay_button.setIcon(render_svg_template_icon("broadcast", color=view_model['accent_color']))
        self.replay_button.setIconSize(QSize(16, 16))
        self.replay_button.setText("")
        self.replay_button.setToolTip("重播")
        self.replay_button.setFixedSize(16, 16)
        
        self.broadcast_button.setIcon(render_svg_template_icon("replay", color=view_model['accent_color']))
        self.broadcast_button.setIconSize(QSize(16, 16))
        self.broadcast_button.setText("")
        self.broadcast_button.setToolTip("重放到虚拟设备")
        self.broadcast_button.setFixedSize(16, 16)
        
        btn_style = f"""
            QPushButton {{
                color: {view_model['accent_color']};
                font-size: 11px;
                font-weight: bold;
                background: transparent;
                border: none;
                padding: 0px;
                text-align: left;
            }}
            QPushButton:hover {{
                text-decoration: underline;
                color: {view_model['button_hover_color']};
            }}
        """
        self.replay_button.setStyleSheet(btn_style)
        self.broadcast_button.setStyleSheet(btn_style)
        self.timestamp_label.setStyleSheet(
            f"color:{view_model['timestamp_color']};font-size:10px;padding-top:16px;"
        )
        self.primary_label.setStyleSheet(
            f"color:{view_model['primary_text_color']};font-size:13px;font-weight:600;line-height:1.65;"
        )
        
        if view_model['meter_align'] == 'right':
            self.action_layout.setAlignment(Qt.AlignRight)
        else:
            self.action_layout.setAlignment(Qt.AlignLeft)
        
        if view_model['speaker_label']:
            self.speaker_label.setText(view_model['speaker_label'])
            self.speaker_label.setStyleSheet(f"color:{view_model['accent_color']};font-size:11px;font-weight:600;")
            align_speaker = Qt.AlignRight if view_model['bubble_align'] == 'right' else Qt.AlignLeft
            self.speaker_label.setAlignment(align_speaker)
            self.speaker_label.show()
        else:
            self.speaker_label.hide()
            
        self.primary_label.setText(view_model['primary_html'])
        if view_model['secondary_html']:
            self.secondary_label.setText(view_model['secondary_html'])
            self.secondary_label.setStyleSheet(f"color:{view_model['translation_color']};font-size:12px;margin-top:4px;")
            self.secondary_label.show()
        else:
            self.secondary_label.hide()
            
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            
        spacer = QSpacerItem(86, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)
        flexible_spacer = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        if view_model['bubble_align'] == 'left':
            self.main_layout.addWidget(self.timestamp_label, 0, Qt.AlignTop)
            self.main_layout.addWidget(self.bubble_frame, 0, Qt.AlignTop)
            self.main_layout.addSpacerItem(flexible_spacer)
            self.main_layout.addSpacerItem(spacer)
        else:
            self.main_layout.addSpacerItem(spacer)
            self.main_layout.addSpacerItem(flexible_spacer)
            self.main_layout.addWidget(self.bubble_frame, 0, Qt.AlignTop)
            self.main_layout.addWidget(self.timestamp_label, 0, Qt.AlignTop)

logger = get_logger(__name__)
APP_DISPLAY_NAME = get_app_display_name()
EMOTION_PRESETS: tuple[tuple[str, str | None, str, str], ...] = (
    ("自然", None, "calm", "😌"),
    ("温柔", "用温柔、自然、放松的语气说，声音柔和一些。", "calm", "🙂"),
    ("开心", "用开心、轻快、带一点笑意的语气说，情绪更明亮。", "happy", "😊"),
    ("悲伤", "用伤感、克制、轻声的语气说，情绪更低沉。", "sad", "😢"),
    ("愤怒", "用愤怒、激动、爆发力更强的语气说，但保持吐字清晰。", "angry", "😠"),
    ("紧张", "用紧张、压低声音、略带试探的语气说，情绪更真实。", "afraid", "😨"),
)
INPUT_MODES: tuple[tuple[str, str], ...] = (
    ("麦克风变声", "microphone"),
    ("文本合成", "text"),
)
VOICE_OUTPUT_MODELS: tuple[tuple[str, str], ...] = (
    ("Index-TTS", "indextts"),
    ("MeanVC2", "meanvc2"),
)
OUTPUT_LANGUAGE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("中文", "zh"),
    ("英文", "en"),
    ("日语", "ja"),
)
TRANSLATION_PROVIDER_OPTIONS: tuple[tuple[str, str], ...] = (
    ("百度翻译", "baidu-translate"),
    ("DeepL", "deepl"),
    ("Google Cloud", "google-cloud-translate"),
    ("OpenAI", "openai"),
    ("DeepSeek", "deepseek"),
    ("自定义兼容接口", "openai-compatible"),
)
DOWNLOADABLE_MODEL_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("Fun-ASR 量化模型", "aklwss/Fun-ASR-Nano-2512-gguf", "aklwss/Fun-ASR-Nano-2512-gguf"),
    ("MeanVC2", "aklwss/MeanVC2", "aklwss/MeanVC2"),
    ("2.5 多语言模型", "IndexTeam/IndexTTS-2.5", "IndexTeam/IndexTTS-2.5"),
    ("emotion2vec+", "iic/emotion2vec_plus_base", "iic/emotion2vec_plus_base"),
)
DOWNLOADABLE_MODEL_DESCRIPTIONS: dict[str, str] = {
    "aklwss/Fun-ASR-Nano-2512-gguf": "用于实时语音识别的 Fun-ASR Nano 量化模型。",
    "aklwss/MeanVC2": "用于实时变声的 MeanVC2 运行模型与检查点文件。",
    "IndexTeam/IndexTTS-2.5": "支持中文、英文、日语统一下载与切换。",
    "iic/emotion2vec_plus_base": "用于自动情感识别，供 emotion2vec+ 实时分析与情感控制使用。",
}
# Index-TTS 八情感: [高兴, 愤怒, 悲伤, 恐惧, 反感, 低落, 惊讶, 自然]
INDEX_TTS_EMOTIONS: tuple[tuple[str, str, str], ...] = (
    ("高兴", "happy", "😊"),
    ("愤怒", "angry", "😠"),
    ("悲伤", "sad", "😢"),
    ("恐惧", "afraid", "😨"),
    ("反感", "disgusted", "😒"),
    ("低落", "melancholic", "😞"),
    ("惊讶", "surprised", "😮"),
    ("自然", "calm", "😌"),
)

VIRTUAL_DEVICE_KEYWORDS = (
    "vb-audio",
    "vb cable",
    "cable input",
    "cable output",
    "voicemeeter",
    "virtual audio cable",
    "vac",
)
PREFERRED_VIRTUAL_OUTPUT_KEYWORDS = (
    "cable input",
    "output (vb-audio point)",
    "voicemeeter input",
    "output",
)


class SystemPerfWorker(QObject):
    metrics_ready = Signal(object)

    @Slot()
    def collect(self) -> None:
        try:
            cpu = get_cpu_utilization()
            ram = get_memory_utilization()
            gpu = get_gpu_utilization()
            gpu_mem_used = get_gpu_memory_used_gb()
            gpu_mem_total = get_gpu_total_memory_gb()

            payload = {
                "cpu": cpu,
                "ram": ram,
                "gpu": gpu,
                "gpu_mem_used": gpu_mem_used,
                "gpu_mem_total": gpu_mem_total,
            }
            self.metrics_ready.emit(payload)
        except Exception as exc:
            logger.warning("后台采集系统监控失败: {}", exc)
            self.metrics_ready.emit(
                {
                    "cpu": None,
                    "ram": None,
                    "gpu": None,
                    "gpu_mem_used": None,
                    "gpu_mem_total": None,
                }
            )




class ToggleSwitch(QCheckBox):
    _TRACK_WIDTH = 36
    _TRACK_HEIGHT = 20
    _THUMB_SIZE = 14
    _TEXT_GAP = 8

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(28)

    def sizeHint(self) -> QSize:
        metrics = self.fontMetrics()
        text_width = metrics.horizontalAdvance(self.text()) if self.text() else 0
        width = self._TRACK_WIDTH + (self._TEXT_GAP + text_width if text_width else 0) + 6
        height = max(self._TRACK_HEIGHT, metrics.height()) + 8
        return QSize(width, height)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        track_x = 1
        track_y = (self.height() - self._TRACK_HEIGHT) // 2
        thumb_y = (self.height() - self._THUMB_SIZE) // 2
        checked = self.isChecked()
        enabled = self.isEnabled()
        app = QApplication.instance()
        theme_name = str(app.property("app_theme") or DEFAULT_THEME) if app is not None else DEFAULT_THEME
        palette = get_theme_palette(theme_name)

        track_color = QColor(palette["primary_start"] if checked else palette["button_bg"])
        track_border = QColor(palette["primary_border"] if checked else palette["button_border"])
        thumb_color = QColor(palette["slider_thumb_bg"] if enabled else palette["button_disabled_text"])
        text_color = QColor(palette["text"] if enabled else palette["button_disabled_text"])
        if not enabled:
            track_color = QColor(palette["button_disabled_bg"])
            track_border = QColor(palette["button_disabled_border"])

        painter.setPen(QPen(track_border, 1))
        painter.setBrush(track_color)
        painter.drawRoundedRect(track_x, track_y, self._TRACK_WIDTH, self._TRACK_HEIGHT, 10, 10)

        thumb_margin = (self._TRACK_HEIGHT - self._THUMB_SIZE) // 2
        thumb_x = (
            track_x + self._TRACK_WIDTH - self._THUMB_SIZE - thumb_margin
            if checked
            else track_x + thumb_margin
        )
        painter.setPen(Qt.NoPen)
        painter.setBrush(thumb_color)
        painter.drawEllipse(thumb_x, thumb_y, self._THUMB_SIZE, self._THUMB_SIZE)

        label_x = track_x + self._TRACK_WIDTH + self._TEXT_GAP
        label_width = max(0, self.width() - label_x)
        painter.setPen(text_color)
        painter.drawText(label_x, 0, label_width, self.height(), Qt.AlignVCenter | Qt.AlignLeft, self.text())


class MainWindow(QMainWindow):
    live_preload_requested = Signal(object)
    live_run_requested = Signal(object)
    live_text_requested = Signal(object)
    live_reference_audio_changed_requested = Signal(str)
    live_voice_output_model_changed_requested = Signal(str)
    live_translation_settings_changed_requested = Signal(object)
    live_speech_rate_changed_requested = Signal(float)
    live_output_volume_changed_requested = Signal(float)
    live_monitor_output_volume_changed_requested = Signal(float)
    live_replay_requested = Signal(str, bool)
    live_stop_requested = Signal()
    live_shutdown_requested = Signal()
    live_asr_unload_requested = Signal()
    live_emotion_unload_requested = Signal()
    live_asr_reload_requested = Signal()
    live_tts_switch_requested = Signal()
    perf_refresh_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_DISPLAY_NAME}")
        self.resize(1280, 960)
        self.setMinimumSize(1180, 860)

        self.settings = get_settings()
        self._live_worker_thread: QThread | None = None
        self._live_worker: LiveSpeechWorker | None = None
        self._perf_worker_thread: QThread | None = None
        self._perf_worker: SystemPerfWorker | None = None
        self._live_worker_state = LiveWorkerState.INITIALIZING.value
        self._models_ready = False
        self._pending_reference_preload = False
        self._input_devices: list[tuple[str, str]] = []
        self._input_device_name_by_key: dict[str, str] = {}
        self._input_device_sd_index_by_key: dict[str, int] = {}
        self._program_audio_supported = is_process_audio_capture_supported()
        self._program_options: list[tuple[int, str]] = []
        self._output_devices: list[tuple[str, str]] = []
        self._monitor_devices: list[tuple[str, str]] = []
        self._output_device_name_by_key: dict[str, str] = {}
        self._output_device_sd_index_by_key: dict[str, int] = {}
        self._current_theme = DEFAULT_THEME
        self._header_runtime_started_at: float | None = None
        self._header_runtime_timer = QTimer(self)
        self._header_runtime_timer.setInterval(1000)
        self._header_runtime_timer.timeout.connect(self._update_header_runtime_clock)
        self._virtual_device_summary = ""
        self._quick_settings_dialog: QDialog | None = None
        self._quick_settings_auto_emotion_checkbox: QCheckBox | None = None
        self._quick_settings_auto_emotion_strength_slider: QSlider | None = None
        self._quick_settings_auto_emotion_strength_label: QLabel | None = None
        self._quick_settings_model_download_action_button: QPushButton | None = None
        self._quick_settings_tts_model_status_labels: dict[str, QLabel] = {}
        self._quick_settings_tts_model_path_labels: dict[str, QLabel] = {}
        self._quick_settings_tts_model_download_buttons: dict[str, QPushButton] = {}
        self._quick_settings_model_download_progress_bar: QProgressBar | None = None
        self._quick_settings_model_download_detail_label: QLabel | None = None
        self._quick_settings_model_download_worker: ModelDownloadWorker | None = None
        self._quick_settings_model_downloading_repo: str | None = None
        self._quick_settings_translation_node_combo: QComboBox | None = None
        self._quick_settings_edit_translation_button: QPushButton | None = None
        self._quick_settings_test_translation_button: QPushButton | None = None
        self._quick_settings_remove_translation_button: QPushButton | None = None
        self.translation_card_checkbox: QCheckBox | None = None
        self.translation_card_target_language_combo: QComboBox | None = None
        self.translation_card_node_combo: QComboBox | None = None
        self.translation_card_info_button: QPushButton | None = None
        self.voice_output_settings_section: QFrame | None = None
        self.emotion_control_info_button: QPushButton | None = None
        self.tts_other_options_info_button: QPushButton | None = None
        self._quick_settings_torch_compile_checkbox: QCheckBox | None = None
        self._quick_settings_accel_gpt_checkbox: QCheckBox | None = None
        self.experimental_torch_compile_hint_label: QLabel | None = None
        self.emotion_control_hint_label: QLabel | None = None
        self.tts_other_options_hint_label: QLabel | None = None
        self.emotion_control_card: QFrame | None = None
        self.tts_other_options_card: QFrame | None = None
        self.emotion_control_overlay: QFrame | None = None
        self.tts_other_options_overlay: QFrame | None = None
        self._reference_audio_dialog: QFileDialog | None = None
        self._message_dialog: QMessageBox | None = None
        self._manual_index_emo_snapshot: dict[str, object] | None = None
        self._auto_emotion_feature_enabled = False
        self._auto_emotion_strength = 0.75
        self._output_voice_language = self.settings.output_voice_language
        self._voice_output_model = self.settings.voice_output_model
        self._active_program_pid: int | None = None
        self._translation_enabled = self.settings.translation_enabled
        self._translation_target_language = self.settings.translation_target_language
        self._translation_node_id = self.settings.translation_node_id
        self._experimental_torch_compile_enabled = self.settings.experimental_torch_compile_enabled
        self._experimental_accel_gpt_enabled = self.settings.experimental_accel_gpt_enabled
        self._coerce_hidden_language_settings()
        self._active_request_mode: str | None = None
        self._pending_live_restart_mode: str | None = None
        self._is_closing = False
        self._active_text_message_id: str | None = None
        self._chat_messages: list[dict[str, object]] = []
        self._chat_message_seq = 0
        self._workspace_summary_signature: tuple[object, ...] | None = None
        self._perf_poll_in_flight = False

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)
        header = self._build_header()
        root_layout.addWidget(header, stretch=10)
        root_layout.addWidget(self._build_main_content(), stretch=90)
        self.setCentralWidget(root)
        self._on_input_mode_changed()
        self._apply_theme(self._current_theme)
        self._log_startup()
        self._start_live_backend()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._is_closing = True
        self._pending_live_restart_mode = None
        self._shutdown_perf_monitor()
        self._shutdown_live_backend()
        super().closeEvent(event)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._apply_window_chrome_theme()
        self._position_chat_scroll_buttons()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._position_chat_scroll_buttons()
        self._update_mode_locked_overlays_geometry()

    def _log_startup(self) -> None:
        logger.info("图形界面已启动")
        logger.info("产品名称: {}", APP_DISPLAY_NAME)
        vram = get_gpu_total_memory_gb()
        if vram is not None:
            logger.info("GPU 显存: {:.1f} GB", vram)
        else:
            logger.info("未检测到 CUDA GPU")
        logger.info("参考音色: 等待用户选择")
        logger.info("已加载音频设备列表: 输入 {} 个, 输出 {} 个", len(self._input_devices), len(self._output_devices))

    def _build_header(self) -> QWidget:
        hero_card = QFrame()
        hero_card.setObjectName("TopHeader")
        hero_layout = QVBoxLayout(hero_card)
        hero_layout.setContentsMargins(14, 10, 14, 10)
        hero_layout.setSpacing(10)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(10)

        brand_panel = QWidget()
        brand_layout = QHBoxLayout(brand_panel)
        brand_layout.setContentsMargins(0, 0, 0, 0)
        brand_layout.setSpacing(8)

        title = QLabel(APP_DISPLAY_NAME)
        title.setObjectName("HeroTitle")
        subtitle = QLabel("实时语音处理与翻译变声一体化平台")
        subtitle.setObjectName("HeroInlineSubtitle")

        brand_layout.addWidget(title, 0, Qt.AlignVCenter)
        brand_layout.addWidget(subtitle, 0, Qt.AlignVCenter)
        brand_layout.addStretch(1)

        status_panel = QWidget()
        status_panel.setObjectName("HeaderMetrics")
        status_row = QHBoxLayout(status_panel)
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(10)
        self.header_runtime_value, self.header_runtime_clock_value = self._create_runtime_summary_card(status_row, "实时运行中")
        self.header_mode_value = self._create_summary_card(status_row, "当前模式")
        self.header_tts_model_value = self._create_summary_card(status_row, "语音输出模型")
        self.header_auto_emotion_value = self._create_summary_card(status_row, "情感识别")
        self.header_reference_value = self._create_summary_card(status_row, "参考音色")
        header_auto_emotion_card = self.header_auto_emotion_value.parentWidget()
        if header_auto_emotion_card is not None:
            header_auto_emotion_card.setVisible(False)

        actions_panel = QWidget()
        actions_panel.setObjectName("HeaderActions")
        actions_layout = QHBoxLayout(actions_panel)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(10)

        self.theme_combo = QComboBox()
        self.theme_combo.setMinimumWidth(112)
        self.theme_combo.setMinimumHeight(34)
        for label, theme_name in THEME_OPTIONS:
            self.theme_combo.addItem(label, theme_name)
        theme_index = self.theme_combo.findData(self._current_theme)
        if theme_index >= 0:
            self.theme_combo.setCurrentIndex(theme_index)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)

        self.quick_settings_button = QPushButton()
        self.quick_settings_button.setObjectName("IconButton")
        settings_icon = render_svg_template_icon("settings_gear", color=self._theme_palette()["muted_text"])
        if not settings_icon.isNull():
            self.quick_settings_button.setIcon(settings_icon)
        else:
            self.quick_settings_button.setIcon(self._standard_icon(QStyle.SP_FileDialogDetailedView))
        self.quick_settings_button.setToolTip("打开快速设置")
        self.quick_settings_button.setFixedSize(34, 34)
        self.quick_settings_button.clicked.connect(self._open_quick_settings_dialog)
        actions_layout.addWidget(self.quick_settings_button)
        actions_layout.addWidget(self.theme_combo)

        top_row.addWidget(brand_panel, 1)
        top_row.addWidget(actions_panel, 0, Qt.AlignRight | Qt.AlignVCenter)

        hero_layout.addLayout(top_row)
        hero_layout.addWidget(status_panel)
        self._update_header_runtime_clock()
        return hero_card

    def _create_summary_card(self, parent_layout: QHBoxLayout, caption: str) -> QLabel:
        card = QFrame()
        card.setObjectName("SummaryCard")
        card.setMinimumHeight(52)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 7, 12, 7)
        layout.setSpacing(2)
        caption_label = QLabel(caption)
        caption_label.setObjectName("SummaryCaption")
        value_label = QLabel("--")
        value_label.setObjectName("SummaryValue")
        value_label.setWordWrap(True)
        layout.addWidget(caption_label)
        layout.addWidget(value_label)
        parent_layout.addWidget(card, stretch=1)
        return value_label

    def _create_runtime_summary_card(self, parent_layout: QHBoxLayout, caption: str) -> tuple[QLabel, QLabel]:
        card = QFrame()
        card.setObjectName("SummaryCard")
        card.setMinimumHeight(52)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 7, 12, 7)
        layout.setSpacing(3)

        caption_label = QLabel(caption)
        caption_label.setObjectName("SummaryCaption")

        body_row = QHBoxLayout()
        body_row.setContentsMargins(0, 0, 0, 0)
        body_row.setSpacing(8)

        status_wrap = QWidget()
        status_wrap.setObjectName("HeaderRuntimeBadge")
        status_layout = QHBoxLayout(status_wrap)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(6)

        runtime_dot = QLabel("●")
        runtime_dot.setObjectName("HeaderRuntimeDot")
        runtime_status = QLabel("待命中")
        runtime_status.setObjectName("HeaderRuntimeStatus")
        runtime_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        status_layout.addWidget(runtime_dot)
        status_layout.addWidget(runtime_status)

        runtime_clock = QLabel("00:00:00")
        runtime_clock.setObjectName("HeaderRuntimeClock")
        runtime_clock.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        body_row.addWidget(status_wrap, 0, Qt.AlignLeft | Qt.AlignVCenter)
        body_row.addWidget(runtime_clock, 1, Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(caption_label)
        layout.addLayout(body_row)
        parent_layout.addWidget(card, stretch=1)
        return runtime_status, runtime_clock

    def _build_main_content(self) -> QWidget:
        panel = QWidget()
        main_layout = QVBoxLayout(panel)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(12)
        
        devices_card, emotion_card = self._build_changer_panel()
        center_panel = self._build_workspace_panel()
        reference_card, speech_volume_card, tts_other_options_card = self._build_reference_panel()

        # 强制这几列忽略内部控件的最小宽度，完全按网格的 stretch 比例分配宽度
        for w in [devices_card, center_panel, reference_card, emotion_card, speech_volume_card, tts_other_options_card]:
            w.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
            
        # 上半区三张核心卡片再拉长一点，保证输入 / 实时对话 / 输出更舒展。
        devices_card.setMinimumSize(280, 410)
        center_panel.setMinimumSize(420, 410)
        reference_card.setMinimumSize(280, 410)

        # 下半区卡片整体压短，避免抢占主工作区高度。
        emotion_card.setMinimumSize(400, 120)
        speech_volume_card.setMinimumSize(180, 120)
        tts_other_options_card.setMinimumSize(180, 120)

        emotion_card.setMaximumHeight(190)
        speech_volume_card.setMaximumHeight(190)
        tts_other_options_card.setMaximumHeight(190)
        
        # 强制让上面的三个卡片更长一点
        devices_card.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)
        center_panel.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)
        reference_card.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        # 上半部分三列比例：22 / 56 / 22
        top_row.addWidget(devices_card, stretch=22)
        top_row.addWidget(center_panel, stretch=56)
        top_row.addWidget(reference_card, stretch=22)
        
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(12)
        # 下半部分比例：情感控制 60%，语速 20%，TTS 20%
        bottom_row.addWidget(emotion_card, stretch=60)
        bottom_row.addWidget(speech_volume_card, stretch=20)
        bottom_row.addWidget(tts_other_options_card, stretch=20)

        # 垂直比例继续向上半区倾斜，让主链路区域更高、底部配置区更短。
        main_layout.addLayout(top_row, stretch=62)
        main_layout.addLayout(bottom_row, stretch=12)
        
        footer_panel = self._build_footer_panel()
        main_layout.addWidget(footer_panel)
        
        return panel

    @staticmethod
    def _set_compact_label_style(label: QLabel) -> QLabel:
        label.setStyleSheet("font-size: 8px;")
        return label

    @staticmethod
    def _set_section_title_style(label: QLabel) -> QLabel:
        label.setStyleSheet("font-size: 11px; font-weight: 700;")
        return label

    @staticmethod
    def _set_compact_field_style(widget: QWidget, *, min_height: int = 34) -> QWidget:
        widget.setMinimumHeight(min_height)
        widget.setStyleSheet("font-size: 9px; padding: 6px 8px;")
        return widget

    @staticmethod
    def _set_compact_button_style(button: QPushButton, *, min_height: int = 34, min_width: int = 96) -> QPushButton:
        button.setMinimumHeight(min_height)
        button.setMinimumWidth(min_width)
        button.setStyleSheet("font-size: 9px; padding: 5px 8px;")
        return button

    @staticmethod
    def _set_compact_badge_style(label: QLabel) -> QLabel:
        label.setStyleSheet("font-size: 9px; padding: 4px 9px;")
        return label

    @staticmethod
    def _set_toggle_switch_style(checkbox: QCheckBox, *, min_height: int = 28) -> QCheckBox:
        checkbox.setMinimumHeight(min_height)
        checkbox.setStyleSheet("font-size: 9px;")
        return checkbox

    def _theme_palette(self) -> dict[str, str]:
        return get_theme_palette(self._current_theme)

    def _set_inline_info_button_style(self, button: QPushButton) -> QPushButton:
        palette = self._theme_palette()
        button.setFixedSize(20, 20)
        button.setCursor(Qt.PointingHandCursor)
        button.setText("i")
        button.setStyleSheet(
            "font-size: 11px; font-weight: 700; padding: 0px;"
            f"color: {palette['accent_soft']};"
            f"background: {palette['surface_bg_alt']};"
            f"border: 1px solid {palette['surface_border']};"
            "border-radius: 10px;"
        )
        return button

    def _translation_mode_tooltip(self) -> str:
        if self._is_meanvc2_selected():
            return (
                "MeanVC2 模式说明\n"
                "1. 翻译只会显示在聊天记录里\n"
                "2. 不会把翻译结果送进变声输出\n"
                "3. 实际播放的仍然是原始语音经过 MeanVC2 变声后的音频"
            )
        return (
            "Index-TTS 模式说明\n"
            "1. 会先按目标语言翻译文本\n"
            "2. 再用翻译后的结果合成语音输出"
        )

    def _emotion_control_tooltip(self) -> str:
        if self._is_meanvc2_selected():
            return (
                "情感控制说明\n"
                "1. 这些选项只对 Index-TTS 生效\n"
                "2. MeanVC2 是语音转换模型，不使用情感预设、情感向量或自动情感识别"
            )
        return (
            "情感控制说明\n"
            "1. 可在 Index-TTS 下调整情感预设与情感向量\n"
            "2. 自动情感识别会根据麦克风语音动态调整情绪"
        )

    def _tts_other_options_tooltip(self) -> str:
        if self._is_meanvc2_selected():
            return (
                "TTS 其他选项说明\n"
                "1. 这些选项只对 Index-TTS 生效\n"
                "2. MeanVC2 不使用实验情感识别、进阶模式或极速模式"
            )
        return (
            "TTS 其他选项说明\n"
            "1. 实验情感识别仅作用于麦克风输入\n"
            "2. 进阶模式与极速模式属于 Index-TTS 的实验加速选项"
        )

    def _theme_icon_color(self, role: str = "accent") -> str:
        palette = self._theme_palette()
        match role:
            case "primary":
                return palette["primary_text"]
            case "warning":
                return palette["warning_text"]
            case "muted":
                return palette["muted_text"]
            case "title":
                return palette["group_title"]
            case _:
                return palette["accent_soft"]

    def _live_button_icon(self, icon_name: str, role: str = "primary") -> QIcon:
        return render_svg_template_icon(icon_name, color=self._theme_icon_color(role))

    def _build_transcript_scroll_style(self) -> str:
        palette = self._theme_palette()
        return f"""
        QScrollArea#TranscriptView {{
            background: transparent;
            border: none;
        }}
        QScrollArea#TranscriptView QScrollBar:vertical {{
            background: {palette['surface_bg_alt']};
            width: 32px;
            margin: 36px 0px 36px 0px;
            border-radius: 16px;
            border: 1px solid {palette['surface_border']};
        }}
        QScrollArea#TranscriptView QScrollBar::handle:vertical {{
            background: {palette['scroll_handle']};
            min-height: 32px;
            border-radius: 12px;
            margin: 2px 4px;
        }}
        QScrollArea#TranscriptView QScrollBar::handle:vertical:hover {{
            background: {palette['scroll_handle_hover']};
        }}
        QScrollArea#TranscriptView QScrollBar::sub-line:vertical,
        QScrollArea#TranscriptView QScrollBar::add-line:vertical {{
            height: 0px;
            background: transparent;
            border: none;
        }}
        QScrollArea#TranscriptView QScrollBar::add-page:vertical,
        QScrollArea#TranscriptView QScrollBar::sub-page:vertical {{
            background: transparent;
        }}
        """

    def _build_transcript_scroll_button_style(self) -> str:
        palette = self._theme_palette()
        return f"""
            QPushButton#TranscriptScrollButton {{
                background: {palette['button_bg']};
                border: 1px solid {palette['button_border']};
                border-radius: 16px;
                padding: 0px;
            }}
            QPushButton#TranscriptScrollButton:hover {{
                background: {palette['button_hover']};
                border: 1px solid {palette['button_hover_border']};
            }}
            QPushButton#TranscriptScrollButton:pressed {{
                background: {palette['button_pressed']};
                border: 1px solid {palette['button_pressed_border']};
            }}
            QPushButton#TranscriptScrollButton:disabled {{
                background: {palette['button_disabled_bg']};
                border: 1px solid {palette['button_disabled_border']};
            }}
        """

    def _refresh_theme_sensitive_ui(self) -> None:
        palette = self._theme_palette()

        if hasattr(self, "quick_settings_button"):
            settings_icon = render_svg_template_icon("settings_gear", color=self._theme_icon_color("muted"))
            if not settings_icon.isNull():
                self.quick_settings_button.setIcon(settings_icon)

        if hasattr(self, "reference_browse_button"):
            self.reference_browse_button.setIcon(
                render_svg_template_icon("select_voice", color=self._theme_icon_color("accent"))
            )
        if hasattr(self, "translation_card_info_button") and self.translation_card_info_button is not None:
            self._set_inline_info_button_style(self.translation_card_info_button)
            self.translation_card_info_button.setToolTip(self._translation_mode_tooltip())
        if hasattr(self, "emotion_control_info_button") and self.emotion_control_info_button is not None:
            self._set_inline_info_button_style(self.emotion_control_info_button)
            self.emotion_control_info_button.setToolTip(self._emotion_control_tooltip())
        if hasattr(self, "tts_other_options_info_button") and self.tts_other_options_info_button is not None:
            self._set_inline_info_button_style(self.tts_other_options_info_button)
            self.tts_other_options_info_button.setToolTip(self._tts_other_options_tooltip())

        if hasattr(self, "use_virtual_output_button"):
            self.use_virtual_output_button.setIcon(
                render_svg_template_icon("play", color=self._theme_icon_color("accent"))
            )

        if hasattr(self, "workspace_clear_button"):
            self.workspace_clear_button.setIcon(
                render_svg_template_icon("clear_history", color=self._theme_icon_color("muted"))
            )
        if hasattr(self, "workspace_export_button"):
            self.workspace_export_button.setIcon(
                render_svg_template_icon("export_history", color=self._theme_icon_color("muted"))
            )

        if hasattr(self, "chat_history_area"):
            self.chat_history_area.setStyleSheet(self._build_transcript_scroll_style())
        if hasattr(self, "_chat_messages"):
            self._render_chat_messages()
        if hasattr(self, "chat_scroll_up_button"):
            self.chat_scroll_up_button.setIcon(
                render_svg_template_icon("scroll_up", color=self._theme_icon_color("muted"))
            )
            self.chat_scroll_up_button.setStyleSheet(self._build_transcript_scroll_button_style())
        if hasattr(self, "chat_scroll_down_button"):
            self.chat_scroll_down_button.setIcon(
                render_svg_template_icon("scroll_down", color=self._theme_icon_color("muted"))
            )
            self.chat_scroll_down_button.setStyleSheet(self._build_transcript_scroll_button_style())

        if hasattr(self, "footer_panel"):
            self.footer_panel.setStyleSheet(
                (
                    "QFrame#ContentCard { "
                    f"background-color: {palette['surface_bg']}; "
                    "border-radius: 8px; "
                    f"border: 1px solid {palette['surface_border']}; "
                    "}"
                )
            )
        if hasattr(self, "sys_perf_label"):
            self.sys_perf_label.setStyleSheet(f"color: {palette['muted_text']}; font-size: 11px;")
        for label_name in ("asr_delay_label", "trans_delay_label", "tts_delay_label", "total_delay_label"):
            label = getattr(self, label_name, None)
            if label is not None:
                label.setStyleSheet(f"color: {palette['muted_text']}; font-size: 11px;")

        if hasattr(self, "live_status_label"):
            raw_text = str(self.live_status_label.property("raw_text") or "待命中")
            state = str(self.live_status_label.property("state") or "ready")
            self._set_live_status_label_state(raw_text, state)

        if hasattr(self, "live_start_button"):
            self._refresh_live_start_button()
        self._sync_voice_output_model_controls()

    def _build_changer_panel(self) -> tuple[QFrame, QFrame]:
        self.input_mode_combo = QComboBox()
        for label, value in INPUT_MODES:
            self.input_mode_combo.addItem(label, value)
        self.input_mode_combo.currentIndexChanged.connect(self._on_input_mode_changed)
        self._set_compact_field_style(self.input_mode_combo)

        self.tts_engine_combo = QComboBox()
        for label, value in VOICE_OUTPUT_MODELS:
            self.tts_engine_combo.addItem(label, value)
        self._set_compact_field_style(self.tts_engine_combo)
        voice_model_index = self.tts_engine_combo.findData(self._voice_output_model)
        self.tts_engine_combo.setCurrentIndex(voice_model_index if voice_model_index >= 0 else 0)
        self.tts_engine_combo.currentIndexChanged.connect(self._on_voice_output_model_changed)

        self.live_reference_audio_edit = QLineEdit("")
        self.live_reference_audio_edit.setPlaceholderText("请先选择克隆参考音频")
        self._set_compact_field_style(self.live_reference_audio_edit)

        self.emotion_preset_combo = QComboBox()
        for label, instruction, _tone, _emoji in EMOTION_PRESETS:
            self.emotion_preset_combo.addItem(label, instruction)
        self.emotion_custom_edit = QLineEdit()
        self.emotion_custom_edit.setPlaceholderText("可选，例如：更像悄悄说、带一点笑、语速慢一点")

        # --- Index-TTS 情感向量控件 ---
        self.index_emotion_mode_combo = QComboBox()
        self.index_emotion_mode_combo.addItem("预设文本", "preset")
        self.index_emotion_mode_combo.addItem("情感向量", "vector")
        self.index_emotion_mode_combo.currentIndexChanged.connect(self._on_index_emotion_mode_changed)

        self.index_emo_sliders: dict[str, QSlider] = {}
        self.index_emo_value_labels: dict[str, QLabel] = {}
        for cn_name, en_name, _emoji in INDEX_TTS_EMOTIONS:
            slider = QSlider(Qt.Vertical)
            slider.setRange(0, 20)  # 0.00 ~ 1.00, step 0.05
            slider.setValue(0)
            # Default Qt vertical slider is bottom-to-top. 
            # If we don't invert it, 0 is at bottom.
            # If we invert it, 0 is at top. We want 0 at bottom (min at bottom, max at top).
            # So we DO NOT use setInvertedAppearance(True) here.
            slider.setTickInterval(4)
            slider.setTickPosition(QSlider.NoTicks)
            self.index_emo_sliders[en_name] = slider
            label = QLabel("0.00")
            label.setAlignment(Qt.AlignCenter)
            self.index_emo_value_labels[en_name] = label

        self.index_emo_alpha_slider = QSlider(Qt.Horizontal)
        self.index_emo_alpha_slider.setRange(0, 20)  # 0.00 ~ 1.00, step 0.05
        self.index_emo_alpha_slider.setValue(20)
        self.index_emo_alpha_slider.setTickInterval(4)
        self.index_emo_alpha_slider.setTickPosition(QSlider.TicksBelow)
        self.index_emo_alpha_label = QLabel("1.00")
        self.index_emo_alpha_label.setObjectName("InlineValueLabel")
        self.index_emo_alpha_label.setStyleSheet("font-size: 9px;")
        self.index_emo_alpha_label.setFixedWidth(36)
        self.index_emo_alpha_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.index_emo_alpha_slider.valueChanged.connect(
            lambda _v: self.index_emo_alpha_label.setText(f"{self.index_emo_alpha_slider.value() * 0.05:.2f}")
        )
        self.speech_rate_title_label = QLabel("Index-TTS 语速", objectName="CompactLabel")
        self.speech_rate_slider = QSlider(Qt.Horizontal)
        self.speech_rate_slider.setRange(0, 20)  # 0.50 ~ 1.50, step 0.05
        self.speech_rate_slider.setValue(10)
        self.speech_rate_slider.setTickInterval(2)
        self.speech_rate_slider.setTickPosition(QSlider.TicksBelow)
        self.speech_rate_label = QLabel("1.00x")
        self.speech_rate_label.setObjectName("InlineValueLabel")
        self.speech_rate_label.setStyleSheet("font-size: 9px;")
        self.speech_rate_label.setFixedWidth(44)
        self.speech_rate_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.speech_rate_slider.valueChanged.connect(
            lambda v: self.speech_rate_label.setText(f"{self._current_speech_rate():.2f}x")
        )
        self.speech_rate_slider.valueChanged.connect(self._on_speech_rate_changed)

        self.input_device_combo = QComboBox()
        self.capture_source_mic_checkbox = ToggleSwitch("麦克风")
        self.capture_source_program_checkbox = ToggleSwitch("程序")
        self._set_toggle_switch_style(self.capture_source_mic_checkbox)
        self._set_toggle_switch_style(self.capture_source_program_checkbox)
        self.capture_source_mic_checkbox.setChecked(True)
        self.capture_source_program_checkbox.setChecked(False)
        self.capture_source_program_checkbox.setEnabled(self._program_audio_supported)
        if not self._program_audio_supported:
            self.capture_source_program_checkbox.setToolTip("当前系统不支持程序监听。")
        self.capture_source_mic_checkbox.toggled.connect(self._on_input_mode_changed)
        self.capture_source_program_checkbox.toggled.connect(self._on_input_mode_changed)
        self.program_combo = QComboBox()
        self.refresh_program_button = QPushButton("刷新程序")
        self.refresh_program_button.setObjectName("SecondaryActionButton")
        self.refresh_program_button.clicked.connect(self._refresh_running_programs)
        self._set_compact_button_style(self.refresh_program_button, min_width=86)
        self.output_device_combo = QComboBox()
        self.monitor_output_device_combo = QComboBox()
        self._set_compact_field_style(self.program_combo)
        self._set_compact_field_style(self.input_device_combo)
        self._set_compact_field_style(self.output_device_combo)
        self._set_compact_field_style(self.monitor_output_device_combo)
        self.input_level_meter = QProgressBar()
        self.input_level_meter.setObjectName("LevelMeter")
        self.input_level_meter.setRange(0, 100)
        self.input_level_meter.setValue(0)
        self.input_level_meter.setTextVisible(False)
        self.input_level_state_label = QLabel("待输入")
        self.input_level_state_label.setObjectName("InlineValueLabel")
        self.input_level_state_label.setStyleSheet("font-size: 9px;")
        self.monitor_output_volume_slider = QSlider(Qt.Horizontal)
        self.monitor_output_volume_slider.setRange(0, 100)
        self.monitor_output_volume_slider.setValue(65)
        self.monitor_output_volume_label = QLabel("0.65")
        self.monitor_output_volume_label.setObjectName("InlineValueLabel")
        self.monitor_output_volume_label.setStyleSheet("font-size: 9px;")
        self.monitor_output_volume_slider.setToolTip("调节耳返输出音量。")
        self.monitor_output_volume_slider.valueChanged.connect(self._on_monitor_output_volume_changed)
        self.output_gain_slider = QSlider(Qt.Horizontal)
        self.output_gain_slider.setRange(0, 100)
        self.output_gain_slider.setValue(80)
        self.output_gain_slider.setToolTip("调节主输出设备的播放音量。")
        self.output_gain_label = QLabel("0.80")
        self.output_gain_label.setObjectName("InlineValueLabel")
        self.output_gain_label.setStyleSheet("font-size: 9px;")
        self.output_gain_slider.valueChanged.connect(self._on_output_volume_changed)
        self.use_virtual_output_button = QPushButton("虚拟播放")
        self.use_virtual_output_button.setObjectName("SecondaryActionButton")
        self.use_virtual_output_button.clicked.connect(self._select_virtual_output_device)
        self.use_virtual_output_button.setIcon(render_svg_template_icon("play", color=self._theme_icon_color("accent")))
        self.use_virtual_output_button.setIconSize(QSize(14, 14))
        self._set_compact_button_style(self.use_virtual_output_button, min_width=98)
        self.live_status_label = QLabel("")
        self.live_status_label.setObjectName("LiveStatusLabel")
        self.live_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.live_status_label.setStyleSheet("font-size: 10px; padding: 8px;")
        self.live_status_label.setProperty("raw_text", "待命中")
        self.live_status_label.setProperty("state", "ready")

        self.live_start_button = QPushButton("准备中")
        self.live_start_button.setObjectName("PrimaryActionButton")
        self.live_start_button.setMinimumHeight(48)
        self.live_start_button.setStyleSheet("font-size: 12px; padding: 8px 11px;")
        self.live_start_button.setEnabled(False)
        self.live_start_button.clicked.connect(self._on_live_start_clicked)
        self.live_start_button.setIcon(render_svg_template_icon("preparing", color=self._theme_icon_color("warning")))

        self._media_devices = QMediaDevices(self)
        self._media_devices.audioInputsChanged.connect(self._refresh_audio_devices)
        self._media_devices.audioOutputsChanged.connect(self._refresh_audio_devices)

        self._refresh_audio_devices()
        self._refresh_running_programs()
        self.live_reference_audio_edit.textChanged.connect(self._refresh_workspace_summary)
        self.capture_source_mic_checkbox.toggled.connect(self._refresh_workspace_summary)
        self.capture_source_program_checkbox.toggled.connect(self._refresh_workspace_summary)
        self.program_combo.currentIndexChanged.connect(self._on_program_selection_changed)
        self.tts_engine_combo.currentIndexChanged.connect(self._refresh_workspace_summary)
        self.input_device_combo.currentIndexChanged.connect(self._refresh_workspace_summary)
        self.output_device_combo.currentIndexChanged.connect(self._refresh_workspace_summary)
        self.monitor_output_device_combo.currentIndexChanged.connect(self._refresh_workspace_summary)

        self._perf_timer = QTimer(self)
        self._perf_timer.timeout.connect(self._request_sys_perf_update)
        self._perf_timer.start(2000)
        self._start_perf_monitor()
        self._request_sys_perf_update()

        devices_card = self._build_devices_panel()
        emotion_card = self._build_emotion_panel()
        
        return devices_card, emotion_card

    def _start_perf_monitor(self) -> None:
        if self._perf_worker_thread is not None:
            return
        self._perf_worker_thread = QThread(self)
        self._perf_worker = SystemPerfWorker()
        self._perf_worker.moveToThread(self._perf_worker_thread)
        self.perf_refresh_requested.connect(self._perf_worker.collect)
        self._perf_worker.metrics_ready.connect(self._on_sys_perf_ready)
        self._perf_worker_thread.start()

    def _shutdown_perf_monitor(self) -> None:
        if hasattr(self, "_perf_timer") and self._perf_timer.isActive():
            self._perf_timer.stop()
        if self._perf_worker_thread is None:
            return
        if self._perf_worker is not None:
            try:
                self.perf_refresh_requested.disconnect(self._perf_worker.collect)
            except (RuntimeError, TypeError):
                pass
            try:
                self._perf_worker.metrics_ready.disconnect(self._on_sys_perf_ready)
            except (RuntimeError, TypeError):
                pass
        self._perf_worker_thread.requestInterruption()
        self._perf_worker_thread.quit()
        if not self._perf_worker_thread.wait(1200):
            logger.warning("系统监控线程关闭超时，强制终止后台采样")
            self._perf_worker_thread.terminate()
            self._perf_worker_thread.wait(1000)
        self._perf_worker_thread.deleteLater()
        self._perf_worker = None
        self._perf_worker_thread = None
        self._perf_poll_in_flight = False

    @Slot()
    def _request_sys_perf_update(self) -> None:
        if self._perf_worker_thread is None or self._perf_worker is None:
            return
        if self._perf_poll_in_flight:
            return
        self._perf_poll_in_flight = True
        self.perf_refresh_requested.emit()

    @Slot(object)
    def _on_sys_perf_ready(self, payload: object) -> None:
        self._perf_poll_in_flight = False
        if not hasattr(self, "sys_perf_label"):
            return
        if not isinstance(payload, dict):
            return
        cpu = payload.get("cpu")
        ram = payload.get("ram")
        gpu = payload.get("gpu")
        gpu_mem_used = payload.get("gpu_mem_used")
        gpu_mem_total = payload.get("gpu_mem_total")
        cpu_str = f"{cpu}%" if cpu is not None else "n/a"
        ram_str = f"{ram}%" if ram is not None else "n/a"
        gpu_str = f"{gpu}%" if gpu is not None else "n/a"
        if gpu_mem_used is not None and gpu_mem_total is not None:
            gpu_mem_str = f"{float(gpu_mem_used):.1f}/{float(gpu_mem_total):.1f} GB"
        else:
            gpu_mem_str = "n/a"
        self._set_text_if_changed(
            self.sys_perf_label,
            f"CPU: {cpu_str}  RAM: {ram_str}  GPU: {gpu_str}  显存: {gpu_mem_str}",
        )

    def _build_devices_panel(self) -> QFrame:
        card, layout = self._create_content_card("输入 / 监听设置", "")
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        source_layout = QVBoxLayout()
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(8)
        self.capture_source_label = self._set_compact_label_style(QLabel("监听来源", objectName="CompactLabel"))
        source_layout.addWidget(self.capture_source_label)
        source_row = QHBoxLayout()
        source_row.setContentsMargins(0, 0, 0, 0)
        source_row.setSpacing(12)
        source_row.addWidget(self.capture_source_mic_checkbox)
        source_row.addWidget(self.capture_source_program_checkbox)
        source_row.addStretch(1)
        source_layout.addLayout(source_row)
        layout.addLayout(source_layout)

        program_layout = QVBoxLayout()
        program_layout.setContentsMargins(0, 0, 0, 0)
        program_layout.setSpacing(8)
        self.program_combo_label = self._set_compact_label_style(QLabel("目标程序", objectName="CompactLabel"))
        program_layout.addWidget(self.program_combo_label)
        program_row = QHBoxLayout()
        program_row.setContentsMargins(0, 0, 0, 0)
        program_row.setSpacing(6)
        program_row.addWidget(self.program_combo, stretch=1)
        program_row.addWidget(self.refresh_program_button)
        program_layout.addLayout(program_row)
        layout.addLayout(program_layout)

        input_layout = QVBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(8)
        self.input_device_label_widget = self._set_compact_label_style(QLabel("输入设备", objectName="CompactLabel"))
        input_layout.addWidget(self.input_device_label_widget)
        input_layout.addWidget(self.input_device_combo)
        input_layout.addWidget(self._set_compact_label_style(QLabel("输入电平", objectName="CompactLabel")))
        input_meter_row = QHBoxLayout()
        input_meter_row.setContentsMargins(0, 0, 0, 0)
        input_meter_row.setSpacing(6)
        input_meter_row.addWidget(self.input_level_meter, stretch=1)
        input_meter_row.addWidget(self.input_level_state_label)
        input_layout.addLayout(input_meter_row)
        layout.addLayout(input_layout)

        monitor_layout = QVBoxLayout()
        monitor_layout.setContentsMargins(0, 0, 0, 0)
        monitor_layout.setSpacing(8)
        monitor_layout.addWidget(self._set_compact_label_style(QLabel("耳返设备（可选）", objectName="CompactLabel")))
        monitor_layout.addWidget(self.monitor_output_device_combo)
        layout.addLayout(monitor_layout)

        layout.addStretch(1)
        return card

    def _build_footer_panel(self) -> QFrame:
        card = QFrame()
        card.setObjectName("ContentCard")
        palette = self._theme_palette()
        card.setStyleSheet(
            "QFrame#ContentCard { "
            f"background-color: {palette['surface_bg']}; "
            "border-radius: 8px; "
            f"border: 1px solid {palette['surface_border']}; "
            "}"
        )
        self.footer_panel = card
        
        layout = QHBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(24)

        # 1. 左侧：系统状态
        sys_status_layout = QVBoxLayout()
        sys_status_layout.setSpacing(4)
        sys_status_title = QLabel("系统状态")
        sys_status_title.setStyleSheet("font-weight: bold; font-size: 11px;")
        sys_status_layout.addWidget(sys_status_title)
        
        status_indicator_layout = QHBoxLayout()
        status_indicator_layout.setSpacing(6)
        
        # CPU/RAM/GPU/显存 状态标签
        self.sys_perf_label = QLabel("CPU: --%  RAM: --%  GPU: --%  显存: --/-- GB")
        self.sys_perf_label.setStyleSheet(f"color: {self._theme_palette()['muted_text']}; font-size: 11px;")
        sys_status_layout.addWidget(self.sys_perf_label)
        
        # 移除底部的 status_indicator_layout 和 self.live_status_label
        sys_status_layout.addStretch(1)
        layout.addLayout(sys_status_layout)

        # 2. 中间：延迟信息
        delay_layout = QHBoxLayout()
        delay_layout.setSpacing(16)
        
        self.asr_delay_label = QLabel("ASR 延迟: --ms")
        self.trans_delay_label = QLabel("翻译延迟: --ms")
        self.tts_delay_label = QLabel("TTS 延迟: --ms")
        self.total_delay_label = QLabel("总延迟: --ms")
        
        for lbl in [self.asr_delay_label, self.trans_delay_label, self.tts_delay_label, self.total_delay_label]:
            lbl.setStyleSheet(f"color: {self._theme_palette()['muted_text']}; font-size: 11px;")
            delay_layout.addWidget(lbl)
            
        layout.addStretch(1)
        layout.addLayout(delay_layout)
        layout.addStretch(1)

        # 3. 右侧：操作按钮
        action_layout = QHBoxLayout()
        action_layout.setSpacing(12)
        
        self.live_start_button.setMinimumHeight(40)
        self.live_start_button.setMinimumWidth(120)
        
        action_layout.addWidget(self.live_start_button)
        
        layout.addLayout(action_layout)
        
        return card

    def _build_emotion_panel(self) -> QFrame:
        card, layout = self._create_content_card("情感控制", "")
        self.emotion_control_card = card
        header_row = getattr(card, "_header_row", None)
        if isinstance(header_row, QHBoxLayout):
            self.emotion_control_info_button = QPushButton()
            self.emotion_control_info_button.setObjectName("InlineInfoButton")
            self.emotion_control_info_button.setFlat(True)
            self._set_inline_info_button_style(self.emotion_control_info_button)
            self.emotion_control_info_button.setToolTip(self._emotion_control_tooltip())
            header_row.insertWidget(1, self.emotion_control_info_button, alignment=Qt.AlignVCenter)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.emotion_control_hint_label = QLabel("")
        self.emotion_control_hint_label.setObjectName("SectionHint")
        self.emotion_control_hint_label.setWordWrap(True)
        self.emotion_control_hint_label.setVisible(False)
        layout.addWidget(self.emotion_control_hint_label)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        # --- Left Sidebar ---
        sidebar = QFrame()
        sidebar.setObjectName("EmotionSidebar")
        sidebar.setFixedWidth(88)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(6)

        self.nav_preset_btn = QPushButton("情绪预设")
        self.nav_preset_btn.setObjectName("EmotionNavButton")
        self.nav_preset_btn.setCheckable(True)
        self.nav_preset_btn.setChecked(True)
        self.nav_preset_btn.clicked.connect(lambda: self._set_index_emotion_mode("preset"))

        self.nav_vector_btn = QPushButton("情感向量")
        self.nav_vector_btn.setObjectName("EmotionNavButton")
        self.nav_vector_btn.setCheckable(True)
        self.nav_vector_btn.clicked.connect(lambda: self._set_index_emotion_mode("vector"))

        sidebar_layout.addWidget(self.nav_preset_btn)
        sidebar_layout.addWidget(self.nav_vector_btn)
        sidebar_layout.addStretch(1)

        content_layout.addWidget(sidebar)

        # --- Right Stack ---
        self.emotion_stack = QStackedWidget()

        # Page 0: Preset
        preset_page = QWidget()
        preset_layout = QVBoxLayout(preset_page)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(8)

        preset_title = QLabel("选择预设")
        preset_title.setObjectName("CompactLabel")
        preset_layout.addWidget(preset_title)

        preset_grid = QGridLayout()
        preset_grid.setHorizontalSpacing(4)
        preset_grid.setVerticalSpacing(4)
        self.emotion_preset_buttons: list[QPushButton] = []
        for index, (label, _instruction, tone, emoji) in enumerate(EMOTION_PRESETS):
            button = QPushButton(f"{emoji}{label}")
            button.setObjectName("EmotionPresetButton")
            button.setProperty("tone", tone)
            button.setCheckable(True)
            button.setMinimumHeight(22)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setToolTip(label)
            button.clicked.connect(lambda checked=False, idx=index: self.emotion_preset_combo.setCurrentIndex(idx))
            preset_grid.addWidget(button, 0, index)
            preset_grid.setColumnStretch(index, 1)
            self.emotion_preset_buttons.append(button)
        preset_layout.addLayout(preset_grid)

        custom_title = QLabel("自定义语气")
        custom_title.setObjectName("CompactLabel")
        preset_layout.addWidget(custom_title)
        preset_layout.addWidget(self.emotion_custom_edit)

        auto_strength_row = QHBoxLayout()
        auto_strength_row.setContentsMargins(0, 0, 0, 0)
        auto_strength_row.setSpacing(6)
        auto_strength_row.addWidget(self._set_compact_label_style(QLabel("情感强度", objectName="CompactLabel")))
        self.auto_emotion_card_strength_slider = QSlider(Qt.Horizontal)
        self.auto_emotion_card_strength_slider.setRange(8, 20)
        self.auto_emotion_card_strength_slider.setTickInterval(1)
        self.auto_emotion_card_strength_slider.setTickPosition(QSlider.NoTicks)
        self.auto_emotion_card_strength_slider.setValue(int(round(self._auto_emotion_strength / 0.05)))
        self.auto_emotion_card_strength_label = QLabel(f"{self._auto_emotion_strength:.2f}")
        self.auto_emotion_card_strength_label.setObjectName("InlineValueLabel")
        self.auto_emotion_card_strength_label.setStyleSheet("font-size: 8px;")
        self.auto_emotion_card_strength_label.setFixedWidth(32)
        self.auto_emotion_card_strength_slider.valueChanged.connect(self._on_auto_emotion_strength_card_changed)
        auto_strength_row.addWidget(self.auto_emotion_card_strength_slider, stretch=1)
        auto_strength_row.addWidget(self.auto_emotion_card_strength_label)
        preset_layout.addLayout(auto_strength_row)
        preset_layout.addStretch(1)
        self.emotion_stack.addWidget(preset_page)

        # Page 1: Vector
        vector_page = QWidget()
        vector_layout = QVBoxLayout(vector_page)
        vector_layout.setContentsMargins(0, 0, 0, 0)
        vector_layout.setSpacing(8)

        vector_grid = QGridLayout()
        vector_grid.setContentsMargins(0, 0, 0, 0)
        vector_grid.setHorizontalSpacing(8)
        vector_grid.setVerticalSpacing(6)
        
        self.index_emo_rows = []
        for cn_name, en_name, emoji in INDEX_TTS_EMOTIONS:
            slider = self.index_emo_sliders[en_name]
            value_label = self.index_emo_value_labels[en_name]
            slider.setOrientation(Qt.Horizontal)
            slider.setFixedHeight(18)
            slider.setMinimumWidth(88)
            slider.setMaximumWidth(120)

            def _make_on_slider(s: QSlider, lbl: QLabel, item_title: str, item_widget: QWidget):
                def _on_move(_value: int | None = None) -> None:
                    current_text = f"{s.value() * 0.05:.2f}"
                    lbl.setText(current_text)
                    item_widget.setToolTip(f"{item_title}: {current_text}")
                return _on_move

            item_container = QFrame()
            item_container.setObjectName("EmotionVectorItem")
            item_layout = QHBoxLayout(item_container)
            item_layout.setContentsMargins(8, 5, 8, 5)
            item_layout.setSpacing(6)
            emoji_label = QLabel(emoji)
            emoji_label.setObjectName("EmotionEmojiBadge")
            emoji_label.setProperty("tone", en_name)
            emoji_label.setAlignment(Qt.AlignCenter)
            value_label.setObjectName("EmotionSliderValue")
            cn_label = QLabel(cn_name)
            cn_label.setObjectName("EmotionSliderName")
            emoji_label.setFixedSize(22, 22)
            cn_label.setFixedWidth(28)
            value_label.setFixedWidth(34)
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            item_layout.addWidget(emoji_label)
            item_layout.addWidget(cn_label)
            item_layout.addWidget(slider, stretch=1)
            item_layout.addWidget(value_label)
            slider.valueChanged.connect(_make_on_slider(slider, value_label, cn_name, item_container))
            
            value_label.setText(f"{slider.value() * 0.05:.2f}")
            item_container.setToolTip(f"{cn_name}: {value_label.text()}")
            index = len(self.index_emo_rows)
            vector_grid.addWidget(item_container, index // 2, index % 2)
            self.index_emo_rows.append(item_container)
            
        vector_layout.addLayout(vector_grid)

        alpha_row = QHBoxLayout()
        alpha_row.setContentsMargins(0, 0, 0, 0)
        alpha_row.setSpacing(6)
        alpha_row.addWidget(self._set_compact_label_style(QLabel("情感强度", objectName="CompactLabel")))
        self.index_emo_alpha_slider.setTickPosition(QSlider.NoTicks)
        alpha_row.addWidget(self.index_emo_alpha_slider, stretch=1)
        alpha_row.addWidget(self.index_emo_alpha_label)
        vector_layout.addLayout(alpha_row)

        self.emotion_stack.addWidget(vector_page)

        content_layout.addWidget(self.emotion_stack, stretch=1)
        layout.addLayout(content_layout)

        self.emotion_preset_combo.currentIndexChanged.connect(self._sync_emotion_preset_buttons)
        self._on_index_emotion_mode_changed()
        self._sync_emotion_preset_buttons()
        self._on_auto_emotion_toggled()
        self.emotion_control_overlay = self._create_mode_locked_overlay(card)
        return card

    def _build_reference_panel(self) -> tuple[QFrame, QFrame, QFrame]:
        card, layout = self._create_content_card("输出 / 变声 / TTS 控制", "")
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        def create_section(title_text: str) -> tuple[QFrame, QVBoxLayout]:
            section = QFrame()
            section.setObjectName("SummaryCard")
            section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            section_layout = QVBoxLayout(section)
            section_layout.setContentsMargins(12, 10, 12, 10)
            section_layout.setSpacing(7)
            section_layout.addWidget(self._set_section_title_style(QLabel(title_text, objectName="CompactLabel")))
            return section, section_layout

        reference_section, reference_layout = create_section("参考音色")

        reference_row = QHBoxLayout()
        reference_row.setContentsMargins(0, 0, 0, 0)
        reference_row.setSpacing(8)
        reference_row.addWidget(self.live_reference_audio_edit, stretch=1)
        self.reference_browse_button = QPushButton("选择音色")
        self.reference_browse_button.setObjectName("OutputActionButton")
        self.reference_browse_button.setIcon(render_svg_template_icon("select_voice", color=self._theme_icon_color("accent")))
        self.reference_browse_button.setIconSize(QSize(14, 14))
        self.reference_browse_button.clicked.connect(self._browse_live_reference_audio)
        self._set_compact_button_style(self.reference_browse_button, min_width=98)
        reference_row.addWidget(self.reference_browse_button)
        reference_layout.addLayout(reference_row)
        layout.addWidget(reference_section)

        tts_section, tts_layout = create_section("语音输出设置")
        self.voice_output_settings_section = tts_section
        tts_grid = QGridLayout()
        tts_grid.setContentsMargins(0, 0, 0, 0)
        tts_grid.setHorizontalSpacing(10)
        tts_grid.setVerticalSpacing(8)
        tts_grid.addWidget(self._set_compact_label_style(QLabel("语音输出模型", objectName="CompactLabel")), 0, 0)
        tts_grid.addWidget(self.tts_engine_combo, 1, 0, 1, 2)
        tts_layout.addLayout(tts_grid)
        layout.addWidget(tts_section)

        translation_section, translation_layout = create_section("翻译")
        translation_header = QHBoxLayout()
        translation_header.setContentsMargins(0, 0, 0, 0)
        translation_header.setSpacing(6)
        translation_header.addWidget(self._set_compact_label_style(QLabel("自动翻译", objectName="CompactLabel")))
        self.translation_card_info_button = QPushButton()
        self.translation_card_info_button.setObjectName("InlineInfoButton")
        self.translation_card_info_button.setFlat(True)
        self._set_inline_info_button_style(self.translation_card_info_button)
        self.translation_card_info_button.setToolTip(self._translation_mode_tooltip())
        translation_header.addWidget(self.translation_card_info_button, alignment=Qt.AlignVCenter)
        translation_header.addStretch(1)
        self.translation_card_checkbox = ToggleSwitch("启用")
        self._set_toggle_switch_style(self.translation_card_checkbox)
        self.translation_card_checkbox.setChecked(self._translation_enabled)
        self.translation_card_checkbox.toggled.connect(self._on_translation_card_toggled)
        translation_header.addWidget(self.translation_card_checkbox)
        translation_layout.addLayout(translation_header)

        translation_grid = QGridLayout()
        translation_grid.setContentsMargins(0, 0, 0, 0)
        translation_grid.setHorizontalSpacing(10)
        translation_grid.setVerticalSpacing(8)
        translation_grid.addWidget(self._set_compact_label_style(QLabel("目标语言", objectName="CompactLabel")), 0, 0)
        translation_grid.addWidget(self._set_compact_label_style(QLabel("翻译节点", objectName="CompactLabel")), 0, 1)
        self.translation_card_target_language_combo = QComboBox()
        for label, language_code in OUTPUT_LANGUAGE_OPTIONS:
            self.translation_card_target_language_combo.addItem(f"翻译到{label}", language_code)
        self._set_compact_field_style(self.translation_card_target_language_combo, min_height=42)
        self.translation_card_target_language_combo.currentIndexChanged.connect(self._on_translation_target_language_changed)
        translation_grid.addWidget(self.translation_card_target_language_combo, 1, 0)
        self.translation_card_node_combo = QComboBox()
        self._set_compact_field_style(self.translation_card_node_combo, min_height=42)
        self.translation_card_node_combo.currentIndexChanged.connect(self._on_translation_card_node_changed)
        translation_grid.addWidget(self.translation_card_node_combo, 1, 1)
        translation_layout.addLayout(translation_grid)
        layout.addWidget(translation_section)

        output_section, output_layout = create_section("输出设备")
        output_row = QHBoxLayout()
        output_row.setContentsMargins(0, 0, 0, 0)
        output_row.setSpacing(8)
        output_row.addWidget(self.output_device_combo, stretch=1)
        self.use_virtual_output_button.setObjectName("OutputActionButton")
        self._set_compact_button_style(self.use_virtual_output_button, min_width=98)
        output_row.addWidget(self.use_virtual_output_button)
        output_layout.addLayout(output_row)
        layout.addWidget(output_section)

        self._refresh_translation_node_combo()
        layout.addStretch(1)

        speech_volume_card = self._build_speech_volume_card()
        tts_other_options_card = self._build_tts_other_options_card()
        
        return card, speech_volume_card, tts_other_options_card

    def _build_speech_volume_card(self) -> QFrame:
        card, layout = self._create_content_card("原生语速 / 音量", "")
        card.setToolTip("Index-TTS 语速使用原生 duration_factor；输出音量和耳返输出音量会作用于 Index-TTS 和 MeanVC2 输出。")
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        rate_layout = QVBoxLayout()
        rate_layout.setContentsMargins(0, 0, 0, 0)
        rate_layout.setSpacing(6)
        rate_header = QHBoxLayout()
        rate_header.setContentsMargins(0, 0, 0, 0)
        rate_header.setSpacing(6)
        rate_header.addWidget(self._set_compact_label_style(self.speech_rate_title_label))
        rate_header.addStretch(1)
        rate_header.addWidget(self.speech_rate_label)
        rate_layout.addLayout(rate_header)
        rate_layout.addWidget(self.speech_rate_slider)
        layout.addLayout(rate_layout)

        output_volume_layout = QVBoxLayout()
        output_volume_layout.setContentsMargins(0, 0, 0, 0)
        output_volume_layout.setSpacing(6)
        output_volume_header = QHBoxLayout()
        output_volume_header.setContentsMargins(0, 0, 0, 0)
        output_volume_header.setSpacing(6)
        output_volume_header.addWidget(self._set_compact_label_style(QLabel("输出音量", objectName="CompactLabel")))
        output_volume_header.addStretch(1)
        output_volume_header.addWidget(self.output_gain_label)
        output_volume_layout.addLayout(output_volume_header)
        output_volume_layout.addWidget(self.output_gain_slider)
        layout.addLayout(output_volume_layout)

        monitor_output_volume_layout = QVBoxLayout()
        monitor_output_volume_layout.setContentsMargins(0, 0, 0, 0)
        monitor_output_volume_layout.setSpacing(6)
        monitor_output_volume_header = QHBoxLayout()
        monitor_output_volume_header.setContentsMargins(0, 0, 0, 0)
        monitor_output_volume_header.setSpacing(6)
        monitor_output_volume_header.addWidget(self._set_compact_label_style(QLabel("耳返输出音量", objectName="CompactLabel")))
        monitor_output_volume_header.addStretch(1)
        monitor_output_volume_header.addWidget(self.monitor_output_volume_label)
        monitor_output_volume_layout.addLayout(monitor_output_volume_header)
        monitor_output_volume_layout.addWidget(self.monitor_output_volume_slider)
        layout.addLayout(monitor_output_volume_layout)

        layout.addStretch(1)
        return card

    def _build_tts_other_options_card(self) -> QFrame:
        card, layout = self._create_content_card("TTS 其他选项", "")
        self.tts_other_options_card = card
        header_row = getattr(card, "_header_row", None)
        if isinstance(header_row, QHBoxLayout):
            self.tts_other_options_info_button = QPushButton()
            self.tts_other_options_info_button.setObjectName("InlineInfoButton")
            self.tts_other_options_info_button.setFlat(True)
            self._set_inline_info_button_style(self.tts_other_options_info_button)
            self.tts_other_options_info_button.setToolTip(self._tts_other_options_tooltip())
            header_row.insertWidget(1, self.tts_other_options_info_button, alignment=Qt.AlignVCenter)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.tts_other_options_hint_label = QLabel("")
        self.tts_other_options_hint_label.setObjectName("SectionHint")
        self.tts_other_options_hint_label.setWordWrap(True)
        self.tts_other_options_hint_label.setVisible(False)
        layout.addWidget(self.tts_other_options_hint_label)

        self.auto_emotion_card_checkbox = ToggleSwitch("启用实验情感识别")
        self._set_toggle_switch_style(self.auto_emotion_card_checkbox)
        self.auto_emotion_card_checkbox.setChecked(self._auto_emotion_feature_enabled)
        self.auto_emotion_card_checkbox.toggled.connect(self._on_auto_emotion_card_toggled)
        layout.addWidget(self.auto_emotion_card_checkbox)

        auto_emotion_hint = QLabel("勾选麦克风监听时可用，会跟随语音情绪自动调整。")
        auto_emotion_hint.setObjectName("SectionHint")
        auto_emotion_hint.setWordWrap(True)
        auto_emotion_hint.setStyleSheet("font-size: 8px;")
        layout.addWidget(auto_emotion_hint)

        self.experimental_torch_compile_card_checkbox = ToggleSwitch("实验加速（进阶模式）")
        self._set_toggle_switch_style(self.experimental_torch_compile_card_checkbox)
        self.experimental_torch_compile_card_checkbox.setChecked(self._experimental_torch_compile_enabled)
        self.experimental_torch_compile_card_checkbox.toggled.connect(self._on_experimental_torch_compile_card_toggled)
        layout.addWidget(self.experimental_torch_compile_card_checkbox)

        self.experimental_torch_compile_hint_label = QLabel("")
        self.experimental_torch_compile_hint_label.setObjectName("SectionHint")
        self.experimental_torch_compile_hint_label.setWordWrap(True)
        self.experimental_torch_compile_hint_label.setStyleSheet("font-size: 8px;")
        layout.addWidget(self.experimental_torch_compile_hint_label)

        self.experimental_accel_gpt_card_checkbox = ToggleSwitch("实验加速（极速模式）")
        self._set_toggle_switch_style(self.experimental_accel_gpt_card_checkbox)
        self.experimental_accel_gpt_card_checkbox.setChecked(self._experimental_accel_gpt_enabled)
        self.experimental_accel_gpt_card_checkbox.toggled.connect(self._on_experimental_accel_gpt_card_toggled)
        layout.addWidget(self.experimental_accel_gpt_card_checkbox)

        accel_hint = QLabel("更激进的提速方案，通常更占显存，开启后需要重启。")
        accel_hint.setObjectName("SectionHint")
        accel_hint.setWordWrap(True)
        accel_hint.setStyleSheet("font-size: 8px;")
        layout.addWidget(accel_hint)

        self._sync_tts_other_option_controls()
        layout.addStretch(1)
        self.tts_other_options_overlay = self._create_mode_locked_overlay(card)
        return card

    def _build_workspace_panel(self) -> QFrame:
        card, layout = self._create_content_card("", "")
        title_item = layout.takeAt(0)
        if title_item is not None and title_item.widget() is not None:
            title_item.widget().deleteLater()

        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.setSpacing(10)
        self.workspace_title_label = QLabel("实时对话与翻译")
        self.workspace_title_label.setObjectName("CardTitle")
        self.workspace_counter_label = QLabel("0 条记录")
        self.workspace_counter_label.setObjectName("WorkspaceStat")
        toolbar_row.addWidget(self.workspace_title_label)
        toolbar_row.addWidget(self.workspace_counter_label)
        toolbar_row.addStretch(1)
        self.workspace_clear_button = QPushButton("清空记录")
        self.workspace_clear_button.setObjectName("WorkspaceGhostButton")
        self._set_compact_button_style(self.workspace_clear_button, min_height=34, min_width=88)
        self.workspace_clear_button.setIcon(render_svg_template_icon("clear_history", color=self._theme_icon_color("muted")))
        self.workspace_clear_button.setIconSize(QSize(14, 14))
        self.workspace_clear_button.setToolTip("清空当前对话记录")
        self.workspace_clear_button.clicked.connect(self._clear_workspace_chat_history)
        self.workspace_export_button = QPushButton("导出记录")
        self.workspace_export_button.setObjectName("WorkspaceGhostButton")
        self._set_compact_button_style(self.workspace_export_button, min_height=34, min_width=88)
        self.workspace_export_button.setIcon(render_svg_template_icon("export_history", color=self._theme_icon_color("muted")))
        self.workspace_export_button.setIconSize(QSize(14, 14))
        self.workspace_export_button.setToolTip("导出当前对话记录")
        self.workspace_export_button.clicked.connect(self._export_workspace_chat_history)
        toolbar_row.addWidget(self.workspace_clear_button)
        toolbar_row.addWidget(self.workspace_export_button)
        layout.addLayout(toolbar_row)

        self.transcript_intro_label = QLabel("中间区域按双语记录流展示：程序监听只做转写并标记说话人，只有麦克风结果会进入后续 TTS。")
        self.transcript_intro_label.setObjectName("SectionHint")
        self.transcript_intro_label.setWordWrap(True)
        self.transcript_intro_label.setVisible(False)
        layout.addWidget(self.transcript_intro_label)

        self.workspace_placeholder_label = QLabel("待命中，可同时开启麦克风与目标程序监听；也可以在下方发送文字。")
        self.workspace_placeholder_label.setObjectName("WorkspaceStateLine")
        self.workspace_placeholder_label.setProperty("state", "ready")
        self.workspace_placeholder_label.setWordWrap(True)
        self.workspace_placeholder_label.setVisible(False)
        layout.addWidget(self.workspace_placeholder_label)

        self.chat_history_area = QScrollArea()
        self.chat_history_area.setObjectName("TranscriptView")
        self.chat_history_area.setWidgetResizable(True)
        self.chat_history_area.setFrameShape(QFrame.NoFrame)
        
        self.chat_history_area.setStyleSheet(self._build_transcript_scroll_style())
        
        self.chat_history_container = QWidget()
        self.chat_history_container.setStyleSheet("QWidget { background: transparent; }")
        self.chat_history_layout = QVBoxLayout(self.chat_history_container)
        self.chat_history_layout.setContentsMargins(0, 16, 0, 16)
        self.chat_history_layout.setSpacing(0)
        self.chat_history_layout.addStretch()
        
        self.chat_history_area.setWidget(self.chat_history_container)
        scroll_bar = self.chat_history_area.verticalScrollBar()
        scroll_up_icon = render_svg_template_icon("scroll_up", color=self._theme_icon_color("muted"))
        scroll_down_icon = render_svg_template_icon("scroll_down", color=self._theme_icon_color("muted"))

        scroll_button_parent = scroll_bar if scroll_bar is not None else self.chat_history_area

        self.chat_scroll_up_button = QPushButton(scroll_button_parent)
        self.chat_scroll_up_button.setObjectName("TranscriptScrollButton")
        self.chat_scroll_up_button.setCursor(Qt.PointingHandCursor)
        self.chat_scroll_up_button.setIcon(scroll_up_icon)
        self.chat_scroll_up_button.setIconSize(QSize(18, 18))
        self.chat_scroll_up_button.setFixedSize(28, 28)
        self.chat_scroll_up_button.setToolTip("向上滚动")
        self.chat_scroll_up_button.clicked.connect(lambda: self._scroll_chat_history(-1))

        self.chat_scroll_down_button = QPushButton(scroll_button_parent)
        self.chat_scroll_down_button.setObjectName("TranscriptScrollButton")
        self.chat_scroll_down_button.setCursor(Qt.PointingHandCursor)
        self.chat_scroll_down_button.setIcon(scroll_down_icon)
        self.chat_scroll_down_button.setIconSize(QSize(18, 18))
        self.chat_scroll_down_button.setFixedSize(28, 28)
        self.chat_scroll_down_button.setToolTip("向下滚动")
        self.chat_scroll_down_button.clicked.connect(lambda: self._scroll_chat_history(1))

        scroll_button_style = self._build_transcript_scroll_button_style()
        self.chat_scroll_up_button.setStyleSheet(scroll_button_style)
        self.chat_scroll_down_button.setStyleSheet(scroll_button_style)
        self.chat_scroll_up_button.raise_()
        self.chat_scroll_down_button.raise_()
        layout.addWidget(self.chat_history_area, stretch=4)

        if scroll_bar is not None:
            scroll_bar.valueChanged.connect(self._refresh_chat_scroll_buttons)
            scroll_bar.rangeChanged.connect(lambda *_: (self._refresh_chat_scroll_buttons(), self._position_chat_scroll_buttons()))
        QTimer.singleShot(0, self._position_chat_scroll_buttons)
        self._refresh_chat_scroll_buttons()
        
        # We keep a dictionary to map message_id to widget
        self._chat_message_widgets = {}

        composer_row = QHBoxLayout()
        composer_row.setContentsMargins(0, 0, 0, 0)
        composer_row.setSpacing(8)
        self.workspace_text_input = QLineEdit()
        self.workspace_text_input.setPlaceholderText("可选：输入文本进行发送（支持翻译与 TTS）")
        self.workspace_text_input.textChanged.connect(self._refresh_workspace_text_metrics)
        self.workspace_text_input.returnPressed.connect(self._on_workspace_text_submit)
        composer_row.addWidget(self.workspace_text_input, stretch=1)
        self.workspace_send_button = QPushButton("发送")
        self.workspace_send_button.setObjectName("PrimaryActionButton")
        self._set_compact_button_style(self.workspace_send_button, min_height=36, min_width=104)
        self.workspace_send_button.clicked.connect(self._on_workspace_text_submit)
        composer_row.addWidget(self.workspace_send_button)
        layout.addLayout(composer_row)
        self.workspace_wave_hint = QLabel("待命中")
        self.workspace_wave_hint.setObjectName("WorkspaceStat")
        self.workspace_wave_hint.setVisible(False)
        layout.addWidget(self.workspace_wave_hint)

        self.workspace_waveform_frame = QFrame()
        self.workspace_waveform_frame.setObjectName("WaveformCard")
        waveform_layout = QVBoxLayout(self.workspace_waveform_frame)
        waveform_layout.setContentsMargins(0, 0, 0, 0)
        waveform_layout.setSpacing(0)
        self.waveform_widget = WaveformWidget()
        waveform_layout.addWidget(self.waveform_widget)
        layout.addWidget(self.workspace_waveform_frame, stretch=2)
        self.workspace_waveform_frame.setVisible(False)
        self._render_chat_messages()
        self._refresh_workspace_text_metrics()
        return card

    def _next_chat_message_id(self) -> str:
        self._chat_message_seq += 1
        return f"chat-{self._chat_message_seq}"

    def _scroll_chat_history(self, direction: int) -> None:
        if not hasattr(self, "chat_history_area"):
            return
        scroll_bar = self.chat_history_area.verticalScrollBar()
        if scroll_bar is None:
            return
        step = max(96, int(scroll_bar.pageStep() * 0.85))
        scroll_bar.setValue(scroll_bar.value() + (step * direction))
        self._refresh_chat_scroll_buttons()
        self._position_chat_scroll_buttons()

    def _position_chat_scroll_buttons(self) -> None:
        if not hasattr(self, "chat_history_area"):
            return
        if not hasattr(self, "chat_scroll_up_button") or not hasattr(self, "chat_scroll_down_button"):
            return
        scroll_bar = self.chat_history_area.verticalScrollBar()
        if scroll_bar is None:
            return

        bar_geom = scroll_bar.geometry()
        if not bar_geom.isValid() or bar_geom.width() <= 0:
            self.chat_scroll_up_button.hide()
            self.chat_scroll_down_button.hide()
            return

        button_size = self.chat_scroll_up_button.width()
        x = max(0, (scroll_bar.width() - button_size) // 2)
        top_y = 2
        bottom_y = scroll_bar.height() - self.chat_scroll_down_button.height() - 2

        self.chat_scroll_up_button.move(x, top_y)
        self.chat_scroll_down_button.move(x, max(top_y + self.chat_scroll_up_button.height() + 4, bottom_y))
        self.chat_scroll_up_button.raise_()
        self.chat_scroll_down_button.raise_()
        self.chat_scroll_up_button.show()
        self.chat_scroll_down_button.show()

    def _refresh_chat_scroll_buttons(self) -> None:
        if not hasattr(self, "chat_history_area"):
            return
        scroll_bar = self.chat_history_area.verticalScrollBar()
        if scroll_bar is None:
            return
        can_scroll = scroll_bar.maximum() > scroll_bar.minimum()
        if hasattr(self, "chat_scroll_up_button"):
            self.chat_scroll_up_button.setEnabled(can_scroll and scroll_bar.value() > scroll_bar.minimum())
        if hasattr(self, "chat_scroll_down_button"):
            self.chat_scroll_down_button.setEnabled(can_scroll and scroll_bar.value() < scroll_bar.maximum())
        self._position_chat_scroll_buttons()

    @staticmethod
    def _format_chat_text(text: str) -> str:
        return escape(text).replace("\n", "<br>")

    @staticmethod
    def _resolve_chat_role(input_mode: str, speaker_label: str) -> str:
        normalized_mode = str(input_mode or "").strip().lower()
        normalized_speaker = str(speaker_label or "").strip()
        if normalized_mode == "program":
            return "program"
        if normalized_mode == "dual" and normalized_speaker and normalized_speaker != "麦克风":
            return "program"
        if normalized_mode in {"microphone", "text"}:
            return "self"
        if normalized_speaker == "麦克风":
            return "self"
        return "self"

    def _resolve_chat_card_palette(self, chat_role: str, state: str) -> tuple[str, str, str, str, str, str]:
        palette = self._theme_palette()
        if state == "pending":
            return (
                palette["warning_bg"],
                palette["warning_border"],
                palette["warning_text"],
                palette["warning_text"],
                palette["text"],
                palette["muted_text"],
            )
        if state == "error":
            return (
                palette["error_bg"],
                palette["error_border"],
                palette["error_text"],
                palette["error_text"],
                palette["text"],
                palette["muted_text"],
            )
        match chat_role:
            case "program":
                return (
                    palette["tts_official_bg"],
                    palette["tts_official_border"],
                    palette["tts_official_text"],
                    palette["tts_official_text"],
                    palette["text"],
                    palette["muted_text"],
                )
            case "self":
                return (
                    palette["success_bg"],
                    palette["success_border"],
                    palette["success_text"],
                    palette["success_text"],
                    palette["text"],
                    palette["muted_text"],
                )
            case _:
                return (
                    palette["surface_bg"],
                    palette["surface_border"],
                    palette["accent_soft"],
                    palette["accent_soft"],
                    palette["text"],
                    palette["muted_text"],
                )

    @staticmethod
    def _resolve_chat_alignment(chat_role: str) -> tuple[str, str, str]:
        if chat_role == "program":
            return "left", "left", "left"
        return "right", "right", "right"

    @staticmethod
    def _resolve_chat_translated_block(translated_display: str, state: str, *, has_translation_target: bool) -> str:
        translated_block = translated_display or ("翻译中..." if state == "pending" and has_translation_target else "")
        if not translated_block and state == "error":
            return "翻译失败"
        return translated_block

    @staticmethod
    def _resolve_chat_text_blocks(
        source_text: str,
        translated_display: str,
        state: str,
        *,
        has_translation_target: bool,
    ) -> tuple[str, str]:
        translated_block = MainWindow._resolve_chat_translated_block(
            translated_display,
            state,
            has_translation_target=has_translation_target,
        )
        if translated_display:
            return translated_display, source_text
        if state == "pending" and translated_block:
            return translated_block, source_text
        return source_text, translated_block

    @staticmethod
    def _resolve_chat_meter(input_mode: str) -> str:
        return "▏▎▍▌▍▎" if input_mode in {"microphone", "program", "dual"} else "▏▎▍"

    def _build_chat_message_view_model(self, item: dict[str, object]) -> dict[str, str] | None:
        source_text = str(item.get("source_text") or "").strip()
        if not source_text:
            return None
        translated_text = item.get("translated_text")
        translated_display = str(translated_text).strip() if translated_text is not None else ""
        state = str(item.get("state") or "ready")
        input_mode = str(item.get("input_mode") or "text")
        has_translation_target = bool(str(item.get("target_language") or "").strip())
        timestamp = str(item.get("created_at") or "--:--:--")
        speaker_label = str(item.get("speaker_label") or "").strip()
        chat_role = self._resolve_chat_role(input_mode, speaker_label)
        primary_text, secondary_text = self._resolve_chat_text_blocks(
            source_text,
            translated_display,
            state,
            has_translation_target=has_translation_target,
        )
        (
            card_bg,
            card_border,
            translation_color,
            accent_color,
            primary_text_color,
            timestamp_color,
        ) = self._resolve_chat_card_palette(chat_role, state)
        bubble_align, timestamp_align, meter_align = self._resolve_chat_alignment(chat_role)
        return {
            "message_id": str(item.get("message_id") or ""),
            "timestamp": timestamp,
            "speaker_label": speaker_label,
            "card_bg": card_bg,
            "card_border": card_border,
            "translation_color": translation_color,
            "accent_color": accent_color,
            "primary_text_color": primary_text_color,
            "timestamp_color": timestamp_color,
            "button_hover_color": primary_text_color,
            "bubble_align": bubble_align,
            "timestamp_align": timestamp_align,
            "meter_align": meter_align,
            "decorative_meter": self._resolve_chat_meter(input_mode),
            "primary_html": self._format_chat_text(primary_text),
            "secondary_html": self._format_chat_text(secondary_text),
        }

    def _format_chat_language_label(self, language: str | None, *, fallback: str) -> str:
        normalized = str(language or "").strip().lower()
        if normalized in {"zh", "en", "ja"}:
            return self._format_output_language_label(normalized)
        return fallback

    def _upsert_chat_message(self, payload: dict[str, object]) -> str:
        message_id = str(payload.get("message_id") or "").strip() or self._next_chat_message_id()
        entry = next((item for item in self._chat_messages if item.get("message_id") == message_id), None)
        if entry is None:
            entry = {"message_id": message_id}
            self._chat_messages.append(entry)
        entry.update(
            {
                "message_id": message_id,
                "created_at": str(payload.get("created_at") or entry.get("created_at") or time.strftime("%H:%M:%S")),
                "input_mode": str(payload.get("input_mode") or entry.get("input_mode") or "text"),
                "source_text": str(payload.get("source_text") or entry.get("source_text") or "").strip(),
                "speaker_label": str(payload.get("speaker_label") or entry.get("speaker_label") or "").strip(),
                "translated_text": (
                    None
                    if payload.get("translated_text") is None
                    else str(payload.get("translated_text") or "").strip()
                ),
                "source_language": str(payload.get("source_language") or entry.get("source_language") or "").strip() or None,
                "target_language": str(payload.get("target_language") or entry.get("target_language") or "").strip() or None,
                "state": str(payload.get("state") or entry.get("state") or "ready"),
            }
        )
        self._render_chat_messages()
        self._refresh_workspace_text_metrics()
        return message_id

    def _render_chat_messages(self) -> None:
        if not hasattr(self, "chat_history_area"):
            return
            
        if not self._chat_messages:
            # clear all widgets
            for msg_id, widget in self._chat_message_widgets.items():
                self.chat_history_layout.removeWidget(widget)
                widget.deleteLater()
            self._chat_message_widgets.clear()
            self._refresh_chat_scroll_buttons()
            return

        for item in self._chat_messages:
            msg_id = item.get("message_id")
            if not msg_id:
                continue
            view_model = self._build_chat_message_view_model(item)
            if view_model is None:
                continue
                
            if msg_id in self._chat_message_widgets:
                widget = self._chat_message_widgets[msg_id]
                widget.update_view_model(view_model)
            else:
                widget = ChatMessageWidget()
                widget.replay_clicked.connect(self._on_chat_replay_clicked)
                widget.broadcast_clicked.connect(self._on_chat_broadcast_clicked)
                widget.update_view_model(view_model)
                self._chat_message_widgets[msg_id] = widget
                # insert before stretch
                self.chat_history_layout.insertWidget(self.chat_history_layout.count() - 1, widget)

        # scroll to bottom
        scroll_bar = self.chat_history_area.verticalScrollBar()
        if scroll_bar is not None:
            QTimer.singleShot(10, lambda: (scroll_bar.setValue(scroll_bar.maximum()), self._refresh_chat_scroll_buttons()))

    def _on_chat_replay_clicked(self, message_id: str) -> None:
        logger.info(f"Replay requested for message: {message_id}")
        if self._live_worker is None:
            self._show_message(QMessageBox.Information, "暂未就绪", "请先启动主程序，再尝试重播。")
            return
        self.live_replay_requested.emit(message_id, False)
        
    def _on_chat_broadcast_clicked(self, message_id: str) -> None:
        logger.info(f"Broadcast requested for message: {message_id}")
        if self._live_worker is None:
            self._show_message(QMessageBox.Information, "暂未就绪", "请先启动主程序，再尝试发送。")
            return
        self.live_replay_requested.emit(message_id, True)

    def _clear_workspace_chat_history(self) -> None:
        self._chat_messages.clear()
        self._render_chat_messages()
        self._refresh_workspace_text_metrics()
        if hasattr(self, "workspace_placeholder_label"):
            self.workspace_placeholder_label.setProperty("state", "ready")
            self._set_text_if_changed(self.workspace_placeholder_label, "记录已清空，可继续开启麦克风或发送文字。")
            self._refresh_widget_style(self.workspace_placeholder_label)

    def _export_workspace_chat_history(self) -> None:
        if not self._chat_messages:
            self._show_message(QMessageBox.Information, "暂无记录", "当前没有可导出的对话记录。")
            return
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出对话记录",
            str(Path.cwd() / "chat_history.zip"),
            "ZIP 压缩包 (*.zip);;所有文件 (*)",
        )
        if not file_path:
            return
        export_path = Path(file_path)
        if export_path.suffix.lower() != ".zip":
            export_path = export_path.with_suffix(".zip")
        export_items: list[dict[str, object]] = []
        for item in self._chat_messages:
            source_text = str(item.get("source_text") or "").strip()
            if not source_text:
                continue
            view_model = self._build_chat_message_view_model(item)
            if view_model is None:
                continue
            message_id = str(item.get("message_id") or "").strip()
            audio_path = self._resolve_export_cached_audio_path(message_id)
            archive_audio_name = None
            if audio_path:
                archive_audio_name = f"audio/{Path(audio_path).name}"
            export_items.append(
                {
                    "item": item,
                    "view_model": view_model,
                    "audio_path": audio_path,
                    "archive_audio_name": archive_audio_name,
                }
            )
        try:
            html_content = self._build_workspace_chat_export_html(export_items)
            with zipfile.ZipFile(export_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("index.html", html_content)
                for export_item in export_items:
                    audio_path = export_item.get("audio_path")
                    archive_audio_name = export_item.get("archive_audio_name")
                    if isinstance(audio_path, str) and audio_path and isinstance(archive_audio_name, str) and archive_audio_name:
                        archive.write(audio_path, archive_audio_name)
        except Exception as exc:
            logger.error("导出对话记录失败: {}", exc)
            self._show_message(QMessageBox.Critical, "导出失败", f"无法导出记录：{exc}")
            return
        self._show_message(QMessageBox.Information, "导出成功", f"已导出到：{export_path}")

    @staticmethod
    def _resolve_export_cached_audio_path(message_id: str) -> str | None:
        normalized_message_id = str(message_id or "").strip()
        if not normalized_message_id:
            return None
        os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
        for ext in (".ogg", ".wav"):
            candidate = Path(AUDIO_CACHE_DIR) / f"{normalized_message_id}{ext}"
            if candidate.exists():
                return str(candidate)
        return None

    def _build_workspace_chat_export_html(self, export_items: list[dict[str, object]]) -> str:
        cards: list[str] = []
        for export_item in export_items:
            item = export_item.get("item")
            view_model = export_item.get("view_model")
            if not isinstance(item, dict) or not isinstance(view_model, dict):
                continue
            bubble_align = str(view_model.get("bubble_align") or "left")
            message_role = "self" if bubble_align == "right" else "program"
            speaker_label = str(item.get("speaker_label") or "").strip()
            timestamp = str(item.get("created_at") or "--:--:--")
            primary_html = str(view_model.get("primary_html") or "")
            secondary_html = str(view_model.get("secondary_html") or "")
            state = str(item.get("state") or "ready").strip().lower()
            archive_audio_name = export_item.get("archive_audio_name")
            audio_block = ""
            if isinstance(archive_audio_name, str) and archive_audio_name:
                audio_block = (
                    '<div class="audio-block">'
                    f'<audio controls preload="none" src="{escape(archive_audio_name)}"></audio>'
                    f'<div class="audio-caption">{escape(Path(archive_audio_name).name)}</div>'
                    "</div>"
                )
            speaker_html = f'<div class="speaker">{escape(speaker_label)}</div>' if speaker_label else ""
            secondary_block = f'<div class="secondary">{secondary_html}</div>' if secondary_html else ""
            state_attr = escape(state or "ready")
            cards.append(
                f'''
<article class="message {message_role} state-{state_attr}">
  <div class="meta">
    <span class="time">{escape(timestamp)}</span>
  </div>
  <div class="bubble">
    {speaker_html}
    <div class="primary">{primary_html}</div>
    {secondary_block}
    {audio_block}
  </div>
</article>'''.strip()
            )
        cards_html = "\n".join(cards) or '<div class="empty">当前没有可导出的记录。</div>'
        exported_at = escape(time.strftime("%Y-%m-%d %H:%M:%S"))
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Revoice 导出记录</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --card: #ffffff;
      --border: #d9e1ec;
      --text: #17212f;
      --muted: #5b6b82;
      --self-bg: #ecfdf3;
      --self-border: #9dd9b0;
      --program-bg: #eef4ff;
      --program-border: #b8ccff;
      --pending-bg: #fff8e7;
      --pending-border: #f1d18a;
      --error-bg: #fff0f0;
      --error-border: #efb0b0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .page {{
      width: min(980px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 40px;
    }}
    .hero {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 20px 22px;
      box-shadow: 0 10px 30px rgba(22, 33, 47, 0.06);
      margin-bottom: 18px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 24px;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.7;
    }}
    .messages {{
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .message {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .message .meta {{
      color: var(--muted);
      font-size: 12px;
      padding: 0 6px;
    }}
    .message.self {{
      align-items: flex-end;
    }}
    .message.program {{
      align-items: flex-start;
    }}
    .message .bubble {{
      width: min(760px, 100%);
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 16px 18px;
      box-shadow: 0 8px 24px rgba(22, 33, 47, 0.05);
    }}
    .message.self .bubble {{
      background: var(--self-bg);
      border-color: var(--self-border);
    }}
    .message.program .bubble {{
      background: var(--program-bg);
      border-color: var(--program-border);
    }}
    .message.state-pending .bubble {{
      background: var(--pending-bg);
      border-color: var(--pending-border);
    }}
    .message.state-error .bubble {{
      background: var(--error-bg);
      border-color: var(--error-border);
    }}
    .speaker {{
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .primary {{
      font-size: 15px;
      font-weight: 700;
      line-height: 1.8;
      word-break: break-word;
    }}
    .secondary {{
      margin-top: 8px;
      font-size: 13px;
      line-height: 1.8;
      color: #4d6383;
      word-break: break-word;
    }}
    .audio-block {{
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px dashed rgba(91, 107, 130, 0.28);
    }}
    .audio-block audio {{
      width: 100%;
    }}
    .audio-caption {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }}
    .empty {{
      background: var(--card);
      border: 1px dashed var(--border);
      border-radius: 16px;
      padding: 24px;
      text-align: center;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>Revoice 导出记录</h1>
      <p>导出时间：{exported_at}，共 {len(export_items)} 条记录。压缩包内包含本页 HTML 以及可重播的音频文件。</p>
    </section>
    <section class="messages">
      {cards_html}
    </section>
  </main>
</body>
</html>
"""

    @staticmethod
    def _format_output_language_label(language: str) -> str:
        if language == "zh":
            return "中文"
        if language == "en":
            return "英文"
        if language == "ja":
            return "日语"
        if language == "es":
            return "西班牙语"
        if language == "ar":
            return "阿拉伯语"
        return "其他"

    @staticmethod
    def _infer_output_language_state(language: str) -> str:
        if language in {"zh", "en", "ja"}:
            return "official"
        return "custom"

    @staticmethod
    def _known_output_languages() -> tuple[str, ...]:
        return tuple(code for _label, code in OUTPUT_LANGUAGE_OPTIONS)

    def _coerce_hidden_language_settings(self) -> None:
        allowed_languages = set(self._known_output_languages())
        fallback_output_language = OUTPUT_LANGUAGE_OPTIONS[0][1]
        output_language = self._output_voice_language if self._output_voice_language in allowed_languages else fallback_output_language
        translation_target_language = (
            self._translation_target_language if self._translation_target_language in allowed_languages else output_language
        )

        if output_language != self._output_voice_language:
            previous_language = self._output_voice_language
            self._output_voice_language = output_language
            try:
                set_output_voice_language(output_language)
            except Exception as exc:
                logger.warning("回退已隐藏的输出语言失败: {} -> {} ({})", previous_language, output_language, exc)

        if translation_target_language != self._translation_target_language:
            previous_language = self._translation_target_language
            self._translation_target_language = translation_target_language
            try:
                set_translation_target_language(translation_target_language)
            except Exception as exc:
                logger.warning(
                    "回退已隐藏的翻译目标语言失败: {} -> {} ({})",
                    previous_language,
                    translation_target_language,
                    exc,
                )

    def _current_index_tts_output_status(self) -> tuple[bool, str]:
        if self._current_voice_output_model() != "indextts":
            return True, "MeanVC2 当前不依赖 Index-TTS 语种模型。"
        settings = get_settings()
        model_dir = settings.index_tts_model_dir
        missing = get_index_tts_missing_files(model_dir)
        detail = (
            f"IndexTTS-2.5 多语言模型\n"
            f"仓库：{settings.index_tts_model_repo}\n"
            f"目录：{model_dir}"
        )
        if not missing:
            return True, f"IndexTTS-2.5 多语言模型已就绪。\n{detail}"
        missing_preview = "、".join(missing[:4])
        if len(missing) > 4:
            missing_preview += " 等"
        return False, f"当前 IndexTTS-2.5 模型资源不完整。\n缺失：{missing_preview}\n{detail}"

    def _reset_quick_settings_dialog_refs(self) -> None:
        self._quick_settings_dialog = None
        self._quick_settings_model_download_action_button = None
        self._quick_settings_tts_model_status_labels = {}
        self._quick_settings_tts_model_path_labels = {}
        self._quick_settings_tts_model_download_buttons = {}
        self._quick_settings_model_download_progress_bar = None
        self._quick_settings_model_download_detail_label = None
        self._quick_settings_translation_node_combo = None
        self._quick_settings_edit_translation_button = None
        self._quick_settings_test_translation_button = None
        self._quick_settings_remove_translation_button = None

    def _dispose_quick_settings_dialog(self) -> None:
        dialog = self._quick_settings_dialog
        self._reset_quick_settings_dialog_refs()
        if dialog is not None:
            dialog.deleteLater()

    def _quick_settings_progress_bar_style(self) -> str:
        palette = self._theme_palette()
        return (
            "QProgressBar {"
            f" background: {palette['surface_bg_alt']};"
            f" color: {palette['text']};"
            f" border: 1px solid {palette['surface_border']};"
            " border-radius: 9px;"
            " min-height: 18px;"
            " text-align: center;"
            " font-size: 11px;"
            " font-weight: 600;"
            "}"
            "QProgressBar::chunk {"
            f" background: {palette['primary_start']};"
            " border-radius: 8px;"
            "}"
        )

    def _ensure_quick_settings_dialog(self) -> QDialog:
        if self._quick_settings_dialog is not None:
            return self._quick_settings_dialog

        dialog = QDialog(self)
        dialog.setWindowTitle("快速设置")
        dialog.setModal(True)
        dialog.resize(880, 640)

        root_layout = QVBoxLayout(dialog)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        dialog.setStyleSheet(
            f"QDialog {{ background: {self._theme_palette()['window_bg']}; color: {self._theme_palette()['text']}; }}"
        )

        content_row = QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(14)
        root_layout.addLayout(content_row, stretch=1)

        sidebar = QFrame(dialog)
        sidebar.setObjectName("SummaryCard")
        sidebar.setFixedWidth(210)
        sidebar.setStyleSheet(
            f"QFrame#SummaryCard {{ background: {self._theme_palette()['surface_bg']}; "
            f"border: 1px solid {self._theme_palette()['surface_border']}; border-radius: 14px; }}"
        )
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(10)

        title_label = self._set_section_title_style(QLabel("设置", sidebar))
        title_label.setStyleSheet("font-size: 18px; font-weight: 700;")
        sidebar_layout.addWidget(title_label)

        nav_button_column = QVBoxLayout()
        nav_button_column.setContentsMargins(0, 12, 0, 0)
        nav_button_column.setSpacing(8)
        sidebar_layout.addLayout(nav_button_column)
        sidebar_layout.addStretch(1)

        content_row.addWidget(sidebar, alignment=Qt.AlignTop)

        page_stack = QStackedWidget(dialog)
        content_row.addWidget(page_stack, stretch=1)

        model_page = QWidget(dialog)
        model_page_layout = QVBoxLayout(model_page)
        model_page_layout.setContentsMargins(0, 0, 0, 0)
        model_page_layout.setSpacing(16)

        model_header_row = QHBoxLayout()
        model_header_row.setContentsMargins(0, 0, 0, 0)
        model_header_row.setSpacing(12)

        model_title_icon = QLabel(dialog)
        model_title_icon.setPixmap(
            render_svg_template_icon("download", color=self._theme_palette()["accent_soft"]).pixmap(20, 20)
        )
        model_title = self._set_section_title_style(QLabel("模型下载列表", dialog))
        model_title.setStyleSheet("font-size: 16px; font-weight: 700;")

        model_header_row.addWidget(model_title_icon)
        model_header_row.addWidget(model_title)
        model_header_row.addStretch(1)

        model_subtitle = QLabel("这里可以手动下载 ASR、MeanVC2、IndexTTS-2.5 和 emotion2vec+，程序启动时不会自动拉取这些可选模型。", dialog)
        model_subtitle.setStyleSheet(
            f"color: {self._theme_palette()['muted_text']}; font-size: 12px;"
        )
        model_header_row.addWidget(model_subtitle, alignment=Qt.AlignVCenter)
        model_page_layout.addLayout(model_header_row)

        model_panel = QFrame(dialog)
        model_panel.setObjectName("SummaryCard")
        model_panel.setStyleSheet(
            f"QFrame#SummaryCard {{ background: {self._theme_palette()['surface_bg']}; "
            f"border: 1px solid {self._theme_palette()['surface_border']}; border-radius: 8px; }}"
        )
        model_panel_layout = QVBoxLayout(model_panel)
        model_panel_layout.setContentsMargins(18, 18, 18, 18)
        model_panel_layout.setSpacing(12)

        for model_name, repo, repo_label in DOWNLOADABLE_MODEL_OPTIONS:
            card = QFrame(model_panel)
            card.setObjectName("SummaryCard")
            card.setStyleSheet(
                f"QFrame#SummaryCard {{ background: {self._theme_palette()['surface_bg_alt']}; "
                f"border: 1px solid {self._theme_palette()['surface_border']}; border-radius: 10px; }}"
            )
            card_layout = QHBoxLayout(card)
            card_layout.setContentsMargins(16, 16, 16, 16)
            card_layout.setSpacing(14)

            icon_label = QLabel(card)
            icon_label.setPixmap(
                render_svg_template_icon("box", color=self._theme_palette()["accent_soft"]).pixmap(18, 18)
            )
            card_layout.addWidget(icon_label, alignment=Qt.AlignTop)

            text_column = QVBoxLayout()
            text_column.setContentsMargins(0, 0, 0, 0)
            text_column.setSpacing(6)

            title_row = QHBoxLayout()
            title_row.setContentsMargins(0, 0, 0, 0)
            title_row.setSpacing(8)

            model_name_label = QLabel(model_name, card)
            model_name_label.setStyleSheet("font-size: 15px; font-weight: 700;")
            status_badge = QLabel("检测中", card)
            status_badge.setStyleSheet(
                f"background: {self._theme_palette()['button_bg']}; color: {self._theme_palette()['muted_text']}; "
                f"border: 1px solid {self._theme_palette()['surface_border']}; border-radius: 10px; "
                "padding: 2px 8px; font-size: 11px; font-weight: 600;"
            )

            title_row.addWidget(model_name_label)
            title_row.addWidget(status_badge, alignment=Qt.AlignVCenter)
            title_row.addStretch(1)

            repo_label_widget = QLabel(repo_label, card)
            repo_label_widget.setStyleSheet(
                f"color: {self._theme_palette()['muted_text']}; font-size: 12px;"
            )
            description_label = QLabel(DOWNLOADABLE_MODEL_DESCRIPTIONS.get(repo, ""), card)
            description_label.setWordWrap(True)
            description_label.setStyleSheet(
                f"color: {self._theme_palette()['muted_text']}; font-size: 12px;"
            )
            path_label = QLabel("-", card)
            path_label.setWordWrap(True)
            path_label.setStyleSheet(
                f"color: {self._theme_palette()['muted_text']}; font-size: 12px;"
            )

            text_column.addLayout(title_row)
            text_column.addWidget(repo_label_widget)
            if description_label.text():
                text_column.addWidget(description_label)
            text_column.addWidget(path_label)
            card_layout.addLayout(text_column, stretch=1)

            download_button = self._set_compact_button_style(
                QPushButton("下载模型", card),
                min_height=38,
                min_width=120,
            )
            download_button.setIcon(
                render_svg_template_icon("cloud_download", color=self._theme_palette()["text"])
            )
            download_button.setProperty("repo", repo)
            download_button.clicked.connect(
                lambda _checked=False, current_repo=repo: self._start_quick_settings_model_download(current_repo)
            )
            card_layout.addWidget(download_button, alignment=Qt.AlignVCenter)

            self._quick_settings_tts_model_status_labels[repo] = status_badge
            self._quick_settings_tts_model_path_labels[repo] = path_label
            self._quick_settings_tts_model_download_buttons[repo] = download_button
            model_panel_layout.addWidget(card)

        progress_row = QHBoxLayout()
        progress_row.setContentsMargins(0, 4, 0, 0)
        progress_row.setSpacing(12)

        model_download_progress_bar = QProgressBar(model_panel)
        model_download_progress_bar.setTextVisible(True)
        model_download_progress_bar.setValue(0)
        model_download_progress_bar.setFormat("%p%")
        model_download_progress_bar.setStyleSheet(self._quick_settings_progress_bar_style())
        progress_row.addWidget(model_download_progress_bar, stretch=1)

        model_download_action_button = self._set_compact_button_style(
            QPushButton("刷新状态", model_panel),
            min_height=36,
            min_width=96,
        )
        model_download_action_button.setIcon(
            render_svg_template_icon("reload", color=self._theme_palette()["text"])
        )
        model_download_action_button.clicked.connect(self._refresh_quick_settings_model_catalog)
        progress_row.addWidget(model_download_action_button)
        model_panel_layout.addLayout(progress_row)

        model_download_detail_label = QLabel("这里会显示模型下载进度。", model_panel)
        model_download_detail_label.setWordWrap(True)
        model_download_detail_label.setStyleSheet(
            f"color: {self._theme_palette()['muted_text']}; font-size: 12px;"
        )
        model_panel_layout.addWidget(model_download_detail_label)

        model_hint_label = QLabel(
            "ASR、MeanVC2、emotion2vec+ 下载完成后会显示“已下载”；IndexTTS-2.5 需要主模型和附属模型都补齐后才会显示就绪。",
            model_panel,
        )
        model_hint_label.setWordWrap(True)
        model_hint_label.setStyleSheet(
            f"color: {self._theme_palette()['muted_text']}; font-size: 12px;"
        )
        model_panel_layout.addWidget(model_hint_label)
        model_page_layout.addWidget(model_panel)
        model_page_layout.addStretch(1)
        page_stack.addWidget(model_page)

        translation_page = QWidget(dialog)
        translation_page_layout = QVBoxLayout(translation_page)
        translation_page_layout.setContentsMargins(0, 0, 0, 0)
        translation_page_layout.setSpacing(16)

        translation_header_row = QHBoxLayout()
        translation_header_row.setContentsMargins(0, 0, 0, 0)
        translation_header_row.setSpacing(12)

        translation_title_icon = QLabel(dialog)
        translation_title_icon.setPixmap(render_svg_template_icon("globe", color=self._theme_palette()['accent_soft']).pixmap(20, 20))
        translation_title = self._set_section_title_style(QLabel("翻译节点", dialog))
        translation_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        
        translation_header_row.addWidget(translation_title_icon)
        translation_header_row.addWidget(translation_title)
        translation_header_row.addStretch(1)
        
        translation_subtitle = QLabel("配置在线翻译服务接口", dialog)
        translation_subtitle.setStyleSheet(f"color: {self._theme_palette()['muted_text']}; font-size: 12px;")
        translation_header_row.addWidget(translation_subtitle, alignment=Qt.AlignVCenter)
        
        translation_page_layout.addLayout(translation_header_row)

        translation_panel = QFrame(dialog)
        translation_panel.setObjectName("SummaryCard")
        translation_panel.setStyleSheet(f"QFrame#SummaryCard {{ background: {self._theme_palette()['surface_bg']}; border: 1px solid {self._theme_palette()['surface_border']}; border-radius: 8px; }}")
        translation_panel_layout = QVBoxLayout(translation_panel)
        translation_panel_layout.setContentsMargins(18, 18, 18, 18)
        translation_panel_layout.setSpacing(16)

        api_label = QLabel("翻译开放平台 API", translation_panel)
        api_label.setStyleSheet("font-size: 13px;")
        translation_panel_layout.addWidget(api_label)

        translation_widget = QWidget(dialog)
        translation_layout = QVBoxLayout(translation_widget)
        translation_layout.setContentsMargins(0, 0, 0, 0)
        translation_layout.setSpacing(12)
        translation_node_combo = QComboBox(translation_widget)
        translation_node_combo.setMinimumHeight(40)
        translation_node_combo.setStyleSheet(f"QComboBox {{ border: 1px solid {self._theme_palette()['surface_border']}; border-radius: 6px; padding: 5px 10px; background: {self._theme_palette()['app_bg']}; font-size: 13px; }} QComboBox::drop-down {{ border: none; }}")
        
        # Populate translation nodes
        def _refresh_translation_node_combo():
            translation_node_combo.clear()
            nodes = get_settings().translation_nodes
            for i, node in enumerate(nodes):
                provider_label = next((label for label, code in TRANSLATION_PROVIDER_OPTIONS if code == node.provider), node.provider)
                display_text = f"{provider_label} - {node.name}" if node.name else provider_label
                translation_node_combo.addItem(display_text, node.node_id)
                if node.node_id == get_settings().translation_node_id:
                    translation_node_combo.setCurrentIndex(i)
        _refresh_translation_node_combo()
        
        translation_node_combo.currentIndexChanged.connect(
            lambda: set_translation_node_id(translation_node_combo.currentData()) if translation_node_combo.currentData() else None
        )

        translation_layout.addWidget(translation_node_combo)

        translation_buttons_row = QHBoxLayout()
        translation_buttons_row.setContentsMargins(0, 0, 0, 0)
        translation_buttons_row.setSpacing(8)
        
        add_translation_button = self._set_compact_button_style(QPushButton(" + 新增", dialog), min_height=36, min_width=88)
        add_translation_button.setStyleSheet(f"QPushButton {{ background: {self._theme_palette()['primary_start']}; color: white; border-radius: 6px; font-weight: 600; font-size: 12px; padding: 5px 8px; }} QPushButton:hover {{ background: {self._theme_palette()['primary_hover_start']}; }}")
        
        edit_translation_button = self._set_compact_button_style(QPushButton(" 编辑", dialog), min_height=36, min_width=72)
        test_translation_button = self._set_compact_button_style(QPushButton(" 测试", dialog), min_height=36, min_width=72)
        remove_translation_button = self._set_compact_button_style(QPushButton(" 删除", dialog), min_height=36, min_width=72)
        remove_translation_button.setStyleSheet(f"QPushButton {{ background: {self._theme_palette()['error_bg']}; color: {self._theme_palette()['error_text']}; border: 1px solid {self._theme_palette()['error_border']}; border-radius: 6px; font-weight: 600; font-size: 12px; padding: 5px 8px; }} QPushButton:hover {{ background: {self._theme_palette()['error_border']}; }}")
        
        translation_buttons_row.addWidget(add_translation_button)
        translation_buttons_row.addWidget(edit_translation_button)
        translation_buttons_row.addWidget(test_translation_button)
        translation_buttons_row.addWidget(remove_translation_button)
        translation_buttons_row.addStretch(1)

        translation_layout.addLayout(translation_buttons_row)
        translation_panel_layout.addWidget(translation_widget)
        
        hint_label = QLabel("💡 支持 百度翻译、DeepL、Google Cloud Translate、OpenAI、DeepSeek 等兼容接口。", translation_panel)
        hint_label.setStyleSheet(f"color: {self._theme_palette()['muted_text']}; font-size: 12px;")
        translation_panel_layout.addStretch(1)
        translation_panel_layout.addWidget(hint_label)
        
        translation_page_layout.addWidget(translation_panel)
        translation_page_layout.addStretch(1)
        page_stack.addWidget(translation_page)

        palette = self._theme_palette()

        def build_nav_button(text: str, icon_name: str, page_index: int) -> QPushButton:
            button = QPushButton(f"  {text}", sidebar)
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumHeight(46)
            button.setIcon(render_svg_template_icon(icon_name, color=palette['text']))
            button.setIconSize(QSize(18, 18))
            button.setStyleSheet(
                f"QPushButton {{"
                f" text-align:left; padding:11px 14px; font-size:13px; font-weight:600;"
                f" border-radius:12px; border:1px solid transparent;"
                f" background:transparent; color:{palette['text']};"
                f"}}"
                f"QPushButton:hover {{"
                f" background:{palette['button_bg']}; border:1px solid {palette['button_border']};"
                f"}}"
                f"QPushButton:checked {{"
                f" background:{palette['hero_chip_bg']}; border:1px solid {palette['hero_chip_bg']};"
                f" color:{palette['hero_chip_text']};"
                f"}}"
            )
            button.clicked.connect(lambda _checked=False, index=page_index: page_stack.setCurrentIndex(index))
            nav_button_column.addWidget(button)
            return button

        nav_buttons = [
            build_nav_button("模型下载", "download", 0),
            build_nav_button("翻译节点", "globe", 1),
        ]

        def sync_nav_buttons(index: int) -> None:
            for button_index, button in enumerate(nav_buttons):
                button.blockSignals(True)
                button.setChecked(button_index == index)
                button.blockSignals(False)

        page_stack.currentChanged.connect(sync_nav_buttons)
        page_stack.setCurrentIndex(0)
        sync_nav_buttons(0)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        root_layout.addWidget(buttons)

        self._quick_settings_dialog = dialog
        self._quick_settings_model_download_action_button = model_download_action_button
        self._quick_settings_model_download_progress_bar = model_download_progress_bar
        self._quick_settings_model_download_detail_label = model_download_detail_label
        self._quick_settings_translation_node_combo = translation_node_combo
        self._quick_settings_edit_translation_button = edit_translation_button
        self._quick_settings_test_translation_button = test_translation_button
        self._quick_settings_remove_translation_button = remove_translation_button
        add_translation_button.clicked.connect(self._add_translation_node)
        edit_translation_button.clicked.connect(self._edit_selected_translation_node)
        test_translation_button.clicked.connect(self._test_selected_translation_node)
        remove_translation_button.clicked.connect(self._remove_selected_translation_node)
        apply_window_chrome_theme(dialog, self._current_theme)
        return dialog

    def _open_quick_settings_dialog(self) -> None:
        self._dispose_quick_settings_dialog()
        dialog = self._ensure_quick_settings_dialog()
        self._prepare_quick_settings_dialog_state()
        if dialog.exec() != QDialog.Accepted:
            return
        self._apply_quick_settings_dialog_state()

    @staticmethod
    def _available_downloadable_model_repos() -> tuple[str, ...]:
        return tuple(repo for _label, repo, _repo_label in DOWNLOADABLE_MODEL_OPTIONS)

    def _reset_quick_settings_model_progress(self, detail: str) -> None:
        if self._quick_settings_model_download_progress_bar is not None:
            self._quick_settings_model_download_progress_bar.setRange(0, 100)
            self._quick_settings_model_download_progress_bar.setValue(0)
            self._quick_settings_model_download_progress_bar.setFormat("%p%")
        if self._quick_settings_model_download_detail_label is not None:
            self._quick_settings_model_download_detail_label.setText(detail)

    def _refresh_quick_settings_model_catalog(self) -> None:
        catalog = {
            item.repo: item
            for item in list_downloadable_model_catalog(list(self._available_downloadable_model_repos()))
        }
        for repo in self._available_downloadable_model_repos():
            item = catalog.get(repo)
            if item is None:
                continue
            status_label = self._quick_settings_tts_model_status_labels.get(repo)
            path_label = self._quick_settings_tts_model_path_labels.get(repo)
            button = self._quick_settings_tts_model_download_buttons.get(repo)
            missing_files = tuple(get_index_tts_missing_files(item.target_dir)) if item.kind == "index_tts" else ()
            fully_installed = bool(item.installed) and not bool(missing_files)
            if path_label is not None:
                if fully_installed:
                    path_label.setText(str(item.target_dir))
                else:
                    preview = "、".join(missing_files[:3])
                    suffix = " 等" if len(missing_files) > 3 else ""
                    path_label.setText(
                        f"{item.target_dir}\n缺少：{preview}{suffix}" if missing_files else str(item.target_dir)
                    )
            is_downloading = repo == self._quick_settings_model_downloading_repo
            if status_label is not None:
                if fully_installed:
                    status_label.setText("已下载")
                    status_label.setStyleSheet(
                        f"background: {self._theme_palette()['success_bg']}; color: {self._theme_palette()['success_text']}; "
                        f"border: 1px solid {self._theme_palette()['success_border']}; border-radius: 10px; "
                        "padding: 2px 8px; font-size: 11px; font-weight: 600;"
                    )
                elif is_downloading:
                    status_label.setText("下载中")
                    status_label.setStyleSheet(
                        f"background: {self._theme_palette()['button_bg']}; color: {self._theme_palette()['text']}; "
                        f"border: 1px solid {self._theme_palette()['button_hover_border']}; border-radius: 10px; "
                        "padding: 2px 8px; font-size: 11px; font-weight: 600;"
                    )
                else:
                    status_label.setText("未下载")
                    status_label.setStyleSheet(
                        f"background: {self._theme_palette()['button_bg']}; color: {self._theme_palette()['muted_text']}; "
                        f"border: 1px solid {self._theme_palette()['surface_border']}; border-radius: 10px; "
                        "padding: 2px 8px; font-size: 11px; font-weight: 600;"
                    )
            if button is None:
                continue
            if fully_installed:
                button.setText("已经下载")
                button.setIcon(
                    render_svg_template_icon("check_circle", color=self._theme_palette()["success_text"])
                )
                button.setEnabled(False)
            elif is_downloading:
                button.setText("下载中...")
                button.setIcon(render_svg_template_icon("download", color=self._theme_palette()["text"]))
                button.setEnabled(False)
            else:
                button.setText("下载模型")
                button.setIcon(
                    render_svg_template_icon("cloud_download", color=self._theme_palette()["text"])
                )
                button.setEnabled(self._quick_settings_model_downloading_repo is None)
        if self._quick_settings_model_download_action_button is not None:
            self._quick_settings_model_download_action_button.setEnabled(
                self._quick_settings_model_downloading_repo is None
            )

    def _on_quick_settings_model_download_progress(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        if self._quick_settings_model_download_progress_bar is not None:
            maximum = int(payload.get("maximum", 0) or 0)
            value = int(payload.get("value", 0) or 0)
            indeterminate = bool(payload.get("indeterminate", False))
            if indeterminate or maximum <= 0:
                self._quick_settings_model_download_progress_bar.setRange(0, 0)
            else:
                self._quick_settings_model_download_progress_bar.setRange(0, maximum)
                self._quick_settings_model_download_progress_bar.setValue(max(0, min(value, maximum)))
        detail = str(payload.get("detail", "")).strip()
        sub_label = str(payload.get("sub_label", "")).strip()
        if self._quick_settings_model_download_detail_label is not None:
            self._quick_settings_model_download_detail_label.setText(detail or sub_label or "正在下载模型...")

    def _start_quick_settings_model_download(self, repo: str) -> None:
        normalized_repo = str(repo or "").strip()
        if not normalized_repo:
            return
        if self._quick_settings_model_download_worker is not None:
            self._show_message(QMessageBox.Information, "下载进行中", "当前已有模型下载任务，请等待完成后再试。")
            return
        self._quick_settings_model_downloading_repo = normalized_repo
        self._reset_quick_settings_model_progress(f"正在准备下载 {normalized_repo} ...")
        self._refresh_quick_settings_model_catalog()
        worker = ModelDownloadWorker(normalized_repo, self)
        worker.status.connect(
            lambda message: (
                self._quick_settings_model_download_detail_label.setText(message)
                if self._quick_settings_model_download_detail_label is not None
                else None
            )
        )
        worker.progress.connect(self._on_quick_settings_model_download_progress)
        worker.finished.connect(self._on_quick_settings_model_download_finished)
        self._quick_settings_model_download_worker = worker
        worker.start()

    def _on_quick_settings_model_download_finished(self, repo: str, success: bool, message: str) -> None:
        worker = self._quick_settings_model_download_worker
        self._quick_settings_model_download_worker = None
        self._quick_settings_model_downloading_repo = None
        if worker is not None:
            worker.deleteLater()
        self._refresh_quick_settings_model_catalog()
        if self._quick_settings_model_download_progress_bar is not None:
            if success:
                self._quick_settings_model_download_progress_bar.setRange(0, 100)
                self._quick_settings_model_download_progress_bar.setValue(100)
                self._quick_settings_model_download_progress_bar.setFormat("%p%")
            else:
                self._quick_settings_model_download_progress_bar.setRange(0, 100)
                self._quick_settings_model_download_progress_bar.setValue(0)
                self._quick_settings_model_download_progress_bar.setFormat("%p%")
        if self._quick_settings_model_download_detail_label is not None:
            self._quick_settings_model_download_detail_label.setText(message)
        if success:
            self.settings = get_settings()
            self._refresh_workspace_summary()
        else:
            self._show_message(QMessageBox.Warning, "下载失败", f"{repo} 下载失败：{message}")

    def _prepare_quick_settings_dialog_state(self) -> None:
        self._refresh_quick_settings_model_catalog()
        if self._quick_settings_model_downloading_repo is None:
            self._reset_quick_settings_model_progress("这里会显示模型下载进度。")
        for combo in self._translation_target_language_combos():
            self._set_combo_current_data_silently(combo, self._translation_target_language)
        self._refresh_translation_node_combo()
        if self._quick_settings_torch_compile_checkbox is not None:
            self._quick_settings_torch_compile_checkbox.setChecked(self._experimental_torch_compile_enabled)
        if self._quick_settings_accel_gpt_checkbox is not None:
            self._quick_settings_accel_gpt_checkbox.setChecked(self._experimental_accel_gpt_enabled)
        self._sync_quick_settings_auto_emotion_controls()

    def _collect_quick_settings_dialog_values(self) -> dict[str, object]:
        return {
            "translation_enabled": self._translation_enabled,
            "translation_target_language": self._translation_target_language,
            "translation_node_id": (self._translation_node_id or "") or None,
            "torch_compile_enabled": (
                self._quick_settings_torch_compile_checkbox.isChecked()
                if self._quick_settings_torch_compile_checkbox is not None
                else self._experimental_torch_compile_enabled
            ),
            "accel_gpt_enabled": (
                self._quick_settings_accel_gpt_checkbox.isChecked()
                if self._quick_settings_accel_gpt_checkbox is not None
                else self._experimental_accel_gpt_enabled
            ),
        }

    def _validate_quick_settings_dialog_values(self, values: dict[str, object]) -> bool:
        new_translation_enabled = bool(values["translation_enabled"])
        new_translation_target_language = str(values["translation_target_language"])
        new_translation_node_id = values["translation_node_id"]
        if new_translation_target_language not in self._known_output_languages():
            self._show_message(QMessageBox.Warning, "保存失败", "当前翻译目标语言无效。")
            return False
        if new_translation_enabled and not new_translation_node_id:
            self._show_message(QMessageBox.Warning, "保存失败", "启用翻译前，请先添加并选择一个翻译节点。")
            return False
        return True

    def _apply_quick_settings_values(self, values: dict[str, object]) -> tuple[bool, list[str], list[str]]:
        new_translation_enabled = bool(values["translation_enabled"])
        new_translation_target_language = str(values["translation_target_language"])
        new_translation_node_id = values["translation_node_id"]
        new_torch_compile_enabled = bool(values["torch_compile_enabled"])
        new_accel_gpt_enabled = bool(values["accel_gpt_enabled"])
        changed_features: list[str] = []
        restart_reasons: list[str] = []

        if new_translation_enabled != self._translation_enabled:
            try:
                set_translation_enabled(new_translation_enabled)
            except Exception as exc:
                logger.exception("保存翻译开关失败")
                self._show_message(QMessageBox.Critical, "保存失败", f"无法保存翻译开关：{exc}")
                return False, [], []
            self._translation_enabled = new_translation_enabled

        if new_translation_target_language != self._translation_target_language:
            try:
                set_translation_target_language(new_translation_target_language)
            except Exception as exc:
                logger.exception("保存翻译目标语言失败")
                self._show_message(QMessageBox.Critical, "保存失败", f"无法保存翻译目标语言：{exc}")
                return False, [], []
            self._translation_target_language = new_translation_target_language

        if new_translation_node_id != self._translation_node_id:
            try:
                set_translation_node_id(new_translation_node_id)
            except Exception as exc:
                logger.exception("保存翻译节点失败")
                self._show_message(QMessageBox.Critical, "保存失败", f"无法保存翻译节点：{exc}")
                return False, [], []
            self._translation_node_id = new_translation_node_id
        self._sync_translation_card_controls()
        self._emit_live_translation_settings()

        if restart_reasons:
            self.settings = get_settings()

        if new_torch_compile_enabled != self._experimental_torch_compile_enabled:
            try:
                set_experimental_torch_compile_enabled(new_torch_compile_enabled)
            except Exception as exc:
                logger.exception("保存进阶模式设置失败")
                self._show_message(QMessageBox.Critical, "保存失败", f"无法保存进阶模式设置：{exc}")
                return False, [], []
            self._experimental_torch_compile_enabled = new_torch_compile_enabled
            changed_features.append("进阶模式")

        if new_accel_gpt_enabled != self._experimental_accel_gpt_enabled:
            try:
                set_experimental_accel_gpt_enabled(new_accel_gpt_enabled)
            except Exception as exc:
                logger.exception("保存极速模式设置失败")
                self._show_message(QMessageBox.Critical, "保存失败", f"无法保存极速模式设置：{exc}")
                return False, [], []
            self._experimental_accel_gpt_enabled = new_accel_gpt_enabled
            changed_features.append("极速模式")
        return True, changed_features, restart_reasons

    def _apply_quick_settings_dialog_state(self) -> None:
        if self._quick_settings_auto_emotion_checkbox is not None:
            self._auto_emotion_feature_enabled = self._quick_settings_auto_emotion_checkbox.isChecked()
        if self._quick_settings_auto_emotion_strength_slider is not None:
            self._auto_emotion_strength = self._quick_settings_auto_emotion_strength_slider.value() * 0.05
        self._on_auto_emotion_toggled()
        self._sync_tts_other_option_controls()
        self._refresh_workspace_summary()
        values = self._collect_quick_settings_dialog_values()
        if not self._validate_quick_settings_dialog_values(values):
            return
        success, changed_features, restart_reasons = self._apply_quick_settings_values(values)
        if not success:
            return

        if changed_features:
            self.settings = get_settings()
            self._refresh_workspace_summary()
            self._show_experimental_restart_notice(
                feature_names=changed_features,
            )
            return

        if restart_reasons:
            self.settings = get_settings()
            if restart_reasons:
                self._show_model_restart_notice(restart_reasons)
                return

        self.settings = get_settings()
        self._refresh_workspace_summary()

    def _standard_icon(self, icon_kind: QStyle.StandardPixmap):
        return self.style().standardIcon(icon_kind)

    def _ensure_message_dialog(self) -> QMessageBox:
        if self._message_dialog is not None:
            return self._message_dialog
        dialog = QMessageBox(self)
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.setModal(True)
        self._message_dialog = dialog
        return dialog

    def _show_message(self, icon: QMessageBox.Icon, title: str, text: str) -> None:
        dialog = self._ensure_message_dialog()
        dialog.setIcon(icon)
        dialog.setWindowTitle(title)
        dialog.setText(text)
        dialog.setInformativeText("")
        apply_window_chrome_theme(dialog, self._current_theme)
        dialog.exec()

    @staticmethod
    def _is_meanvc2_setup_error(message: str) -> bool:
        normalized = str(message or "").strip()
        return "MeanVC2 尚未准备完成" in normalized or "缺少 MeanVC2 运行文件" in normalized

    @staticmethod
    def _is_asr_setup_error(message: str) -> bool:
        normalized = str(message or "").strip()
        return (
            "量化 ASR 产物不存在" in normalized
            or "Fun-ASR-Nano-2512 GGUF 模型未找到" in normalized
        )

    def _show_meanvc2_setup_guide(self, message: str) -> None:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Information)
        dialog.setWindowTitle("请先下载 MeanVC2 模型")
        dialog.setText("当前选择的是 MeanVC2，但本地运行文件还没有准备完成。")
        dialog.setInformativeText(
            "建议先打开“快速设置 -> 模型下载”，下载 `aklwss/MeanVC2`。\n\n"
            "下载完成后，如果仍提示缺少 `wavlm_large_finetune.pth`，请按 vendor/MeanVC2/initialization.py 中的说明手动补齐。"
        )
        open_button = dialog.addButton("打开模型下载", QMessageBox.AcceptRole)
        dialog.addButton("知道了", QMessageBox.RejectRole)
        dialog.setDetailedText(message)
        dialog.setModal(True)
        apply_window_chrome_theme(dialog, self._current_theme)
        dialog.exec()
        if dialog.clickedButton() is open_button:
            self._open_quick_settings_dialog()

    def _show_asr_setup_guide(self, message: str) -> None:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Information)
        dialog.setWindowTitle("请先下载 ASR 模型")
        dialog.setText("当前实时模式还需要语音识别模型，但本地 ASR 量化产物还没有准备完成。")
        dialog.setInformativeText(
            "建议先打开“快速设置 -> 模型下载”，下载 `aklwss/Fun-ASR-Nano-2512-gguf`。\n\n"
            "下载完成后，再重新开始麦克风或程序监听。"
        )
        open_button = dialog.addButton("打开模型下载", QMessageBox.AcceptRole)
        dialog.addButton("知道了", QMessageBox.RejectRole)
        dialog.setDetailedText(message)
        dialog.setModal(True)
        apply_window_chrome_theme(dialog, self._current_theme)
        dialog.exec()
        if dialog.clickedButton() is open_button:
            self._open_quick_settings_dialog()

    def _show_experimental_restart_notice(self, *, feature_names: list[str]) -> None:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Information)
        dialog.setWindowTitle("需要重启")
        feature_summary = "、".join(feature_names)
        dialog.setText(
            f"实验选项（{feature_summary}）已更新，软件现在会退出。请重新启动后再使用。"
        )
        dialog.setInformativeText(
            self._build_experimental_restart_detail(feature_names)
        )
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.setModal(True)
        apply_window_chrome_theme(dialog, self._current_theme)
        dialog.exec()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _show_model_restart_notice(self, reasons: list[str]) -> None:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Information)
        dialog.setWindowTitle("需要重启")
        reason_summary = "、".join(reasons)
        dialog.setText(f"输出语言设置（{reason_summary}）已更新，软件现在会退出。请重新启动后继续使用。")
        dialog.setInformativeText(
            "下次启动时，程序会按新的输出语言检查当前实际使用的模型；可选的 Index-TTS 模型仍可在设置页里手动下载。"
        )
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.setModal(True)
        apply_window_chrome_theme(dialog, self._current_theme)
        dialog.exec()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _build_experimental_restart_detail(self, feature_names: list[str]) -> str:
        detail_parts: list[str] = []
        if "进阶模式" in feature_names:
            detail_parts.append(self._experimental_torch_compile_restart_detail())
        if "极速模式" in feature_names:
            detail_parts.append(
                "极速模式通常会占用更多显存；如果出现卡顿、报错或显存不足，建议关闭。"
            )
        if not detail_parts:
            return "实验设置已更新，重启后将按新的配置运行。"
        return " ".join(detail_parts)

    @staticmethod
    def _translation_node_label(node) -> str:
        provider = getattr(node, "provider", "openai-compatible")
        return f"{node.name} ({provider} / {node.model})"

    def _primary_translation_node_combo(self) -> QComboBox | None:
        if self._quick_settings_dialog is not None and self._quick_settings_dialog.isVisible():
            return self._quick_settings_translation_node_combo or self.translation_card_node_combo
        return self.translation_card_node_combo or self._quick_settings_translation_node_combo

    def _translation_node_combos(self) -> list[QComboBox]:
        combos: list[QComboBox] = []
        for combo in (self.translation_card_node_combo, self._quick_settings_translation_node_combo):
            if combo is not None and combo not in combos:
                combos.append(combo)
        return combos

    def _translation_enabled_checkboxes(self) -> list[QCheckBox]:
        return [self.translation_card_checkbox] if self.translation_card_checkbox is not None else []

    def _translation_target_language_combos(self) -> list[QComboBox]:
        return [self.translation_card_target_language_combo] if self.translation_card_target_language_combo is not None else []

    def _current_voice_output_model(self) -> str:
        return str(self._voice_output_model or "indextts").strip() or "indextts"

    def _is_meanvc2_selected(self) -> bool:
        return self._current_voice_output_model() == "meanvc2"

    @staticmethod
    def _set_combo_current_data_silently(combo: QComboBox | None, data: object) -> None:
        if combo is None:
            return
        index = combo.findData(data)
        if index < 0:
            index = 0
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _selected_translation_node_id(self) -> str:
        combo = self._primary_translation_node_combo()
        if combo is None:
            return ""
        return str(combo.currentData() or "").strip()

    def _sync_translation_node_actions(self, *_args) -> None:
        has_selected_node = bool(self._selected_translation_node_id())
        for button in (
            self._quick_settings_edit_translation_button,
            self._quick_settings_test_translation_button,
            self._quick_settings_remove_translation_button,
        ):
            if button is not None:
                button.setEnabled(has_selected_node)

    def _sync_translation_card_controls(self) -> None:
        selected_id = self._translation_node_id or ""
        meanvc2_selected = self._is_meanvc2_selected()
        for checkbox in self._translation_enabled_checkboxes():
            self._set_checkbox_silently(checkbox, self._translation_enabled)
        for combo in self._translation_target_language_combos():
            self._set_combo_current_data_silently(combo, self._translation_target_language)
        for combo in self._translation_node_combos():
            self._set_combo_current_data_silently(combo, selected_id)
        if self.translation_card_info_button is not None:
            self.translation_card_info_button.setToolTip(self._translation_mode_tooltip())
        if self.translation_card_target_language_combo is not None:
            self.translation_card_target_language_combo.setEnabled(True)
            self.translation_card_target_language_combo.setToolTip(
                "MeanVC2 仅影响聊天记录里的翻译文本；Index-TTS 会用这里的目标语言生成语音。"
                if meanvc2_selected
                else ""
            )
        if self.translation_card_node_combo is not None:
            self.translation_card_node_combo.setEnabled(bool(self.settings.translation_nodes))
            self.translation_card_node_combo.setToolTip(
                "MeanVC2 会使用这里的翻译节点生成记录里的翻译文本。"
                if meanvc2_selected and bool(self.settings.translation_nodes)
                else ("请先在设置中配置翻译节点。" if not bool(self.settings.translation_nodes) else "")
            )

        self._sync_translation_node_actions()

    def _sync_emotion_controls(self) -> None:
        meanvc2_selected = self._is_meanvc2_selected()
        disabled_tooltip = "MeanVC2 当前不使用 Index-TTS 情感控制。"
        if self.emotion_control_info_button is not None:
            self.emotion_control_info_button.setToolTip(self._emotion_control_tooltip())
        if self.emotion_control_hint_label is not None:
            self.emotion_control_hint_label.clear()
            self.emotion_control_hint_label.setVisible(False)
        self._set_mode_locked_card_visual(
            card=self.emotion_control_card,
            overlay=self.emotion_control_overlay,
            hint_label=self.emotion_control_hint_label,
            locked=meanvc2_selected,
            badge_text="仅 Index-TTS",
            overlay_message="当前使用 MeanVC2 语音转换模型，请切换到 Index-TTS 后再调整情感预设、情感向量和自动情感识别。",
        )
        for widget_name in (
            "nav_preset_btn",
            "nav_vector_btn",
            "emotion_preset_combo",
            "emotion_custom_edit",
            "index_emotion_mode_combo",
            "emotion_stack",
            "index_emo_alpha_slider",
            "index_emo_alpha_label",
            "auto_emotion_card_strength_slider",
            "auto_emotion_card_strength_label",
        ):
            if hasattr(self, widget_name):
                widget = getattr(self, widget_name)
                widget.setEnabled(not meanvc2_selected)
                if hasattr(widget, "setToolTip"):
                    widget.setToolTip(disabled_tooltip if meanvc2_selected else "")
        if hasattr(self, "emotion_preset_buttons"):
            for button in self.emotion_preset_buttons:
                button.setEnabled(not meanvc2_selected)
                button.setToolTip(disabled_tooltip if meanvc2_selected else "")
        if hasattr(self, "index_emo_rows"):
            for row in self.index_emo_rows:
                row.setEnabled(not meanvc2_selected)
                row.setToolTip(disabled_tooltip if meanvc2_selected else "")
        if hasattr(self, "index_emo_sliders"):
            for slider in self.index_emo_sliders.values():
                slider.setEnabled(not meanvc2_selected)
                slider.setToolTip(disabled_tooltip if meanvc2_selected else "")
        if hasattr(self, "index_emo_value_labels"):
            for label in self.index_emo_value_labels.values():
                label.setEnabled(not meanvc2_selected)

    def _sync_voice_output_model_controls(self) -> None:
        meanvc2_selected = self._is_meanvc2_selected()
        output_language_enabled = not meanvc2_selected
        index_tts_ready, index_tts_message = self._current_index_tts_output_status()
        output_language_tooltip = (
            "MeanVC2 为语音转换模型，当前不需要选择输出语言。"
            if meanvc2_selected
            else index_tts_message
        )
        self._set_mode_locked_section_visual(
            self.voice_output_settings_section,
            locked=meanvc2_selected,
            tooltip=output_language_tooltip,
        )
        if hasattr(self, "tts_engine_combo"):
            self.tts_engine_combo.setToolTip(
                (
                    f"Index-TTS 适合文本转语音；MeanVC2 适合语音转换。\n\n{index_tts_message}"
                    if not meanvc2_selected and not index_tts_ready
                    else "Index-TTS 适合文本转语音；MeanVC2 适合语音转换。"
                )
            )
        if self.translation_card_checkbox is not None:
            self.translation_card_checkbox.setEnabled(True)
            self.translation_card_checkbox.setToolTip(
                "MeanVC2 开启后只会为聊天记录生成翻译，不影响实际变声输出。"
                if meanvc2_selected
                else ""
            )
        if self.translation_card_target_language_combo is not None:
            self.translation_card_target_language_combo.setEnabled(True)
            self.translation_card_target_language_combo.setToolTip(
                "MeanVC2 会按这里的目标语言生成聊天记录里的翻译文本。"
                if meanvc2_selected
                else ""
            )
        self._sync_translation_card_controls()
        self._sync_emotion_controls()
        self._sync_tts_other_option_controls()
        self._refresh_workspace_summary()

    def _refresh_translation_node_combo(self) -> None:
        combos = self._translation_node_combos()
        if not combos:
            return
        for combo in combos:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("不使用翻译节点", "")
            for node in self.settings.translation_nodes:
                combo.addItem(self._translation_node_label(node), node.node_id)
            combo.blockSignals(False)
        self._sync_translation_card_controls()

    @staticmethod
    def _preset_translation_node_values(provider: str) -> dict[str, str]:
        if provider == "baidu-translate":
            return {
                "provider": "baidu-translate",
                "name": "百度翻译开放平台",
                "base_url": "",
                "model": "https://api.fanyi.baidu.com/api/trans/vip/translate",
                "api_key": "",
            }
        if provider == "deepl":
            return {
                "provider": "deepl",
                "name": "DeepL",
                "base_url": "https://api-free.deepl.com",
                "model": "quality_optimized",
                "api_key": "",
            }
        if provider == "google-cloud-translate":
            return {
                "provider": "google-cloud-translate",
                "name": "Google Cloud Translate",
                "base_url": "",
                "model": "global",
                "api_key": "",
            }
        if provider == "openai":
            return {
                "provider": "openai",
                "name": "OpenAI",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "api_key": "",
            }
        if provider == "deepseek":
            return {
                "provider": "deepseek",
                "name": "DeepSeek",
                "base_url": "https://api.deepseek.com/v1",
                "model": "deepseek-chat",
                "api_key": "",
            }
        return {
            "provider": "openai-compatible",
            "name": "自定义节点",
            "base_url": "https://your-endpoint.example.com/v1",
            "model": "",
            "api_key": "",
        }

    @staticmethod
    def _translation_provider_field_labels(provider: str) -> tuple[str, str, str, str]:
        match provider:
            case "baidu-translate":
                return (
                    "APP ID",
                    "API 地址",
                    "密钥",
                    "百度翻译开放平台按 demo 签名调用；请填写 APP ID 和密钥，API 地址可保留默认值。",
                )
            case "deepl":
                return (
                    "Server URL",
                    "模型类型",
                    "API Key",
                    "DeepL 需要 API Key；Server URL 可保留默认 free 地址，模型类型可用 quality_optimized。",
                )
            case "google-cloud-translate":
                return (
                    "Project ID",
                    "Location",
                    "凭证 JSON",
                    "Google Cloud 需要填写 Project ID；凭证可填服务账号 JSON 路径或完整 JSON 内容。",
                )
            case "deepseek":
                return (
                    "Base URL",
                    "模型名",
                    "API Key",
                    "DeepSeek 使用 OpenAI 兼容接口，通常填写 https://api.deepseek.com/v1。",
                )
            case "openai":
                return (
                    "Base URL",
                    "模型名",
                    "API Key",
                    "OpenAI 使用标准 /v1 接口，模型可用 gpt-4o-mini 等。",
                )
            case _:
                return (
                    "Base URL",
                    "模型名",
                    "API Key",
                    "兼容接口需提供可用的 chat/completions 地址与模型名。",
                )

    def _open_translation_node_editor(self, *, provider: str = "openai", existing_node=None) -> dict[str, str] | None:
        defaults = self._preset_translation_node_values(provider)
        dialog = QDialog(self)
        dialog.setWindowTitle("翻译节点")
        dialog.setModal(True)
        form = QFormLayout(dialog)
        form.setSpacing(10)

        provider_combo = QComboBox(dialog)
        for label, value in TRANSLATION_PROVIDER_OPTIONS:
            provider_combo.addItem(label, value)
        name_edit = QLineEdit(dialog)
        base_url_edit = QLineEdit(dialog)
        model_edit = QLineEdit(dialog)
        api_key_edit = QLineEdit(dialog)
        api_key_edit.setEchoMode(QLineEdit.Password)
        base_url_label = QLabel("Base URL", dialog)
        model_label = QLabel("模型名", dialog)
        api_key_label = QLabel("API Key", dialog)
        provider_hint = QLabel("", dialog)
        provider_hint.setObjectName("SectionHint")
        provider_hint.setWordWrap(True)

        if existing_node is not None:
            name_edit.setText(existing_node.name)
            base_url_edit.setText(existing_node.base_url)
            model_edit.setText(existing_node.model)
            api_key_edit.setText(existing_node.api_key)
            provider_value = existing_node.provider
        else:
            name_edit.setText(defaults["name"])
            base_url_edit.setText(defaults["base_url"])
            model_edit.setText(defaults["model"])
            api_key_edit.setText(defaults["api_key"])
            provider_value = defaults["provider"]

        def apply_provider_defaults(provider_name: str, *, override_values: bool) -> None:
            values = self._preset_translation_node_values(provider_name)
            base_label, model_name_label, key_label, hint_text = self._translation_provider_field_labels(provider_name)
            base_url_label.setText(base_label)
            model_label.setText(model_name_label)
            api_key_label.setText(key_label)
            provider_hint.setText(hint_text)
            if override_values:
                name_edit.setText(values["name"])
                base_url_edit.setText(values["base_url"])
                model_edit.setText(values["model"])

        provider_index = provider_combo.findData(provider_value)
        provider_combo.setCurrentIndex(provider_index if provider_index >= 0 else 0)
        apply_provider_defaults(provider_value, override_values=existing_node is None)
        if existing_node is None:
            provider_combo.currentIndexChanged.connect(
                lambda _index: apply_provider_defaults(
                    str(provider_combo.currentData() or "openai-compatible"),
                    override_values=True,
                )
            )

        form.addRow("节点类型", provider_combo)
        form.addRow("节点名称", name_edit)
        form.addRow(base_url_label, base_url_edit)
        form.addRow(model_label, model_edit)
        form.addRow(api_key_label, api_key_edit)
        form.addRow(provider_hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        apply_window_chrome_theme(dialog, self._current_theme)
        if dialog.exec() != QDialog.Accepted:
            return None

        name = name_edit.text().strip()
        base_url = base_url_edit.text().strip()
        model = model_edit.text().strip()
        api_key = api_key_edit.text().strip()
        provider_value = str(provider_combo.currentData() or "openai-compatible").strip() or "openai-compatible"
        match provider_value:
            case "baidu-translate":
                missing_required = not name or not base_url or not api_key
                if not model:
                    model = self._preset_translation_node_values("baidu-translate")["model"]
                error_text = "百度翻译节点至少需要节点名称、APP ID 和密钥。"
            case "deepl":
                missing_required = not name or not api_key
                if not base_url:
                    base_url = self._preset_translation_node_values("deepl")["base_url"]
                if not model:
                    model = self._preset_translation_node_values("deepl")["model"]
                error_text = "DeepL 节点至少需要节点名称和 API Key。"
            case "google-cloud-translate":
                missing_required = not name or not base_url
                if not model:
                    model = self._preset_translation_node_values("google-cloud-translate")["model"]
                error_text = "Google Cloud Translate 节点至少需要节点名称和 Project ID。"
            case _:
                missing_required = not name or not base_url or not model
                error_text = "节点名称、Base URL 和模型名不能为空。"
        if missing_required:
            self._show_message(QMessageBox.Warning, "保存失败", error_text)
            return None
        return {
            "node_id": getattr(existing_node, "node_id", uuid4().hex),
            "name": name,
            "provider": provider_value,
            "base_url": base_url,
            "model": model,
            "api_key": api_key,
        }

    def _save_translation_nodes_from_dicts(self, nodes: list[dict[str, object]], *, selected_node_id: str | None = None) -> None:
        set_translation_nodes(nodes)
        if selected_node_id is not None:
            set_translation_node_id(selected_node_id)
            self._translation_node_id = selected_node_id
        self.settings = get_settings()
        if self._translation_node_id and all(node.node_id != self._translation_node_id for node in self.settings.translation_nodes):
            set_translation_node_id(None)
            self._translation_node_id = None
        if not self.settings.translation_nodes and self._translation_enabled:
            set_translation_enabled(False)
            self._translation_enabled = False
        self._refresh_translation_node_combo()
        self._refresh_workspace_summary()
        self._emit_live_translation_settings()

    def _add_translation_node(self) -> None:
        node_payload = self._open_translation_node_editor()
        if node_payload is None:
            return
        existing_nodes = [
            {
                "node_id": node.node_id,
                "name": node.name,
                "provider": node.provider,
                "base_url": node.base_url,
                "model": node.model,
                "api_key": node.api_key,
            }
            for node in self.settings.translation_nodes
        ]
        existing_nodes.append(node_payload)
        self._save_translation_nodes_from_dicts(existing_nodes, selected_node_id=str(node_payload["node_id"]))

    def _edit_selected_translation_node(self) -> None:
        selected_id = self._selected_translation_node_id()
        if not selected_id:
            self._show_message(QMessageBox.Information, "暂无节点", "请先选择一个翻译节点。")
            return
        existing_node = next((node for node in self.settings.translation_nodes if node.node_id == selected_id), None)
        if existing_node is None:
            self._show_message(QMessageBox.Warning, "节点不存在", "当前翻译节点已不存在，请重新选择。")
            self._translation_node_id = None
            self._refresh_translation_node_combo()
            return
        node_payload = self._open_translation_node_editor(provider=existing_node.provider, existing_node=existing_node)
        if node_payload is None:
            return
        updated_nodes = []
        for node in self.settings.translation_nodes:
            if node.node_id == selected_id:
                updated_nodes.append(node_payload)
            else:
                updated_nodes.append(
                    {
                        "node_id": node.node_id,
                        "name": node.name,
                        "provider": node.provider,
                        "base_url": node.base_url,
                        "model": node.model,
                        "api_key": node.api_key,
                    }
                )
        self._save_translation_nodes_from_dicts(updated_nodes, selected_node_id=selected_id)

    def _test_selected_translation_node(self) -> None:
        selected_id = self._selected_translation_node_id()
        if not selected_id:
            self._show_message(QMessageBox.Information, "暂无节点", "请先选择一个翻译节点。")
            return

        target_language = self._translation_target_language
        sample_text = "这是一条翻译节点连通性测试，请翻译得自然一些。"
        source_language = "zh"
        if target_language == "zh":
            sample_text = "This is a translation node connectivity test. Please keep it natural for speech."
            source_language = "en"

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            service = TranslationService()
            result = service.translate(
                sample_text,
                target_language=target_language,
                source_language=source_language,
                node_id=selected_id,
            )
        except Exception as exc:
            logger.exception("测试翻译节点失败")
            self._show_message(QMessageBox.Critical, "测试失败", f"翻译节点连接失败：{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        translated_text = result.text.strip() or "(返回空文本)"
        self._show_message(
            QMessageBox.Information,
            "测试成功",
            f"翻译节点 {result.node_name or '当前节点'} 可以正常访问。\n测试结果：{translated_text}",
        )

    def _remove_selected_translation_node(self) -> None:
        selected_id = self._selected_translation_node_id()
        if not selected_id:
            self._show_message(QMessageBox.Information, "暂无节点", "当前没有可删除的翻译节点。")
            return
        remaining_nodes = [
            {
                "node_id": node.node_id,
                "name": node.name,
                "provider": node.provider,
                "base_url": node.base_url,
                "model": node.model,
                "api_key": node.api_key,
            }
            for node in self.settings.translation_nodes
            if node.node_id != selected_id
        ]
        next_selected = remaining_nodes[0]["node_id"] if remaining_nodes else None
        self._save_translation_nodes_from_dicts(remaining_nodes, selected_node_id=next_selected)
        if not remaining_nodes:
            self._translation_enabled = False
            self._sync_translation_card_controls()
            self._emit_live_translation_settings()

    @staticmethod
    def _set_text_if_changed(widget: QLabel | QPushButton, text: str) -> None:
        if widget.text() != text:
            widget.setText(text)

    @staticmethod
    def _set_text_edit_if_changed(widget: QTextEdit, text: str) -> None:
        if widget.toPlainText() != text:
            widget.setPlainText(text)

    def _set_live_status_label_state(self, text: str, state: str) -> None:
        display_text = f"● {text}" if state in ("busy", "ready") else f"● {text}" if text else ""
        palette = self._theme_palette()
        if state == "busy":
            color = palette["success_text"]
        elif state == "error":
            color = palette["error_text"]
        else:
            color = palette["muted_text"]
            
        styled_text = f'<span style="color: {color};">{display_text}</span>'
        
        text_changed = self.live_status_label.text() != styled_text
        state_changed = self.live_status_label.property("state") != state
        if text_changed:
            self.live_status_label.setText(styled_text)
            self.live_status_label.setProperty("raw_text", text)
        if state_changed:
            self.live_status_label.setProperty("state", state)
            self._refresh_widget_style(self.live_status_label)

    def _set_input_mode(self, mode_value: str) -> None:
        target_index = self.input_mode_combo.findData(mode_value)
        if target_index >= 0 and target_index != self.input_mode_combo.currentIndex():
            self.input_mode_combo.setCurrentIndex(target_index)

    def _apply_theme(self, theme_name: str) -> None:
        self._current_theme = theme_name
        app = QApplication.instance()
        if app is not None:
            app.setProperty("app_theme", theme_name)
            app.setStyleSheet(get_theme_stylesheet(theme_name))
        if hasattr(self, "waveform_widget"):
            self.waveform_widget.set_theme(theme_name)
        if self._quick_settings_dialog is not None and not self._quick_settings_dialog.isVisible():
            self._dispose_quick_settings_dialog()
        self._refresh_theme_sensitive_ui()
        self._apply_window_chrome_theme()

    def _apply_window_chrome_theme(self) -> None:
        apply_window_chrome_theme(self, self._current_theme)
        if self._quick_settings_dialog is not None:
            apply_window_chrome_theme(self._quick_settings_dialog, self._current_theme)
        if self._message_dialog is not None:
            apply_window_chrome_theme(self._message_dialog, self._current_theme)

    def _on_theme_changed(self) -> None:
        selected = self.theme_combo.currentData()
        self._apply_theme(str(selected or DEFAULT_THEME))

    def _on_voice_output_model_changed(self) -> None:
        selected_model = str(self.tts_engine_combo.currentData() or "").strip()
        if not selected_model or selected_model == self._voice_output_model:
            self._sync_voice_output_model_controls()
            return
        try:
            set_voice_output_model(selected_model)
        except Exception as exc:
            logger.exception("保存语音输出模型失败")
            self._show_message(QMessageBox.Critical, "保存失败", f"无法保存语音输出模型：{exc}")
            previous_index = self.tts_engine_combo.findData(self._voice_output_model)
            if previous_index >= 0:
                self.tts_engine_combo.blockSignals(True)
                self.tts_engine_combo.setCurrentIndex(previous_index)
                self.tts_engine_combo.blockSignals(False)
            self._sync_voice_output_model_controls()
            return
        self._voice_output_model = selected_model
        self.settings = get_settings()
        self._sync_translation_card_controls()
        self._sync_voice_output_model_controls()
        self._refresh_workspace_summary()
        if self._is_live_worker_active():
            self._pending_live_restart_mode = self._active_request_mode
            self._stop_live_microphone_for_settings_refresh("语音输出模型已切换，正在立即重启处理流...")
        elif self._live_worker_thread is not None and self._models_ready:
            self._models_ready = False
            self._refresh_live_start_button()
            self._set_live_status("正在切换语音输出模型...", "busy")
            self.live_tts_switch_requested.emit()

    def _on_translation_card_toggled(self, checked: bool) -> None:
        enabled = bool(checked)
        selected_node_id = self._selected_translation_node_id()
        if enabled and not selected_node_id:
            if self.settings.translation_nodes:
                selected_node_id = self.settings.translation_nodes[0].node_id
                self._translation_node_id = selected_node_id
            else:
                self._show_message(QMessageBox.Information, "缺少翻译节点", "请先新增百度翻译、DeepL、Google Cloud 或兼容接口翻译节点。")
                self._translation_enabled = False
                self._sync_translation_card_controls()
                return
        try:
            set_translation_enabled(enabled)
            if selected_node_id != self._translation_node_id:
                set_translation_node_id(selected_node_id or None)
                self._translation_node_id = selected_node_id or None
        except Exception as exc:
            logger.exception("保存翻译设置失败")
            self._show_message(QMessageBox.Critical, "保存失败", f"无法保存翻译设置：{exc}")
            self._sync_translation_card_controls()
            return
        self._translation_enabled = enabled
        self.settings = get_settings()
        self._refresh_translation_node_combo()
        self._refresh_workspace_summary()
        self._emit_live_translation_settings()

    def _on_translation_target_language_changed(self, _index: int) -> None:
        if self._is_meanvc2_selected():
            self._sync_voice_output_model_controls()
            return
        combo = self.sender()
        selected_language = ""
        if isinstance(combo, QComboBox):
            selected_language = str(combo.currentData() or "").strip()
        if not selected_language:
            selected_language = self._translation_target_language
        if selected_language == self._translation_target_language:
            return
        try:
            set_translation_target_language(selected_language)
        except Exception as exc:
            logger.exception("保存翻译目标语言失败")
            self._show_message(QMessageBox.Critical, "保存失败", f"无法保存翻译目标语言：{exc}")
            self._sync_translation_card_controls()
            return
        self._translation_target_language = selected_language
        self.settings = get_settings()
        self._sync_translation_card_controls()
        self._refresh_workspace_summary()
        self._emit_live_translation_settings()

    def _on_translation_card_node_changed(self, _index: int) -> None:
        selected_node_id = self._selected_translation_node_id() or None
        if selected_node_id == self._translation_node_id:
            return
        try:
            set_translation_node_id(selected_node_id)
        except Exception as exc:
            logger.exception("保存翻译节点失败")
            self._show_message(QMessageBox.Critical, "保存失败", f"无法保存翻译节点：{exc}")
            self._refresh_translation_node_combo()
            return
        self._translation_node_id = selected_node_id
        if self._translation_enabled and not selected_node_id:
            try:
                set_translation_enabled(False)
            except Exception as exc:
                logger.exception("关闭翻译开关失败")
                self._show_message(QMessageBox.Critical, "保存失败", f"无法更新翻译开关：{exc}")
                self._refresh_translation_node_combo()
                return
            self._translation_enabled = False
        self.settings = get_settings()
        self._sync_translation_card_controls()
        self._refresh_workspace_summary()
        self._emit_live_translation_settings()

    def _emit_live_translation_settings(self) -> None:
        if self._live_worker_thread is None:
            return
        self.live_translation_settings_changed_requested.emit(
            {
                "translation_enabled": self._translation_enabled,
                "translation_target_language": self._translation_target_language,
                "translation_node_id": self._translation_node_id,
            }
        )

    def _on_output_volume_changed(self, value: int) -> None:
        self._set_text_if_changed(self.output_gain_label, f"{value / 100:.2f}")
        self._refresh_workspace_summary()
        self.live_output_volume_changed_requested.emit(self._current_output_volume())

    def _on_speech_rate_changed(self, value: int) -> None:
        _ = value
        self._refresh_workspace_summary()
        self.live_speech_rate_changed_requested.emit(self._current_speech_rate())

    def _current_output_volume(self) -> float:
        return self.output_gain_slider.value() / 100.0

    def _on_monitor_output_volume_changed(self, value: int) -> None:
        self._set_text_if_changed(self.monitor_output_volume_label, f"{value / 100:.2f}")
        self._refresh_workspace_summary()
        self.live_monitor_output_volume_changed_requested.emit(self._current_monitor_output_volume())

    def _current_monitor_output_volume(self) -> float:
        return self.monitor_output_volume_slider.value() / 100.0

    def _refresh_input_level_state(self) -> None:
        if not hasattr(self, "input_level_meter") or not hasattr(self, "input_level_state_label"):
            return
        if self._current_input_mode() == "text":
            self.input_level_meter.setValue(0)
            self._set_text_if_changed(self.input_level_state_label, "文本模式")
            return
        if self._live_worker_state == LiveWorkerState.RUNNING.value:
            self.input_level_meter.setValue(72)
            self._set_text_if_changed(
                self.input_level_state_label,
                self._mode_input_level_text(self._active_request_mode, running=True),
            )
            return
        if self._live_worker_state == LiveWorkerState.STOPPING.value:
            self.input_level_meter.setValue(18)
            self._set_text_if_changed(self.input_level_state_label, "停止中")
            return
        self.input_level_meter.setValue(0)
        self._set_text_if_changed(
            self.input_level_state_label,
            self._mode_input_level_text(self._current_input_mode(), running=False),
        )

    def _create_content_card(self, title_text: str, hint_text: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("ContentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)
        title = QLabel(title_text)
        title.setObjectName("CardTitle")
        header_row.addWidget(title)
        header_row.addStretch(1)
        badge = QLabel("")
        badge.setObjectName("CardBadge")
        badge.setVisible(False)
        badge.setStyleSheet("font-size: 9px; padding: 4px 9px; font-weight: 700;")
        header_row.addWidget(badge, alignment=Qt.AlignVCenter)
        layout.addLayout(header_row)
        card._header_row = header_row  # type: ignore[attr-defined]
        card._title_badge = badge  # type: ignore[attr-defined]
        if hint_text:
            hint = QLabel(hint_text)
            hint.setObjectName("SectionHint")
            hint.setWordWrap(True)
            layout.addWidget(hint)
        return card, layout

    def _create_mode_locked_overlay(self, card: QFrame) -> QFrame:
        overlay = QFrame(card)
        overlay.hide()
        overlay.setCursor(Qt.ForbiddenCursor)
        overlay_layout = QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(18, 18, 18, 18)
        overlay_layout.setSpacing(8)
        overlay_layout.addStretch(1)
        title = QLabel("MeanVC2 模式下不可用", overlay)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 13px; font-weight: 700;")
        subtitle = QLabel("仅 Index-TTS 支持这些控制项", overlay)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setObjectName("SectionHint")
        overlay_layout.addWidget(title)
        overlay_layout.addWidget(subtitle)
        overlay_layout.addStretch(1)
        overlay._title_label = title  # type: ignore[attr-defined]
        overlay._subtitle_label = subtitle  # type: ignore[attr-defined]
        return overlay

    def _set_mode_locked_card_visual(
        self,
        *,
        card: QFrame | None,
        overlay: QFrame | None,
        hint_label: QLabel | None,
        locked: bool,
        badge_text: str,
        overlay_message: str,
    ) -> None:
        if card is None or overlay is None:
            return
        palette = self._theme_palette()
        badge = getattr(card, "_title_badge", None)
        if isinstance(badge, QLabel):
            badge.setText(badge_text if locked else "")
            badge.setVisible(locked)
            badge.setStyleSheet(
                "font-size: 9px; padding: 4px 9px; font-weight: 700;"
                f"background: {palette['warning_bg']};"
                f"color: {palette['warning_text']};"
                f"border: 1px solid {palette['warning_border']};"
                "border-radius: 10px;"
            )

        if locked:
            card.setStyleSheet(
                f"QFrame#ContentCard {{"
                f" background: {palette['surface_bg_alt']};"
                f" border: 2px dashed {palette['warning_border']};"
                " border-radius: 14px;"
                "}"
            )
            overlay.hide()
        else:
            card.setStyleSheet("")
            overlay.hide()

    def _set_mode_locked_section_visual(
        self,
        section: QFrame | None,
        *,
        locked: bool,
        tooltip: str = "",
    ) -> None:
        if section is None:
            return
        palette = self._theme_palette()
        if locked:
            section.setStyleSheet(
                f"QFrame#SummaryCard {{"
                f" background: {palette['surface_bg_alt']};"
                f" border: 2px dashed {palette['warning_border']};"
                " border-radius: 12px;"
                "}"
            )
            section.setToolTip(tooltip)
        else:
            section.setStyleSheet("")
            section.setToolTip(tooltip)

    def _layout_mode_locked_overlay(
        self,
        card: QFrame,
        overlay: QFrame,
        hint_label: QLabel | None,
    ) -> None:
        top = 48
        if hint_label is not None and hint_label.isVisible():
            top = max(top, int(hint_label.geometry().bottom()) + 10)
        overlay.setGeometry(12, top, max(0, card.width() - 24), max(0, card.height() - top - 12))

    def _update_mode_locked_overlays_geometry(self) -> None:
        if self.emotion_control_card is not None and self.emotion_control_overlay is not None:
            self._layout_mode_locked_overlay(
                self.emotion_control_card,
                self.emotion_control_overlay,
                self.emotion_control_hint_label,
            )
        if self.tts_other_options_card is not None and self.tts_other_options_overlay is not None:
            self._layout_mode_locked_overlay(
                self.tts_other_options_card,
                self.tts_other_options_overlay,
                self.tts_other_options_hint_label,
            )

    def _ensure_reference_audio_dialog(self) -> QFileDialog:
        if self._reference_audio_dialog is not None:
            return self._reference_audio_dialog

        dialog = QFileDialog(self, "选择实时参考音频")
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("音频文件 (*.wav *.mp3 *.ogg *.flac);;所有文件 (*)")
        dialog.setViewMode(QFileDialog.Detail)
        self._reference_audio_dialog = dialog
        return dialog

    def _browse_live_reference_audio(self) -> None:
        dialog = self._ensure_reference_audio_dialog()
        current_path = self.live_reference_audio_edit.text().strip()
        if current_path:
            current_file = Path(current_path)
            dialog.setDirectory(str(current_file.parent if current_file.parent.exists() else Path.cwd()))
            if current_file.exists():
                dialog.selectFile(str(current_file))
        else:
            dialog.setDirectory(str(Path.cwd()))

        if dialog.exec() != QDialog.Accepted:
            return

        selected_files = dialog.selectedFiles()
        path = selected_files[0] if selected_files else ""
        if path:
            self.live_reference_audio_edit.setText(path)
            if (
                self._is_live_worker_active()
                and self._active_request_mode == "microphone"
                and self._is_meanvc2_selected()
            ):
                self._set_live_status("正在切换 MeanVC2 参考音色，新音色将在当前实时流中生效...", "busy")
                self.live_reference_audio_changed_requested.emit(path)
                return
            if self._models_ready:
                self._models_ready = False
                self._refresh_live_start_button()
                self._set_live_status(
                    "正在加载参考音色（MeanVC2 首次加载可能需要 5-10 秒）..." if self._is_meanvc2_selected() else "正在加载参考音色并预热...",
                    "busy",
                )
                self.live_preload_requested.emit(self._build_live_preload_payload(path))
            else:
                self._pending_reference_preload = True
                self._refresh_live_start_button()
                self._set_live_status(
                    "参考音色已选择，模型加载完成后自动更新（MeanVC2 首次加载可能需要 5-10 秒）"
                    if self._is_meanvc2_selected()
                    else "参考音色已选择，模型加载完成后自动更新",
                    "busy",
                )

    def _build_tts_instruction(self) -> str | None:
        preset_instruction = self.emotion_preset_combo.currentData()
        custom_instruction = self.emotion_custom_edit.text().strip()
        parts = [part for part in (preset_instruction, custom_instruction) if isinstance(part, str) and part.strip()]
        if not parts:
            return None
        return " ".join(parts)

    def _is_auto_emotion_feature_enabled(self) -> bool:
        return bool(self._auto_emotion_feature_enabled)

    def _is_auto_emotion_runtime_enabled(self) -> bool:
        return self._is_auto_emotion_feature_enabled() and self._current_input_mode() in {"microphone", "dual"}

    def _capture_manual_index_emotion_snapshot(self) -> None:
        if self._manual_index_emo_snapshot is not None:
            return
        self._manual_index_emo_snapshot = {
            "mode": self.index_emotion_mode_combo.currentData(),
            "vector": [self.index_emo_sliders[en_name].value() for _cn_name, en_name, _emoji in INDEX_TTS_EMOTIONS],
            "alpha": self.index_emo_alpha_slider.value(),
        }

    def _restore_manual_index_emotion_snapshot(self) -> None:
        snapshot = self._manual_index_emo_snapshot
        self._manual_index_emo_snapshot = None
        if not isinstance(snapshot, dict):
            return
        raw_vector = snapshot.get("vector")
        raw_alpha = snapshot.get("alpha")
        raw_mode = snapshot.get("mode")
        if isinstance(raw_vector, list):
            for (_cn_name, en_name, _emoji), value in zip(INDEX_TTS_EMOTIONS, raw_vector, strict=False):
                slider = self.index_emo_sliders[en_name]
                slider.setValue(max(slider.minimum(), min(slider.maximum(), int(value))))
        if isinstance(raw_alpha, int):
            self.index_emo_alpha_slider.setValue(
                max(self.index_emo_alpha_slider.minimum(), min(self.index_emo_alpha_slider.maximum(), raw_alpha))
            )
        if isinstance(raw_mode, str):
            self._set_index_emotion_mode(raw_mode)

    def _build_index_emo_vector(self) -> tuple[list[float] | None, float]:
        """Collect Index-TTS emotion vector from sliders (or None if not in vector mode)."""
        mode_value = self.index_emotion_mode_combo.currentData()
        slider_values = [self.index_emo_sliders[en_name].value() for _cn_name, en_name, _emoji in INDEX_TTS_EMOTIONS]
        alpha_value = self.index_emo_alpha_slider.value()
        if self._is_auto_emotion_runtime_enabled() and isinstance(self._manual_index_emo_snapshot, dict):
            snapshot_mode = self._manual_index_emo_snapshot.get("mode")
            snapshot_vector = self._manual_index_emo_snapshot.get("vector")
            snapshot_alpha = self._manual_index_emo_snapshot.get("alpha")
            if isinstance(snapshot_mode, str):
                mode_value = snapshot_mode
            if isinstance(snapshot_vector, list):
                slider_values = [int(value) for value in snapshot_vector[: len(INDEX_TTS_EMOTIONS)]]
            if isinstance(snapshot_alpha, int):
                alpha_value = snapshot_alpha
        if mode_value != "vector":
            return None, 1.0
        emo_vector = [value * 0.05 for value in slider_values]
        if all(v <= 0.0 for v in emo_vector):
            return None, 1.0
        emo_alpha = alpha_value * 0.05
        return emo_vector, emo_alpha

    def _current_speech_rate(self) -> float:
        return 0.5 + self.speech_rate_slider.value() * 0.05

    def _current_input_mode(self) -> str:
        if hasattr(self, "capture_source_mic_checkbox") and hasattr(self, "capture_source_program_checkbox"):
            use_mic = self.capture_source_mic_checkbox.isChecked()
            use_program = self.capture_source_program_checkbox.isChecked() and self._program_audio_supported
            if use_mic and use_program:
                return "dual"
            if use_program:
                return "program"
            if use_mic:
                return "microphone"
            return "none"
        return "dual" if self._program_audio_supported else "microphone"

    def _set_capture_source_selection(self, input_mode: str) -> None:
        use_mic = input_mode in {"microphone", "dual"}
        use_program = input_mode in {"program", "dual"} and self._program_audio_supported
        if hasattr(self, "capture_source_mic_checkbox"):
            mic_blocker = QSignalBlocker(self.capture_source_mic_checkbox)
            self.capture_source_mic_checkbox.setChecked(use_mic)
            del mic_blocker
        if hasattr(self, "capture_source_program_checkbox"):
            program_blocker = QSignalBlocker(self.capture_source_program_checkbox)
            self.capture_source_program_checkbox.setChecked(use_program)
            del program_blocker

    def _validate_live_input_mode_change(self, input_mode: str) -> str | None:
        if input_mode == "none":
            return None
        reference_audio = self.live_reference_audio_edit.text().strip() if hasattr(self, "live_reference_audio_edit") else ""
        if input_mode != "program" and not reference_audio:
            return "请选择实时页面使用的参考音频。"
        if input_mode in {"program", "dual"}:
            program_pid = self.program_combo.currentData() if hasattr(self, "program_combo") else None
            if not isinstance(program_pid, int):
                return "请先选择一个正在发声的程序。"
        return None

    def _handle_live_input_mode_change(self, requested_mode: str) -> None:
        active_mode = self._active_request_mode
        if active_mode not in {"microphone", "program", "dual"}:
            return
        if requested_mode == active_mode:
            return
        if requested_mode in {"program", "dual"}:
            self._refresh_running_programs()
        validation_message = self._validate_live_input_mode_change(requested_mode)
        if validation_message is not None:
            self._set_capture_source_selection(active_mode)
            self._show_message(QMessageBox.Warning, "无法切换监听来源", validation_message)
            return
        self._pending_live_restart_mode = requested_mode if requested_mode in {"microphone", "program", "dual"} else None
        if self._live_worker_state == LiveWorkerState.RUNNING.value:
            status_text = (
                "已关闭所有监听来源，正在停止当前处理流..."
                if requested_mode == "none"
                else f"监听来源已切换为{self._mode_start_button_text(requested_mode).replace('开启', '')}，正在更新处理流..."
            )
            self._stop_live_microphone_for_settings_refresh(status_text)

    @staticmethod
    def _mode_intro_text(input_mode: str) -> str:
        match input_mode:
            case "none":
                return "中间区域按双语记录流展示：请至少勾选麦克风、程序中的一个监听来源；下方仍可直接发送文字。"
            case "dual":
                return "中间区域按双语记录流展示：会同时监听麦克风与目标程序音频；程序监听只做转写并标记说话人，只有麦克风结果会进入后续 TTS。"
            case "program":
                return "中间区域按双语记录流展示：可监听某个正在运行的程序音频，由 ASR 转写并标记说话人，不会送入 TTS。"
            case _:
                return "中间区域按双语记录流展示：上方原文，下方翻译；麦克风结果与文字输入会汇总在这里，且只有麦克风结果会进入 TTS。"

    @staticmethod
    def _mode_placeholder_text(input_mode: str) -> str:
        match input_mode:
            case "none":
                return "待命中，请至少勾选麦克风或程序监听；也可以在下方发送文字。"
            case "dual":
                return "待命中，可同时开启麦克风与目标程序监听；也可以在下方发送文字。"
            case "program":
                return "待命中，可选择正在发声的程序并开始监听；也可以在下方发送文字。"
            case _:
                return "待命中，可直接开启麦克风，也可以在下方发送文字。"

    @staticmethod
    def _mode_stop_button_text(input_mode: str | None) -> str:
        match input_mode:
            case "none":
                return "关闭监听"
            case "dual":
                return "关闭双路监听"
            case "program":
                return "关闭程序监听"
            case _:
                return "关闭麦克风"

    @staticmethod
    def _mode_start_button_text(input_mode: str) -> str:
        match input_mode:
            case "none":
                return "选择监听来源"
            case "dual":
                return "开启双路监听"
            case "program":
                return "开启程序监听"
            case _:
                return "开启麦克风"

    @staticmethod
    def _mode_input_level_text(input_mode: str | None, *, running: bool) -> str:
        if not running:
            if input_mode == "none":
                return "待选择"
            return "待选程序" if input_mode in {"program", "dual"} else "待输入"
        match input_mode:
            case "none":
                return "未选择"
            case "dual":
                return "双路监听中"
            case "program":
                return "程序监听中"
            case _:
                return "监听中"

    def _refresh_running_programs(self) -> None:
        if not hasattr(self, "program_combo"):
            return
        previous_pid = self.program_combo.currentData()
        self.program_combo.clear()
        self._program_options.clear()
        if not self._program_audio_supported:
            self.program_combo.addItem("当前系统不支持程序监听", None)
            self.program_combo.setEnabled(False)
            self.refresh_program_button.setEnabled(False)
            self._refresh_live_start_button()
            self._refresh_workspace_summary()
            return
        try:
            programs = list_running_audio_programs()
        except Exception as exc:
            logger.exception("刷新运行中程序失败")
            self.program_combo.addItem(f"刷新失败: {exc}", None)
            self.program_combo.setEnabled(False)
            self.refresh_program_button.setEnabled(True)
            self._refresh_live_start_button()
            self._refresh_workspace_summary()
            return
        if not programs:
            self.program_combo.addItem("当前没有检测到正在发声的程序", None)
            self.program_combo.setEnabled(False)
            self.refresh_program_button.setEnabled(True)
            self._refresh_live_start_button()
            self._refresh_workspace_summary()
            return
        for program in programs:
            self._program_options.append((program.pid, program.display_label))
            self.program_combo.addItem(program.display_label, program.pid)
        self.program_combo.setEnabled(True)
        target_index = self.program_combo.findData(previous_pid)
        if target_index < 0:
            target_index = 0
        self.program_combo.setCurrentIndex(target_index)
        self._refresh_live_start_button()
        self._refresh_workspace_summary()

    def _sync_capture_source_controls(self) -> None:
        mode = self._current_input_mode()
        use_microphone = mode in {"microphone", "dual"}
        use_program = mode in {"program", "dual"}
        if hasattr(self, "input_device_combo"):
            self.input_device_combo.setEnabled(use_microphone)
        if hasattr(self, "input_device_label_widget"):
            self.input_device_label_widget.setEnabled(use_microphone)
        if hasattr(self, "program_combo"):
            self.program_combo.setEnabled(
                use_program
                and self.program_combo.count() > 0
                and self._program_audio_supported
            )
        if hasattr(self, "program_combo_label"):
            self.program_combo_label.setEnabled(use_program)
        if hasattr(self, "refresh_program_button"):
            self.refresh_program_button.setEnabled(use_program and self._program_audio_supported)
        if use_program:
            self._refresh_running_programs()

    def _set_index_emotion_mode(self, mode_value: str) -> None:
        if (
            mode_value != "vector"
            and self._is_auto_emotion_runtime_enabled()
        ):
            mode_value = "vector"
        target_index = self.index_emotion_mode_combo.findData(mode_value)
        if target_index >= 0 and target_index != self.index_emotion_mode_combo.currentIndex():
            self.index_emotion_mode_combo.setCurrentIndex(target_index)

    def _on_index_emotion_mode_changed(self) -> None:
        if (
            self._is_auto_emotion_runtime_enabled()
            and self.index_emotion_mode_combo.currentData() != "vector"
        ):
            self._set_index_emotion_mode("vector")
            return
        mode = self.index_emotion_mode_combo.currentData()
        is_vector = mode == "vector"
        if hasattr(self, "nav_preset_btn") and hasattr(self, "nav_vector_btn"):
            self.nav_preset_btn.setChecked(not is_vector)
            self.nav_vector_btn.setChecked(is_vector)
        if hasattr(self, "emotion_stack"):
            self.emotion_stack.setCurrentIndex(1 if is_vector else 0)

    def _apply_auto_emotion_vector(self, vector: list[float] | tuple[float, ...] | None, alpha: float | None) -> None:
        if vector is None or not self._is_auto_emotion_runtime_enabled():
            return
        self._capture_manual_index_emotion_snapshot()
        self._set_index_emotion_mode("vector")
        scale = max(0.0, float(alpha)) if alpha is not None else 1.0
        for (_cn_name, en_name, _emoji), value in zip(INDEX_TTS_EMOTIONS, vector, strict=False):
            slider = self.index_emo_sliders[en_name]
            effective_value = max(0.0, float(value)) * scale
            slider_value = max(slider.minimum(), min(slider.maximum(), int(round(effective_value / 0.05))))
            slider.setValue(slider_value)
        if alpha is not None:
            alpha_value = max(
                self.index_emo_alpha_slider.minimum(),
                min(self.index_emo_alpha_slider.maximum(), int(round(float(alpha) / 0.05))),
            )
            self.index_emo_alpha_slider.setValue(alpha_value)

    def _on_auto_emotion_toggled(self, *_args) -> None:
        feature_enabled = self._is_auto_emotion_feature_enabled()
        runtime_enabled = self._is_auto_emotion_runtime_enabled()
        if hasattr(self, "nav_preset_btn"):
            self.nav_preset_btn.setEnabled(not runtime_enabled)
        if hasattr(self, "emotion_preset_combo"):
            self.emotion_preset_combo.setEnabled(not runtime_enabled)
        if hasattr(self, "emotion_custom_edit"):
            self.emotion_custom_edit.setEnabled(not runtime_enabled)
        if runtime_enabled:
            self._capture_manual_index_emotion_snapshot()
            self._set_index_emotion_mode("vector")
        else:
            self._restore_manual_index_emotion_snapshot()
        if hasattr(self, "header_auto_emotion_value"):
            header_auto_emotion_card = self.header_auto_emotion_value.parentWidget()
            if header_auto_emotion_card is not None:
                header_auto_emotion_card.setVisible(feature_enabled)
            self._set_text_if_changed(
                self.header_auto_emotion_value,
                "等待识别" if runtime_enabled else ("待命" if feature_enabled else "实验关闭"),
            )

    def _update_auto_emotion_ui(self, _summary: str, state: str) -> None:
        if not self._is_auto_emotion_feature_enabled():
            return
        if hasattr(self, "header_auto_emotion_value"):
            status_text = {
                "ready": "已识别",
                "busy": "识别中",
                "error": "已回退",
                "idle": "待命",
            }.get(state, "待命")
            self._set_text_if_changed(self.header_auto_emotion_value, status_text)

    def _sync_emotion_preset_buttons(self) -> None:
        if not hasattr(self, "emotion_preset_buttons"):
            return
        current_index = self.emotion_preset_combo.currentIndex()
        for idx, btn in enumerate(self.emotion_preset_buttons):
            btn.setChecked(idx == current_index)

    def _disable_auto_emotion_for_text_mode(self) -> None:
        if self._current_input_mode() != "text":
            return
        if self._auto_emotion_feature_enabled:
            self._auto_emotion_feature_enabled = False
        if self._quick_settings_auto_emotion_checkbox is not None:
            self._quick_settings_auto_emotion_checkbox.setChecked(False)
        self._sync_tts_other_option_controls()

    @staticmethod
    def _set_checkbox_silently(checkbox: QCheckBox | None, checked: bool) -> None:
        if checkbox is None:
            return
        checkbox.blockSignals(True)
        checkbox.setChecked(checked)
        checkbox.blockSignals(False)

    @staticmethod
    def _set_slider_silently(slider: QSlider | None, value: int) -> None:
        if slider is None:
            return
        slider.blockSignals(True)
        slider.setValue(value)
        slider.blockSignals(False)

    def _experimental_torch_compile_hint_text(self) -> str:
        return "开启后需要重启。该模式会尝试使用 torch.compile / inductor autotune；如遇卡顿、报错或显存压力较大，建议关闭。"

    def _experimental_torch_compile_restart_detail(self) -> str:
        if not self._experimental_torch_compile_enabled:
            return "进阶模式已关闭，重启后会直接使用常规推理路径。"
        return "进阶模式已开启，重启后会尝试使用 torch.compile / inductor autotune。"

    def _sync_experimental_torch_compile_controls(self) -> None:
        hint_text = self._experimental_torch_compile_hint_text()
        if hasattr(self, "experimental_torch_compile_hint_label") and self.experimental_torch_compile_hint_label is not None:
            self._set_text_if_changed(self.experimental_torch_compile_hint_label, hint_text)
            self.experimental_torch_compile_hint_label.setToolTip(hint_text)
        if hasattr(self, "experimental_torch_compile_card_checkbox"):
            self.experimental_torch_compile_card_checkbox.setToolTip(hint_text)
        if self._quick_settings_torch_compile_checkbox is not None:
            self._quick_settings_torch_compile_checkbox.setToolTip(hint_text)

    def _sync_tts_other_option_controls(self) -> None:
        is_text_mode = False
        meanvc2_selected = self._is_meanvc2_selected()
        disabled_tooltip = "MeanVC2 当前不使用这些 Index-TTS 选项。"
        speech_rate_tooltip = (
            "MeanVC2 当前不支持原生语速控制。"
            if meanvc2_selected
            else "仅对 Index-TTS 生效，使用原生 duration_factor 控制语速。"
        )
        if self.tts_other_options_info_button is not None:
            self.tts_other_options_info_button.setToolTip(self._tts_other_options_tooltip())
        if self.tts_other_options_hint_label is not None:
            self.tts_other_options_hint_label.clear()
            self.tts_other_options_hint_label.setVisible(False)
        self._set_mode_locked_card_visual(
            card=self.tts_other_options_card,
            overlay=self.tts_other_options_overlay,
            hint_label=self.tts_other_options_hint_label,
            locked=meanvc2_selected,
            badge_text="仅 Index-TTS",
            overlay_message="当前使用 MeanVC2 语音转换模型，请切换到 Index-TTS 后再调整实验情感与 TTS 加速选项。",
        )
        if hasattr(self, "auto_emotion_card_checkbox"):
            self._set_checkbox_silently(self.auto_emotion_card_checkbox, self._auto_emotion_feature_enabled)
            self.auto_emotion_card_checkbox.setEnabled(not is_text_mode and not meanvc2_selected)
            self.auto_emotion_card_checkbox.setToolTip(
                disabled_tooltip
                if meanvc2_selected
                else ("情感识别仅作用于麦克风输入。" if is_text_mode else "")
            )
        slider_value = int(round(self._auto_emotion_strength / 0.05))
        if hasattr(self, "auto_emotion_card_strength_slider"):
            self._set_slider_silently(self.auto_emotion_card_strength_slider, slider_value)
            self.auto_emotion_card_strength_slider.setEnabled(not is_text_mode and not meanvc2_selected)
            self.auto_emotion_card_strength_slider.setToolTip(disabled_tooltip if meanvc2_selected else "")
        if hasattr(self, "auto_emotion_card_strength_label"):
            self._set_text_if_changed(self.auto_emotion_card_strength_label, f"{self._auto_emotion_strength:.2f}")
            self.auto_emotion_card_strength_label.setEnabled(not is_text_mode and not meanvc2_selected)
        if hasattr(self, "speech_rate_slider"):
            self.speech_rate_slider.setEnabled(not meanvc2_selected)
            self.speech_rate_slider.setToolTip(speech_rate_tooltip)
        if hasattr(self, "speech_rate_label"):
            self.speech_rate_label.setEnabled(not meanvc2_selected)
            self.speech_rate_label.setToolTip(speech_rate_tooltip)
        if hasattr(self, "speech_rate_title_label"):
            self.speech_rate_title_label.setEnabled(not meanvc2_selected)
            self.speech_rate_title_label.setToolTip(speech_rate_tooltip)
        if hasattr(self, "experimental_torch_compile_card_checkbox"):
            self._set_checkbox_silently(
                self.experimental_torch_compile_card_checkbox,
                self._experimental_torch_compile_enabled,
            )
            self.experimental_torch_compile_card_checkbox.setEnabled(not meanvc2_selected)
            self.experimental_torch_compile_card_checkbox.setToolTip(disabled_tooltip if meanvc2_selected else "")
        self._sync_experimental_torch_compile_controls()
        if hasattr(self, "experimental_accel_gpt_card_checkbox"):
            self._set_checkbox_silently(
                self.experimental_accel_gpt_card_checkbox,
                self._experimental_accel_gpt_enabled,
            )
            self.experimental_accel_gpt_card_checkbox.setEnabled(not meanvc2_selected)
            self.experimental_accel_gpt_card_checkbox.setToolTip(disabled_tooltip if meanvc2_selected else "")

    def _on_auto_emotion_card_toggled(self, checked: bool) -> None:
        self._auto_emotion_feature_enabled = bool(checked) and self._current_input_mode() in {"microphone", "dual"}
        if self._quick_settings_auto_emotion_checkbox is not None:
            self._set_checkbox_silently(self._quick_settings_auto_emotion_checkbox, self._auto_emotion_feature_enabled)
        self._sync_tts_other_option_controls()
        self._on_auto_emotion_toggled()
        self._refresh_workspace_summary()

    def _on_auto_emotion_strength_card_changed(self, value: int) -> None:
        self._auto_emotion_strength = value * 0.05
        if self._quick_settings_auto_emotion_strength_slider is not None:
            self._set_slider_silently(self._quick_settings_auto_emotion_strength_slider, value)
        if self._quick_settings_auto_emotion_strength_label is not None:
            self._set_text_if_changed(self._quick_settings_auto_emotion_strength_label, f"{self._auto_emotion_strength:.2f}")
        self._sync_tts_other_option_controls()
        self._refresh_workspace_summary()

    def _on_experimental_torch_compile_card_toggled(self, checked: bool) -> None:
        if checked == self._experimental_torch_compile_enabled:
            return
        try:
            set_experimental_torch_compile_enabled(checked)
        except Exception as exc:
            logger.exception("保存进阶模式设置失败")
            self._show_message(QMessageBox.Critical, "保存失败", f"无法保存进阶模式设置：{exc}")
            if hasattr(self, "experimental_torch_compile_card_checkbox"):
                self._set_checkbox_silently(
                    self.experimental_torch_compile_card_checkbox,
                    self._experimental_torch_compile_enabled,
                )
            return
        self._experimental_torch_compile_enabled = checked
        if self._quick_settings_torch_compile_checkbox is not None:
            self._set_checkbox_silently(self._quick_settings_torch_compile_checkbox, checked)
        self.settings = get_settings()
        self._refresh_workspace_summary()
        self._show_experimental_restart_notice(feature_names=["进阶模式"])

    def _on_experimental_accel_gpt_card_toggled(self, checked: bool) -> None:
        if checked == self._experimental_accel_gpt_enabled:
            return
        try:
            set_experimental_accel_gpt_enabled(checked)
        except Exception as exc:
            logger.exception("保存极速模式设置失败")
            self._show_message(QMessageBox.Critical, "保存失败", f"无法保存极速模式设置：{exc}")
            if hasattr(self, "experimental_accel_gpt_card_checkbox"):
                self._set_checkbox_silently(
                    self.experimental_accel_gpt_card_checkbox,
                    self._experimental_accel_gpt_enabled,
                )
            return
        self._experimental_accel_gpt_enabled = checked
        if self._quick_settings_accel_gpt_checkbox is not None:
            self._set_checkbox_silently(self._quick_settings_accel_gpt_checkbox, checked)
        self.settings = get_settings()
        self._refresh_workspace_summary()
        self._show_experimental_restart_notice(feature_names=["极速模式"])

    def _sync_quick_settings_auto_emotion_controls(self) -> None:
        is_text_mode = False
        if self._quick_settings_auto_emotion_checkbox is not None:
            if is_text_mode:
                self._quick_settings_auto_emotion_checkbox.setChecked(False)
            self._quick_settings_auto_emotion_checkbox.setEnabled(not is_text_mode)
            self._quick_settings_auto_emotion_checkbox.setToolTip(
                "情感识别仅作用于麦克风输入。" if is_text_mode else ""
            )
        if self._quick_settings_auto_emotion_strength_slider is not None:
            self._quick_settings_auto_emotion_strength_slider.setEnabled(not is_text_mode)
        if self._quick_settings_auto_emotion_strength_label is not None:
            self._quick_settings_auto_emotion_strength_label.setEnabled(not is_text_mode)

    def _on_input_mode_changed(self) -> None:
        requested_mode = self._current_input_mode()
        if requested_mode in {"program", "dual"}:
            self._refresh_running_programs()
        if self._is_live_worker_active():
            self._handle_live_input_mode_change(requested_mode)
        self._sync_capture_source_controls()
        self._sync_quick_settings_auto_emotion_controls()
        self._sync_tts_other_option_controls()
        self._sync_voice_output_model_controls()
        self._refresh_input_level_state()
        self._refresh_devices_info()
        current_mode = self._current_input_mode()
        if hasattr(self, "transcript_intro_label"):
            self.transcript_intro_label.setText(self._mode_intro_text(current_mode))
        if hasattr(self, "workspace_placeholder_label"):
            self.workspace_placeholder_label.setProperty("state", "ready")
            self.workspace_placeholder_label.setText(self._mode_placeholder_text(current_mode))
            self._refresh_widget_style(self.workspace_placeholder_label)
        if hasattr(self, "workspace_waveform_frame"):
            self.workspace_waveform_frame.setVisible(False)
        if hasattr(self, "workspace_text_input"):
            self.workspace_text_input.setFocus()
        self._refresh_live_start_button()
        if (
            self._live_worker_thread is not None
            and self._live_worker_state not in {LiveWorkerState.RUNNING.value, LiveWorkerState.STOPPING.value}
            and current_mode != "none"
        ):
            self._models_ready = False
            self._refresh_live_start_button()
            self._set_live_status("正在刷新监听引擎...", "busy")
            self.live_preload_requested.emit(self._build_live_preload_payload())
        self._on_auto_emotion_toggled()
        self._refresh_workspace_text_metrics()
        self._refresh_workspace_summary()

    def _on_program_selection_changed(self) -> None:
        self._refresh_workspace_summary()
        self._refresh_live_start_button()
        requested_mode = self._current_input_mode()
        if not self._is_live_worker_active():
            return
        if self._live_worker_state != LiveWorkerState.RUNNING.value:
            return
        if requested_mode not in {"program", "dual"}:
            return
        if self._active_request_mode not in {"program", "dual"}:
            return
        selected_pid = self.program_combo.currentData() if hasattr(self, "program_combo") else None
        if not isinstance(selected_pid, int):
            return
        if selected_pid == self._active_program_pid:
            return
        validation_message = self._validate_live_input_mode_change(requested_mode)
        if validation_message is not None:
            return
        self._pending_live_restart_mode = requested_mode
        self._stop_live_microphone_for_settings_refresh("程序监听目标已更新，正在同步处理流...")

    def _refresh_audio_devices(self) -> None:
        self._reset_audio_device_collections()
        try:
            snapshot, sd_devices, sd_hostapis = self._load_audio_device_snapshot()
        except Exception as exc:
            logger.error("读取音频设备失败: {}", exc)
            self._show_audio_device_refresh_error()
            return

        self._initialize_audio_device_combos()
        logger.info(
            "音频设备枚举完成: Windows 输入 {} 个, Windows 输出 {} 个, sounddevice 端点 {} 个",
            len(snapshot.inputs),
            len(snapshot.outputs),
            len(sd_devices),
        )
        deduped_inputs, deduped_sd_inputs, deduped_outputs = self._build_deduped_audio_device_maps(snapshot, sd_devices, sd_hostapis)
        virtual_input_count = self._populate_audio_input_devices(deduped_inputs, deduped_sd_inputs)
        virtual_output_count = self._populate_audio_output_devices(snapshot, deduped_outputs)
        self._select_default_audio_devices(snapshot)
        self._apply_preferred_virtual_output()
        self._update_virtual_device_hint(virtual_input_count, virtual_output_count)
        self._refresh_workspace_summary()

    def _reset_audio_device_collections(self) -> None:
        self.input_device_combo.clear()
        self.output_device_combo.clear()
        self.monitor_output_device_combo.clear()
        self._input_devices.clear()
        self._input_device_name_by_key.clear()
        self._input_device_sd_index_by_key.clear()
        self._output_devices.clear()
        self._monitor_devices.clear()
        self._output_device_name_by_key.clear()
        self._output_device_sd_index_by_key.clear()

    @staticmethod
    def _load_audio_device_snapshot():
        snapshot = list_windows_audio_endpoints()
        sd_devices = sd.query_devices()
        sd_hostapis = sd.query_hostapis()
        return snapshot, sd_devices, sd_hostapis

    def _show_audio_device_refresh_error(self) -> None:
        self.input_device_combo.addItem("读取失败")
        self.output_device_combo.addItem("读取失败")
        self.monitor_output_device_combo.addItem("读取失败")
        self._refresh_workspace_summary()

    def _initialize_audio_device_combos(self) -> None:
        self.input_device_combo.addItem("系统默认输入", None)
        self.output_device_combo.addItem("系统默认输出", None)
        self.monitor_output_device_combo.addItem("关闭耳返", None)

    def _build_deduped_audio_device_maps(self, snapshot, sd_devices, sd_hostapis):
        deduped_inputs: dict[str, tuple[str, str]] = {}
        deduped_sd_inputs: dict[str, tuple[int, str, str]] = {}
        deduped_outputs: dict[str, tuple[int, str, str]] = {}

        for endpoint in snapshot.inputs:
            name = endpoint.name.strip() or "输入设备"
            device_key = endpoint.endpoint_id
            dedupe_key = self._device_dedupe_key(name)
            if dedupe_key not in deduped_inputs:
                deduped_inputs[dedupe_key] = (device_key, name)

        for index, device in enumerate(sd_devices):
            hostapi_name = self._sounddevice_hostapi_name(device, sd_hostapis)
            if int(device.get("max_input_channels", 0)) > 0:
                input_name = str(device.get("name") or "").strip() or f"输入设备 {index}"
                input_dedupe_key = self._device_dedupe_key(input_name)
                candidate_input = (index, input_name, hostapi_name)
                current_input = deduped_sd_inputs.get(input_dedupe_key)
                if self._should_prefer_sounddevice_entry(current_input, candidate_input):
                    deduped_sd_inputs[input_dedupe_key] = candidate_input
            if int(device.get("max_output_channels", 0)) <= 0:
                continue
            name = str(device.get("name") or "").strip() or f"输出设备 {index}"
            dedupe_key = self._device_dedupe_key(name)
            candidate_output = (index, name, hostapi_name)
            current_output = deduped_outputs.get(dedupe_key)
            if self._should_prefer_sounddevice_entry(current_output, candidate_output):
                deduped_outputs[dedupe_key] = candidate_output
        return deduped_inputs, deduped_sd_inputs, deduped_outputs

    def _populate_audio_input_devices(
        self,
        deduped_inputs: dict[str, tuple[str, str]],
        deduped_sd_inputs: dict[str, tuple[int, str, str]],
    ) -> int:
        virtual_input_count = 0
        for device_key, name in deduped_inputs.values():
            label = self._format_device_label(device_key, name)
            if self._is_virtual_device_name(name):
                virtual_input_count += 1
            self._input_devices.append((device_key, label))
            self._input_device_name_by_key[device_key] = name
            matched_sd_index = self._match_sounddevice_input_index(name, deduped_sd_inputs)
            if matched_sd_index is not None:
                self._input_device_sd_index_by_key[device_key] = matched_sd_index
            self.input_device_combo.addItem(label, device_key)
        return virtual_input_count

    def _populate_audio_output_devices(
        self,
        snapshot,
        deduped_outputs: dict[str, tuple[int, str, str]],
    ) -> int:
        virtual_output_count = 0
        for endpoint in snapshot.outputs:
            name = endpoint.name.strip() or "输出设备"
            matched_sd_index = self._match_sounddevice_output_index(name, deduped_outputs)
            if matched_sd_index is None:
                continue
            label = self._format_output_device_label(endpoint.endpoint_id, name)
            if self._is_virtual_device_name(name):
                virtual_output_count += 1
            self._output_devices.append((endpoint.endpoint_id, label))
            self._monitor_devices.append((endpoint.endpoint_id, label))
            self._output_device_name_by_key[endpoint.endpoint_id] = name
            self._output_device_sd_index_by_key[endpoint.endpoint_id] = matched_sd_index
            self.output_device_combo.addItem(label, endpoint.endpoint_id)
            self.monitor_output_device_combo.addItem(label, endpoint.endpoint_id)
        return virtual_output_count

    def _select_default_audio_devices(self, snapshot) -> None:
        if snapshot.default_input_id:
            input_index = self.input_device_combo.findData(snapshot.default_input_id)
            if input_index >= 0:
                self.input_device_combo.setCurrentIndex(input_index)
        if snapshot.default_output_id:
            output_index = self.output_device_combo.findData(snapshot.default_output_id)
            if output_index >= 0:
                self.output_device_combo.setCurrentIndex(output_index)
            monitor_index = self.monitor_output_device_combo.findData(snapshot.default_output_id)
            if monitor_index >= 0:
                self.monitor_output_device_combo.setCurrentIndex(monitor_index)

    @staticmethod
    def _is_virtual_device_name(name: str) -> bool:
        normalized = name.casefold()
        return any(keyword in normalized for keyword in VIRTUAL_DEVICE_KEYWORDS)

    @staticmethod
    def _device_dedupe_key(name: str) -> str:
        normalized = " ".join(name.casefold().split())
        suffixes = (
            ", windows wasapi",
            ", windows directsound",
            ", mme",
            ", wdm-ks",
        )
        for suffix in suffixes:
            if normalized.endswith(suffix):
                return normalized[: -len(suffix)].rstrip(" ,")
        return normalized

    @staticmethod
    def _sounddevice_hostapi_name(device: dict[str, object], hostapis: object) -> str:
        try:
            hostapi_index = int(device.get("hostapi", -1))
        except (TypeError, ValueError):
            return ""
        if not isinstance(hostapis, (list, tuple)) or hostapi_index < 0 or hostapi_index >= len(hostapis):
            return ""
        hostapi = hostapis[hostapi_index]
        if not isinstance(hostapi, dict):
            return ""
        return str(hostapi.get("name") or "").strip()

    @staticmethod
    def _sounddevice_hostapi_score(hostapi_name: str) -> int:
        normalized = hostapi_name.casefold()
        if "wasapi" in normalized:
            return 40
        if "wdm-ks" in normalized:
            return 30
        if "directsound" in normalized:
            return 20
        if "mme" in normalized:
            return 10
        return 0

    @classmethod
    def _should_prefer_sounddevice_entry(
        cls,
        current: tuple[int, str, str] | None,
        candidate: tuple[int, str, str],
    ) -> bool:
        if current is None:
            return True
        current_score = cls._sounddevice_hostapi_score(current[2])
        candidate_score = cls._sounddevice_hostapi_score(candidate[2])
        if candidate_score != current_score:
            return candidate_score > current_score
        return candidate[0] < current[0]

    @classmethod
    def _match_sounddevice_input_index(
        cls,
        device_name: str,
        sounddevice_inputs: dict[str, tuple[int, str, str]],
    ) -> int | None:
        target_key = cls._device_dedupe_key(device_name)
        exact = sounddevice_inputs.get(target_key)
        if exact is not None:
            return exact[0]
        for dedupe_key, (index, _name, _hostapi_name) in sounddevice_inputs.items():
            if target_key and (target_key in dedupe_key or dedupe_key in target_key):
                return index
        return None

    @classmethod
    def _match_sounddevice_output_index(
        cls,
        device_name: str,
        sounddevice_outputs: dict[str, tuple[int, str, str]],
    ) -> int | None:
        target_key = cls._device_dedupe_key(device_name)
        exact = sounddevice_outputs.get(target_key)
        if exact is not None:
            return exact[0]
        for dedupe_key, (index, _name, _hostapi_name) in sounddevice_outputs.items():
            if target_key and (target_key in dedupe_key or dedupe_key in target_key):
                return index
        return None

    def _format_device_label(self, device_key: str, name: str) -> str:
        prefix = "[虚拟]" if self._is_virtual_device_name(name) else "[设备]"
        return f"{prefix} {name}"

    def _format_output_device_label(self, device_key: str, name: str) -> str:
        prefix = "[虚拟]" if self._is_virtual_device_name(name) else "[设备]"
        return f"{prefix} {name}"

    @staticmethod
    def _extract_device_display_name(label: str) -> str:
        if "] " in label:
            return label.split("] ", 1)[1].strip()
        return label.strip()

    def _update_virtual_device_hint(self, virtual_input_count: int, virtual_output_count: int) -> None:
        if virtual_input_count == 0 and virtual_output_count == 0:
            self._virtual_device_summary = "未检测到虚拟音频设备。"
            self._refresh_devices_info()
            return
        preferred_output = self._find_preferred_virtual_output_index()
        recommended = ""
        if preferred_output is not None:
            recommended = f" 推荐：{self.output_device_combo.itemText(preferred_output)}。"
        self._virtual_device_summary = (
            f"虚拟设备 输入 {virtual_input_count} / 输出 {virtual_output_count}。"
            f"{recommended}"
        )
        self._refresh_devices_info()

    def _refresh_devices_info(self) -> None:
        if not hasattr(self, "devices_info_label"):
            return
        mode_hint = "麦克风实时处理常驻可用，文字输入走中间聊天输入框。"
        virtual_hint = self._virtual_device_summary or "未检测到虚拟音频设备。"
        self._set_text_if_changed(self.devices_info_label, f"{virtual_hint} {mode_hint}")

    def _select_virtual_output_device(self) -> None:
        preferred_index = self._find_preferred_virtual_output_index()
        if preferred_index is not None:
            label = self.output_device_combo.itemText(preferred_index)
            self.output_device_combo.setCurrentIndex(preferred_index)
            logger.info("已自动切换到虚拟播放设备: {}", label)
            self._set_live_status("已切换到虚拟播放设备", "ready")
            return
        self._show_message(QMessageBox.Information, "未找到虚拟设备", "当前没有检测到可用的虚拟播放设备。")

    def _find_preferred_virtual_output_index(self) -> int | None:
        best_index: int | None = None
        best_score = -1
        for index in range(self.output_device_combo.count()):
            label = self.output_device_combo.itemText(index)
            if not label.startswith("[虚拟]"):
                continue
            score = self._score_virtual_output_label(label)
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _apply_preferred_virtual_output(self) -> None:
        preferred_index = self._find_preferred_virtual_output_index()
        if preferred_index is None:
            return
        self.output_device_combo.setCurrentIndex(preferred_index)

    @staticmethod
    def _score_virtual_output_label(label: str) -> int:
        normalized = label.casefold()
        for rank, keyword in enumerate(PREFERRED_VIRTUAL_OUTPUT_KEYWORDS):
            if keyword in normalized:
                return 100 - rank
        return 1

    def _build_live_preload_payload(self, reference_audio_path: str | None = None) -> dict[str, str]:
        reference_path = (
            self.live_reference_audio_edit.text().strip()
            if reference_audio_path is None
            else str(reference_audio_path).strip()
        )
        return {
            "reference_audio_path": reference_path,
            "input_mode": self._current_input_mode(),
            "voice_output_model": self._current_voice_output_model(),
        }

    def _on_workspace_text_submit(self) -> None:
        if self._is_meanvc2_selected():
            self._show_message(QMessageBox.Information, "暂不支持", "MeanVC2 当前是语音转换模型，暂不支持文字直出。请先切回 Index-TTS。")
            return
        input_text = self.workspace_text_input.text().strip() if hasattr(self, "workspace_text_input") else ""
        if not input_text:
            self._show_message(QMessageBox.Warning, "缺少输入文本", "请先在下方输入要发送的文字。")
            return
        if self._live_worker_state == LiveWorkerState.STOPPING.value:
            self._show_message(QMessageBox.Information, "正在停止", "麦克风正在停止，请稍后再发送文字。")
            return
        if self._live_worker_state == LiveWorkerState.RUNNING.value:
            if self._active_request_mode not in {"microphone", "program", "dual"}:
                self._show_message(QMessageBox.Information, "文字处理中", "上一条文字仍在处理中，请稍候。")
                return
        reference_audio = self.live_reference_audio_edit.text().strip()
        if not reference_audio:
            self._show_message(QMessageBox.Warning, "缺少参考音频", "请选择实时页面使用的参考音频。")
            return
        if not self._models_ready or self._live_worker_thread is None:
            self._show_message(QMessageBox.Information, "模型加载中", "模型尚未加载完成，请稍后再试。")
            return

        pending_id = self._next_chat_message_id()
        self._active_text_message_id = pending_id
        self._upsert_chat_message(
            {
                "message_id": pending_id,
                "input_mode": "text",
                "source_text": input_text,
                "translated_text": None,
                "source_language": detect_text_language(input_text),
                "target_language": self._translation_target_language if self._translation_enabled else None,
                "state": "pending",
            }
        )
        self.workspace_text_input.clear()
        if self._live_worker_state == LiveWorkerState.RUNNING.value and self._active_request_mode in {"microphone", "program", "dual"}:
            payload = {
                "input_mode": "text",
                "input_text": input_text,
                "ui_message_id": pending_id,
                "reference_audio_path": self.live_reference_audio_edit.text().strip(),
                "voice_output_model": self._current_voice_output_model(),
                "translation_enabled": self._translation_enabled,
                "translation_target_language": self._translation_target_language,
                "translation_node_id": self._translation_node_id,
                "instruction": self._build_tts_instruction(),
            }
            self.live_text_requested.emit(payload)
        else:
            self._start_live_speech(input_mode="text", input_text=input_text, ui_message_id=pending_id)

    @Slot()
    def _on_live_start_clicked(self) -> None:
        if self._live_worker_state == LiveWorkerState.RUNNING.value and self._active_request_mode in {"microphone", "program", "dual"}:
            self.live_start_button.setEnabled(False)
            self.live_start_button.setText("正在停止...")
            self.live_stop_requested.emit()
            return
        if self._live_worker_state == LiveWorkerState.STOPPING.value:
            return

        self._start_live_speech(input_mode=self._current_input_mode())

    def _start_live_speech(
        self,
        *,
        input_mode: str = "microphone",
        input_text: str = "",
        ui_message_id: str | None = None,
    ) -> None:
        if input_mode in {"program", "dual"}:
            self._refresh_running_programs()
        reference_audio = self.live_reference_audio_edit.text().strip()
        if input_mode == "none":
            # #region debug-point A:ui-no-input-mode
            _debug_report_program_listen_ui(
                "A",
                "UI blocked live start because no input mode was selected.",
                {"input_mode": input_mode},
            )
            # #endregion
            self._show_message(QMessageBox.Warning, "缺少监听来源", "请至少勾选麦克风或程序中的一个监听来源。")
            return
        if input_mode != "program" and not reference_audio:
            self._show_message(QMessageBox.Warning, "缺少参考音频", "请选择实时页面使用的参考音频。")
            return
        if input_mode in {"program", "dual"} and not isinstance(self.program_combo.currentData(), int):
            # #region debug-point A:ui-missing-program
            _debug_report_program_listen_ui(
                "A",
                "UI blocked live start because selected program pid was invalid.",
                {
                    "input_mode": input_mode,
                    "program_current_data": repr(self.program_combo.currentData()),
                    "program_current_text": self.program_combo.currentText().strip(),
                },
            )
            # #endregion
            self._show_message(QMessageBox.Warning, "缺少目标程序", "请先选择一个正在发声的程序。")
            return
        if input_mode == "text" and not input_text:
            self._show_message(QMessageBox.Warning, "缺少输入文本", "请先输入要发送的文字内容。")
            return
        if not self._models_ready or self._live_worker_thread is None:
            self._show_message(QMessageBox.Information, "模型加载中", "模型尚未加载完成，请稍后再试。")
            return

        self._active_request_mode = input_mode
        if input_mode in {"microphone", "program", "dual"}:
            self._set_text_if_changed(
                self.live_start_button,
                self._mode_stop_button_text(input_mode),
            )
            self.live_start_button.setIcon(self._live_button_icon("stop", "primary"))
            self.live_start_button.setEnabled(True)
        else:
            self.live_start_button.setEnabled(False)
            
        self._set_live_status("正在开始处理...", "busy")
        self.waveform_widget.clear_waveform()
        if self._is_auto_emotion_runtime_enabled() and input_mode != "text":
            self._update_auto_emotion_ui("", "busy")
        else:
            self._update_auto_emotion_ui("", "idle")
        emo_vector, emo_alpha = self._build_index_emo_vector()
        payload = {
            "input_mode": input_mode,
            "input_text": input_text,
            "reference_audio_path": reference_audio,
            "language": "auto",
            "output_language": self._output_voice_language,
            "translation_enabled": self._translation_enabled,
            "translation_target_language": self._translation_target_language,
            "translation_node_id": self._translation_node_id,
            "instruction": self._build_tts_instruction(),
            "max_new_tokens": DEFAULT_TTS_MAX_NEW_TOKENS,
            "input_device": (
                self._input_device_sd_index_by_key.get(self.input_device_combo.currentData())
                if isinstance(self.input_device_combo.currentData(), str)
                else None
            ),
            "input_device_label": self._extract_device_display_name(self.input_device_combo.currentText()),
            "program_pid": (
                self.program_combo.currentData()
                if input_mode in {"program", "dual"} and isinstance(self.program_combo.currentData(), int)
                else None
            ),
            "program_label": (
                self.program_combo.currentText().strip()
                if input_mode in {"program", "dual"}
                else None
            ),
            "output_device": (
                self.output_device_combo.currentData()
                if isinstance(self.output_device_combo.currentData(), str)
                else None
            ),
            "output_device_label": self._extract_device_display_name(self.output_device_combo.currentText()),
            "monitor_output_device": (
                self.monitor_output_device_combo.currentData()
                if isinstance(self.monitor_output_device_combo.currentData(), str)
                else None
            ),
            "monitor_output_device_label": self._extract_device_display_name(
                self.monitor_output_device_combo.currentText()
            ),
            "output_volume": self._current_output_volume(),
            "monitor_output_volume": self._current_monitor_output_volume(),
            "index_emo_vector": emo_vector,
            "index_emo_alpha": emo_alpha,
            "auto_emotion_enabled": self._is_auto_emotion_runtime_enabled() and input_mode != "text",
            "auto_emotion_strength": self._auto_emotion_strength,
            "speech_rate": self._current_speech_rate(),
            "ui_message_id": ui_message_id,
        }
        # #region debug-point A:ui-live-start-payload
        _debug_report_program_listen_ui(
            "A",
            "UI emitted live start payload.",
            {
                "input_mode": input_mode,
                "program_pid": payload.get("program_pid"),
                "program_label": payload.get("program_label"),
                "models_ready": self._models_ready,
                "worker_thread_ready": self._live_worker_thread is not None,
            },
        )
        # #endregion
        self._active_program_pid = payload.get("program_pid") if isinstance(payload.get("program_pid"), int) else None
        self.live_run_requested.emit(
            payload
        )

    def _start_live_backend(self) -> None:
        if self._live_worker_thread is not None:
            return
        self._models_ready = False
        self._live_worker_state = LiveWorkerState.INITIALIZING.value
        self._refresh_live_start_button()
        self._set_live_status("正在初始化引擎...", "busy")
        self._live_worker_thread = QThread(self)
        self._live_worker = LiveSpeechWorker(
            initial_reference_audio_path=self.live_reference_audio_edit.text().strip(),
        )
        self._live_worker.moveToThread(self._live_worker_thread)
        self.live_preload_requested.connect(self._live_worker.preload_models)
        self.live_run_requested.connect(self._live_worker.run_once)
        self.live_text_requested.connect(self._live_worker.enqueue_text_request, Qt.DirectConnection)
        self.live_reference_audio_changed_requested.connect(self._live_worker.update_reference_audio_path, Qt.DirectConnection)
        self.live_voice_output_model_changed_requested.connect(
            self._live_worker.update_voice_output_model,
            Qt.DirectConnection,
        )
        self.live_translation_settings_changed_requested.connect(
            self._live_worker.update_translation_settings,
            Qt.DirectConnection,
        )
        self.live_speech_rate_changed_requested.connect(self._live_worker.update_speech_rate, Qt.DirectConnection)
        self.live_output_volume_changed_requested.connect(self._live_worker.update_output_volume, Qt.DirectConnection)
        self.live_monitor_output_volume_changed_requested.connect(self._live_worker.update_monitor_output_volume, Qt.DirectConnection)
        self.live_replay_requested.connect(self._live_worker.replay_cached_audio, Qt.DirectConnection)
        self.live_stop_requested.connect(self._live_worker.stop_live, Qt.DirectConnection)
        self.live_shutdown_requested.connect(self._live_worker.shutdown)
        self.live_asr_unload_requested.connect(self._live_worker.unload_asr)
        self.live_emotion_unload_requested.connect(self._live_worker.unload_emotion)
        self.live_asr_reload_requested.connect(self._live_worker.reload_asr)
        self.live_tts_switch_requested.connect(self._live_worker.switch_tts_model)
        self._live_worker.ready.connect(self._on_live_ready)
        self._live_worker.status.connect(self._on_worker_status)
        self._live_worker.runtime_state.connect(self._on_live_runtime_state)
        self._live_worker.emotion_state.connect(self._on_worker_emotion_state)
        self._live_worker.chat_message.connect(self._on_live_chat_message)
        self._live_worker.transcript.connect(self._on_live_transcript)
        self._live_worker.waveform.connect(self._on_waveform_chunk)
        self._live_worker.perf_metrics.connect(self._on_perf_metrics)
        self._live_worker.finished.connect(self._on_live_finished)
        self._live_worker.error.connect(self._on_worker_error)
        self._live_worker_thread.start()
        self._emit_live_translation_settings()
        self.live_speech_rate_changed_requested.emit(self._current_speech_rate())
        self.live_output_volume_changed_requested.emit(self._current_output_volume())
        self.live_monitor_output_volume_changed_requested.emit(self._current_monitor_output_volume())
        self.live_preload_requested.emit(self._build_live_preload_payload())

    def _shutdown_live_backend(self) -> None:
        if self._live_worker_thread is None:
            return
        self._pending_live_restart_mode = None
        worker = self._live_worker
        if worker is not None:
            worker.stop_live()
        self.live_shutdown_requested.emit()
        self._live_worker_thread.requestInterruption()
        self._live_worker_thread.quit()
        if not self._live_worker_thread.wait(3000):
            logger.warning("实时处理线程关闭超时，强制终止后台任务")
            self._live_worker_thread.terminate()
            self._live_worker_thread.wait(1000)
        if worker is not None:
            try:
                worker.shutdown()
            except Exception:
                logger.exception("实时处理资源二次释放失败")
            worker.deleteLater()
        self._live_worker_thread.deleteLater()
        self._live_worker = None
        self._live_worker_thread = None
        self._live_worker_state = LiveWorkerState.INITIALIZING.value
        self._models_ready = False
        self._refresh_live_start_button()

    @Slot()
    def _on_live_ready(self) -> None:
        self._models_ready = True
        if self._pending_reference_preload and self.live_reference_audio_edit.text().strip():
            self._pending_reference_preload = False
            self._models_ready = False
            self._refresh_live_start_button()
            self._set_live_status(
                "正在加载参考音色（MeanVC2 首次加载可能需要 5-10 秒）..." if self._is_meanvc2_selected() else "正在加载参考音色并预热...",
                "busy",
            )
            self.live_preload_requested.emit(self._build_live_preload_payload())
            return
        self._refresh_live_start_button()
        self._refresh_workspace_text_metrics()
        reference_audio_ready = bool(self.live_reference_audio_edit.text().strip())
        if self._is_meanvc2_selected() and reference_audio_ready:
            self._set_live_status("MeanVC2 已加载完成，可开始变声", "ready")
        elif reference_audio_ready:
            self._set_live_status("参考音色已加载并预热完成，可开始合成", "ready")
        else:
            self._set_live_status("引擎已就绪，等待开始", "ready")
        logger.info("UI 模型预加载完成")

    @Slot(str)
    def _on_live_runtime_state(self, state: str) -> None:
        self._live_worker_state = state or LiveWorkerState.INITIALIZING.value
        self._refresh_live_start_button()
        self._refresh_input_level_state()
        self._sync_header_runtime_timer()
        self._refresh_workspace_text_metrics()
        self._refresh_workspace_summary()

    @Slot(object)
    def _on_perf_metrics(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        
        asr_lat = payload.get("asr_latency", 0.0)
        trans_lat = payload.get("trans_latency", 0.0)
        tts_lat = payload.get("tts_latency", 0.0)
        total_lat = payload.get("total_latency", 0.0)
        
        def format_ms(val):
            return f"{val:.1f}ms" if val and val > 0 else "--ms"
            
        if hasattr(self, "asr_delay_label"):
            self._set_text_if_changed(self.asr_delay_label, f"ASR 延迟: {format_ms(asr_lat)}")
        if hasattr(self, "trans_delay_label"):
            self._set_text_if_changed(self.trans_delay_label, f"翻译延迟: {format_ms(trans_lat)}")
        if hasattr(self, "tts_delay_label"):
            self._set_text_if_changed(self.tts_delay_label, f"TTS 延迟: {format_ms(tts_lat)}")
        if hasattr(self, "total_delay_label"):
            self._set_text_if_changed(self.total_delay_label, f"总延迟: {format_ms(total_lat)}")

    @Slot(str)
    def _on_worker_status(self, message: str) -> None:
        self._set_live_status(message, self._infer_status_state(message))
        logger.info("{}", message)

    @Slot(object)
    def _on_worker_emotion_state(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        summary = str(payload.get("summary") or "当前：仍使用手动情感。")
        state = str(payload.get("state") or "idle")
        raw_vector = payload.get("applied_vector")
        applied_alpha = payload.get("applied_alpha")
        if isinstance(raw_vector, (list, tuple)) and self._is_auto_emotion_runtime_enabled():
            self._apply_auto_emotion_vector(list(raw_vector), float(applied_alpha) if applied_alpha is not None else None)
        self._update_auto_emotion_ui(summary, state)

    @Slot(str)
    def _on_live_finished(self, transcript: str) -> None:
        finished_mode = self._active_request_mode
        restart_mode = self._pending_live_restart_mode
        self._pending_live_restart_mode = None
        if self._is_closing:
            restart_mode = None
        if finished_mode == "text":
            self.live_asr_unload_requested.emit()
            self.live_emotion_unload_requested.emit()
            self._active_text_message_id = None
        if restart_mode in {"microphone", "program", "dual"}:
            self._set_live_status("正在按新语音模型重启处理流...", "busy")
        else:
            self._set_live_status("变声完成", "ready")
        if finished_mode == "text":
            self._update_auto_emotion_ui("", "idle")
            if hasattr(self, "workspace_text_input"):
                self.workspace_text_input.setFocus()
        self._active_request_mode = None
        self._refresh_live_start_button()
        if hasattr(self, "workspace_wave_hint"):
            self._set_text_if_changed(self.workspace_wave_hint, "波形已刷新")
        if hasattr(self, "workspace_placeholder_label") and finished_mode in {"microphone", "program", "dual"}:
            self.workspace_placeholder_label.setProperty("state", "ready")
            self._set_text_if_changed(
                self.workspace_placeholder_label,
                "双路监听已停止，可继续选择程序并使用麦克风；也可以在下方发送文字。"
                if finished_mode == "dual"
                else ("程序监听已停止，可继续选择其他程序；也可以在下方发送文字。"
                if finished_mode == "program"
                else "麦克风已停止，可继续说话；也可以在下方发送文字。"),
            )
            self._refresh_widget_style(self.workspace_placeholder_label)
        logger.info("实时链路处理完成")
        if restart_mode in {"microphone", "program", "dual"} and self._live_worker_thread is not None:
            self.live_tts_switch_requested.emit()
            QTimer.singleShot(0, lambda mode=restart_mode: self._start_live_speech(input_mode=mode))

    @Slot(object)
    def _on_live_chat_message(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        self._upsert_chat_message(payload)
        if str(payload.get("input_mode") or "") == "text" and str(payload.get("state") or "ready") != "pending":
            self._active_text_message_id = None

    @Slot(str)
    def _on_live_transcript(self, transcript: str) -> None:
        if self._active_request_mode not in {"microphone", "program", "dual"}:
            return
        if not hasattr(self, "workspace_placeholder_label"):
            return
        if transcript.strip():
            self.workspace_placeholder_label.setProperty("state", "ready")
            self._set_text_if_changed(self.workspace_placeholder_label, f"最新识别：{transcript}")
            self._refresh_widget_style(self.workspace_placeholder_label)

    @Slot(object)
    def _on_waveform_chunk(self, audio) -> None:
        self.waveform_widget.append_audio_chunk(audio)
        if hasattr(self, "workspace_wave_hint"):
            self._set_text_if_changed(self.workspace_wave_hint, "接收音频中")

    def _refresh_workspace_text_metrics(self) -> None:
        if not hasattr(self, "workspace_counter_label"):
            return
        self._set_text_if_changed(self.workspace_counter_label, f"{len(self._chat_messages)} 条记录")
        if hasattr(self, "workspace_send_button") and hasattr(self, "workspace_text_input"):
            has_text = bool(self.workspace_text_input.text().strip())
            self.workspace_send_button.setEnabled(self._models_ready and has_text)

    @Slot(str)
    def _on_worker_error(self, message: str) -> None:
        # #region debug-point D:ui-worker-error
        _debug_report_program_listen_ui(
            "D",
            "UI received worker error.",
            {
                "active_request_mode": self._active_request_mode,
                "message": message,
            },
        )
        # #endregion
        if self._active_text_message_id:
            self._upsert_chat_message(
                {
                    "message_id": self._active_text_message_id,
                    "state": "error",
                    "translated_text": f"发送失败：{message}",
                }
            )
            self._active_text_message_id = None
        self._active_request_mode = None
        self._set_live_status("处理失败", "error")
        self._update_auto_emotion_ui("", "error")
        self._refresh_live_start_button()
        self._refresh_workspace_text_metrics()
        if hasattr(self, "workspace_placeholder_label"):
            self.workspace_placeholder_label.setProperty("state", "error")
            self._set_text_if_changed(self.workspace_placeholder_label, f"处理失败：{message}")
            self._refresh_widget_style(self.workspace_placeholder_label)
        logger.error("界面任务失败: {}", message)
        if self._is_meanvc2_setup_error(message):
            self._set_live_status("MeanVC2 尚未准备完成，请先下载模型", "error")
            self._show_meanvc2_setup_guide(message)
            return
        if self._is_asr_setup_error(message):
            self._set_live_status("ASR 尚未准备完成，请先下载模型", "error")
            self._show_asr_setup_guide(message)
            return
        self._show_message(QMessageBox.Critical, "错误", message)

    def _set_live_status(self, text: str, state: str) -> None:
        self._set_live_status_label_state(text, state)
        self._refresh_workspace_summary(status_override=text)

    def _clear_stale_text_runtime_state(self) -> bool:
        if self._active_request_mode != "text":
            return False
        if self._active_text_message_id:
            return False
        self._active_request_mode = None
        if self._live_worker_state == LiveWorkerState.RUNNING.value:
            self._live_worker_state = LiveWorkerState.READY.value
        self._refresh_input_level_state()
        self._sync_header_runtime_timer()
        return True

    def _is_live_worker_active(self) -> bool:
        return self._live_worker_state in {
            LiveWorkerState.RUNNING.value,
            LiveWorkerState.STOPPING.value,
        }

    def _stop_live_microphone_for_settings_refresh(self, status_text: str) -> None:
        if self._active_request_mode not in {"microphone", "program", "dual"}:
            return
        if self._live_worker_state == LiveWorkerState.RUNNING.value:
            self.live_start_button.setEnabled(False)
            self._set_text_if_changed(self.live_start_button, "正在停止...")
            self.live_start_button.setIcon(self._live_button_icon("reload", "warning"))
            self._set_live_status(status_text, "busy")
            self.live_stop_requested.emit()

    def _refresh_live_start_button(self) -> None:
        # 刷新开始按钮
        self._clear_stale_text_runtime_state()
        current_mode = self._current_input_mode()
        index_tts_ready, index_tts_message = self._current_index_tts_output_status()
        if self._live_worker_state == LiveWorkerState.STOPPING.value:
            self._set_text_if_changed(self.live_start_button, "正在停止...")
            self.live_start_button.setIcon(self._live_button_icon("reload", "warning"))
            self.live_start_button.setEnabled(False)
            self.live_start_button.setToolTip("")
            return
        if self._live_worker_state == LiveWorkerState.RUNNING.value:
            if self._active_request_mode in {"microphone", "program", "dual"}:
                self._set_text_if_changed(
                    self.live_start_button,
                    self._mode_stop_button_text(self._active_request_mode),
                )
                self.live_start_button.setIcon(self._live_button_icon("stop", "primary"))
                self.live_start_button.setEnabled(True)
            else:
                self._set_text_if_changed(self.live_start_button, "文字处理中")
                self.live_start_button.setIcon(self._live_button_icon("reload", "warning"))
                self.live_start_button.setEnabled(False)
            self.live_start_button.setToolTip("")
            return
        if current_mode != "program" and not index_tts_ready:
            self._set_text_if_changed(self.live_start_button, "缺少模型")
            self.live_start_button.setIcon(self._live_button_icon("preparing", "warning"))
            self.live_start_button.setEnabled(False)
            self.live_start_button.setToolTip(index_tts_message)
            return
        if not self._models_ready:
            self._set_text_if_changed(self.live_start_button, "准备中")
            self.live_start_button.setIcon(self._live_button_icon("preparing", "warning"))
            self.live_start_button.setEnabled(False)
            self.live_start_button.setToolTip("")
            return
        self._set_text_if_changed(
            self.live_start_button,
            self._mode_start_button_text(current_mode),
        )
        self.live_start_button.setIcon(self._live_button_icon("play", "primary"))
        self.live_start_button.setToolTip("")
        if current_mode == "none":
            self.live_start_button.setEnabled(False)
            return
        if current_mode in {"program", "dual"}:
            self.live_start_button.setEnabled(isinstance(self.program_combo.currentData(), int))
        else:
            self.live_start_button.setEnabled(True)

    def _build_workspace_summary_context(self, *, status_override: str | None = None) -> dict[str, object]:
        input_mode = self._format_header_mode_label(self._current_input_mode())
        reference_path = self.live_reference_audio_edit.text().strip()
        reference_name = Path(reference_path).name if reference_path else "未选择"
        output_label = self.output_device_combo.currentText().strip() or "系统默认输出"
        runtime_text = status_override or str(self.live_status_label.property("raw_text") or "").strip() or "待命中"
        output_language = self._output_voice_language
        voice_output_model = self._current_voice_output_model()
        index_tts_ready, index_tts_message = self._current_index_tts_output_status()
        runtime_header_text = self._format_header_runtime_label(runtime_text)
        runtime_state = (
            "busy"
            if self._live_worker_state in {LiveWorkerState.RUNNING.value, LiveWorkerState.STOPPING.value} or not self._models_ready
            else self._infer_status_state(runtime_text)
        )
        if voice_output_model == "indextts" and not index_tts_ready and self._live_worker_state != LiveWorkerState.RUNNING.value:
            runtime_header_text = "缺少模型"
            runtime_state = "error"
        tts_model_label = self._format_header_voice_output_model_label(voice_output_model, output_language)
        tts_model_state = (
            "error"
            if voice_output_model == "indextts" and not index_tts_ready
            else self._infer_voice_output_model_state(voice_output_model, output_language)
        )
        supported_count = len(self.settings.supported_index_tts_model_repos)
        monitor_label = self.monitor_output_device_combo.currentText().strip() or "关闭耳返"
        output_volume = round(self._current_output_volume(), 2)
        monitor_output_volume = round(self._current_monitor_output_volume(), 2)
        return {
            "input_mode": input_mode,
            "reference_path": reference_path,
            "reference_name": reference_name,
            "output_label": output_label,
            "runtime_text": runtime_text,
            "runtime_header_text": runtime_header_text,
            "runtime_state": runtime_state,
            "voice_output_model": voice_output_model,
            "output_language": output_language,
            "tts_model_label": tts_model_label,
            "tts_model_state": tts_model_state,
            "tts_model_tooltip": index_tts_message,
            "tts_output_ready": index_tts_ready,
            "supported_count": supported_count,
            "monitor_label": monitor_label,
            "output_volume": output_volume,
            "monitor_output_volume": monitor_output_volume,
        }

    @staticmethod
    def _workspace_summary_signature_from_context(context: dict[str, object]) -> tuple[object, ...]:
        summary_signature = (
            context["input_mode"],
            context["reference_name"],
            context["output_label"],
            context["runtime_text"],
            context["runtime_state"],
            context["tts_model_label"],
            context["tts_model_state"],
            context["tts_output_ready"],
            context["supported_count"],
            context["monitor_label"],
            context["output_volume"],
            context["monitor_output_volume"],
        )
        return summary_signature

    def _refresh_workspace_reference_meta(self, context: dict[str, object]) -> None:
        if hasattr(self, "reference_meta_label"):
            availability_suffix = ""
            if not bool(context["tts_output_ready"]) and context["voice_output_model"] == "indextts":
                availability_suffix = " | 当前语种模型：缺失"
            self._set_text_if_changed(
                self.reference_meta_label,
                f"语音输出模型：{context['tts_model_label']} | 当前参考：{context['reference_name']} | "
                f"播放：{context['output_label']} | 耳返：{context['monitor_label']} | "
                f"输出音量：{context['output_volume']:.2f} | 耳返输出音量：{context['monitor_output_volume']:.2f} | "
                f"可选 Index-TTS 模型：{context['supported_count']} 个"
                f"{availability_suffix}"
            )
            self.reference_meta_label.setToolTip(str(context["tts_model_tooltip"]))

    def _refresh_workspace_badges(self, context: dict[str, object]) -> None:
        return

    def _refresh_workspace_placeholder(self, context: dict[str, object]) -> None:
        if hasattr(self, "workspace_placeholder_label") and self.workspace_placeholder_label.isVisible():
            placeholder_state = "ready"
            placeholder_text = self._mode_placeholder_text(self._current_input_mode())
            if not bool(context["tts_output_ready"]) and context["voice_output_model"] == "indextts":
                placeholder_state = "error"
                placeholder_text = str(context["tts_model_tooltip"]).splitlines()[0]
            elif context["runtime_state"] == "busy":
                placeholder_text = str(context["runtime_text"])
            elif context["runtime_state"] == "error":
                placeholder_state = "error"
                placeholder_text = str(context["runtime_text"])
            self.workspace_placeholder_label.setProperty("state", placeholder_state)
            self._set_text_if_changed(self.workspace_placeholder_label, placeholder_text)
            self._refresh_widget_style(self.workspace_placeholder_label)

    def _refresh_workspace_summary(self, *_args, status_override: str | None = None) -> None:
        context = self._build_workspace_summary_context(status_override=status_override)
        summary_signature = self._workspace_summary_signature_from_context(context)
        if summary_signature == self._workspace_summary_signature:
            return
        self._workspace_summary_signature = summary_signature

        self._set_text_if_changed(self.header_mode_value, str(context["input_mode"]))
        self._set_text_if_changed(self.header_runtime_value, str(context["runtime_header_text"]))
        self._set_widget_state(self.header_runtime_value, str(context["runtime_state"]))
        self._set_text_if_changed(self.header_tts_model_value, str(context["tts_model_label"]))
        self._set_widget_state(self.header_tts_model_value, str(context["tts_model_state"]))
        self.header_tts_model_value.setToolTip(str(context["tts_model_tooltip"]))
        self._set_text_if_changed(self.header_reference_value, str(context["reference_name"]))
        self._sync_header_runtime_timer()
        self._refresh_workspace_reference_meta(context)
        self._refresh_workspace_badges(context)
        self._refresh_workspace_placeholder(context)

    @staticmethod
    def _infer_status_state(message: str) -> str:
        if any(keyword in message for keyword in ("失败", "错误")):
            return "error"
        if any(keyword in message for keyword in ("完成", "已加载", "已就绪", "待命", "已切换", "已更新")):
            return "ready"
        return "busy"

    def _format_header_mode_label(self, input_mode: str) -> str:
        match input_mode:
            case "none":
                return "未选择监听来源 + 文字（含翻译）" if self._translation_enabled else "未选择监听来源 + 文字"
            case "dual":
                return "麦克风 + 程序 + 文字（含翻译）" if self._translation_enabled else "麦克风 + 程序 + 文字"
            case "program":
                return "程序监听 + 文字（含翻译）" if self._translation_enabled else "程序监听 + 文字"
            case _:
                return "麦克风 + 文字（含翻译）" if self._translation_enabled else "麦克风 + 文字"

    def _format_header_voice_output_model_label(self, model_name: str, language: str) -> str:
        if model_name == "meanvc2":
            return "MeanVC2"
        return "Index-TTS"

    def _infer_voice_output_model_state(self, model_name: str, language: str) -> str:
        if model_name == "meanvc2":
            return "custom"
        return "official"

    def _format_header_runtime_label(self, runtime_text: str) -> str:
        if self._live_worker_state == LiveWorkerState.RUNNING.value:
            match self._current_input_mode():
                case "none":
                    return "待选择"
                case "dual":
                    return "双路监听中"
                case "program":
                    return "程序监听中"
                case _:
                    return "实时运行中"
        if self._live_worker_state == LiveWorkerState.STOPPING.value:
            return "正在停止"
        if not self._models_ready:
            return "准备中"
        state = self._infer_status_state(runtime_text)
        if state == "error":
            return "处理失败"
        if state == "ready":
            if self._current_input_mode() == "none":
                return "待选择"
            return "已就绪"
        return runtime_text or "待命中"

    def _sync_header_runtime_timer(self) -> None:
        is_running = self._live_worker_state == LiveWorkerState.RUNNING.value
        if is_running:
            if self._header_runtime_started_at is None:
                self._header_runtime_started_at = time.monotonic()
                self._update_header_runtime_clock()
            if not self._header_runtime_timer.isActive():
                self._header_runtime_timer.start()
            return
        if self._header_runtime_timer.isActive():
            self._header_runtime_timer.stop()
        self._header_runtime_started_at = None
        self._update_header_runtime_clock()

    def _update_header_runtime_clock(self) -> None:
        if not hasattr(self, "header_runtime_clock_value"):
            return
        elapsed_seconds = 0
        if self._header_runtime_started_at is not None:
            elapsed_seconds = max(0, int(time.monotonic() - self._header_runtime_started_at))
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        clock_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        self._set_text_if_changed(self.header_runtime_clock_value, clock_text)

    @staticmethod
    def _refresh_widget_style(widget: QWidget) -> None:
        style = widget.style()
        if style is None:
            return
        style.unpolish(widget)
        style.polish(widget)
        widget.update()

    @classmethod
    def _set_widget_state(cls, widget: QWidget, state: str) -> None:
        if widget.property("state") == state:
            return
        widget.setProperty("state", state)
        cls._refresh_widget_style(widget)
