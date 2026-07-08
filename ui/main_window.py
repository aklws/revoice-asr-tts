from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import Q_ARG, QMetaObject, QThread, Qt, Signal
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtGui import QIcon
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
    QPushButton,
    QSlider,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QStyle,
)

from app import get_settings
from app.core.config import DEFAULT_TTS_MAX_NEW_TOKENS
from app.core.logging import get_logger
from app.core.paths import get_bundle_file
from app.core.runtime import get_gpu_total_memory_gb
from ui.theme import DEFAULT_THEME, THEME_OPTIONS, get_theme_stylesheet
from ui.window_chrome import apply_window_chrome_theme
from ui.waveform_widget import WaveformWidget
from ui.workers import LiveSpeechWorker


logger = get_logger(__name__)
APP_VERSION = "v0.0.1"
APP_DISPLAY_NAME = f"Revoice ASR-TTS {APP_VERSION}"
SETTINGS_ICON_PATH = get_bundle_file("ui", "assets", "settings_gear.svg")
EMOTION_PRESETS: tuple[tuple[str, str | None], ...] = (
    ("自然", None),
    ("温柔", "用温柔、自然、放松的语气说，声音柔和一些。"),
    ("开心", "用开心、轻快、带一点笑意的语气说，情绪更明亮。"),
    ("悲伤", "用伤感、克制、轻声的语气说，情绪更低沉。"),
    ("愤怒", "用愤怒、激动、爆发力更强的语气说，但保持吐字清晰。"),
    ("紧张", "用紧张、压低声音、略带试探的语气说，情绪更真实。"),
)
INPUT_MODES: tuple[tuple[str, str], ...] = (
    ("麦克风变声", "microphone"),
    ("文本合成", "text"),
)
TTS_ENGINES: tuple[tuple[str, str], ...] = (
    ("Index-TTS", "indextts"),
)
# Index-TTS 八情感: [高兴, 愤怒, 悲伤, 恐惧, 反感, 低落, 惊讶, 自然]
INDEX_TTS_EMOTIONS: tuple[tuple[str, str], ...] = (
    ("高兴", "happy"),
    ("愤怒", "angry"),
    ("悲伤", "sad"),
    ("恐惧", "afraid"),
    ("反感", "disgusted"),
    ("低落", "melancholic"),
    ("惊讶", "surprised"),
    ("自然", "calm"),
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


class MainWindow(QMainWindow):
    live_preload_requested = Signal(object)
    live_run_requested = Signal(object)
    live_stop_requested = Signal()
    live_shutdown_requested = Signal()
    live_asr_unload_requested = Signal()
    live_asr_reload_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_DISPLAY_NAME}")
        self.resize(1280, 900)
        self.setMinimumSize(1180, 800)

        self.settings = get_settings()
        self._live_worker_thread: QThread | None = None
        self._live_worker: LiveSpeechWorker | None = None
        self._models_ready = False
        self._pending_reference_preload = False
        self._input_devices: list[tuple[str, str]] = []
        self._input_device_name_by_key: dict[str, str] = {}
        self._output_devices: list[tuple[int, str]] = []
        self._monitor_devices: list[tuple[int, str]] = []
        self._current_theme = DEFAULT_THEME
        self._virtual_device_summary = ""
        self._quick_settings_dialog: QDialog | None = None
        self._quick_settings_input_mode_combo: QComboBox | None = None
        self._quick_settings_auto_emotion_checkbox: QCheckBox | None = None
        self._quick_settings_auto_emotion_strength_slider: QSlider | None = None
        self._quick_settings_auto_emotion_strength_label: QLabel | None = None
        self._reference_audio_dialog: QFileDialog | None = None
        self._message_dialog: QMessageBox | None = None
        self._manual_index_emo_snapshot: dict[str, object] | None = None
        self._auto_emotion_feature_enabled = False
        self._auto_emotion_strength = 0.75

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)
        root_layout.addWidget(self._build_header())
        root_layout.addWidget(self._build_main_content(), stretch=1)
        self.setCentralWidget(root)
        self._on_input_mode_changed()
        self._apply_theme(self._current_theme)

        self._log_startup()
        self._start_live_backend()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._shutdown_live_backend()
        super().closeEvent(event)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._apply_window_chrome_theme()

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
        hero_layout.setContentsMargins(18, 14, 18, 14)
        hero_layout.setSpacing(10)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(10)
        eyebrow = QLabel("实时语音变声器")
        eyebrow.setObjectName("HeroEyebrow")
        nav_row.addWidget(eyebrow)
        nav_row.addStretch(1)
        self.theme_combo = QComboBox()
        self.theme_combo.setMinimumWidth(124)
        for label, theme_name in THEME_OPTIONS:
            self.theme_combo.addItem(label, theme_name)
        theme_index = self.theme_combo.findData(self._current_theme)
        if theme_index >= 0:
            self.theme_combo.setCurrentIndex(theme_index)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)

        self.quick_settings_button = QPushButton()
        self.quick_settings_button.setObjectName("IconButton")
        if SETTINGS_ICON_PATH.exists():
            self.quick_settings_button.setIcon(QIcon(str(SETTINGS_ICON_PATH)))
        else:
            self.quick_settings_button.setIcon(self._standard_icon(QStyle.SP_FileDialogDetailedView))
        self.quick_settings_button.setToolTip("打开快速设置")
        self.quick_settings_button.clicked.connect(self._open_quick_settings_dialog)
        nav_row.addWidget(self.quick_settings_button)
        nav_row.addWidget(self.theme_combo)

        title = QLabel(APP_DISPLAY_NAME)
        title.setObjectName("HeroTitle")
        subtitle = QLabel("实时识别、参考音色克隆与低延迟回放的一体化变声器")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("HeroSubtitle")

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)
        title_column = QVBoxLayout()
        title_column.setSpacing(2)
        title_column.addWidget(title)
        title_column.addWidget(subtitle)
        bottom_row.addLayout(title_column, stretch=1)

        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        self.header_mode_value = self._create_summary_card(status_row, "当前模式")
        self.header_runtime_value = self._create_summary_card(status_row, "引擎状态")
        self.header_auto_emotion_value = self._create_summary_card(status_row, "情感识别")
        self.header_reference_value = self._create_summary_card(status_row, "参考音色")
        header_auto_emotion_card = self.header_auto_emotion_value.parentWidget()
        if header_auto_emotion_card is not None:
            header_auto_emotion_card.setVisible(False)
        bottom_row.addLayout(status_row, stretch=1)

        hero_layout.addLayout(nav_row)
        hero_layout.addLayout(bottom_row)
        return hero_card

    def _create_summary_card(self, parent_layout: QHBoxLayout, caption: str) -> QLabel:
        card = QFrame()
        card.setObjectName("SummaryCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        caption_label = QLabel(caption)
        caption_label.setObjectName("SummaryCaption")
        value_label = QLabel("--")
        value_label.setObjectName("SummaryValue")
        value_label.setWordWrap(True)
        layout.addWidget(caption_label)
        layout.addWidget(value_label)
        parent_layout.addWidget(card, stretch=1)
        return value_label

    def _build_main_content(self) -> QWidget:
        panel = QWidget()
        grid = QGridLayout(panel)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        grid.addWidget(self._build_changer_panel(), 0, 0)
        grid.addWidget(self._build_reference_panel(), 0, 1)
        grid.addWidget(self._build_emotion_panel(), 1, 0)
        grid.addWidget(self._build_workspace_panel(), 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        return panel

    def _build_changer_panel(self) -> QWidget:
        self.input_mode_combo = QComboBox()
        for label, value in INPUT_MODES:
            self.input_mode_combo.addItem(label, value)
        self.input_mode_combo.currentIndexChanged.connect(self._on_input_mode_changed)

        self.tts_engine_combo = QComboBox()
        for label, value in TTS_ENGINES:
            self.tts_engine_combo.addItem(label, value)

        self.live_reference_audio_edit = QLineEdit("")
        self.live_reference_audio_edit.setPlaceholderText("请先选择克隆参考音频")

        self.emotion_preset_combo = QComboBox()
        for label, instruction in EMOTION_PRESETS:
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
        for cn_name, en_name in INDEX_TTS_EMOTIONS:
            slider = QSlider(Qt.Vertical)
            slider.setRange(0, 24)  # 0.00 ~ 1.20, step 0.05
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
        self.index_emo_alpha_label.setFixedWidth(38)
        self.index_emo_alpha_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.index_emo_alpha_slider.valueChanged.connect(
            lambda _v: self.index_emo_alpha_label.setText(f"{self.index_emo_alpha_slider.value() * 0.05:.2f}")
        )

        self.speech_rate_slider = QSlider(Qt.Horizontal)
        self.speech_rate_slider.setRange(0, 20)  # 0.50 ~ 1.50, step 0.05
        self.speech_rate_slider.setValue(10)
        self.speech_rate_slider.setTickInterval(2)
        self.speech_rate_slider.setTickPosition(QSlider.TicksBelow)
        self.speech_rate_label = QLabel("1.00x")
        self.speech_rate_label.setFixedWidth(44)
        self.speech_rate_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.speech_rate_slider.valueChanged.connect(
            lambda v: self.speech_rate_label.setText(f"{self._current_speech_rate():.2f}x")
        )

        self.input_device_combo = QComboBox()
        self.output_device_combo = QComboBox()
        self.monitor_output_device_combo = QComboBox()
        self.refresh_devices_button = QPushButton("刷新")
        self.refresh_devices_button.setObjectName("SecondaryActionButton")
        self.refresh_devices_button.clicked.connect(self._refresh_audio_devices)
        self.refresh_devices_button.setIcon(self._standard_icon(QStyle.SP_BrowserReload))
        self.use_virtual_output_button = QPushButton("虚拟播放")
        self.use_virtual_output_button.setObjectName("SecondaryActionButton")
        self.use_virtual_output_button.clicked.connect(self._select_virtual_output_device)
        self.use_virtual_output_button.setIcon(self._standard_icon(QStyle.SP_ArrowForward))
        self.live_status_label = QLabel("待命中")
        self.live_status_label.setObjectName("LiveStatusLabel")
        self.live_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.live_start_button = QPushButton("准备中")
        self.live_start_button.setObjectName("PrimaryActionButton")
        self.live_start_button.setMinimumHeight(50)
        self.live_start_button.setEnabled(False)
        self.live_start_button.clicked.connect(self._on_live_start_clicked)
        self.live_start_button.setIcon(self._standard_icon(QStyle.SP_MediaPlay))

        self._refresh_audio_devices()
        self.live_reference_audio_edit.textChanged.connect(self._refresh_workspace_summary)
        self.tts_engine_combo.currentIndexChanged.connect(self._refresh_workspace_summary)
        self.output_device_combo.currentIndexChanged.connect(self._refresh_workspace_summary)
        self.monitor_output_device_combo.currentIndexChanged.connect(self._refresh_workspace_summary)
        return self._build_devices_panel()

    def _build_devices_panel(self) -> QFrame:
        card, layout = self._create_content_card("设备路由", "")
        device_grid = QGridLayout()
        device_grid.setHorizontalSpacing(10)
        device_grid.setVerticalSpacing(8)
        device_grid.addWidget(self._build_compact_field("播放设备", self.output_device_combo), 0, 0)
        device_grid.addWidget(self._build_compact_field("录音设备", self.input_device_combo), 0, 1)
        device_grid.addWidget(self._build_compact_field("耳返设备", self.monitor_output_device_combo), 1, 0)
        action_card = QWidget()
        action_layout = QHBoxLayout(action_card)
        action_layout.setContentsMargins(0, 16, 0, 0)
        action_layout.setSpacing(8)
        action_layout.addWidget(self.refresh_devices_button)
        action_layout.addWidget(self.use_virtual_output_button)
        device_grid.addWidget(action_card, 1, 1)
        layout.addLayout(device_grid)
        
        # --- Voiceprint Filter Section ---
        vp_card = QFrame()
        vp_card.setObjectName("EmotionPanelCard")
        vp_layout = QVBoxLayout(vp_card)
        vp_layout.setContentsMargins(12, 10, 12, 10)
        vp_layout.setSpacing(8)
        
        vp_header = QHBoxLayout()
        self.enable_voiceprint_checkbox = QCheckBox("开启声纹锁定")
        self.enable_voiceprint_checkbox.setObjectName("CompactLabel")
        self.record_voiceprint_button = QPushButton("录制我的声纹")
        self.record_voiceprint_button.setObjectName("SecondaryActionButton")
        self.record_voiceprint_button.clicked.connect(self._on_record_voiceprint_clicked)
        
        vp_header.addWidget(self.enable_voiceprint_checkbox)
        vp_header.addStretch(1)
        vp_header.addWidget(self.record_voiceprint_button)
        vp_layout.addLayout(vp_header)
        
        vp_slider_row = QHBoxLayout()
        vp_slider_row.addWidget(QLabel("严格度", objectName="CompactLabel"))
        self.voiceprint_threshold_slider = QSlider(Qt.Horizontal)
        self.voiceprint_threshold_slider.setRange(30, 95) # 0.30 to 0.95
        self.voiceprint_threshold_slider.setValue(65)
        self.voiceprint_threshold_label = QLabel("0.65")
        self.voiceprint_threshold_slider.valueChanged.connect(
            lambda v: self.voiceprint_threshold_label.setText(f"{v / 100:.2f}")
        )
        vp_slider_row.addWidget(self.voiceprint_threshold_slider, stretch=1)
        vp_slider_row.addWidget(self.voiceprint_threshold_label)
        vp_layout.addLayout(vp_slider_row)
        
        self.user_voiceprint: list[float] | None = None
        layout.addWidget(vp_card)
        
        self.devices_info_label = QLabel()
        self.devices_info_label.setObjectName("SectionHint")
        self.devices_info_label.setWordWrap(True)
        layout.addWidget(self.devices_info_label)
        return card

    def _build_emotion_panel(self) -> QFrame:
        card, layout = self._create_content_card("情感与语速", "")
        card.setMinimumHeight(380)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(20)

        # --- Left Sidebar ---
        sidebar = QFrame()
        sidebar.setObjectName("EmotionSidebar")
        sidebar.setFixedWidth(130)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(8)

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

        speech_label = QLabel("全局语速")
        speech_label.setObjectName("CompactLabel")
        sidebar_layout.addWidget(speech_label)
        
        speech_val_layout = QHBoxLayout()
        speech_val_layout.addWidget(self.speech_rate_slider)
        speech_val_layout.addWidget(self.speech_rate_label)
        sidebar_layout.addLayout(speech_val_layout)

        content_layout.addWidget(sidebar)

        # --- Right Stack ---
        self.emotion_stack = QStackedWidget()

        # Page 0: Preset
        preset_page = QWidget()
        preset_layout = QVBoxLayout(preset_page)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(16)

        preset_title = QLabel("选择预设")
        preset_title.setObjectName("CompactLabel")
        preset_layout.addWidget(preset_title)

        preset_grid = QGridLayout()
        preset_grid.setSpacing(10)
        self.emotion_preset_buttons: list[QPushButton] = []
        for index, (label, _instruction) in enumerate(EMOTION_PRESETS):
            button = QPushButton(label)
            button.setObjectName("EmotionPresetButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, idx=index: self.emotion_preset_combo.setCurrentIndex(idx))
            preset_grid.addWidget(button, index // 3, index % 3)
            self.emotion_preset_buttons.append(button)
        preset_layout.addLayout(preset_grid)

        custom_title = QLabel("自定义语气")
        custom_title.setObjectName("CompactLabel")
        preset_layout.addWidget(custom_title)
        preset_layout.addWidget(self.emotion_custom_edit)
        preset_layout.addStretch(1)
        self.emotion_stack.addWidget(preset_page)

        # Page 1: Vector
        vector_page = QWidget()
        vector_layout = QVBoxLayout(vector_page)
        vector_layout.setContentsMargins(0, 0, 0, 0)
        vector_layout.setSpacing(16)

        slider_row = QHBoxLayout()
        slider_row.setContentsMargins(0, 0, 0, 0)
        slider_row.setSpacing(12)
        
        self.index_emo_rows = []
        for cn_name, en_name in INDEX_TTS_EMOTIONS:
            slider = self.index_emo_sliders[en_name]
            value_label = self.index_emo_value_labels[en_name]
            slider.setFixedHeight(210)
            slider.setFixedWidth(34)

            def _make_on_slider(s: QSlider, lbl: QLabel, item_title: str, item_widget: QWidget):
                def _on_move(_value: int | None = None) -> None:
                    current_text = f"{s.value() * 0.05:.2f}"
                    lbl.setText(current_text)
                    item_widget.setToolTip(f"{item_title}: {current_text}")
                return _on_move

            item_container = QWidget()
            item_layout = QVBoxLayout(item_container)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(8)
            value_label.setObjectName("EmotionSliderValue")
            cn_label = QLabel(cn_name)
            cn_label.setObjectName("EmotionSliderName")
            item_layout.addWidget(value_label, alignment=Qt.AlignHCenter)
            item_layout.addWidget(slider, alignment=Qt.AlignHCenter)
            item_layout.addWidget(cn_label, alignment=Qt.AlignHCenter)
            slider.valueChanged.connect(_make_on_slider(slider, value_label, cn_name, item_container))
            
            # Note: We don't invert value here, Qt handles display based on invertedAppearance
            # We just need to ensure the value mapping is correct
            value_label.setText(f"{slider.value() * 0.05:.2f}")
            item_container.setToolTip(f"{cn_name}: {value_label.text()}")
            slider_row.addWidget(item_container)
            self.index_emo_rows.append(item_container)
            
        vector_layout.addLayout(slider_row)

        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel("情感强度", objectName="CompactLabel"))
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
        return card

    def _build_reference_panel(self) -> QFrame:
        card, layout = self._create_content_card("参考音频", "右上用于设置参考音频、查看引擎状态并启动当前任务。")

        reference_row = QHBoxLayout()
        reference_row.setSpacing(8)
        reference_row.addWidget(self.live_reference_audio_edit, stretch=1)
        self.reference_ready_badge = QLabel("待选择")
        self.reference_ready_badge.setObjectName("StatusBadge")
        reference_row.addWidget(self.reference_ready_badge)
        browse_button = QPushButton("选择参考音频")
        browse_button.setObjectName("SecondaryActionButton")
        browse_button.setIcon(self._standard_icon(QStyle.SP_DialogOpenButton))
        browse_button.clicked.connect(self._browse_live_reference_audio)
        reference_row.addWidget(browse_button)
        layout.addLayout(reference_row)

        self.reference_meta_label = QLabel("当前尚未选择参考音频。")
        self.reference_meta_label.setObjectName("SectionHint")
        self.reference_meta_label.setWordWrap(True)
        layout.addWidget(self.reference_meta_label)

        self.reference_engine_label = QLabel()
        self.reference_engine_label.setObjectName("SummaryValue")
        self.reference_engine_label.setWordWrap(True)
        layout.addWidget(self.reference_engine_label)

        layout.addWidget(self.live_status_label)
        layout.addWidget(self.live_start_button)
        return card

    def _build_compact_field(self, title_text: str, widget: QWidget) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        title = QLabel(title_text)
        title.setObjectName("CompactLabel")
        layout.addWidget(title)
        layout.addWidget(widget)
        return container

    def _build_workspace_panel(self) -> QFrame:
        card, layout = self._create_content_card("文本与波形", "右下同时展示文本输入或识别结果，以及当前生成的声波。")
        workspace_canvas = QFrame()
        workspace_canvas.setObjectName("WorkspaceCanvas")
        canvas_layout = QVBoxLayout(workspace_canvas)
        canvas_layout.setContentsMargins(14, 14, 14, 14)
        canvas_layout.setSpacing(12)

        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        self.workspace_mode_badge = QLabel("麦克风变声")
        self.workspace_mode_badge.setObjectName("WorkspaceBadge")
        self.transcript_intro_label = QLabel("实时识别文本会汇总在这里，方便边说边看。")
        self.transcript_intro_label.setObjectName("SectionHint")
        self.transcript_intro_label.setWordWrap(True)
        header_row.addWidget(self.workspace_mode_badge)
        header_row.addWidget(self.transcript_intro_label, stretch=1)
        canvas_layout.addLayout(header_row)

        self.live_transcript_view = QTextEdit()
        self.live_transcript_view.setObjectName("TranscriptView")
        self.live_transcript_view.setReadOnly(True)
        self.live_transcript_view.setPlaceholderText("这里会显示实时识别结果。")
        self.live_transcript_view.textChanged.connect(self._refresh_workspace_text_metrics)
        canvas_layout.addWidget(self.live_transcript_view, stretch=4)

        footer_row = QHBoxLayout()
        footer_row.setSpacing(8)
        waveform_title = QLabel("输出波形")
        waveform_title.setObjectName("CompactLabel")
        self.workspace_counter_label = QLabel("0 字")
        self.workspace_counter_label.setObjectName("WorkspaceStat")
        self.workspace_wave_hint = QLabel("等待开始处理")
        self.workspace_wave_hint.setObjectName("WorkspaceStat")
        footer_row.addWidget(waveform_title)
        footer_row.addStretch(1)
        footer_row.addWidget(self.workspace_counter_label)
        footer_row.addWidget(self.workspace_wave_hint)
        canvas_layout.addLayout(footer_row)

        waveform_frame = QFrame()
        waveform_frame.setObjectName("WaveformCard")
        waveform_layout = QVBoxLayout(waveform_frame)
        waveform_layout.setContentsMargins(0, 0, 0, 0)
        waveform_layout.setSpacing(0)
        self.waveform_widget = WaveformWidget()
        waveform_layout.addWidget(self.waveform_widget)
        canvas_layout.addWidget(waveform_frame, stretch=2)

        layout.addWidget(workspace_canvas, stretch=1)
        self._refresh_workspace_text_metrics()
        return card

    def _ensure_quick_settings_dialog(self) -> QDialog:
        if self._quick_settings_dialog is not None:
            return self._quick_settings_dialog

        dialog = QDialog(self)
        dialog.setWindowTitle("快速设置")
        dialog.setModal(True)
        dialog.resize(360, 220)

        form = QFormLayout(dialog)
        form.setSpacing(12)

        input_mode_combo = QComboBox(dialog)
        for label, value in INPUT_MODES:
            input_mode_combo.addItem(label, value)
        form.addRow("输入方式", input_mode_combo)

        experimental_widget = QWidget(dialog)
        experimental_layout = QVBoxLayout(experimental_widget)
        experimental_layout.setContentsMargins(0, 0, 0, 0)
        experimental_layout.setSpacing(8)

        auto_emotion_checkbox = QCheckBox("启用实验性情感识别（跟随麦克风情绪）", dialog)
        experimental_layout.addWidget(auto_emotion_checkbox)

        auto_strength_row = QHBoxLayout()
        auto_strength_row.setContentsMargins(0, 0, 0, 0)
        auto_strength_row.setSpacing(6)
        auto_strength_row.addWidget(QLabel("强度", objectName="CompactLabel"))
        auto_emotion_strength_slider = QSlider(Qt.Horizontal, dialog)
        auto_emotion_strength_slider.setRange(8, 20)  # 0.40 ~ 1.00
        auto_emotion_strength_slider.setTickInterval(1)
        auto_emotion_strength_label = QLabel("0.75", dialog)
        auto_emotion_strength_label.setFixedWidth(36)
        auto_emotion_strength_slider.valueChanged.connect(
            lambda v: auto_emotion_strength_label.setText(f"{v * 0.05:.2f}")
        )
        auto_strength_row.addWidget(auto_emotion_strength_slider, stretch=1)
        auto_strength_row.addWidget(auto_emotion_strength_label)
        experimental_layout.addLayout(auto_strength_row)
        form.addRow("实验选项", experimental_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        self._quick_settings_dialog = dialog
        self._quick_settings_input_mode_combo = input_mode_combo
        self._quick_settings_auto_emotion_checkbox = auto_emotion_checkbox
        self._quick_settings_auto_emotion_strength_slider = auto_emotion_strength_slider
        self._quick_settings_auto_emotion_strength_label = auto_emotion_strength_label
        apply_window_chrome_theme(dialog, self._current_theme)
        return dialog

    def _open_quick_settings_dialog(self) -> None:
        dialog = self._ensure_quick_settings_dialog()
        if self._quick_settings_input_mode_combo is None:
            return
        self._quick_settings_input_mode_combo.setCurrentIndex(self.input_mode_combo.currentIndex())
        if self._quick_settings_auto_emotion_checkbox is not None:
            self._quick_settings_auto_emotion_checkbox.setChecked(self._auto_emotion_feature_enabled)
        if self._quick_settings_auto_emotion_strength_slider is not None:
            self._quick_settings_auto_emotion_strength_slider.setValue(int(round(self._auto_emotion_strength / 0.05)))

        if dialog.exec() != QDialog.Accepted:
            return

        self.input_mode_combo.setCurrentIndex(self._quick_settings_input_mode_combo.currentIndex())
        if self._quick_settings_auto_emotion_checkbox is not None:
            self._auto_emotion_feature_enabled = self._quick_settings_auto_emotion_checkbox.isChecked()
        if self._quick_settings_auto_emotion_strength_slider is not None:
            self._auto_emotion_strength = self._quick_settings_auto_emotion_strength_slider.value() * 0.05
        self._on_auto_emotion_toggled()
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
    def _set_text_if_changed(widget: QLabel | QPushButton, text: str) -> None:
        if widget.text() != text:
            widget.setText(text)

    @staticmethod
    def _set_text_edit_if_changed(widget: QTextEdit, text: str) -> None:
        if widget.toPlainText() != text:
            widget.setPlainText(text)

    def _set_live_status_label_state(self, text: str, state: str) -> None:
        text_changed = self.live_status_label.text() != text
        state_changed = self.live_status_label.property("state") != state
        if text_changed:
            self.live_status_label.setText(text)
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
            app.setStyleSheet(get_theme_stylesheet(theme_name))
        if hasattr(self, "waveform_widget"):
            self.waveform_widget.set_theme(theme_name)
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

    def _create_content_card(self, title_text: str, hint_text: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("ContentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        title = QLabel(title_text)
        title.setObjectName("CardTitle")
        layout.addWidget(title)
        if hint_text:
            hint = QLabel(hint_text)
            hint.setObjectName("SectionHint")
            hint.setWordWrap(True)
            layout.addWidget(hint)
        return card, layout

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
            if self._models_ready:
                self._models_ready = False
                self._refresh_live_start_button()
                self._set_live_status("正在预热参考音色...", "busy")
                self.live_preload_requested.emit(self._build_live_preload_payload(path))
            else:
                self._pending_reference_preload = True
                self._refresh_live_start_button()
                self._set_live_status("参考音色已选择，模型加载完成后自动预热", "busy")

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
        return self._is_auto_emotion_feature_enabled() and self._current_input_mode() == "microphone"

    def _capture_manual_index_emotion_snapshot(self) -> None:
        if self._manual_index_emo_snapshot is not None:
            return
        self._manual_index_emo_snapshot = {
            "mode": self.index_emotion_mode_combo.currentData(),
            "vector": [self.index_emo_sliders[en_name].value() for _cn_name, en_name in INDEX_TTS_EMOTIONS],
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
            for (_cn_name, en_name), value in zip(INDEX_TTS_EMOTIONS, raw_vector, strict=False):
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
        slider_values = [self.index_emo_sliders[en_name].value() for _cn_name, en_name in INDEX_TTS_EMOTIONS]
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
        mode = self.input_mode_combo.currentData()
        return str(mode or "microphone")

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
        for (_cn_name, en_name), value in zip(INDEX_TTS_EMOTIONS, vector, strict=False):
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

    def _on_input_mode_changed(self) -> None:
        is_text_mode = self._current_input_mode() == "text"
        self.input_device_combo.setEnabled(not is_text_mode)
        self._refresh_devices_info()
        if hasattr(self, "live_transcript_view") and hasattr(self, "transcript_intro_label"):
            if is_text_mode:
                self.live_transcript_view.setReadOnly(False)
                self.live_transcript_view.setPlaceholderText("文本模式下，在这里直接输入要合成的内容。")
                self.transcript_intro_label.setText("这里是文本模式的主输入区。直接输入内容后点击“开始合成”。")
                if hasattr(self, "workspace_mode_badge"):
                    self.workspace_mode_badge.setText("文本合成")
                if hasattr(self, "workspace_wave_hint"):
                    self.workspace_wave_hint.setText("输入文本后开始合成")
            else:
                self.live_transcript_view.setReadOnly(True)
                self.live_transcript_view.setPlaceholderText("这里会显示实时识别结果。")
                self.transcript_intro_label.setText("实时识别文本会汇总在这里，方便边说边看。")
                if hasattr(self, "workspace_mode_badge"):
                    self.workspace_mode_badge.setText("麦克风变声")
                if hasattr(self, "workspace_wave_hint"):
                    self.workspace_wave_hint.setText("等待实时识别与语音输出")
        if is_text_mode:
            if hasattr(self, "live_transcript_view"):
                self.live_transcript_view.setFocus()
            self.live_asr_unload_requested.emit()
        else:
            self.live_asr_reload_requested.emit()
        self._refresh_live_start_button()
        self._on_auto_emotion_toggled()
        self._refresh_workspace_text_metrics()
        self._refresh_workspace_summary()

    def _refresh_audio_devices(self) -> None:
        self.input_device_combo.clear()
        self.output_device_combo.clear()
        self.monitor_output_device_combo.clear()
        self._input_devices.clear()
        self._input_device_name_by_key.clear()
        self._output_devices.clear()
        self._monitor_devices.clear()

        try:
            input_devices = list(QMediaDevices.audioInputs())
            default_input_key = bytes(QMediaDevices.defaultAudioInput().id()).hex()
            sd_devices = sd.query_devices()
            default_output = getattr(sd.default, "device", (None, None))[1]
        except Exception as exc:
            logger.error("读取音频设备失败: {}", exc)
            self.input_device_combo.addItem("读取失败")
            self.output_device_combo.addItem("读取失败")
            self.monitor_output_device_combo.addItem("读取失败")
            self._refresh_workspace_summary()
            return

        self.input_device_combo.addItem("系统默认输入", None)
        self.output_device_combo.addItem("系统默认输出", None)
        self.monitor_output_device_combo.addItem("关闭耳返", None)

        logger.info("音频设备枚举完成: Qt 输入 {} 个, sounddevice 输出 {} 个", len(input_devices), len(sd_devices))

        deduped_inputs: dict[str, tuple[str, str]] = {}
        deduped_outputs: dict[str, tuple[int, str]] = {}

        for device in input_devices:
            name = str(device.description()).strip() or "输入设备"
            device_key = bytes(device.id()).hex()
            dedupe_key = self._device_dedupe_key(name)
            if dedupe_key not in deduped_inputs:
                deduped_inputs[dedupe_key] = (device_key, name)

        for index, device in enumerate(sd_devices):
            if int(device.get("max_output_channels", 0)) <= 0:
                continue
            name = str(device.get("name") or "").strip() or f"输出设备 {index}"
            dedupe_key = self._device_dedupe_key(name)
            if dedupe_key not in deduped_outputs:
                deduped_outputs[dedupe_key] = (index, name)

        virtual_input_count = 0
        for device_key, name in deduped_inputs.values():
            label = self._format_device_label(device_key, name)
            if self._is_virtual_device_name(name):
                virtual_input_count += 1
            self._input_devices.append((device_key, label))
            self._input_device_name_by_key[device_key] = name
            self.input_device_combo.addItem(label, device_key)

        virtual_output_count = 0
        for device_index, name in deduped_outputs.values():
            label = self._format_output_device_label(device_index, name)
            if self._is_virtual_device_name(name):
                virtual_output_count += 1
            self._output_devices.append((device_index, label))
            self._monitor_devices.append((device_index, label))
            self.output_device_combo.addItem(label, device_index)
            self.monitor_output_device_combo.addItem(label, device_index)

        if default_input_key:
            input_index = self.input_device_combo.findData(default_input_key)
            if input_index >= 0:
                self.input_device_combo.setCurrentIndex(input_index)
        if default_output is not None and int(default_output) >= 0:
            output_index = self.output_device_combo.findData(int(default_output))
            if output_index >= 0:
                self.output_device_combo.setCurrentIndex(output_index)
            monitor_index = self.monitor_output_device_combo.findData(int(default_output))
            if monitor_index >= 0:
                self.monitor_output_device_combo.setCurrentIndex(monitor_index)

        self._apply_preferred_virtual_output()
        self._update_virtual_device_hint(virtual_input_count, virtual_output_count)
        self._refresh_workspace_summary()

    @staticmethod
    def _is_virtual_device_name(name: str) -> bool:
        normalized = name.casefold()
        return any(keyword in normalized for keyword in VIRTUAL_DEVICE_KEYWORDS)

    @staticmethod
    def _device_dedupe_key(name: str) -> str:
        normalized = " ".join(name.casefold().split())
        suffixes = (
            ", windows wasapi",
        )
        for suffix in suffixes:
            if normalized.endswith(suffix):
                return normalized[: -len(suffix)].rstrip(" ,")
        return normalized

    def _format_device_label(self, device_key: str, name: str) -> str:
        prefix = "[虚拟]" if self._is_virtual_device_name(name) else "[设备]"
        short_key = device_key[:8] if device_key else "default"
        return f"{prefix} [{short_key}] {name}"

    def _format_output_device_label(self, device_index: int, name: str) -> str:
        prefix = "[虚拟]" if self._is_virtual_device_name(name) else "[设备]"
        return f"{prefix} [{device_index}] {name}"

    @staticmethod
    def _extract_device_display_name(label: str) -> str:
        parts = label.split("] ", 2)
        if len(parts) >= 3:
            return parts[2].strip()
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
        mode_hint = (
            "文本模式下录音设备不参与。"
            if self._current_input_mode() == "text"
            else "麦克风模式下录音、播放、耳返都会参与。"
        )
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

    def _on_record_voiceprint_clicked(self) -> None:
        if self._live_worker is not None and self._live_worker._busy:
            self._show_message(QMessageBox.Warning, "正在变声", "请先关闭麦克风再录制声纹。")
            return
            
        self.record_voiceprint_button.setEnabled(False)
        self._set_text_if_changed(self.record_voiceprint_button, "请朗读：「测试一二三，开启声纹锁定」(3秒)")
        
        def _record_task() -> None:
            try:
                # Record 3 seconds at 16000 Hz
                duration = 3.0
                sr = 16000
                
                # Resolve sounddevice input index from current combo box text
                current_device_key = self.input_device_combo.currentData()
                device_name = ""
                if isinstance(current_device_key, str) and current_device_key:
                    device_name = self._input_device_name_by_key.get(current_device_key, "")
                if not device_name:
                    device_name = self._extract_device_display_name(self.input_device_combo.currentText())
                
                sd_device = None
                if device_name:
                    for i, dev in enumerate(sd.query_devices()):
                        if dev['max_input_channels'] > 0 and device_name in dev['name']:
                            sd_device = i
                            break
                            
                # sd.rec is non-blocking, but we use sd.wait() to block the thread
                recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32', device=sd_device)
                sd.wait()
                
                # Extract voiceprint
                if not self._models_ready or self._live_worker is None or self._live_worker.index_tts_service is None:
                    # Need to ensure TTS is loaded
                    self._start_live_backend()
                    while self._live_worker is None or self._live_worker.index_tts_service is None:
                        time.sleep(0.1)
                        if self._live_worker and self._live_worker.index_tts_service:
                            break
                            
                vp = self._live_worker.index_tts_service.extract_voiceprint(recording.flatten(), sr)
                self.user_voiceprint = vp
                
                # Update UI
                self._update_record_btn_text("录制完成！")
                time.sleep(2)
                self._update_record_btn_text("重新录制声纹")
                
                # Automatically enable the checkbox
                self._enable_voiceprint_cb(True)
                
            except Exception as e:
                self._update_record_btn_text("录制失败")
                logger.error("录制声纹失败: {}", e)

        threading.Thread(target=_record_task, daemon=True).start()

    def _update_record_btn_text(self, text: str) -> None:
        QMetaObject.invokeMethod(
            self.record_voiceprint_button,
            "setText",
            Qt.QueuedConnection,
            Q_ARG(str, text)
        )
        if "完成" in text or "重新" in text or "失败" in text:
            QMetaObject.invokeMethod(
                self.record_voiceprint_button,
                "setEnabled",
                Qt.QueuedConnection,
                Q_ARG(bool, True)
            )

    def _enable_voiceprint_cb(self, checked: bool) -> None:
        QMetaObject.invokeMethod(
            self.enable_voiceprint_checkbox,
            "setChecked",
            Qt.QueuedConnection,
            Q_ARG(bool, checked)
        )

    def _build_live_preload_payload(self, reference_audio_path: str | None = None) -> dict[str, str]:
        reference_path = (
            self.live_reference_audio_edit.text().strip()
            if reference_audio_path is None
            else str(reference_audio_path).strip()
        )
        return {
            "reference_audio_path": reference_path,
            "input_mode": self._current_input_mode(),
        }

    def _on_live_start_clicked(self) -> None:
        input_mode = self._current_input_mode()
        if input_mode == "microphone" and self._live_worker is not None and self._live_worker._busy:
            self.live_start_button.setEnabled(False)
            self.live_start_button.setText("正在停止...")
            self._live_worker.stop_live()
            return
        
        self._start_live_speech()

    def _start_live_speech(self) -> None:
        reference_audio = self.live_reference_audio_edit.text().strip()
        input_mode = self._current_input_mode()
        input_text = self.live_transcript_view.toPlainText().strip() if input_mode == "text" else ""
        if not reference_audio:
            self._show_message(QMessageBox.Warning, "缺少参考音频", "请选择实时页面使用的参考音频。")
            return
        if input_mode == "text" and not input_text:
            self._show_message(QMessageBox.Warning, "缺少输入文本", "文本模式下请输入要合成的文本。")
            return
        if not self._models_ready or self._live_worker_thread is None:
            self._show_message(QMessageBox.Information, "模型加载中", "模型尚未加载完成，请稍后再试。")
            return

        if input_mode != "text":
            self._set_text_if_changed(self.live_start_button, "关闭麦克风")
            self.live_start_button.setIcon(self._standard_icon(QStyle.SP_MediaStop))
            self.live_start_button.setEnabled(True)  # Keep enabled so user can stop it
            self.live_transcript_view.clear()
        else:
            self.live_start_button.setEnabled(False)
            
        self._set_live_status("正在开始处理...", "busy")
        self.waveform_widget.clear_waveform()
        if self._is_auto_emotion_runtime_enabled() and input_mode != "text":
            self._update_auto_emotion_ui("", "busy")
        else:
            self._update_auto_emotion_ui("", "idle")
        emo_vector, emo_alpha = self._build_index_emo_vector()
        self.live_run_requested.emit(
            {
                "input_mode": input_mode,
                "input_text": input_text,
                "reference_audio_path": reference_audio,
                "language": "Chinese",
                "instruction": self._build_tts_instruction(),
                "max_new_tokens": DEFAULT_TTS_MAX_NEW_TOKENS,
                "input_device": self.input_device_combo.currentData(),
                "input_device_label": self._extract_device_display_name(self.input_device_combo.currentText()),
                "output_device": self.output_device_combo.currentData(),
                "output_device_label": self._extract_device_display_name(self.output_device_combo.currentText()),
                "monitor_output_device": self.monitor_output_device_combo.currentData(),
                "monitor_output_device_label": self._extract_device_display_name(
                    self.monitor_output_device_combo.currentText()
                ),
                "index_emo_vector": emo_vector,
                "index_emo_alpha": emo_alpha,
                "auto_emotion_enabled": self._is_auto_emotion_runtime_enabled() and input_mode != "text",
                "auto_emotion_strength": self._auto_emotion_strength,
                "speech_rate": self._current_speech_rate(),
                "user_voiceprint": self.user_voiceprint if self.enable_voiceprint_checkbox.isChecked() else None,
                "voiceprint_threshold": self.voiceprint_threshold_slider.value() / 100.0,
            }
        )

    def _start_live_backend(self) -> None:
        if self._live_worker_thread is not None:
            return
        self._models_ready = False
        self._refresh_live_start_button()
        self._set_live_status("正在初始化引擎...", "busy")
        self._live_worker_thread = QThread(self)
        self._live_worker = LiveSpeechWorker(
            initial_reference_audio_path=self.live_reference_audio_edit.text().strip(),
        )
        self._live_worker.moveToThread(self._live_worker_thread)
        self.live_preload_requested.connect(self._live_worker.preload_models)
        self.live_run_requested.connect(self._live_worker.run_once)
        self.live_stop_requested.connect(self._live_worker.stop_live)
        self.live_shutdown_requested.connect(self._live_worker.shutdown)
        self.live_asr_unload_requested.connect(self._live_worker.unload_asr)
        self.live_asr_reload_requested.connect(self._live_worker.reload_asr)
        self._live_worker.ready.connect(self._on_live_ready)
        self._live_worker.status.connect(self._on_worker_status)
        self._live_worker.emotion_state.connect(self._on_worker_emotion_state)
        self._live_worker.transcript.connect(self._on_live_transcript)
        self._live_worker.waveform.connect(self._on_waveform_chunk)
        self._live_worker.finished.connect(self._on_live_finished)
        self._live_worker.error.connect(self._on_worker_error)
        self._live_worker_thread.start()
        self.live_preload_requested.emit(self._build_live_preload_payload())

    def _shutdown_live_backend(self) -> None:
        if self._live_worker_thread is None:
            return
        self.live_shutdown_requested.emit()
        self._live_worker_thread.quit()
        self._live_worker_thread.wait(5000)
        if self._live_worker is not None:
            self._live_worker.deleteLater()
        self._live_worker_thread.deleteLater()
        self._live_worker = None
        self._live_worker_thread = None
        self._models_ready = False
        self._refresh_live_start_button()

    def _on_live_ready(self) -> None:
        self._models_ready = True
        if self._pending_reference_preload and self.live_reference_audio_edit.text().strip():
            self._pending_reference_preload = False
            self._models_ready = False
            self._refresh_live_start_button()
            self._set_live_status("正在预热参考音色...", "busy")
            self.live_preload_requested.emit(self._build_live_preload_payload())
            return
        self._refresh_live_start_button()
        self._set_live_status("引擎已就绪，等待开始", "ready")
        logger.info("UI 模型预加载完成")

    def _on_worker_status(self, message: str) -> None:
        self._set_live_status(message, self._infer_status_state(message))
        logger.info("{}", message)

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

    def _on_live_finished(self, transcript: str) -> None:
        if self._current_input_mode() == "text":
            self._set_text_edit_if_changed(self.live_transcript_view, transcript)
        self._set_live_status("变声完成", "ready")
        if self._current_input_mode() == "text":
            self._update_auto_emotion_ui("", "idle")
        self._refresh_live_start_button()
        if hasattr(self, "workspace_wave_hint"):
            self._set_text_if_changed(self.workspace_wave_hint, "波形已刷新")
        logger.info("实时链路处理完成")

    def _on_live_transcript(self, transcript: str) -> None:
        self._set_text_edit_if_changed(self.live_transcript_view, transcript)

    def _on_waveform_chunk(self, audio) -> None:
        self.waveform_widget.append_audio_chunk(audio)
        if hasattr(self, "workspace_wave_hint"):
            self._set_text_if_changed(self.workspace_wave_hint, "接收音频中")

    def _refresh_workspace_text_metrics(self) -> None:
        if not hasattr(self, "live_transcript_view") or not hasattr(self, "workspace_counter_label"):
            return
        text = self.live_transcript_view.toPlainText()
        compact = "".join(text.split())
        self._set_text_if_changed(self.workspace_counter_label, f"{len(compact)} 字")

    def _on_worker_error(self, message: str) -> None:
        self._set_live_status("处理失败", "error")
        self._update_auto_emotion_ui("", "error")
        self._refresh_live_start_button()
        logger.error("界面任务失败: {}", message)
        self._show_message(QMessageBox.Critical, "错误", message)

    def _set_live_status(self, text: str, state: str) -> None:
        self._set_live_status_label_state(text, state)
        self._refresh_workspace_summary(status_override=text)

    def _refresh_live_start_button(self) -> None:
        input_mode = self._current_input_mode()
        if input_mode == "microphone" and self._live_worker is not None and self._live_worker._busy:
            self._set_text_if_changed(self.live_start_button, "关闭麦克风")
            self.live_start_button.setIcon(self._standard_icon(QStyle.SP_MediaStop))
            self.live_start_button.setEnabled(True)
            return
        if not self._models_ready:
            self._set_text_if_changed(self.live_start_button, "准备中")
            self.live_start_button.setIcon(self._standard_icon(QStyle.SP_BrowserReload))
            self.live_start_button.setEnabled(False)
            return
        if input_mode == "text":
            self._set_text_if_changed(self.live_start_button, "开始合成")
        else:
            self._set_text_if_changed(self.live_start_button, "开始变声")
        self.live_start_button.setIcon(self._standard_icon(QStyle.SP_MediaPlay))
        self.live_start_button.setEnabled(True)

    def _refresh_workspace_summary(self, *_args, status_override: str | None = None) -> None:
        input_mode = "文本合成" if self._current_input_mode() == "text" else "麦克风变声"
        engine_name = self.tts_engine_combo.currentText().strip() or "未选择"
        reference_path = self.live_reference_audio_edit.text().strip()
        reference_name = Path(reference_path).name if reference_path else "未选择"
        output_label = self.output_device_combo.currentText().strip() or "系统默认输出"
        runtime_text = status_override or self.live_status_label.text().strip() or "待命中"

        self._set_text_if_changed(self.header_mode_value, input_mode)
        self._set_text_if_changed(self.header_runtime_value, runtime_text)
        self._set_text_if_changed(self.header_reference_value, reference_name)
        if hasattr(self, "reference_meta_label"):
            self._set_text_if_changed(
                self.reference_meta_label,
                f"当前参考：{reference_name} | 播放：{output_label} | 耳返："
                f"{self.monitor_output_device_combo.currentText().strip() or '关闭耳返'}"
            )
        if hasattr(self, "reference_engine_label"):
            self._set_text_if_changed(self.reference_engine_label, f"{engine_name} | {runtime_text}")
        if hasattr(self, "reference_ready_badge"):
            if not reference_path:
                badge_text = "待选择"
            elif any(keyword in runtime_text for keyword in ("预热", "初始化", "准备", "开始处理")):
                badge_text = "处理中"
            elif any(keyword in runtime_text for keyword in ("已就绪", "完成", "已加载")):
                badge_text = "已就绪"
            else:
                badge_text = "已选择"
            self._set_text_if_changed(self.reference_ready_badge, badge_text)

    @staticmethod
    def _infer_status_state(message: str) -> str:
        if any(keyword in message for keyword in ("失败", "错误")):
            return "error"
        if any(keyword in message for keyword in ("完成", "已加载", "已就绪", "待命", "已切换")):
            return "ready"
        return "busy"

    @staticmethod
    def _refresh_widget_style(widget: QWidget) -> None:
        style = widget.style()
        if style is None:
            return
        style.unpolish(widget)
        style.polish(widget)
        widget.update()
