from __future__ import annotations


DEFAULT_THEME = "dark"
THEME_OPTIONS: tuple[tuple[str, str], ...] = (
    ("深色", "dark"),
    ("浅色", "light"),
)

_THEME_PALETTES: dict[str, dict[str, str]] = {
    "dark": {
        "widget_bg": "#0f1115",
        "window_bg": "#090b10",
        "group_bg": "#131722",
        "surface_bg": "#101621",
        "surface_bg_alt": "#0d131d",
        "group_border": "#222938",
        "surface_border": "#283246",
        "group_title": "#f0f0f0",
        "text": "#e5e7eb",
        "muted_text": "#95a2b8",
        "input_border": "#2b3345",
        "focus_border": "#5c8dff",
        "selection_bg": "#3159c9",
        "selection_text": "#f8fafc",
        "button_bg": "#1a2130",
        "button_hover": "#20293b",
        "button_pressed": "#171e2b",
        "button_border": "#2f3a50",
        "button_hover_border": "#47608f",
        "button_pressed_border": "#5e79ab",
        "button_disabled_bg": "#131925",
        "button_disabled_text": "#6b7280",
        "button_disabled_border": "#212838",
        "primary_start": "#3b82f6",
        "primary_end": "#1d4ed8",
        "primary_hover_start": "#5a9bff",
        "primary_hover_end": "#2563eb",
        "primary_border": "#5d92ff",
        "primary_hover_border": "#8ab1ff",
        "scroll_handle": "#2e3950",
        "scroll_handle_hover": "#465778",
        "splitter": "#1c2332",
        "hero_title": "#f8fafc",
        "combo_popup_bg": "#121826",
        "hero_card_bg": "#0f1420",
        "hero_card_border": "#24314a",
        "hero_chip_bg": "#161f31",
        "hero_chip_border": "#31435f",
        "hero_chip_text": "#dbe7ff",
        "accent_soft": "#7fb0ff",
        "success_bg": "#0f1c19",
        "success_border": "#25493d",
        "success_text": "#b7f7d8",
        "warning_bg": "#1a1720",
        "warning_border": "#4a3b66",
        "warning_text": "#d6ccff",
        "error_bg": "#231316",
        "error_border": "#6a2e37",
        "error_text": "#ffc9d0",
        "log_bg": "#0a0e16",
        "editor_bg": "#0b0f17",
    },
    "light": {
        "widget_bg": "#f3f6fb",
        "window_bg": "#edf2f9",
        "group_bg": "#ffffff",
        "surface_bg": "#ffffff",
        "surface_bg_alt": "#f6f9ff",
        "group_border": "#d7e0ef",
        "surface_border": "#d6e0ef",
        "group_title": "#172033",
        "text": "#1f2937",
        "muted_text": "#607089",
        "input_border": "#cfdae9",
        "focus_border": "#5b8def",
        "selection_bg": "#a9c4ff",
        "selection_text": "#10203a",
        "button_bg": "#ffffff",
        "button_hover": "#f1f5fb",
        "button_pressed": "#e7edf8",
        "button_border": "#c7d2e0",
        "button_hover_border": "#aab9cc",
        "button_pressed_border": "#92a5bf",
        "button_disabled_bg": "#eef2f7",
        "button_disabled_text": "#8b97a8",
        "button_disabled_border": "#d6deea",
        "primary_start": "#3b82f6",
        "primary_end": "#2563eb",
        "primary_hover_start": "#4f8df7",
        "primary_hover_end": "#3572ee",
        "primary_border": "#2c66dc",
        "primary_hover_border": "#1f57ca",
        "scroll_handle": "#c2ccda",
        "scroll_handle_hover": "#aab7c9",
        "splitter": "#d3dbe7",
        "hero_title": "#0f172a",
        "combo_popup_bg": "#f9fbff",
        "hero_card_bg": "#ffffff",
        "hero_card_border": "#dbe5f5",
        "hero_chip_bg": "#eef4ff",
        "hero_chip_border": "#cdddf8",
        "hero_chip_text": "#24457d",
        "accent_soft": "#3b82f6",
        "success_bg": "#eefcf5",
        "success_border": "#bee7d2",
        "success_text": "#175b3b",
        "warning_bg": "#f4f1ff",
        "warning_border": "#d6cdf9",
        "warning_text": "#4d3a8f",
        "error_bg": "#fff1f3",
        "error_border": "#f0c5cf",
        "error_text": "#8a2338",
        "log_bg": "#f8fbff",
        "editor_bg": "#ffffff",
    },
}


def get_theme_stylesheet(theme_name: str) -> str:
    palette = _THEME_PALETTES.get(theme_name, _THEME_PALETTES[DEFAULT_THEME])
    return """
QWidget {{
    background: {widget_bg};
    color: {text};
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 13px;
}}

QMainWindow {{
    background: {window_bg};
}}

QScrollArea {{
    background: transparent;
    border: none;
}}

QGroupBox {{
    background: {group_bg};
    border: 1px solid {group_border};
    border-radius: 12px;
    margin-top: 14px;
    padding: 16px 14px 14px 14px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: {group_title};
    background: transparent;
    font-size: 14px;
}}

QLineEdit,
QPlainTextEdit,
QTextEdit,
QSpinBox,
QComboBox {{
    background: {editor_bg};
    border: 1px solid {input_border};
    border-radius: 10px;
    padding: 10px 12px;
    selection-background-color: {selection_bg};
    selection-color: {selection_text};
}}

QPlainTextEdit,
QTextEdit {{
    padding: 12px;
}}

QLineEdit:focus,
QPlainTextEdit:focus,
QTextEdit:focus,
QSpinBox:focus,
QComboBox:focus {{
    border: 1px solid {focus_border};
}}

QComboBox::drop-down {{
    border: none;
    width: 28px;
    background: transparent;
}}

QComboBox::down-arrow {{
    width: 10px;
    height: 10px;
}}

QComboBox QAbstractItemView {{
    background: {combo_popup_bg};
    color: {text};
    border: 1px solid {input_border};
    selection-background-color: {selection_bg};
    selection-color: {selection_text};
}}

QPushButton {{
    background: {button_bg};
    color: {text};
    border: 1px solid {button_border};
    border-radius: 10px;
    padding: 9px 14px;
    font-weight: 600;
}}

QPushButton:hover {{
    background: {button_hover};
    border: 1px solid {button_hover_border};
}}

QPushButton:pressed {{
    background: {button_pressed};
    border: 1px solid {button_pressed_border};
}}

QPushButton:disabled {{
    background: {button_disabled_bg};
    color: {button_disabled_text};
    border: 1px solid {button_disabled_border};
}}

QLabel {{
    color: {text};
    background: transparent;
}}

QLabel#HeroTitle {{
    color: {hero_title};
    font-size: 28px;
    font-weight: 800;
}}

QLabel#HeroSubtitle,
QLabel#SectionHint,
QLabel#VirtualHint {{
    color: {muted_text};
    font-size: 12px;
}}

QFrame#TopHeader {{
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 {hero_card_bg},
        stop: 1 {surface_bg}
    );
    border: 1px solid {hero_card_border};
    border-radius: 14px;
}}

QLabel#HeroEyebrow {{
    color: {accent_soft};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}}

QLabel#StatChip {{
    color: {hero_chip_text};
    background: {hero_chip_bg};
    border: 1px solid {hero_chip_border};
    border-radius: 8px;
    padding: 4px 8px;
    font-size: 12px;
    font-weight: 600;
}}

QLabel#SectionLead {{
    color: {text};
    font-size: 15px;
    font-weight: 600;
}}

QFrame#SummaryCard,
QFrame#DetailCard,
QFrame#DashboardTile {{
    background: {surface_bg_alt};
    border: 1px solid {surface_border};
    border-radius: 12px;
}}

QFrame#DashboardTile:hover {{
    border: 1px solid {button_hover_border};
}}

QFrame#ControlDeck,
QFrame#ContentCard {{
    background: {surface_bg};
    border: 1px solid {surface_border};
    border-radius: 14px;
}}

QLabel#SummaryCaption,
QLabel#DetailCaption,
QLabel#DashboardCaption {{
    color: {muted_text};
    font-size: 11px;
    font-weight: 600;
}}

QLabel#SummaryValue,
QLabel#DetailValue,
QLabel#DashboardValue {{
    color: {text};
    font-size: 14px;
    font-weight: 700;
}}

QLabel#CompactLabel {{
    color: {muted_text};
    font-size: 11px;
    font-weight: 600;
    padding-left: 2px;
}}

QLabel#StatusBadge {{
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 {primary_start},
        stop: 1 {primary_end}
    );
    color: #ffffff;
    border: 1px solid {primary_border};
    border-radius: 16px;
    padding: 8px 14px;
    font-size: 12px;
    font-weight: 700;
    min-width: 64px;
}}

QFrame#EmotionPanelCard,
QFrame#EmotionSliderCard,
QFrame#EmotionMixerCard {{
    background: {surface_bg_alt};
    border: 1px solid {surface_border};
    border-radius: 12px;
}}

QFrame#EmotionSidebar {{
    border-right: 1px solid {surface_border};
    padding-right: 12px;
}}

QFrame#EmotionSliderCard:hover {{
    border: 1px solid {button_hover_border};
}}

QFrame#WorkspaceCanvas {{
    background: {surface_bg_alt};
    border: 1px solid {surface_border};
    border-radius: 14px;
}}

QLabel#WorkspaceBadge {{
    background: {hero_chip_bg};
    color: {hero_chip_text};
    border: 1px solid {hero_chip_border};
    border-radius: 9px;
    padding: 7px 12px;
    font-size: 12px;
    font-weight: 700;
    min-width: 74px;
}}

QLabel#WorkspaceStat {{
    color: {muted_text};
    font-size: 11px;
    font-weight: 600;
}}

QLabel#CardTitle {{
    color: {group_title};
    font-size: 16px;
    font-weight: 700;
}}

QLabel#SectionHint {{
    line-height: 1.5;
}}

QLabel#LiveStatusLabel {{
    background: {warning_bg};
    color: {warning_text};
    border: 1px solid {warning_border};
    border-radius: 10px;
    padding: 12px;
    font-size: 15px;
    font-weight: 700;
}}

QLabel#LiveStatusLabel[state="ready"] {{
    background: {success_bg};
    color: {success_text};
    border: 1px solid {success_border};
}}

QLabel#LiveStatusLabel[state="error"] {{
    background: {error_bg};
    color: {error_text};
    border: 1px solid {error_border};
}}

QLabel#ModeBadge {{
    background: {hero_chip_bg};
    color: {hero_chip_text};
    border: 1px solid {hero_chip_border};
    border-radius: 9px;
    padding: 8px 12px;
    font-size: 12px;
    font-weight: 700;
    min-width: 86px;
}}

QPushButton#ModeToggleButton {{
    background: {surface_bg_alt};
    color: {muted_text};
    border: 1px solid {surface_border};
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 12px;
    font-weight: 700;
}}

QPushButton#ModeToggleButton:hover {{
    color: {text};
    border: 1px solid {button_hover_border};
}}

QPushButton#ModeToggleButton:checked {{
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 {primary_start},
        stop: 1 {primary_end}
    );
    color: #ffffff;
    border: 1px solid {primary_border};
}}

QTextEdit#TranscriptView,
QPlainTextEdit#LogView,
QFrame#WaveformCard {{
    background: {log_bg};
    border: 1px solid {group_border};
    border-radius: 12px;
}}

QTextEdit#TranscriptView {{
    padding: 10px 12px;
    selection-background-color: {primary_start};
}}

QLabel#EmotionSliderName {{
    color: {text};
    font-size: 12px;
    font-weight: 600;
}}

QLabel#EmotionSliderValue {{
    color: {muted_text};
    font-size: 13px;
    font-weight: 700;
    min-width: 40px;
}}

QPushButton#PrimaryActionButton {{
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 {primary_start},
        stop: 1 {primary_end}
    );
    color: #ffffff;
    border: 1px solid {primary_border};
    font-size: 16px;
    padding: 14px 18px;
}}

QPushButton#PrimaryActionButton:hover {{
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 {primary_hover_start},
        stop: 1 {primary_hover_end}
    );
    border: 1px solid {primary_hover_border};
}}

QPushButton#SecondaryActionButton {{
    min-width: 72px;
    padding: 8px 10px;
}}

QPushButton#IconButton {{
    min-width: 0;
    min-height: 0;
    padding: 8px;
    border-radius: 10px;
}}

QPushButton#EmotionPresetButton {{
    background: {surface_bg};
    color: {text};
    border: 1px solid {surface_border};
    border-radius: 10px;
    padding: 8px 10px;
    font-size: 12px;
    font-weight: 700;
}}

QPushButton#EmotionPresetButton:hover {{
    border: 1px solid {button_hover_border};
}}

QPushButton#EmotionPresetButton:checked {{
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 {primary_start},
        stop: 1 {primary_end}
    );
    color: #ffffff;
    border: 1px solid {primary_border};
}}

QPushButton#EmotionNavButton {{
    background: transparent;
    color: {muted_text};
    border: none;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: left;
    font-size: 14px;
    font-weight: 600;
}}

QPushButton#EmotionNavButton:hover {{
    background: {surface_bg_alt};
    color: {text};
}}

QPushButton#EmotionNavButton:checked {{
    background: {surface_bg_alt};
    color: {primary_start};
    border-left: 3px solid {primary_start};
    border-radius: 4px;
    font-weight: 700;
}}

QTabWidget#ActivityTabs::pane,
QTabWidget#SettingsTabs::pane {{
    border: 1px solid {group_border};
    border-radius: 12px;
    background: {surface_bg_alt};
    top: -1px;
}}

QTabWidget#ActivityTabs QWidget,
QTabWidget#SettingsTabs QWidget {{
    background: transparent;
}}

QTabBar::tab {{
    background: transparent;
    color: {muted_text};
    border: 1px solid transparent;
    border-bottom: none;
    padding: 9px 14px;
    margin-right: 4px;
    font-weight: 600;
}}

QTabBar::tab:selected {{
    color: {text};
    background: {surface_bg_alt};
    border: 1px solid {group_border};
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
}}

QTabBar::tab:hover:!selected {{
    color: {text};
}}

QSlider::groove:horizontal {{
    height: 6px;
    border-radius: 3px;
    background: {button_disabled_border};
}}

QSlider::sub-page:horizontal {{
    border-radius: 3px;
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 {primary_start},
        stop: 1 {primary_end}
    );
}}

QSlider::handle:horizontal {{
    background: {hero_title};
    border: 2px solid {primary_border};
    width: 18px;
    height: 18px;
    margin: -7px 0;
    border-radius: 7px;
}}

QSlider::groove:vertical {{
    background: {surface_border};
    width: 6px;
    border-radius: 3px;
}}

QSlider::add-page:vertical {{
    background: qlineargradient(
        x1: 0, y1: 1, x2: 0, y2: 0,
        stop: 0 {primary_start},
        stop: 1 {primary_end}
    );
    width: 6px;
    border-radius: 3px;
}}

QSlider::sub-page:vertical {{
    background: {surface_border};
    width: 6px;
    border-radius: 3px;
}}

QSlider::handle:vertical {{
    background: {hero_title};
    border: 2px solid {primary_border};
    width: 18px;
    height: 18px;
    margin: 0 -6px;
    border-radius: 9px;
}}

QSplitter::handle {{
    background: {splitter};
    border-radius: 3px;
}}

QSplitter::handle:horizontal {{
    width: 6px;
    margin: 6px 0;
}}

QSplitter::handle:vertical {{
    height: 6px;
    margin: 0 6px;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 12px;
    margin: 4px 0 4px 0;
}}

QScrollBar::handle:vertical {{
    background: {scroll_handle};
    border-radius: 4px;
    min-height: 24px;
}}

QScrollBar::handle:vertical:hover {{
    background: {scroll_handle_hover};
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: transparent;
    border: none;
}}
""".format(**palette)


APP_QSS = get_theme_stylesheet(DEFAULT_THEME)
