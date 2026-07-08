import ctypes
import sys
from ctypes import wintypes

from PySide6.QtWidgets import QWidget


DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19
DWMWA_BORDER_COLOR = 34
DWMWA_CAPTION_COLOR = 35
DWMWA_TEXT_COLOR = 36
DWMWA_COLOR_DEFAULT = 0xFFFFFFFF

_THEME_CHROME_COLORS: dict[str, dict[str, str]] = {
    "dark": {
        "caption": "#0f1420",
        "text": "#f8fafc",
        "border": "#24314a",
    },
    "light": {
        "caption": "#ffffff",
        "text": "#0f172a",
        "border": "#dbe5f5",
    },
}


def _hex_to_colorref(hex_color: str) -> int:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Unexpected hex color: {hex_color}")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return red | (green << 8) | (blue << 16)


def _set_dwm_attribute(hwnd: int, attribute: int, value: int) -> None:
    dwmapi = ctypes.windll.dwmapi
    attr_value = wintypes.DWORD(value)
    dwmapi.DwmSetWindowAttribute(
        wintypes.HWND(hwnd),
        ctypes.c_uint(attribute),
        ctypes.byref(attr_value),
        ctypes.sizeof(attr_value),
    )


def apply_window_chrome_theme(window: QWidget, theme_name: str) -> None:
    if sys.platform != "win32":
        return
    if not isinstance(window, QWidget):
        return

    hwnd = int(window.winId())
    colors = _THEME_CHROME_COLORS.get(theme_name, _THEME_CHROME_COLORS["dark"])
    use_dark_mode = 1 if theme_name == "dark" else 0

    try:
        _set_dwm_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, use_dark_mode)
    except Exception:
        try:
            _set_dwm_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1, use_dark_mode)
        except Exception:
            return

    try:
        _set_dwm_attribute(hwnd, DWMWA_CAPTION_COLOR, _hex_to_colorref(colors["caption"]))
        _set_dwm_attribute(hwnd, DWMWA_TEXT_COLOR, _hex_to_colorref(colors["text"]))
        _set_dwm_attribute(hwnd, DWMWA_BORDER_COLOR, _hex_to_colorref(colors["border"]))
    except Exception:
        try:
            _set_dwm_attribute(hwnd, DWMWA_BORDER_COLOR, DWMWA_COLOR_DEFAULT)
        except Exception:
            pass
