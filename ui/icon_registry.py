from __future__ import annotations

from functools import lru_cache
import xml.etree.ElementTree as ET

from PySide6.QtCore import QByteArray
from PySide6.QtGui import QIcon, QPixmap

from app.core.paths import get_bundle_file


ICON_REGISTRY_XML_PATH = get_bundle_file("asset", "icons", "icons.xml")


@lru_cache(maxsize=1)
def _load_icon_registry() -> dict[str, str]:
    try:
        root = ET.fromstring(ICON_REGISTRY_XML_PATH.read_text(encoding="utf-8"))
    except (OSError, ET.ParseError):
        return {}

    registry: dict[str, str] = {}
    for icon_node in root.findall("icon"):
        icon_name = str(icon_node.get("name") or "").strip()
        if not icon_name:
            continue
        svg_text = (icon_node.text or "").strip()
        if not svg_text:
            continue
        registry[icon_name] = svg_text
    return registry


def _render_svg_icon(svg: str) -> QIcon:
    pixmap = QPixmap()
    pixmap.loadFromData(QByteArray(svg.encode("utf-8")), "SVG")
    return QIcon(pixmap)


def get_svg_template(name: str) -> str:
    registry = _load_icon_registry()
    return registry[name]


def get_registered_icon(name: str) -> QIcon:
    return _render_svg_icon(get_svg_template(name))


def render_svg_template_icon(name: str, **replacements: str) -> QIcon:
    svg = get_svg_template(name)
    for key, value in replacements.items():
        svg = svg.replace(f"{{{key}}}", value)
    return _render_svg_icon(svg)
