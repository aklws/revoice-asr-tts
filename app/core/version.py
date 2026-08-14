from __future__ import annotations

from functools import cache
from pathlib import Path
import tomllib

from app.core.paths import get_app_root


def _resolve_pyproject_path() -> Path:
    return get_app_root() / "pyproject.toml"


@cache
def get_project_version() -> str:
    pyproject_path = _resolve_pyproject_path()
    try:
        with pyproject_path.open("rb") as fh:
            data = tomllib.load(fh)
        version = str(data.get("project", {}).get("version", "")).strip()
        if version:
            return version
    except Exception:
        pass
    return "0.0.0"


@cache
def get_app_display_name() -> str:
    return f"Revoice ASR-TTS v{get_project_version()}"
