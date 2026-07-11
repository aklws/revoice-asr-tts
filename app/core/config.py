from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.paths import get_app_root

BASE_DIR = get_app_root()
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
DEFAULT_ASR_MODEL_DIR = CHECKPOINTS_DIR / "Qwen3-ASR-0.6B"
DEFAULT_INDEX_TTS_MODEL_DIR = CHECKPOINTS_DIR / "IndexTTS-2"
DEFAULT_EMOTION_MODEL_DIR = CHECKPOINTS_DIR / "emotion2vec_plus_base"
DEFAULT_TTS_DEVICE = "cuda"
DEFAULT_ASR_DEVICE = "cpu"
DEFAULT_EMOTION_DEVICE = "cpu"
DEFAULT_TTS_MAX_NEW_TOKENS = 2048
MIN_LIVE_IDLE_UNLOAD_SEC = 3600.0
DEFAULT_LIVE_ASR_IDLE_UNLOAD_SEC = MIN_LIVE_IDLE_UNLOAD_SEC
DEFAULT_LIVE_TTS_IDLE_UNLOAD_SEC = MIN_LIVE_IDLE_UNLOAD_SEC
DEFAULT_EXPERIMENTAL_TORCH_COMPILE_ENABLED = False
RUNTIME_SETTINGS_FILE = BASE_DIR / "settings.json"


@dataclass(frozen=True, slots=True)
class AppSettings:
    checkpoints_dir: Path
    asr_model_dir: Path
    index_tts_model_dir: Path
    emotion_model_dir: Path
    tts_device: str
    asr_device: str
    emotion_device: str
    tts_max_new_tokens: int
    live_asr_idle_unload_sec: float
    live_tts_idle_unload_sec: float
    experimental_torch_compile_enabled: bool


def _load_runtime_settings() -> dict[str, object]:
    if not RUNTIME_SETTINGS_FILE.exists():
        return {}
    try:
        with RUNTIME_SETTINGS_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_runtime_settings(data: dict[str, object]) -> None:
    RUNTIME_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RUNTIME_SETTINGS_FILE.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)


def _parse_optional_bool(raw: object) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return bool(raw)
    if not isinstance(raw, str):
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def resolve_asr_model_dir() -> Path:
    configured = os.getenv("QWEN_ASR_MODEL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_ASR_MODEL_DIR


def resolve_index_tts_model_dir() -> Path:
    configured = os.getenv("INDEX_TTS_MODEL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_INDEX_TTS_MODEL_DIR


def resolve_emotion_model_dir() -> Path:
    configured = os.getenv("EMOTION_MODEL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_EMOTION_MODEL_DIR


def resolve_tts_device() -> str:
    return str(os.getenv("TTS_DEVICE", DEFAULT_TTS_DEVICE)).strip()


def resolve_asr_device() -> str:
    return str(os.getenv("ASR_DEVICE", DEFAULT_ASR_DEVICE)).strip()


def resolve_emotion_device() -> str:
    return str(os.getenv("EMOTION_DEVICE", DEFAULT_EMOTION_DEVICE)).strip()


def resolve_tts_max_new_tokens() -> int:
    raw = os.getenv("TTS_MAX_NEW_TOKENS")
    if raw is None or not raw.strip():
        return DEFAULT_TTS_MAX_NEW_TOKENS
    return max(1, int(raw))


def _resolve_positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return max(0.0, float(raw))


def resolve_live_asr_idle_unload_sec() -> float:
    return max(
        MIN_LIVE_IDLE_UNLOAD_SEC,
        _resolve_positive_float_env("LIVE_ASR_IDLE_UNLOAD_SEC", DEFAULT_LIVE_ASR_IDLE_UNLOAD_SEC),
    )


def resolve_live_tts_idle_unload_sec() -> float:
    return max(
        MIN_LIVE_IDLE_UNLOAD_SEC,
        _resolve_positive_float_env("LIVE_TTS_IDLE_UNLOAD_SEC", DEFAULT_LIVE_TTS_IDLE_UNLOAD_SEC),
    )


def resolve_experimental_torch_compile_enabled() -> bool:
    env_value = _parse_optional_bool(os.getenv("EXPERIMENTAL_TORCH_COMPILE"))
    if env_value is not None:
        return env_value
    settings_value = _parse_optional_bool(_load_runtime_settings().get("experimental_torch_compile_enabled"))
    if settings_value is not None:
        return settings_value
    return DEFAULT_EXPERIMENTAL_TORCH_COMPILE_ENABLED


def set_experimental_torch_compile_enabled(enabled: bool) -> Path:
    settings = _load_runtime_settings()
    settings["experimental_torch_compile_enabled"] = bool(enabled)
    _save_runtime_settings(settings)
    get_settings.cache_clear()
    return RUNTIME_SETTINGS_FILE


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings(
        checkpoints_dir=CHECKPOINTS_DIR,
        asr_model_dir=resolve_asr_model_dir(),
        index_tts_model_dir=resolve_index_tts_model_dir(),
        emotion_model_dir=resolve_emotion_model_dir(),
        tts_device=resolve_tts_device(),
        asr_device=resolve_asr_device(),
        emotion_device=resolve_emotion_device(),
        tts_max_new_tokens=resolve_tts_max_new_tokens(),
        live_asr_idle_unload_sec=resolve_live_asr_idle_unload_sec(),
        live_tts_idle_unload_sec=resolve_live_tts_idle_unload_sec(),
        experimental_torch_compile_enabled=resolve_experimental_torch_compile_enabled(),
    )
