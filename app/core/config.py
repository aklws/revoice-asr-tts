from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.paths import get_app_root

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv is optional at import time.
    load_dotenv = None

BASE_DIR = get_app_root()


def _load_project_dotenv() -> None:
    if load_dotenv is None:
        return
    load_dotenv(BASE_DIR / ".env", override=False)


_load_project_dotenv()

CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
DEFAULT_ASR_MODEL_DIR = CHECKPOINTS_DIR / "Fun-ASR-Nano-2512"
DEFAULT_ASR_MODEL_REPO = "FunAudioLLM/Fun-ASR-Nano-2512"
DEFAULT_SPEAKER_MODEL_DIR = CHECKPOINTS_DIR / "speech_campplus_sv_zh-cn_16k-common"
DEFAULT_SPEAKER_MODEL_REPO = "iic/speech_campplus_sv_zh-cn_16k-common"
DEFAULT_SPEAKER_MODEL_REVISION = "v1.0.0"
DEFAULT_INDEX_TTS_MODEL_DIR = CHECKPOINTS_DIR / "IndexTTS-2"
DEFAULT_INDEX_TTS_MODEL_REPO = "IndexTeam/IndexTTS-2"
SUPPORTED_INDEX_TTS_MODEL_REPOS: tuple[str, ...] = (
    "IndexTeam/IndexTTS-2",
    "xmccln/IndexTTS-2-Japanese",
)
DEFAULT_OUTPUT_VOICE_LANGUAGE = "zh"
DEFAULT_EMOTION_MODEL_DIR = CHECKPOINTS_DIR / "emotion2vec_plus_base"
DEFAULT_TTS_DEVICE = "cuda"
DEFAULT_ASR_DEVICE = "cpu"
DEFAULT_EMOTION_DEVICE = "cpu"
DEFAULT_TTS_MAX_NEW_TOKENS = 2048
MIN_LIVE_IDLE_UNLOAD_SEC = 3600.0
DEFAULT_LIVE_ASR_IDLE_UNLOAD_SEC = MIN_LIVE_IDLE_UNLOAD_SEC
DEFAULT_LIVE_TTS_IDLE_UNLOAD_SEC = MIN_LIVE_IDLE_UNLOAD_SEC
DEFAULT_EXPERIMENTAL_TORCH_COMPILE_ENABLED = False
DEFAULT_EXPERIMENTAL_ACCEL_GPT_ENABLED = False
DEFAULT_TRANSLATION_ENABLED = False
RUNTIME_SETTINGS_FILE = BASE_DIR / "settings.json"


@dataclass(frozen=True, slots=True)
class TranslationNodeConfig:
    node_id: str
    name: str
    provider: str
    base_url: str
    model: str
    api_key: str


@dataclass(frozen=True, slots=True)
class AppSettings:
    checkpoints_dir: Path
    asr_model_dir: Path
    asr_model_repo: str
    speaker_model_dir: Path
    speaker_model_repo: str
    speaker_model_revision: str
    index_tts_model_dir: Path
    index_tts_model_repo: str
    supported_index_tts_model_repos: tuple[str, ...]
    output_voice_language: str
    emotion_model_dir: Path
    tts_device: str
    asr_device: str
    emotion_device: str
    tts_max_new_tokens: int
    live_asr_idle_unload_sec: float
    live_tts_idle_unload_sec: float
    translation_enabled: bool
    translation_node_id: str | None
    translation_nodes: tuple[TranslationNodeConfig, ...]
    experimental_torch_compile_enabled: bool
    experimental_accel_gpt_enabled: bool


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


def _parse_optional_str(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    return value or None


def _resolve_configured_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def _parse_repo_list(raw: object) -> tuple[str, ...]:
    items: list[str] = []
    if isinstance(raw, str):
        normalized = raw.replace(";", "\n").replace(",", "\n")
        items = [part.strip() for part in normalized.splitlines()]
    elif isinstance(raw, (list, tuple)):
        items = [str(part).strip() for part in raw]

    repos: list[str] = []
    seen: set[str] = set()
    for repo in items:
        if not repo or repo in seen:
            continue
        repos.append(repo)
        seen.add(repo)
    return tuple(repos)


def normalize_output_voice_language(raw: str | None) -> str:
    normalized = str(raw or "").strip().casefold()
    mapping = {
        "zh": "zh",
        "zh-cn": "zh",
        "chinese": "zh",
        "中文": "zh",
        "cn": "zh",
        "en": "en",
        "english": "en",
        "英文": "en",
        "ja": "ja",
        "jp": "ja",
        "japanese": "ja",
        "日语": "ja",
        "日文": "ja",
    }
    return mapping.get(normalized, DEFAULT_OUTPUT_VOICE_LANGUAGE)


def index_tts_repo_for_output_voice_language(language: str) -> str:
    normalized = normalize_output_voice_language(language)
    if normalized == "ja":
        return "xmccln/IndexTTS-2-Japanese"
    return "IndexTeam/IndexTTS-2"


def _infer_output_voice_language_from_repo(repo: str) -> str:
    if repo.strip() == "xmccln/IndexTTS-2-Japanese":
        return "ja"
    return DEFAULT_OUTPUT_VOICE_LANGUAGE


def _parse_translation_nodes(raw: object) -> tuple[TranslationNodeConfig, ...]:
    if not isinstance(raw, list):
        return ()
    nodes: list[TranslationNodeConfig] = []
    seen_ids: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        node_id = _parse_optional_str(item.get("node_id")) or _parse_optional_str(item.get("id"))
        name = _parse_optional_str(item.get("name"))
        provider = _parse_optional_str(item.get("provider")) or "openai-compatible"
        base_url = _parse_optional_str(item.get("base_url"))
        model = _parse_optional_str(item.get("model"))
        api_key = _parse_optional_str(item.get("api_key")) or ""
        if not node_id or not name or not base_url or not model or node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        nodes.append(
            TranslationNodeConfig(
                node_id=node_id,
                name=name,
                provider=provider,
                base_url=base_url,
                model=model,
                api_key=api_key,
            )
        )
    return tuple(nodes)


def resolve_asr_model_dir() -> Path:
    configured = (
        os.getenv("FUNASR_ASR_MODEL_DIR")
        or os.getenv("ASR_MODEL_DIR")
        or os.getenv("QWEN_ASR_MODEL_DIR")
    )
    if configured:
        return _resolve_configured_path(configured)
    return DEFAULT_ASR_MODEL_DIR


def resolve_asr_model_repo() -> str:
    configured = (
        _parse_optional_str(os.getenv("FUNASR_ASR_MODEL_REPO"))
        or _parse_optional_str(os.getenv("ASR_MODEL_REPO"))
    )
    if configured is not None:
        return configured
    settings_value = _parse_optional_str(_load_runtime_settings().get("asr_model_repo"))
    if settings_value is not None:
        return settings_value
    return DEFAULT_ASR_MODEL_REPO


def resolve_speaker_model_dir() -> Path:
    configured = os.getenv("SPEAKER_MODEL_DIR")
    if configured:
        return _resolve_configured_path(configured)
    return DEFAULT_SPEAKER_MODEL_DIR


def resolve_speaker_model_repo() -> str:
    configured = _parse_optional_str(os.getenv("SPEAKER_MODEL_REPO"))
    if configured is not None:
        return configured
    settings_value = _parse_optional_str(_load_runtime_settings().get("speaker_model_repo"))
    if settings_value is not None:
        return settings_value
    return DEFAULT_SPEAKER_MODEL_REPO


def resolve_speaker_model_revision() -> str:
    configured = _parse_optional_str(os.getenv("SPEAKER_MODEL_REVISION"))
    if configured is not None:
        return configured
    settings_value = _parse_optional_str(_load_runtime_settings().get("speaker_model_revision"))
    if settings_value is not None:
        return settings_value
    return DEFAULT_SPEAKER_MODEL_REVISION


def resolve_index_tts_model_repo() -> str:
    env_value = _parse_optional_str(os.getenv("INDEX_TTS_MODEL_REPO"))
    if env_value is not None:
        return env_value
    settings_value = _parse_optional_str(_load_runtime_settings().get("index_tts_model_repo"))
    if settings_value is not None:
        return settings_value
    return DEFAULT_INDEX_TTS_MODEL_REPO


def resolve_output_voice_language() -> str:
    env_value = _parse_optional_str(os.getenv("OUTPUT_VOICE_LANGUAGE"))
    if env_value is not None:
        return normalize_output_voice_language(env_value)
    settings = _load_runtime_settings()
    settings_value = _parse_optional_str(settings.get("output_voice_language"))
    if settings_value is not None:
        return normalize_output_voice_language(settings_value)
    return _infer_output_voice_language_from_repo(resolve_index_tts_model_repo())


def _repo_name_to_dir_name(repo: str) -> str:
    repo_name = repo.rsplit("/", 1)[-1].strip()
    safe_chars = []
    for char in repo_name:
        if char.isalnum() or char in {"-", "_", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("-")
    sanitized = "".join(safe_chars).strip("-._")
    return sanitized or DEFAULT_INDEX_TTS_MODEL_DIR.name


def default_index_tts_model_dir(repo: str | None = None) -> Path:
    effective_repo = repo or DEFAULT_INDEX_TTS_MODEL_REPO
    if effective_repo == DEFAULT_INDEX_TTS_MODEL_REPO:
        return DEFAULT_INDEX_TTS_MODEL_DIR
    return CHECKPOINTS_DIR / _repo_name_to_dir_name(effective_repo)


def resolve_supported_index_tts_model_repos() -> tuple[str, ...]:
    env_repos = _parse_repo_list(os.getenv("INDEX_TTS_MODEL_REPOS"))
    settings_repos = _parse_repo_list(_load_runtime_settings().get("supported_index_tts_model_repos"))
    repos: list[str] = []
    seen: set[str] = set()
    for repo in (*SUPPORTED_INDEX_TTS_MODEL_REPOS, *settings_repos, *env_repos, resolve_index_tts_model_repo()):
        normalized = repo.strip()
        if not normalized or normalized in seen:
            continue
        repos.append(normalized)
        seen.add(normalized)
    return tuple(repos)


def set_index_tts_model_repo(repo: str) -> Path:
    normalized = repo.strip()
    if not normalized:
        raise ValueError("TTS 模型仓库不能为空。")
    settings = _load_runtime_settings()
    settings["index_tts_model_repo"] = normalized
    # 当前 UI 未提供自定义 TTS 模型目录，切换仓库时优先回到该仓库的默认目录。
    settings.pop("index_tts_model_dir", None)
    _save_runtime_settings(settings)
    get_settings.cache_clear()
    return RUNTIME_SETTINGS_FILE


def set_supported_index_tts_model_repos(repos: list[str] | tuple[str, ...]) -> Path:
    normalized = _parse_repo_list(list(repos))
    if not normalized:
        raise ValueError("至少需要保留一个 TTS 模型仓库。")
    settings = _load_runtime_settings()
    settings["supported_index_tts_model_repos"] = list(normalized)
    _save_runtime_settings(settings)
    get_settings.cache_clear()
    return RUNTIME_SETTINGS_FILE


def set_output_voice_language(language: str) -> Path:
    normalized = normalize_output_voice_language(language)
    settings = _load_runtime_settings()
    settings["output_voice_language"] = normalized
    settings["index_tts_model_repo"] = index_tts_repo_for_output_voice_language(normalized)
    settings.pop("index_tts_model_dir", None)
    _save_runtime_settings(settings)
    get_settings.cache_clear()
    return RUNTIME_SETTINGS_FILE


def resolve_index_tts_model_dir() -> Path:
    configured = _parse_optional_str(os.getenv("INDEX_TTS_MODEL_DIR"))
    if configured:
        return _resolve_configured_path(configured)
    settings_value = _parse_optional_str(_load_runtime_settings().get("index_tts_model_dir"))
    if settings_value is not None:
        return _resolve_configured_path(settings_value)
    return default_index_tts_model_dir(resolve_index_tts_model_repo())


def resolve_emotion_model_dir() -> Path:
    configured = os.getenv("EMOTION_MODEL_DIR")
    if configured:
        return _resolve_configured_path(configured)
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


def resolve_translation_enabled() -> bool:
    env_value = _parse_optional_bool(os.getenv("TRANSLATION_ENABLED"))
    if env_value is not None:
        return env_value
    settings_value = _parse_optional_bool(_load_runtime_settings().get("translation_enabled"))
    if settings_value is not None:
        return settings_value
    return DEFAULT_TRANSLATION_ENABLED


def set_translation_enabled(enabled: bool) -> Path:
    settings = _load_runtime_settings()
    settings["translation_enabled"] = bool(enabled)
    _save_runtime_settings(settings)
    get_settings.cache_clear()
    return RUNTIME_SETTINGS_FILE


def resolve_translation_node_id() -> str | None:
    settings_value = _parse_optional_str(_load_runtime_settings().get("translation_node_id"))
    return settings_value


def set_translation_node_id(node_id: str | None) -> Path:
    settings = _load_runtime_settings()
    normalized = _parse_optional_str(node_id)
    if normalized is None:
        settings.pop("translation_node_id", None)
    else:
        settings["translation_node_id"] = normalized
    _save_runtime_settings(settings)
    get_settings.cache_clear()
    return RUNTIME_SETTINGS_FILE


def resolve_translation_nodes() -> tuple[TranslationNodeConfig, ...]:
    return _parse_translation_nodes(_load_runtime_settings().get("translation_nodes"))


def set_translation_nodes(nodes: list[dict[str, object]] | tuple[dict[str, object], ...]) -> Path:
    normalized_nodes = _parse_translation_nodes(list(nodes))
    settings = _load_runtime_settings()
    settings["translation_nodes"] = [
        {
            "node_id": node.node_id,
            "name": node.name,
            "provider": node.provider,
            "base_url": node.base_url,
            "model": node.model,
            "api_key": node.api_key,
        }
        for node in normalized_nodes
    ]
    current_id = _parse_optional_str(settings.get("translation_node_id"))
    valid_ids = {node.node_id for node in normalized_nodes}
    if current_id and current_id not in valid_ids:
        settings.pop("translation_node_id", None)
    _save_runtime_settings(settings)
    get_settings.cache_clear()
    return RUNTIME_SETTINGS_FILE


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


def resolve_experimental_accel_gpt_enabled() -> bool:
    env_value = _parse_optional_bool(os.getenv("EXPERIMENTAL_ACCEL_GPT"))
    if env_value is not None:
        return env_value
    settings_value = _parse_optional_bool(_load_runtime_settings().get("experimental_accel_gpt_enabled"))
    if settings_value is not None:
        return settings_value
    return DEFAULT_EXPERIMENTAL_ACCEL_GPT_ENABLED


def set_experimental_accel_gpt_enabled(enabled: bool) -> Path:
    settings = _load_runtime_settings()
    settings["experimental_accel_gpt_enabled"] = bool(enabled)
    _save_runtime_settings(settings)
    get_settings.cache_clear()
    return RUNTIME_SETTINGS_FILE


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings(
        checkpoints_dir=CHECKPOINTS_DIR,
        asr_model_dir=resolve_asr_model_dir(),
        asr_model_repo=resolve_asr_model_repo(),
        speaker_model_dir=resolve_speaker_model_dir(),
        speaker_model_repo=resolve_speaker_model_repo(),
        speaker_model_revision=resolve_speaker_model_revision(),
        index_tts_model_dir=resolve_index_tts_model_dir(),
        index_tts_model_repo=resolve_index_tts_model_repo(),
        supported_index_tts_model_repos=resolve_supported_index_tts_model_repos(),
        output_voice_language=resolve_output_voice_language(),
        emotion_model_dir=resolve_emotion_model_dir(),
        tts_device=resolve_tts_device(),
        asr_device=resolve_asr_device(),
        emotion_device=resolve_emotion_device(),
        tts_max_new_tokens=resolve_tts_max_new_tokens(),
        live_asr_idle_unload_sec=resolve_live_asr_idle_unload_sec(),
        live_tts_idle_unload_sec=resolve_live_tts_idle_unload_sec(),
        translation_enabled=resolve_translation_enabled(),
        translation_node_id=resolve_translation_node_id(),
        translation_nodes=resolve_translation_nodes(),
        experimental_torch_compile_enabled=resolve_experimental_torch_compile_enabled(),
        experimental_accel_gpt_enabled=resolve_experimental_accel_gpt_enabled(),
    )
