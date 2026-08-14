from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.paths import get_app_root
from app.core.logging import get_logger

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv is optional at import time.
    load_dotenv = None

BASE_DIR = get_app_root()
logger = get_logger(__name__)


def _load_project_dotenv() -> None:
    if load_dotenv is None:
        return
    load_dotenv(BASE_DIR / ".env", override=False)


_load_project_dotenv()

CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
DEFAULT_ASR_MODEL_DIR = CHECKPOINTS_DIR / "Fun-ASR-Nano-2512-gguf"
DEFAULT_ASR_MODEL_REPO = "aklwss/Fun-ASR-Nano-2512-gguf"
SUPPORTED_ASR_MODEL_REPOS: tuple[str, ...] = (
    "aklwss/Fun-ASR-Nano-2512-gguf",
)
DEFAULT_INDEX_TTS_MODEL_DIR = CHECKPOINTS_DIR / "IndexTTS-2.5"
DEFAULT_INDEX_TTS_MODEL_REPO = "IndexTeam/IndexTTS-2.5"
DEFAULT_VOICE_OUTPUT_MODEL = "meanvc2"
SUPPORTED_INDEX_TTS_MODEL_REPOS: tuple[str, ...] = (
    "IndexTeam/IndexTTS-2.5",
)
LEGACY_INDEX_TTS_MODEL_REPOS: tuple[str, ...] = (
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
DEFAULT_TRANSLATION_TARGET_LANGUAGE = DEFAULT_OUTPUT_VOICE_LANGUAGE
PROJECT_ENV_FILE = BASE_DIR / ".env"
MODEL_SETTINGS_FILE = BASE_DIR / "models.json"
LEGACY_PROGRAM_STATE_FILE = BASE_DIR / "runtime_state.json"
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
    supported_asr_model_repos: tuple[str, ...]
    index_tts_model_dir: Path
    index_tts_model_repo: str
    supported_index_tts_model_repos: tuple[str, ...]
    voice_output_model: str
    output_voice_language: str
    emotion_model_dir: Path
    tts_device: str
    asr_device: str
    emotion_device: str
    tts_max_new_tokens: int
    live_asr_idle_unload_sec: float
    live_tts_idle_unload_sec: float
    translation_enabled: bool
    translation_target_language: str
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
    if not isinstance(data, dict):
        return {}
    migrated = _migrate_translation_node_secrets(data)
    if migrated:
        _save_runtime_settings(data)
    return data


def _save_runtime_settings(data: dict[str, object]) -> None:
    RUNTIME_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RUNTIME_SETTINGS_FILE.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)


def _save_model_settings(data: dict[str, object]) -> None:
    MODEL_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_SETTINGS_FILE.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)


def _load_model_settings() -> dict[str, object]:
    try:
        if MODEL_SETTINGS_FILE.exists():
            with MODEL_SETTINGS_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            model_settings = data if isinstance(data, dict) else {}
        else:
            model_settings = {}
    except Exception:
        model_settings = {}

    runtime_settings = _load_runtime_settings()
    legacy_program_state: dict[str, object] = {}
    try:
        if LEGACY_PROGRAM_STATE_FILE.exists():
            with LEGACY_PROGRAM_STATE_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            legacy_program_state = data if isinstance(data, dict) else {}
    except Exception:
        legacy_program_state = {}

    migrated_model_settings = _build_migrated_model_settings(
        model_settings=model_settings,
        runtime_settings=runtime_settings,
        legacy_program_state=legacy_program_state,
    )
    if migrated_model_settings != model_settings:
        _save_model_settings(migrated_model_settings)
        model_settings = migrated_model_settings

    if _cleanup_legacy_model_keys(runtime_settings):
        _save_runtime_settings(runtime_settings)

    if LEGACY_PROGRAM_STATE_FILE.exists():
        try:
            LEGACY_PROGRAM_STATE_FILE.unlink()
        except OSError:
            pass

    return model_settings


def _build_migrated_model_settings(
    *,
    model_settings: dict[str, object],
    runtime_settings: dict[str, object],
    legacy_program_state: dict[str, object],
) -> dict[str, object]:
    asr_repo = _normalize_asr_model_repo(model_settings.get("asr_model_repo"))
    if asr_repo is None:
        asr_repo = _normalize_asr_model_repo(runtime_settings.get("asr_model_repo"))
    if asr_repo is None:
        asr_repo = _normalize_asr_model_repo(os.getenv("FUNASR_ASR_MODEL_REPO") or os.getenv("ASR_MODEL_REPO"))
    if asr_repo is None:
        asr_repo = DEFAULT_ASR_MODEL_REPO

    asr_dir_raw = (
        _parse_optional_str(model_settings.get("asr_model_dir"))
        or _parse_optional_str(runtime_settings.get("asr_model_dir"))
        or _parse_optional_str(os.getenv("FUNASR_ASR_MODEL_DIR") or os.getenv("ASR_MODEL_DIR") or os.getenv("QWEN_ASR_MODEL_DIR"))
    )
    if asr_dir_raw:
        asr_dir = str(_resolve_configured_path(asr_dir_raw))
    elif asr_repo == DEFAULT_ASR_MODEL_REPO:
        asr_dir = str(DEFAULT_ASR_MODEL_DIR)
    else:
        asr_dir = str(CHECKPOINTS_DIR / _repo_name_to_asr_dir_name(asr_repo))

    supported_asr_repos: list[str] = []
    seen_asr_repos: set[str] = set()
    for repo in (
        *SUPPORTED_ASR_MODEL_REPOS,
        *_parse_repo_list(model_settings.get("supported_asr_model_repos")),
        *_parse_repo_list(runtime_settings.get("supported_asr_model_repos")),
        *_parse_repo_list(os.getenv("ASR_MODEL_REPOS")),
        asr_repo,
    ):
        normalized_repo = _normalize_asr_model_repo(repo)
        if not normalized_repo or normalized_repo in seen_asr_repos:
            continue
        supported_asr_repos.append(normalized_repo)
        seen_asr_repos.add(normalized_repo)

    raw_index_tts_repo = _parse_optional_str(model_settings.get("index_tts_model_repo"))
    if raw_index_tts_repo is None:
        raw_index_tts_repo = _parse_optional_str(runtime_settings.get("index_tts_model_repo"))
    if raw_index_tts_repo is None:
        raw_index_tts_repo = _parse_optional_str(os.getenv("INDEX_TTS_MODEL_REPO"))
    index_tts_repo = normalize_index_tts_model_repo(raw_index_tts_repo)
    if index_tts_repo is None:
        index_tts_repo = DEFAULT_INDEX_TTS_MODEL_REPO

    voice_output_model = _parse_optional_str(model_settings.get("voice_output_model"))
    if voice_output_model is None:
        voice_output_model = _parse_optional_str(runtime_settings.get("voice_output_model"))
    if voice_output_model is None:
        voice_output_model = _parse_optional_str(os.getenv("VOICE_OUTPUT_MODEL"))
    normalized_voice_output_model = normalize_voice_output_model(voice_output_model)

    output_voice_language = _parse_optional_str(model_settings.get("output_voice_language"))
    if output_voice_language is None:
        output_voice_language = _parse_optional_str(runtime_settings.get("output_voice_language"))
    if output_voice_language is None:
        output_voice_language = _parse_optional_str(os.getenv("OUTPUT_VOICE_LANGUAGE"))
    default_output_language = _infer_output_voice_language_from_repo(raw_index_tts_repo or index_tts_repo)
    if normalized_voice_output_model == "meanvc2":
        default_output_language = DEFAULT_OUTPUT_VOICE_LANGUAGE
    normalized_output_language = normalize_output_voice_language(output_voice_language or default_output_language)

    index_tts_dir_raw = (
        _parse_optional_str(model_settings.get("index_tts_model_dir"))
        or _parse_optional_str(legacy_program_state.get("index_tts_model_dir"))
        or _parse_optional_str(runtime_settings.get("index_tts_model_dir"))
        or _parse_optional_str(os.getenv("INDEX_TTS_MODEL_DIR"))
    )
    if index_tts_dir_raw:
        resolved_index_tts_dir = _resolve_configured_path(index_tts_dir_raw)
        legacy_default_dirs = {
            default_index_tts_model_dir("IndexTeam/IndexTTS-2"),
            default_index_tts_model_dir("xmccln/IndexTTS-2-Japanese"),
        }
        if resolved_index_tts_dir in legacy_default_dirs:
            index_tts_model_dir = str(default_index_tts_model_dir(index_tts_repo))
        else:
            index_tts_model_dir = str(resolved_index_tts_dir)
    else:
        index_tts_model_dir = str(default_index_tts_model_dir(index_tts_repo))

    supported_index_tts_repos: list[str] = []
    seen_tts_repos: set[str] = set()
    for repo in (
        *SUPPORTED_INDEX_TTS_MODEL_REPOS,
        *_parse_repo_list(model_settings.get("supported_index_tts_model_repos")),
        *_parse_repo_list(runtime_settings.get("supported_index_tts_model_repos")),
        *_parse_repo_list(os.getenv("INDEX_TTS_MODEL_REPOS")),
        index_tts_repo,
    ):
        normalized_repo = normalize_index_tts_model_repo(repo)
        if not normalized_repo or normalized_repo in seen_tts_repos:
            continue
        supported_index_tts_repos.append(normalized_repo)
        seen_tts_repos.add(normalized_repo)

    emotion_dir_raw = (
        _parse_optional_str(model_settings.get("emotion_model_dir"))
        or _parse_optional_str(runtime_settings.get("emotion_model_dir"))
        or _parse_optional_str(os.getenv("EMOTION_MODEL_DIR"))
    )
    emotion_model_dir = str(_resolve_configured_path(emotion_dir_raw)) if emotion_dir_raw else str(DEFAULT_EMOTION_MODEL_DIR)

    return {
        "asr_model_dir": asr_dir,
        "asr_model_repo": asr_repo,
        "supported_asr_model_repos": supported_asr_repos,
        "index_tts_model_dir": index_tts_model_dir,
        "index_tts_model_repo": index_tts_repo,
        "supported_index_tts_model_repos": supported_index_tts_repos,
        "voice_output_model": normalized_voice_output_model,
        "output_voice_language": normalized_output_language,
        "emotion_model_dir": emotion_model_dir,
    }


def _cleanup_legacy_model_keys(settings: dict[str, object]) -> bool:
    changed = False
    for key in (
        "asr_model_dir",
        "asr_model_repo",
        "supported_asr_model_repos",
        "index_tts_model_dir",
        "index_tts_model_repo",
        "supported_index_tts_model_repos",
        "voice_output_model",
        "output_voice_language",
        "emotion_model_dir",
    ):
        if key in settings:
            settings.pop(key, None)
            changed = True
    return changed


def _translation_node_env_var_name(node_id: str) -> str:
    normalized = re.sub(r"[^A-Z0-9_]", "_", str(node_id or "").upper()).strip("_")
    return f"TRANSLATION_NODE_{normalized or 'DEFAULT'}_API_KEY"


def _format_env_assignment_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _set_project_env_value(key: str, value: str | None) -> None:
    normalized_value = None if value is None else str(value).strip()
    env_lines: list[str] = []
    if PROJECT_ENV_FILE.exists():
        try:
            env_lines = PROJECT_ENV_FILE.read_text(encoding="utf-8").splitlines()
        except OSError:
            env_lines = []

    key_pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(key)}\s*=")
    updated_lines: list[str] = []
    found = False
    for line in env_lines:
        if key_pattern.match(line):
            found = True
            if normalized_value is not None:
                updated_lines.append(f"{key}={_format_env_assignment_value(normalized_value)}")
            continue
        updated_lines.append(line)

    if normalized_value is not None and not found:
        if updated_lines and updated_lines[-1].strip():
            updated_lines.append("")
        updated_lines.append(f"{key}={_format_env_assignment_value(normalized_value)}")

    PROJECT_ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROJECT_ENV_FILE.write_text("\n".join(updated_lines).rstrip() + "\n", encoding="utf-8")

    if normalized_value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = normalized_value


def _collect_translation_node_env_names(raw_nodes: object) -> set[str]:
    if not isinstance(raw_nodes, list):
        return set()
    env_names: set[str] = set()
    for item in raw_nodes:
        if not isinstance(item, dict):
            continue
        node_id = _parse_optional_str(item.get("node_id")) or _parse_optional_str(item.get("id"))
        if node_id:
            env_names.add(_translation_node_env_var_name(node_id))
    return env_names


def _migrate_translation_node_secrets(settings: dict[str, object]) -> bool:
    raw_nodes = settings.get("translation_nodes")
    if not isinstance(raw_nodes, list):
        return False

    migrated_nodes: list[object] = []
    changed = False
    for item in raw_nodes:
        if not isinstance(item, dict):
            migrated_nodes.append(item)
            continue

        node = dict(item)
        node_id = _parse_optional_str(node.get("node_id")) or _parse_optional_str(node.get("id"))
        api_key = _parse_optional_str(node.get("api_key")) or ""
        api_key_env = _parse_optional_str(node.get("api_key_env"))

        derived_env_key = _translation_node_env_var_name(node_id) if node_id else None
        target_env_key = api_key_env or derived_env_key

        if target_env_key:
            if api_key:
                _set_project_env_value(target_env_key, api_key)
            if "api_key_env" in node:
                node.pop("api_key_env", None)
                changed = True
        if "api_key" in node:
            node.pop("api_key", None)
            changed = True

        if node != item:
            changed = True
        migrated_nodes.append(node)

    if changed:
        settings["translation_nodes"] = migrated_nodes
    return changed


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
        "es": "es",
        "es-es": "es",
        "spanish": "es",
        "西班牙语": "es",
        "西语": "es",
        "ar": "ar",
        "ar-sa": "ar",
        "arabic": "ar",
        "阿拉伯语": "ar",
    }
    return mapping.get(normalized, DEFAULT_OUTPUT_VOICE_LANGUAGE)


def normalize_voice_output_model(raw: str | None) -> str:
    normalized = str(raw or "").strip().casefold()
    mapping = {
        "indextts": "indextts",
        "index-tts": "indextts",
        "index_tts": "indextts",
        "meanvc2": "meanvc2",
        "mean-vc2": "meanvc2",
        "mean_vc2": "meanvc2",
    }
    return mapping.get(normalized, DEFAULT_VOICE_OUTPUT_MODEL)


def normalize_index_tts_model_repo(raw: str | None) -> str | None:
    value = _parse_optional_str(raw)
    if value is None:
        return None
    if value in {DEFAULT_INDEX_TTS_MODEL_REPO, *LEGACY_INDEX_TTS_MODEL_REPOS}:
        return DEFAULT_INDEX_TTS_MODEL_REPO
    return value


def index_tts_repo_for_output_voice_language(language: str) -> str:
    _ = normalize_output_voice_language(language)
    return DEFAULT_INDEX_TTS_MODEL_REPO


def _infer_output_voice_language_from_repo(repo: str) -> str:
    normalized_repo = str(repo or "").strip()
    if normalized_repo == "xmccln/IndexTTS-2-Japanese":
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
        api_key = ""
        if node_id:
            api_key = str(os.getenv(_translation_node_env_var_name(node_id)) or "").strip()
        if not api_key:
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


def _repo_name_to_asr_dir_name(repo: str) -> str:
    repo_name = str(repo or "").strip().split("/")[-1]
    safe_chars = []
    for char in repo_name:
        if char.isalnum() or char in {"-", "_", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("-")
    sanitized = "".join(safe_chars).strip("-._")
    return sanitized or DEFAULT_ASR_MODEL_DIR.name


def _normalize_asr_model_repo(raw: str | None) -> str | None:
    value = _parse_optional_str(raw)
    if value is None:
        return None
    if value.casefold() in {
        "iic/sensevoicesmall",
        "funaudiollm/sensevoicesmall",
        "sensevoicesmall",
        "funaudiollm/fun-asr-nano-2512",
        "fun-asr-nano-2512",
    }:
        return DEFAULT_ASR_MODEL_REPO
    return value

def resolve_asr_model_dir() -> Path:
    configured = _parse_optional_str(_load_model_settings().get("asr_model_dir"))
    repo = resolve_asr_model_repo()
    if configured:
        resolved = _resolve_configured_path(configured)
        expected_name = DEFAULT_ASR_MODEL_DIR.name if repo == DEFAULT_ASR_MODEL_REPO else _repo_name_to_asr_dir_name(repo)
        if repo == DEFAULT_ASR_MODEL_REPO and resolved.name.casefold() == "sensevoicesmall":
            logger.warning(
                "检测到旧的 SenseVoiceSmall ASR 目录覆盖，已回退到默认 Nano 目录: {}",
                resolved,
            )
            return DEFAULT_ASR_MODEL_DIR
        if resolved.name.casefold() != expected_name.casefold():
            logger.warning(
                "ASR 模型 repo 与目录可能不一致: repo={} dir={} expected_dir_name={}",
                repo,
                resolved,
                expected_name,
            )
        return resolved
    return DEFAULT_ASR_MODEL_DIR if repo == DEFAULT_ASR_MODEL_REPO else CHECKPOINTS_DIR / _repo_name_to_asr_dir_name(repo)


def resolve_asr_model_repo() -> str:
    settings_value = _normalize_asr_model_repo(_load_model_settings().get("asr_model_repo"))
    if settings_value is not None:
        return settings_value
    return DEFAULT_ASR_MODEL_REPO


def set_asr_model_repo(repo: str) -> Path:
    normalized = repo.strip()
    if not normalized:
        raise ValueError("ASR 模型仓库不能为空。")
    settings = _load_model_settings()
    settings["asr_model_repo"] = normalized
    settings["asr_model_dir"] = str(DEFAULT_ASR_MODEL_DIR if normalized == DEFAULT_ASR_MODEL_REPO else CHECKPOINTS_DIR / _repo_name_to_asr_dir_name(normalized))
    _save_model_settings(settings)
    get_settings.cache_clear()
    return MODEL_SETTINGS_FILE


def resolve_supported_asr_model_repos() -> tuple[str, ...]:
    settings_repos = _parse_repo_list(_load_model_settings().get("supported_asr_model_repos"))
    repos: list[str] = []
    seen: set[str] = set()
    for repo in (*SUPPORTED_ASR_MODEL_REPOS, *settings_repos, resolve_asr_model_repo()):
        normalized = _normalize_asr_model_repo(repo)
        if normalized is None:
            continue
        if not normalized or normalized in seen:
            continue
        repos.append(normalized)
        seen.add(normalized)
    return tuple(repos)


def set_supported_asr_model_repos(repos: list[str] | tuple[str, ...]) -> Path:
    normalized = _parse_repo_list(list(repos))
    if not normalized:
        raise ValueError("至少需要保留一个 ASR 模型仓库。")
    settings = _load_model_settings()
    settings["supported_asr_model_repos"] = list(normalized)
    _save_model_settings(settings)
    get_settings.cache_clear()
    return MODEL_SETTINGS_FILE


def resolve_index_tts_model_repo() -> str:
    settings_value = normalize_index_tts_model_repo(_load_model_settings().get("index_tts_model_repo"))
    if settings_value is not None:
        return settings_value
    return DEFAULT_INDEX_TTS_MODEL_REPO


def resolve_voice_output_model() -> str:
    settings_value = _parse_optional_str(_load_model_settings().get("voice_output_model"))
    return normalize_voice_output_model(settings_value)


def resolve_output_voice_language() -> str:
    settings_value = _parse_optional_str(_load_model_settings().get("output_voice_language"))
    if settings_value is not None:
        return normalize_output_voice_language(settings_value)
    return _infer_output_voice_language_from_repo(resolve_index_tts_model_repo())


def set_voice_output_model(model_name: str) -> Path:
    normalized = normalize_voice_output_model(model_name)
    settings = _load_model_settings()
    settings["voice_output_model"] = normalized
    _save_model_settings(settings)
    get_settings.cache_clear()
    return MODEL_SETTINGS_FILE


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
    settings_repos = _parse_repo_list(_load_model_settings().get("supported_index_tts_model_repos"))
    repos: list[str] = []
    seen: set[str] = set()
    for repo in (*SUPPORTED_INDEX_TTS_MODEL_REPOS, *settings_repos, resolve_index_tts_model_repo()):
        normalized = normalize_index_tts_model_repo(repo)
        if not normalized or normalized in seen:
            continue
        repos.append(normalized)
        seen.add(normalized)
    return tuple(repos)


def set_index_tts_model_repo(repo: str) -> Path:
    normalized = normalize_index_tts_model_repo(repo)
    if not normalized:
        raise ValueError("TTS 模型仓库不能为空。")
    settings = _load_model_settings()
    settings["index_tts_model_repo"] = normalized
    settings["index_tts_model_dir"] = str(default_index_tts_model_dir(normalized))
    _save_model_settings(settings)
    get_settings.cache_clear()
    return MODEL_SETTINGS_FILE


def set_supported_index_tts_model_repos(repos: list[str] | tuple[str, ...]) -> Path:
    normalized = tuple(
        repo
        for repo in (normalize_index_tts_model_repo(item) for item in repos)
        if repo
    )
    if not normalized:
        raise ValueError("至少需要保留一个 TTS 模型仓库。")
    settings = _load_model_settings()
    settings["supported_index_tts_model_repos"] = list(normalized)
    _save_model_settings(settings)
    get_settings.cache_clear()
    return MODEL_SETTINGS_FILE


def set_output_voice_language(language: str) -> Path:
    normalized = normalize_output_voice_language(language)
    target_repo = index_tts_repo_for_output_voice_language(normalized)
    settings = _load_model_settings()
    settings["output_voice_language"] = normalized
    settings["index_tts_model_repo"] = target_repo
    settings["index_tts_model_dir"] = str(default_index_tts_model_dir(target_repo))
    _save_model_settings(settings)
    get_settings.cache_clear()
    return MODEL_SETTINGS_FILE


def resolve_index_tts_model_dir() -> Path:
    settings_value = _parse_optional_str(_load_model_settings().get("index_tts_model_dir"))
    if settings_value is not None:
        return _resolve_configured_path(settings_value)
    return default_index_tts_model_dir(resolve_index_tts_model_repo())


def set_index_tts_model_dir(model_dir: str | Path | None) -> Path:
    state = _load_model_settings()
    if model_dir is None:
        state.pop("index_tts_model_dir", None)
    else:
        state["index_tts_model_dir"] = str(Path(model_dir).expanduser())
    _save_model_settings(state)
    get_settings.cache_clear()
    return MODEL_SETTINGS_FILE


def resolve_emotion_model_dir() -> Path:
    configured = _parse_optional_str(_load_model_settings().get("emotion_model_dir"))
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


def resolve_translation_target_language() -> str:
    env_value = _parse_optional_str(os.getenv("TRANSLATION_TARGET_LANGUAGE"))
    if env_value is not None:
        return normalize_output_voice_language(env_value)
    settings = _load_runtime_settings()
    settings_value = _parse_optional_str(settings.get("translation_target_language"))
    if settings_value is not None:
        return normalize_output_voice_language(settings_value)
    output_language = _parse_optional_str(settings.get("output_voice_language"))
    if output_language is not None:
        return normalize_output_voice_language(output_language)
    return DEFAULT_TRANSLATION_TARGET_LANGUAGE


def set_translation_enabled(enabled: bool) -> Path:
    settings = _load_runtime_settings()
    settings["translation_enabled"] = bool(enabled)
    _save_runtime_settings(settings)
    get_settings.cache_clear()
    return RUNTIME_SETTINGS_FILE


def set_translation_target_language(language: str) -> Path:
    normalized = normalize_output_voice_language(language)
    settings = _load_runtime_settings()
    settings["translation_target_language"] = normalized
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
    previous_env_names = _collect_translation_node_env_names(settings.get("translation_nodes"))
    current_env_names: set[str] = set()
    serialized_nodes: list[dict[str, str]] = []
    for node in normalized_nodes:
        env_key = _translation_node_env_var_name(node.node_id)
        api_key = node.api_key.strip()
        serialized_node = {
            "node_id": node.node_id,
            "name": node.name,
            "provider": node.provider,
            "base_url": node.base_url,
            "model": node.model,
        }
        if api_key:
            _set_project_env_value(env_key, api_key)
            current_env_names.add(env_key)
        else:
            _set_project_env_value(env_key, None)
        serialized_nodes.append(serialized_node)
    settings["translation_nodes"] = serialized_nodes
    for env_key in previous_env_names - current_env_names:
        _set_project_env_value(env_key, None)
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
        supported_asr_model_repos=resolve_supported_asr_model_repos(),
        index_tts_model_dir=resolve_index_tts_model_dir(),
        index_tts_model_repo=resolve_index_tts_model_repo(),
        supported_index_tts_model_repos=resolve_supported_index_tts_model_repos(),
        voice_output_model=resolve_voice_output_model(),
        output_voice_language=resolve_output_voice_language(),
        emotion_model_dir=resolve_emotion_model_dir(),
        tts_device=resolve_tts_device(),
        asr_device=resolve_asr_device(),
        emotion_device=resolve_emotion_device(),
        tts_max_new_tokens=resolve_tts_max_new_tokens(),
        live_asr_idle_unload_sec=resolve_live_asr_idle_unload_sec(),
        live_tts_idle_unload_sec=resolve_live_tts_idle_unload_sec(),
        translation_enabled=resolve_translation_enabled(),
        translation_target_language=resolve_translation_target_language(),
        translation_node_id=resolve_translation_node_id(),
        translation_nodes=resolve_translation_nodes(),
        experimental_torch_compile_enabled=resolve_experimental_torch_compile_enabled(),
        experimental_accel_gpt_enabled=resolve_experimental_accel_gpt_enabled(),
    )
