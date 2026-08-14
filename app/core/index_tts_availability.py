from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import (
    index_tts_repo_for_output_voice_language,
    normalize_output_voice_language,
    set_output_voice_language,
)
from app.core.index_tts_files import get_index_tts_missing_files
from app.core.model_setup import list_index_tts_model_catalog


OUTPUT_LANGUAGE_LABELS = {
    "zh": "中文",
    "en": "英文",
    "ja": "日语",
    "es": "西班牙语",
    "ar": "阿拉伯语",
}


@dataclass(frozen=True, slots=True)
class IndexTTSOutputLanguageAvailability:
    language: str
    language_label: str
    repo: str
    display_name: str
    target_dir: Path
    installed: bool
    missing_files: tuple[str, ...]


def get_index_tts_output_language_availability(language: str) -> IndexTTSOutputLanguageAvailability:
    normalized_language = normalize_output_voice_language(language)
    target_repo = index_tts_repo_for_output_voice_language(normalized_language)
    catalog = list_index_tts_model_catalog((target_repo,))
    if not catalog:
        raise RuntimeError(f"无法解析 {normalized_language} 对应的 Index-TTS 模型信息。")
    item = catalog[0]
    missing_files = tuple(get_index_tts_missing_files(item.target_dir))
    return IndexTTSOutputLanguageAvailability(
        language=normalized_language,
        language_label=OUTPUT_LANGUAGE_LABELS.get(normalized_language, normalized_language),
        repo=item.repo,
        display_name=item.display_name,
        target_dir=item.target_dir,
        installed=bool(item.installed) and not bool(missing_files),
        missing_files=missing_files,
    )


def ensure_index_tts_output_language_available(language: str) -> IndexTTSOutputLanguageAvailability:
    availability = get_index_tts_output_language_availability(language)
    if availability.installed:
        return availability
    raise RuntimeError(
        f"Index-TTS 当前没有可用的{availability.language_label}模型。"
        f"缺少文件：{', '.join(availability.missing_files)}。"
        f"请先在“模型下载”里下载或补齐 {availability.display_name}（{availability.repo}），"
        f"然后再切换到{availability.language_label}输出。"
    )


def set_index_tts_output_language(language: str) -> IndexTTSOutputLanguageAvailability:
    availability = ensure_index_tts_output_language_available(language)
    set_output_voice_language(availability.language)
    return availability
