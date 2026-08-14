from __future__ import annotations

import re
from pathlib import Path

from app.core.config import default_index_tts_model_dir


_W2V_BERT_REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
    "configuration.json",
    "model.safetensors",
    "preprocessor_config.json",
)


def is_japanese_index_tts_dir(model_dir: Path) -> bool:
    hints = " ".join((model_dir.name, str(model_dir))).lower()
    return "japanese" in hints or re.search(r"(^|[-_/])ja($|[-_/])", hints) is not None


def _dir_has_required_entries(base_dir: Path, entries: tuple[str, ...]) -> bool:
    return base_dir.exists() and all((base_dir / entry).exists() for entry in entries)


def _path_exists(path: Path) -> bool:
    return path.exists()


def get_index_tts_missing_files(model_dir: Path, *, local_only: bool = False) -> list[str]:
    missing: list[str] = []
    source_roots = [model_dir]

    if is_japanese_index_tts_dir(model_dir) and not local_only:
        official_model_dir = default_index_tts_model_dir("IndexTeam/IndexTTS-2.5")
        if official_model_dir != model_dir:
            source_roots.append(official_model_dir)

    def exists_in_sources(relative_path: str) -> bool:
        return any(_path_exists(root / relative_path) for root in source_roots)

    def dir_exists_in_sources(relative_path: str, required_entries: tuple[str, ...] | None = None) -> bool:
        for root in source_roots:
            candidate = root / relative_path
            if required_entries is None:
                if candidate.exists():
                    return True
                continue
            if _dir_has_required_entries(candidate, required_entries):
                return True
        return False

    if not exists_in_sources("config.yaml"):
        missing.append("config.yaml")

    for relative_path in (
        "gpt.pth",
        "s2mel.pth",
        "codec.pth",
        "multilingual_zh_ja_yue_char_del.tiktoken",
        "wav2vec2bert_stats.pt",
    ):
        if not exists_in_sources(relative_path):
            missing.append(relative_path)

    if not dir_exists_in_sources("hf_cache/w2v-bert-2.0", _W2V_BERT_REQUIRED_FILES):
        missing.append("hf_cache/w2v-bert-2.0")

    for relative_path in (
        "hf_cache/semantic_codec_model.safetensors",
        "hf_cache/campplus_cn_common.bin",
        "hf_cache/bigvgan/config.json",
        "hf_cache/bigvgan/bigvgan_generator.pt",
    ):
        if not exists_in_sources(relative_path):
            missing.append(relative_path)

    return missing
