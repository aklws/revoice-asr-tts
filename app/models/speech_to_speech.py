from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import DEFAULT_TTS_MAX_NEW_TOKENS


@dataclass(frozen=True)
class SpeechToSpeechRequest:
    audio_path: str
    output_path: str | Path
    asr_language: str | None = None
    tts_language: str | None = None
    instruction: str | None = None
    tokens: int | None = None
    quality: str | None = None
    sound_event: str | None = None
    ambient_sound: str | None = None
    max_new_tokens: int = DEFAULT_TTS_MAX_NEW_TOKENS
    do_sample: bool = True
    audio_temperature: float = 1.7
    audio_top_p: float = 0.8
    audio_top_k: int = 25
    audio_repetition_penalty: float = 1.0
    cache_implementation: str | None = None
    seed: int | None = None
