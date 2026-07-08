from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from app.models.asr import ASRRequest
from app.models.speech_to_speech import SpeechToSpeechRequest
from app.models.voice_clone import VoiceCloneRequest
from app.services.asr import ASRResult, ASRService
from app.services.voice_clone import VoiceCloneService


@dataclass(frozen=True)
class SpeechToSpeechResult:
    transcript: str
    transcript_language: str | None
    audio: torch.Tensor
    sample_rate: int
    output_path: Path


class SpeechToSpeechService:
    def __init__(self, asr_service: ASRService, tts_service: VoiceCloneService) -> None:
        self.asr_service = asr_service
        self.tts_service = tts_service

    def load(self) -> None:
        # In low-memory mode we avoid keeping both ASR and TTS on the GPU.
        self.asr_service.load()

    def transcribe(self, audio_path: str, language: str | None = None) -> ASRResult:
        return self.asr_service.transcribe(
            ASRRequest(audio_path=audio_path, language=language)
        )

    def run(self, request: SpeechToSpeechRequest) -> SpeechToSpeechResult:
        self.tts_service.unload()
        asr_result = self.transcribe(request.audio_path, language=request.asr_language)
        self.asr_service.unload()
        tts_request = VoiceCloneRequest(
            text=asr_result.text,
            reference_audio_path=request.audio_path,
            language=request.tts_language or asr_result.language,
            instruction=request.instruction,
            tokens=request.tokens,
            quality=request.quality,
            sound_event=request.sound_event,
            ambient_sound=request.ambient_sound,
            max_new_tokens=request.max_new_tokens,
            do_sample=request.do_sample,
            audio_temperature=request.audio_temperature,
            audio_top_p=request.audio_top_p,
            audio_top_k=request.audio_top_k,
            audio_repetition_penalty=request.audio_repetition_penalty,
            cache_implementation=request.cache_implementation,
            seed=request.seed,
        )
        audio, sample_rate = self.tts_service.clone_voice(tts_request)
        output_path = self.tts_service.save_audio(audio, sample_rate, request.output_path)
        self.tts_service.unload()
        return SpeechToSpeechResult(
            transcript=asr_result.text,
            transcript_language=asr_result.language,
            audio=audio,
            sample_rate=sample_rate,
            output_path=output_path,
        )
