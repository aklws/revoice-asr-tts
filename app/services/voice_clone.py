from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor, GenerationConfig

from app.core.config import DEFAULT_TTS_CODEC_DIR, DEFAULT_TTS_MODEL_DIR, get_settings, resolve_tts_device
from app.core.logging import get_logger
from app.core.paths import get_runtime_dir
from app.core.runtime import (
    release_memory,
    resolve_attn_implementation,
    resolve_device,
    resolve_dtype,
    resolve_generation_cache_implementation,
)
from app.models.voice_clone import VoiceCloneRequest


logger = get_logger(__name__)


@dataclass(frozen=True)
class ReferenceCacheEvent:
    hit: bool
    elapsed_sec: float
    code_frames: int
    codebooks: int
    cache_entries: int


class VoiceCloneService:
    def __init__(self, model_dir: Path, codec_dir: Path | None = None) -> None:
        self.model_dir = model_dir
        self.codec_dir = codec_dir or DEFAULT_TTS_CODEC_DIR
        self.device = resolve_device(resolve_tts_device())
        self.dtype = resolve_dtype(self.device)
        self.attn_implementation = resolve_attn_implementation(self.device)
        self.generation_cache_implementation = resolve_generation_cache_implementation(
            get_settings().tts_cache_implementation,
            device=self.device,
        )
        self.processor: Any | None = None
        self.model: Any | None = None
        self.reference_codes_cache: dict[str, torch.Tensor] = {}
        self.last_reference_cache_event: ReferenceCacheEvent | None = None
        self.reference_cache_hits = 0
        self.reference_cache_misses = 0
        self.reference_cache_encode_sec = 0.0

    def _describe_attention_backend(self) -> str:
        model = self.model
        if model is None:
            return "unloaded"

        declared = getattr(self, "attn_implementation", None)
        top_level = getattr(getattr(model, "config", None), "_attn_implementation", None)
        qwen_attn = None
        local_attn = None

        if hasattr(model, "transformer") and hasattr(model.transformer, "layers") and model.transformer.layers:
            qwen_attn = getattr(model.transformer.layers[0].self_attn, "attn_implementation", None)
        if hasattr(model, "local_transformer") and hasattr(model.local_transformer, "h") and model.local_transformer.h:
            local_attn = getattr(model.local_transformer.h[0].attn, "attn_implementation", None)

        return (
            f"declared={declared}, top_level={top_level}, "
            f"qwen3={qwen_attn}, local_gpt2={local_attn}"
        )

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        if not self.model_dir.exists():
            raise FileNotFoundError(
                "Model directory not found. Set MOSS_TTS_MODEL_DIR or place the model under "
                f"{DEFAULT_TTS_MODEL_DIR}."
            )
        if not self.codec_dir.exists():
            raise FileNotFoundError(
                "MOSS audio tokenizer directory not found. Set MOSS_TTS_CODEC_DIR or place the codec under "
                f"{DEFAULT_TTS_CODEC_DIR}."
            )

        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        self.processor = AutoProcessor.from_pretrained(
            str(self.model_dir),
            codec_path=str(self.codec_dir),
            trust_remote_code=True,
        )
        if hasattr(self.processor, "audio_tokenizer") and hasattr(self.processor.audio_tokenizer, "to"):
            self.processor.audio_tokenizer = self.processor.audio_tokenizer.to(device=self.device)

        self.model = AutoModel.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        logger.info("MOSS-TTS attention backend: {}", self._describe_attention_backend())
        logger.info("MOSS-TTS generation cache: {}", self.generation_cache_implementation)

    def unload(self) -> None:
        if self.model is None and self.processor is None:
            return
        model = self.model
        processor = self.processor
        self.model = None
        self.processor = None
        self.reference_codes_cache.clear()
        self.last_reference_cache_event = None
        self.reference_cache_hits = 0
        self.reference_cache_misses = 0
        self.reference_cache_encode_sec = 0.0
        del model
        del processor
        release_memory()

    def assert_ready(self) -> tuple[Any, Any]:
        if self.processor is None or self.model is None:
            raise RuntimeError("Model service is not loaded.")
        return self.processor, self.model

    @staticmethod
    def _resolve_ffmpeg_executable() -> str | None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg

        candidates = (
            get_runtime_dir("bin", "ffmpeg", "bin", "ffmpeg.exe"),
            get_runtime_dir("bin", "ffmpeg", "ffmpeg.exe"),
        )
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    @classmethod
    def _load_audio_waveform_via_ffmpeg(cls, audio_path: str | os.PathLike[str]) -> tuple[torch.Tensor, int]:
        ffmpeg = cls._resolve_ffmpeg_executable()
        if ffmpeg is None:
            raise RuntimeError("找不到 ffmpeg，无法回退解码参考音频。")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            command = [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-i",
                str(audio_path),
                "-f",
                "wav",
                "-acodec",
                "pcm_f32le",
                str(temp_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                stderr = completed.stderr.strip() or "unknown ffmpeg error"
                raise RuntimeError(f"ffmpeg 解码失败: {stderr}")

            waveform, sample_rate = sf.read(str(temp_path), dtype="float32", always_2d=True)
            wav = torch.from_numpy(waveform).transpose(0, 1).contiguous()
            return wav, int(sample_rate)
        finally:
            temp_path.unlink(missing_ok=True)

    @classmethod
    def _load_audio_waveform(cls, audio_path: str | os.PathLike[str]) -> tuple[torch.Tensor, int]:
        try:
            waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
            wav = torch.from_numpy(waveform).transpose(0, 1).contiguous()
            return wav, int(sample_rate)
        except Exception as exc:
            logger.warning("soundfile 读取参考音频失败，尝试 ffmpeg 回退解码: {} | {}", audio_path, exc)
            try:
                return cls._load_audio_waveform_via_ffmpeg(audio_path)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"读取参考音频失败: {audio_path} | soundfile={exc} | ffmpeg={fallback_exc}"
                ) from fallback_exc

    @staticmethod
    def _reference_cache_key(audio_path: str | os.PathLike[str]) -> str:
        path = Path(audio_path).expanduser().resolve()
        stat = path.stat()
        return f"{path}|{stat.st_mtime_ns}|{stat.st_size}"

    def get_reference_codes(self, audio_path: str | os.PathLike[str]) -> torch.Tensor:
        if self.processor is None or self.model is None:
            self.load()
        processor, _ = self.assert_ready()

        cache_key = self._reference_cache_key(audio_path)
        started = time.perf_counter()
        cached = self.reference_codes_cache.get(cache_key)
        if cached is not None:
            elapsed = time.perf_counter() - started
            self.reference_cache_hits += 1
            self.last_reference_cache_event = ReferenceCacheEvent(
                hit=True,
                elapsed_sec=elapsed,
                code_frames=int(cached.shape[0]),
                codebooks=int(cached.shape[1]),
                cache_entries=len(self.reference_codes_cache),
            )
            return cached

        reference_wav, reference_sr = self._load_audio_waveform(audio_path)
        reference_codes = processor.encode_audios_from_wav(
            reference_wav,
            sampling_rate=reference_sr,
        )[0]
        if isinstance(reference_codes, torch.Tensor):
            reference_codes = reference_codes.detach().to(dtype=torch.long).cpu().contiguous()
        self.reference_codes_cache = {cache_key: reference_codes}
        elapsed = time.perf_counter() - started
        self.reference_cache_misses += 1
        self.reference_cache_encode_sec += elapsed
        self.last_reference_cache_event = ReferenceCacheEvent(
            hit=False,
            elapsed_sec=elapsed,
            code_frames=int(reference_codes.shape[0]),
            codebooks=int(reference_codes.shape[1]),
            cache_entries=len(self.reference_codes_cache),
        )
        return reference_codes

    def warm_reference_cache(self, audio_path: str | os.PathLike[str]) -> None:
        self.get_reference_codes(audio_path)

    def describe_last_reference_cache_event(self) -> str:
        event = self.last_reference_cache_event
        if event is None:
            return "reference cache: unused"
        status = "hit" if event.hit else "miss"
        return (
            f"reference cache {status}: {event.elapsed_sec:.3f}s | "
            f"codes={event.code_frames}x{event.codebooks} | entries={event.cache_entries}"
        )

    def describe_reference_cache_summary(self) -> str:
        return (
            f"reference cache summary: hits={self.reference_cache_hits}, "
            f"misses={self.reference_cache_misses}, "
            f"encode_time={self.reference_cache_encode_sec:.3f}s, "
            f"entries={len(self.reference_codes_cache)}"
        )

    def clone_voice(
        self,
        request: VoiceCloneRequest,
    ) -> tuple[torch.Tensor, int]:
        if self.model is None or self.processor is None:
            self.load()
        processor, model = self.assert_ready()

        if request.seed is not None:
            torch.manual_seed(request.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(request.seed)

        reference_codes = self.get_reference_codes(request.reference_audio_path)

        conversations = [[
            processor.build_user_message(
                text=request.text,
                reference=[reference_codes],
                language=request.language,
                instruction=request.instruction,
                tokens=request.tokens,
                quality=request.quality,
                sound_event=request.sound_event,
                ambient_sound=request.ambient_sound,
            )
        ]]

        batch = processor(conversations, mode="generation")
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        cache_implementation = resolve_generation_cache_implementation(
            request.cache_implementation or self.generation_cache_implementation,
            device=self.device,
        )
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.do_sample,
            "use_cache": True,
            "cache_implementation": cache_implementation,
        }
        if cache_implementation in {"static", "offloaded_static"}:
            generation_kwargs["cache_config"] = {"max_cache_len": int(input_ids.shape[-1]) + int(request.max_new_tokens)}
        generation_config = GenerationConfig(**generation_kwargs)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                audio_temperature=request.audio_temperature,
                audio_top_p=request.audio_top_p,
                audio_top_k=request.audio_top_k,
                audio_repetition_penalty=request.audio_repetition_penalty,
            )

        messages = processor.decode(outputs)
        if not messages or messages[0] is None:
            raise RuntimeError("Model returned an empty generation result.")
        if not messages[0].audio_codes_list:
            raise RuntimeError("Model did not produce audio data.")

        audio = messages[0].audio_codes_list[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu()
        sample_rate = int(processor.model_config.sampling_rate)
        return audio, sample_rate

    @staticmethod
    def save_audio(audio: torch.Tensor, sample_rate: int, output_path: str | os.PathLike[str]) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output), audio.detach().cpu().transpose(0, 1).numpy(), sample_rate)
        return output
