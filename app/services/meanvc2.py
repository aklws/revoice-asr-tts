from __future__ import annotations

import importlib.util
import sys
from functools import cache
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi

from app.core.audio import resample_waveform
from app.core.logging import get_logger
from app.core.paths import get_app_root
from app.core.runtime import release_memory

logger = get_logger(__name__)
MEANVC2_VENDOR_ROOT = get_app_root() / "vendor" / "MeanVC2"
MEANVC2_CHECKPOINT_ROOT = get_app_root() / "checkpoints" / "MeanVC2"
MEANVC2_RUNTIME_ENTRY = MEANVC2_VENDOR_ROOT / "runtime" / "run_rt.py"
MEANVC2_SAMPLE_RATE = 16_000
DEFAULT_MEANVC2_VARIANT = "120ms"
MEANVC2_INIT_SCRIPT = MEANVC2_VENDOR_ROOT / "initialization.py"


def _normalize_model_variant(raw: str | None) -> str:
    normalized = str(raw or "").strip().casefold()
    return "40ms" if normalized == "40ms" else DEFAULT_MEANVC2_VARIANT


def _pick_preferred_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_runtime_assets(model_variant: str) -> dict[str, Path]:
    normalized_variant = _normalize_model_variant(model_variant)
    variant_ckpt = (
        "meanvc2_40ms_40ms.safetensors"
        if normalized_variant == "40ms"
        else "meanvc2_120ms_40ms.safetensors"
    )
    asr_ckpt = "fastu2pp_80ms.pt" if normalized_variant == "40ms" else "fastu2pp_160ms.pt"
    config_name = "config_40ms_40ms.json" if normalized_variant == "40ms" else "config_120ms_40ms.json"
    return {
        "config": _pick_preferred_path(
            MEANVC2_CHECKPOINT_ROOT / config_name,
            MEANVC2_VENDOR_ROOT / "src" / "config" / config_name,
        ),
        "ckpt": _pick_preferred_path(
            MEANVC2_CHECKPOINT_ROOT / variant_ckpt,
            MEANVC2_VENDOR_ROOT / "ckpts" / "pretrained_models" / variant_ckpt,
        ),
        "asr_ckpt": _pick_preferred_path(
            MEANVC2_CHECKPOINT_ROOT / asr_ckpt,
            MEANVC2_VENDOR_ROOT / "preprocess" / "ckpts" / asr_ckpt,
        ),
        "vocoder": _pick_preferred_path(
            MEANVC2_CHECKPOINT_ROOT / "vocos.pt",
            MEANVC2_VENDOR_ROOT / "ckpts" / "vocos" / "vocos.pt",
        ),
        "speaker_model": _pick_preferred_path(
            MEANVC2_CHECKPOINT_ROOT / "wavlm_large_finetune.pth",
            MEANVC2_VENDOR_ROOT / "preprocess" / "ckpts" / "wavlm_large_finetune.pth",
        ),
        "wavlm_config": _pick_preferred_path(
            MEANVC2_CHECKPOINT_ROOT / "wavlm_large_cfg.pt",
            MEANVC2_VENDOR_ROOT / "preprocess" / "ckpts" / "wavlm_large_cfg.pt",
        ),
    }


def _required_runtime_files(model_variant: str) -> list[Path]:
    assets = _resolve_runtime_assets(model_variant)
    return [
        assets["config"],
        assets["ckpt"],
        assets["vocoder"],
        assets["asr_ckpt"],
        assets["speaker_model"],
        assets["wavlm_config"],
    ]


def _validate_runtime_dependencies(model_variant: str) -> None:
    missing_packages = [
        package_name
        for package_name in ("einops", "x_transformers")
        if find_spec(package_name) is None
    ]
    missing_files = [str(path) for path in _required_runtime_files(model_variant) if not path.exists()]
    if not missing_packages and not missing_files:
        return

    details: list[str] = []
    if missing_packages:
        missing_text = ", ".join(missing_packages)
        details.append(
            "缺少 Python 依赖："
            f"{missing_text}\n"
            f"请先执行 `{sys.executable} -m pip install -r vendor/MeanVC2/requirements.txt`。"
        )
    if missing_files:
        missing_text = "\n".join(missing_files)
        init_hint = (
            f"`{sys.executable} {MEANVC2_INIT_SCRIPT} --task train_{_normalize_model_variant(model_variant)}`"
            if MEANVC2_INIT_SCRIPT.exists()
            else "`python vendor/MeanVC2/initialization.py --task train_120ms`"
        )
        details.append(
            "缺少 MeanVC2 运行文件：\n"
            f"{missing_text}\n"
            f"当前会优先查找外部模型目录：{MEANVC2_CHECKPOINT_ROOT}\n"
            f"建议先执行：{init_hint}\n"
            "另外 `wavlm_large_finetune.pth` 需要按 vendor/MeanVC2/initialization.py 中的说明手动放到位。"
        )
    raise RuntimeError("MeanVC2 尚未准备完成。\n" + "\n\n".join(details))


@cache
def _load_vcrunner_module() -> Any:
    if not MEANVC2_RUNTIME_ENTRY.exists():
        raise RuntimeError(f"未找到 MeanVC2 runtime 入口文件: {MEANVC2_RUNTIME_ENTRY}")

    module_name = "_vendor_meanvc2_runtime"
    spec = importlib.util.spec_from_file_location(module_name, MEANVC2_RUNTIME_ENTRY)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 MeanVC2 runtime 模块。")

    module = importlib.util.module_from_spec(spec)
    runtime_dir = str(MEANVC2_RUNTIME_ENTRY.parent)
    if runtime_dir not in sys.path:
        sys.path.insert(0, runtime_dir)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        missing_name = str(getattr(exc, "name", "") or "").strip() or "unknown"
        raise RuntimeError(
            f"MeanVC2 依赖缺失：{missing_name}。请先执行 "
            f"`{sys.executable} -m pip install -r vendor/MeanVC2/requirements.txt`。"
        ) from exc
    return module


def _load_vcrunner_class(model_variant: str) -> type[Any]:
    module = _load_vcrunner_module()
    assets = _resolve_runtime_assets(model_variant)
    normalized_variant = _normalize_model_variant(model_variant)
    other_variant = "40ms" if normalized_variant == "120ms" else "120ms"
    other_assets = _resolve_runtime_assets(other_variant)

    module.VOCODER_PATH = str(assets["vocoder"])
    module.SPEAKER_MODEL_PATH = str(assets["speaker_model"])
    module.WAVLM_CONFIG_PATH = str(assets["wavlm_config"])
    module.MODEL_PATHS["120ms"]["config"] = str(_resolve_runtime_assets("120ms")["config"])
    module.MODEL_PATHS["120ms"]["ckpt"] = str(_resolve_runtime_assets("120ms")["ckpt"])
    module.MODEL_PATHS["120ms"]["asr_ckpt"] = str(_resolve_runtime_assets("120ms")["asr_ckpt"])
    module.MODEL_PATHS["40ms"]["config"] = str(other_assets["config"] if other_variant == "40ms" else assets["config"])
    module.MODEL_PATHS["40ms"]["ckpt"] = str(other_assets["ckpt"] if other_variant == "40ms" else assets["ckpt"])
    module.MODEL_PATHS["40ms"]["asr_ckpt"] = str(other_assets["asr_ckpt"] if other_variant == "40ms" else assets["asr_ckpt"])

    runner_cls = getattr(module, "VCRunner", None)
    if runner_cls is None:
        raise RuntimeError("MeanVC2 runtime 中未找到 VCRunner。")
    return runner_cls


class MeanVC2Service:
    """Utterance-first wrapper around vendor MeanVC2 runtime assets."""

    def __init__(self, *, device: str = "cpu", model_variant: str = DEFAULT_MEANVC2_VARIANT) -> None:
        self.device = "cuda" if str(device or "").strip().casefold().startswith("cuda") else "cpu"
        self.model_variant = _normalize_model_variant(model_variant)
        self.reference_audio_path: str | None = None
        self._runner: Any | None = None
        self._stream_initialized = False

    def load(self, reference_audio_path: str) -> None:
        reference_path = str(reference_audio_path or "").strip()
        if not reference_path:
            raise ValueError("MeanVC2 需要参考音频。")
        if self._runner is not None and self.reference_audio_path == reference_path:
            return
        _validate_runtime_dependencies(self.model_variant)
        runner_cls = _load_vcrunner_class(self.model_variant)
        self._runner = runner_cls(
            target_wav=reference_path,
            device=self.device,
            model=self.model_variant,
        )
        self.reference_audio_path = reference_path
        self._stream_initialized = False
        logger.info("MeanVC2 已加载: variant={} reference={}", self.model_variant, reference_path)

    def unload(self) -> None:
        self._runner = None
        self.reference_audio_path = None
        self._stream_initialized = False
        release_memory()

    def warmup_speaker(self, reference_audio_path: str) -> None:
        self.load(reference_audio_path)

    def get_stream_chunk_samples(self) -> int:
        if self._runner is not None:
            return int(getattr(self._runner, "CHUNK", int(MEANVC2_SAMPLE_RATE * 0.16)))
        runner_cls = _load_vcrunner_class(self.model_variant)
        chunk = getattr(runner_cls, "CHUNK", None)
        if chunk is not None:
            return int(chunk)
        return int(MEANVC2_SAMPLE_RATE * 0.16)

    def _ensure_runner(self, *, reference_audio_path: str | None = None) -> Any:
        if reference_audio_path:
            self.load(reference_audio_path)
        if self._runner is None:
            raise RuntimeError("MeanVC2 尚未加载。")
        return self._runner

    def _extract_bn_features(self, audio: np.ndarray, runner: Any) -> np.ndarray:
        if audio.size == 0:
            return np.empty((0, 256), dtype=np.float32)
        wav_tensor = torch.from_numpy(audio.astype(np.float32, copy=False) * (1 << 15)).unsqueeze(0)
        fbanks = kaldi.fbank(
            wav_tensor,
            frame_length=25,
            frame_shift=10,
            snip_edges=True,
            num_mel_bins=80,
            energy_floor=0.0,
            dither=0.0,
            sample_frequency=MEANVC2_SAMPLE_RATE,
        ).squeeze(0)
        window = int(getattr(runner, "_bn_window", 0) or 0)
        stride = int(getattr(runner, "_bn_stride", 0) or 0)
        offset = int(getattr(runner, "_asr_offset_init", 0) or 0)
        offset_step = int(getattr(runner, "_asr_offset_step", 0) or 0)
        req_cache = int(getattr(runner, "_required_cache_size", 0) or 0)
        att_cache = torch.zeros(6, 4, req_cache, 128)
        cnn_cache = torch.zeros(6, 1, 256, 8)

        bns: list[torch.Tensor] = []
        position = 0
        total_frames = int(fbanks.shape[0])
        while position + window <= total_frames:
            fb = fbanks[position : position + window].unsqueeze(0)
            out, att_cache, cnn_cache = runner.asr(
                fb,
                torch.tensor(offset, dtype=torch.int64),
                torch.tensor(req_cache, dtype=torch.int64),
                att_cache,
                cnn_cache,
            )
            bns.append(out.squeeze(0).detach())
            offset += offset_step
            position += stride

        if not bns:
            return np.empty((0, 256), dtype=np.float32)
        return torch.cat(bns, dim=0).cpu().numpy().astype(np.float32, copy=False)

    def _infer_utterance(self, bn: np.ndarray, runner: Any, *, steps: int = 3, seed: int = 42) -> np.ndarray:
        if bn.size == 0:
            return np.empty(0, dtype=np.float32)
        with torch.inference_mode():
            torch.manual_seed(seed)
            bn_t = torch.from_numpy(bn).to(self.device).unsqueeze(0).transpose(1, 2)
            bn_t = torch.nn.functional.interpolate(
                bn_t,
                size=int(bn_t.shape[2] * 4),
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
            spk_t = runner.vc_spk_emb
            if len(spk_t.shape) == 1:
                spk_t = spk_t.unsqueeze(0)

            seq_len = int(bn_t.shape[1])
            if seq_len <= 0:
                return np.empty(0, dtype=np.float32)
            chunk_size = int(getattr(runner, "chunk_size", 0) or 0)
            block_size = int(getattr(runner, "block_size", 0) or 0)
            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            timesteps = torch.linspace(1.0, 0.0, steps + 1, device=bn_t.device)
            noise = torch.randn(1, seq_len, 80, device=bn_t.device, dtype=bn_t.dtype)
            x_pred_collect: list[torch.Tensor] = []
            offset = 0
            kv_cache = None

            for chunk_id in range(num_chunks):
                start = chunk_id * chunk_size
                mid = min(start + chunk_size, seq_len)
                end = min(mid + block_size, seq_len)
                current_len = mid - start
                cond_in = torch.cat([bn_t[:, start:mid], bn_t[:, mid:end]], dim=1)
                x = noise[:, start:end].clone()

                for index in range(steps):
                    t_val = timesteps[index]
                    r_val = timesteps[index + 1]
                    t_tensor = torch.full((x.size(0),), float(t_val), device=x.device)
                    r_tensor = torch.full((x.size(0),), float(r_val), device=x.device)
                    u, tmp_kv = runner.vc(
                        x,
                        t_tensor,
                        r_tensor,
                        cache=None,
                        cond=cond_in,
                        spks=spk_t,
                        offset=offset,
                        is_inference=True,
                        kv_cache=kv_cache,
                        gtm_kv=runner.vc_gtm_kv,
                    )
                    x = x - (float(t_val) - float(r_val)) * u
                kv_cache = tmp_kv
                x_pred_collect.append(x[:, :current_len])
                offset += current_len

            x_pred = torch.cat(x_pred_collect, dim=1)
            mel = (x_pred.transpose(1, 2) + 1.0) / 2.0
            waveform = runner.vocoder.decode(mel)
        return waveform.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)

    def process_stream_chunk(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        reference_audio_path: str | None = None,
    ) -> tuple[np.ndarray, int]:
        return self.convert_waveform(
            waveform,
            sample_rate,
            reference_audio_path=reference_audio_path,
        )

    def flush_stream(self) -> tuple[np.ndarray, int]:
        self._stream_initialized = False
        return np.empty(0, dtype=np.float32), MEANVC2_SAMPLE_RATE

    def convert_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        reference_audio_path: str | None = None,
    ) -> tuple[np.ndarray, int]:
        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1, dtype=np.float32)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.size == 0:
            return np.empty(0, dtype=np.float32), MEANVC2_SAMPLE_RATE
        if sample_rate != MEANVC2_SAMPLE_RATE:
            audio = resample_waveform(audio, sample_rate, MEANVC2_SAMPLE_RATE)
        runner = self._ensure_runner(reference_audio_path=reference_audio_path)
        bn = self._extract_bn_features(audio, runner)
        if bn.size == 0:
            logger.warning("MeanVC2 未提取到有效 BN 特征: samples={}", int(audio.shape[0]))
            return np.empty(0, dtype=np.float32), MEANVC2_SAMPLE_RATE
        converted = self._infer_utterance(bn, runner)
        if converted.size == 0:
            return np.empty(0, dtype=np.float32), MEANVC2_SAMPLE_RATE
        self._stream_initialized = False
        return np.ascontiguousarray(converted, dtype=np.float32), MEANVC2_SAMPLE_RATE
