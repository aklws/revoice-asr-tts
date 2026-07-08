"""Auto-download missing checkpoints/models at startup."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import threading
import time
from functools import cache
from pathlib import Path
from typing import Callable
from modelscope.hub.callback import ProgressCallback as ModelScopeProgressCallback

from app.core.config import (
    CHECKPOINTS_DIR,
    DEFAULT_EMOTION_MODEL_DIR,
    DEFAULT_INDEX_TTS_MODEL_DIR,
)
from app.core.paths import get_bin_dir
from app.core.logging import get_logger

logger = get_logger(__name__)
StatusCallback = Callable[[str], None]
_MODEL_SETUP_LOCK = threading.RLock()

ProgressPayload = dict[str, object]
ProgressCallback = Callable[[ProgressPayload], None]


@dataclass(frozen=True, slots=True)
class _ModelSpec:
    display_name: str
    target_dir: Path
    markers: tuple[str, ...]
    marker_policy: str = "all"
    repo: str | None = None


_DOWNLOADABLE_MODELS: tuple[_ModelSpec, ...] = (
    _ModelSpec(
        display_name="IndexTTS-2",
        target_dir=DEFAULT_INDEX_TTS_MODEL_DIR,
        repo="IndexTeam/IndexTTS-2",
        markers=("config.yaml",),
    ),
    _ModelSpec(
        display_name="emotion2vec+",
        target_dir=DEFAULT_EMOTION_MODEL_DIR,
        repo="iic/emotion2vec_plus_base",
        markers=("configuration.json", "README.md", "model.pt", "config.yaml"),
        marker_policy="any",
    ),
)
_LOCAL_MODELS: tuple[_ModelSpec, ...] = (
    _ModelSpec(
        display_name="Qwen3-ASR",
        target_dir=get_bin_dir("Qwen3-ASR-Transcribe", "model"),
        markers=(
            "qwen3_asr_encoder_backend.int4.onnx",
            "qwen3_asr_encoder_frontend.int4.onnx",
            "qwen3_asr_llm.q4_k.gguf",
        ),
        marker_policy="all",
    ),
)


@cache
def _load_modelscope_hub_api() -> type[object]:
    module = importlib.import_module("modelscope.hub.api")
    return module.HubApi


@cache
def _load_modelscope_snapshot_download():
    module = importlib.import_module("modelscope.hub.snapshot_download")
    return module.snapshot_download


@cache
def _load_indextts_ensure_models_available():
    module = importlib.import_module("indextts.utils.model_download")
    return module.ensure_models_available


def _emit_status(status_callback: StatusCallback | None, message: str) -> None:
    if status_callback is not None:
        status_callback(message)


def _emit_progress(
    progress_callback: ProgressCallback | None,
    *,
    value: int,
    maximum: int,
    indeterminate: bool,
    detail: str,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        {
            "value": value,
            "maximum": maximum,
            "indeterminate": indeterminate,
            "detail": detail,
        }
    )


class _StartupProgressReporter:
    def __init__(self, progress_callback: ProgressCallback | None, total_steps: int) -> None:
        self.progress_callback = progress_callback
        self.total_steps = max(total_steps, 1)
        self.completed_steps = 0
        self.current_label = ""

    def _emit(
        self,
        *,
        current_value: int = 0,
        current_maximum: int = 0,
        current_indeterminate: bool = False,
        detail: str = "",
    ) -> None:
        if self.progress_callback is None:
            return
        if current_indeterminate or current_maximum <= 0:
            fraction = 0.0
        else:
            fraction = max(0.0, min(float(current_value) / float(current_maximum), 1.0))
        overall_maximum = self.total_steps * 1000
        overall_value = min(overall_maximum, self.completed_steps * 1000 + int(fraction * 1000))
        self.progress_callback(
            {
                "value": overall_value,
                "maximum": overall_maximum,
                "indeterminate": False,
                "detail": detail,
                "sub_label": self.current_label,
                "sub_value": current_value,
                "sub_maximum": current_maximum,
                "sub_indeterminate": current_indeterminate,
            }
        )

    def begin_step(
        self,
        label: str,
        *,
        detail: str,
        current_value: int = 0,
        current_maximum: int = 1,
        current_indeterminate: bool = False,
    ) -> None:
        self.current_label = label
        self._emit(
            current_value=current_value,
            current_maximum=current_maximum,
            current_indeterminate=current_indeterminate,
            detail=detail,
        )

    def update_step(
        self,
        *,
        detail: str,
        current_value: int,
        current_maximum: int,
        current_indeterminate: bool = False,
    ) -> None:
        self._emit(
            current_value=current_value,
            current_maximum=current_maximum,
            current_indeterminate=current_indeterminate,
            detail=detail,
        )

    def finish_step(self, *, detail: str) -> None:
        self._emit(
            current_value=1,
            current_maximum=1,
            current_indeterminate=False,
            detail=detail,
        )
        self.completed_steps = min(self.completed_steps + 1, self.total_steps)
        self._emit(detail=detail)

    def child_progress_callback(self, label: str) -> ProgressCallback:
        self.current_label = label

        def _callback(payload: ProgressPayload) -> None:
            current_value = int(payload.get("value", 0) or 0)
            current_maximum = int(payload.get("maximum", 0) or 0)
            current_indeterminate = bool(payload.get("indeterminate", False))
            detail = str(payload.get("detail", "")).strip() or label
            self.current_label = str(payload.get("sub_label") or label)
            self._emit(
                current_value=current_value,
                current_maximum=current_maximum,
                current_indeterminate=current_indeterminate,
                detail=detail,
            )

        return _callback


def _format_size(num_bytes: int) -> str:
    value = float(max(num_bytes, 0))
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024.0
    return f"{int(num_bytes)} B"


def _has_model_markers(model: _ModelSpec) -> bool:
    if not model.target_dir.exists():
        return False

    existing_count = sum(1 for marker in model.markers if (model.target_dir / marker).exists())
    if model.marker_policy == "any":
        return existing_count > 0 or any(model.target_dir.iterdir())
    return existing_count == len(model.markers)


def _get_modelscope_repo_stats(repo: str) -> tuple[int, int] | None:
    HubApi = _load_modelscope_hub_api()

    try:
        repo_files = HubApi().get_model_files(repo, recursive=True, use_cookies=False)
    except Exception as exc:
        logger.warning("获取 {} 文件清单失败，将退回不确定进度条: {}", repo, exc)
        return None

    file_entries = [item for item in repo_files if item.get("Type") != "tree"]
    total_bytes = sum(int(item.get("Size") or 0) for item in file_entries)
    total_files = len(file_entries)
    return total_bytes, total_files


class _ModelScopeDownloadProgress:
    def __init__(
        self,
        display_name: str,
        total_bytes: int,
        progress_callback: ProgressCallback | None,
    ) -> None:
        self.display_name = display_name
        self.total_bytes = max(total_bytes, 1)
        self.progress_callback = progress_callback
        self.downloaded_bytes = 0
        self.file_progress: dict[str, int] = {}
        self.file_sizes: dict[str, int] = {}
        self._lock = threading.Lock()
        self._last_emit = 0.0

    def _emit_locked(self, current_file: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_emit) < 0.1:
            return
        self._last_emit = now
        current_bytes = self.file_progress.get(current_file, 0)
        current_total = self.file_sizes.get(current_file, 0)
        detail = (
            f"正在下载 {self.display_name}: {current_file}  "
            f"{_format_size(self.downloaded_bytes)} / {_format_size(self.total_bytes)}"
        )
        if current_total > 0:
            detail += f"  |  当前文件 {_format_size(current_bytes)} / {_format_size(current_total)}"
        _emit_progress(
            self.progress_callback,
            value=min(self.downloaded_bytes, self.total_bytes),
            maximum=self.total_bytes,
            indeterminate=False,
            detail=detail,
        )

    def callback_type(self) -> type[ModelScopeProgressCallback]:
        tracker = self

        class AggregatedProgressCallback(ModelScopeProgressCallback):
            def __init__(self, filename: str, file_size: int):
                super().__init__(filename, file_size)
                with tracker._lock:
                    tracker.file_sizes[filename] = max(int(file_size or 0), 0)
                    tracker.file_progress.setdefault(filename, 0)
                    tracker._emit_locked(filename, force=True)

            def update(self, size: int):
                with tracker._lock:
                    file_size = tracker.file_sizes.get(self.filename, 0)
                    previous = tracker.file_progress.get(self.filename, 0)
                    current = previous + int(size or 0)
                    if file_size > 0:
                        current = min(current, file_size)
                    delta = max(current - previous, 0)
                    tracker.file_progress[self.filename] = current
                    tracker.downloaded_bytes += delta
                    tracker._emit_locked(self.filename)

            def end(self):
                with tracker._lock:
                    file_size = tracker.file_sizes.get(self.filename, 0)
                    previous = tracker.file_progress.get(self.filename, 0)
                    if file_size > previous:
                        tracker.file_progress[self.filename] = file_size
                        tracker.downloaded_bytes += file_size - previous
                    tracker._emit_locked(self.filename, force=True)

        return AggregatedProgressCallback


def _modelscope_download(
    display_name: str,
    repo: str,
    target_dir: str,
    progress_callback: ProgressCallback | None = None,
) -> None:
    snapshot_download = _load_modelscope_snapshot_download()

    stats = _get_modelscope_repo_stats(repo)
    progress_callbacks = None
    if stats is None:
        _emit_progress(
            progress_callback,
            value=0,
            maximum=0,
            indeterminate=True,
            detail=f"正在准备下载 {display_name} ({repo}) ...",
        )
    else:
        total_bytes, total_files = stats
        _emit_progress(
            progress_callback,
            value=0,
            maximum=max(total_bytes, 1),
            indeterminate=False,
            detail=(
                f"准备下载 {display_name}，共 {total_files} 个文件，"
                f"约 {_format_size(total_bytes)}"
            ),
        )
        progress_callbacks = [
            _ModelScopeDownloadProgress(
                display_name=display_name,
                total_bytes=total_bytes,
                progress_callback=progress_callback,
            ).callback_type()
        ]

    snapshot_download(repo, local_dir=target_dir, progress_callbacks=progress_callbacks)

    if stats is not None:
        total_bytes, _ = stats
        _emit_progress(
            progress_callback,
            value=max(total_bytes, 1),
            maximum=max(total_bytes, 1),
            indeterminate=False,
            detail=f"{display_name} 下载完成，共 {_format_size(total_bytes)}",
        )


def _download_model(
    model: _ModelSpec,
    status_callback: StatusCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> bool:
    if not model.repo:
        return False
    target = str(model.target_dir)
    message = f"正在下载 {model.display_name} ({model.repo}) ..."
    _emit_status(status_callback, message)
    logger.info(message)
    try:
        _modelscope_download(
            model.display_name,
            model.repo,
            target,
            progress_callback=progress_callback,
        )
        return _has_model_markers(model)
    except Exception as exc:
        logger.warning("{} 下载失败: {}", model.display_name, exc)
        return False


def _ensure_indextts_aux_models(
    model_dir: Path,
    status_callback: StatusCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    ensure_models_available = _load_indextts_ensure_models_available()
    _emit_status(status_callback, "正在准备 IndexTTS 附属模型...")
    ensure_models_available(
        str(model_dir),
        status_callback=status_callback,
        progress_callback=progress_callback,
    )
    _emit_status(status_callback, "IndexTTS 附属模型已就绪")


def ensure_models(
    status_callback: StatusCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Check all required models exist; auto-download missing ones."""
    with _MODEL_SETUP_LOCK:
        progress_reporter = _StartupProgressReporter(
            progress_callback,
            total_steps=len(_LOCAL_MODELS) + len(_DOWNLOADABLE_MODELS) + 1,
        )
        _emit_status(status_callback, "正在检查模型文件...")
        progress_reporter.begin_step(
            "启动检查",
            detail="正在检查模型文件...",
            current_value=0,
            current_maximum=1,
        )
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        for model in _LOCAL_MODELS:
            progress_reporter.begin_step(
                model.display_name,
                detail=f"正在检查 {model.display_name} 模型文件...",
                current_value=0,
                current_maximum=1,
            )
            if _has_model_markers(model):
                ready_message = f"{model.display_name} 已就绪"
                _emit_status(status_callback, ready_message)
                logger.info(ready_message)
                progress_reporter.finish_step(detail=f"{model.display_name} 已就绪")
                continue
            fail_message = f"{model.display_name} 模型未找到，请确认目录: {model.target_dir}"
            logger.error(fail_message)
            raise RuntimeError(fail_message)

        for model in _DOWNLOADABLE_MODELS:
            progress_reporter.begin_step(
                model.display_name,
                detail=f"正在检查 {model.display_name} 模型文件...",
                current_value=0,
                current_maximum=1,
            )
            if _has_model_markers(model):
                ready_message = f"{model.display_name} 已就绪"
                _emit_status(status_callback, ready_message)
                logger.info(ready_message)
                progress_reporter.finish_step(detail=f"{model.display_name} 已就绪")
                continue
            missing_message = f"{model.display_name} 模型未找到，开始自动下载..."
            _emit_status(status_callback, missing_message)
            logger.warning(missing_message)
            if _download_model(
                model,
                status_callback=status_callback,
                progress_callback=progress_reporter.child_progress_callback(model.display_name),
            ):
                done_message = f"{model.display_name} 下载完成"
                _emit_status(status_callback, done_message)
                logger.info(done_message)
                progress_reporter.finish_step(detail=done_message)
            else:
                fail_message = f"{model.display_name} 自动下载失败，请手动放置到 {model.target_dir}"
                logger.error(fail_message)
                raise RuntimeError(fail_message)

        progress_reporter.begin_step(
            "IndexTTS 附属模型",
            detail="正在检查 w2v-bert、semantic codec、CAMPPlus、BigVGAN...",
            current_value=0,
            current_maximum=4,
        )
        _ensure_indextts_aux_models(
            DEFAULT_INDEX_TTS_MODEL_DIR,
            status_callback=status_callback,
            progress_callback=progress_reporter.child_progress_callback("IndexTTS 附属模型"),
        )
        progress_reporter.finish_step(detail="IndexTTS 附属模型已就绪")
