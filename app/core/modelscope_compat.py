from __future__ import annotations

import importlib
import inspect
from functools import cache
from typing import Any, Callable

SnapshotDownloadCallable = Callable[..., str]


@cache
def load_modelscope_snapshot_download() -> SnapshotDownloadCallable:
    try:
        module = importlib.import_module("modelscope_hub.compat")
        return module.snapshot_download
    except Exception:
        module = importlib.import_module("modelscope.hub.snapshot_download")
        return module.snapshot_download


@cache
def load_modelscope_hub_api() -> type[object]:
    module = importlib.import_module("modelscope.hub.api")
    return module.HubApi


def get_model_files(repo: str) -> list[dict[str, Any]]:
    HubApi = load_modelscope_hub_api()
    return HubApi().get_model_files(repo, recursive=True)


def snapshot_download(
    repo: str,
    *,
    local_dir: str,
    revision: str | None = None,
    progress_callbacks: list[type[object]] | None = None,
) -> str:
    download = load_modelscope_snapshot_download()
    signature = inspect.signature(download)

    kwargs: dict[str, Any] = {
        "local_dir": local_dir,
    }
    if revision is not None and "revision" in signature.parameters:
        kwargs["revision"] = revision
    if progress_callbacks is not None and "progress_callbacks" in signature.parameters:
        kwargs["progress_callbacks"] = progress_callbacks

    return download(repo, **kwargs)
