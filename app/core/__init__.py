"""Core configuration and runtime helpers."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import threading
from functools import cache
from pathlib import Path


class IndexTTSVendorLoader:
    """Load vendor/index-tts as a source package without editing sys.path."""

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self._package_root = project_root / "vendor" / "index-tts" / "indextts"
        self._lock = threading.RLock()

    def import_module(self, module_name: str):
        if module_name != "indextts" and not module_name.startswith("indextts."):
            raise ValueError(f"unsupported module name: {module_name}")
        with self._lock:
            self._ensure_package_loaded()
            return importlib.import_module(module_name)

    def _ensure_package_loaded(self) -> None:
        existing = sys.modules.get("indextts")
        if existing is not None:
            return
        init_file = self._package_root / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "indextts",
            init_file,
            submodule_search_locations=[str(self._package_root)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"failed to build import spec for {init_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["indextts"] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop("indextts", None)
            raise


@cache
def get_indextts_vendor_loader() -> IndexTTSVendorLoader:
    return IndexTTSVendorLoader()
