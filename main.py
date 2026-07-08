from __future__ import annotations

import importlib
import multiprocessing
import os
import sys
from functools import cache
from typing import TextIO

# Windowed compiled mode may set stdout/stderr to None.
# Some dependencies, such as transformers, still call .isatty() on them.
_FALLBACK_STD_STREAMS: list[TextIO] = []


def _ensure_standard_streams() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None:
            continue
        fallback = open(os.devnull, "w", encoding="utf-8", buffering=1)
        _FALLBACK_STD_STREAMS.append(fallback)
        setattr(sys, name, fallback)


_ensure_standard_streams()


@cache
def _load_ui_main():
    module = importlib.import_module("ui.app")
    return module.main


def main() -> int:
    return _load_ui_main()()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
