from __future__ import annotations

import sys
from typing import TextIO

from loguru import logger


def _resolve_level(message: str, file: TextIO | None) -> str:
    normalized = message.strip()
    upper = normalized.upper()
    if file is sys.stderr:
        return "ERROR"
    if upper.startswith(("ERROR", "[FAILED]", "FAILED")):
        return "ERROR"
    if upper.startswith(("WARNING", "[WARNING]", "WARN")):
        return "WARNING"
    return "INFO"


def print_loguru(*args: object, sep: str = " ", end: str = "\n", file: TextIO | None = None, flush: bool = False) -> None:
    del flush
    message = sep.join(str(arg) for arg in args) + end
    logger.log(_resolve_level(message, file), message.rstrip("\n"))
