from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class ASRRequest:
    audio_path: str = ""
    audio_ndarray: tuple[np.ndarray, int] | None = None
    language: str | None = None
    return_language: bool = True
    context: str | None = None
