from __future__ import annotations

import math

import numpy as np
from scipy import signal


def resample_waveform(waveform: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if source_sample_rate == target_sample_rate or waveform.size == 0:
        return waveform.astype(np.float32, copy=False)
    source_length = int(waveform.shape[0])
    if source_length <= 1:
        return waveform.astype(np.float32, copy=False)
    waveform_f32 = waveform.astype(np.float32, copy=False)
    gcd = math.gcd(int(source_sample_rate), int(target_sample_rate))
    up = int(target_sample_rate) // gcd
    down = int(source_sample_rate) // gcd
    resampled = signal.resample_poly(waveform_f32, up, down)
    return np.ascontiguousarray(resampled.astype(np.float32, copy=False))
