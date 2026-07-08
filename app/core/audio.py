from __future__ import annotations

import numpy as np


def resample_waveform(waveform: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if source_sample_rate == target_sample_rate or waveform.size == 0:
        return waveform.astype(np.float32, copy=False)
    source_length = int(waveform.shape[0])
    if source_length <= 1:
        return waveform.astype(np.float32, copy=False)
    target_length = max(1, int(round(source_length * float(target_sample_rate) / float(source_sample_rate))))
    waveform_f32 = waveform.astype(np.float32, copy=False)
    source_positions = np.arange(source_length, dtype=np.float32)
    target_step = np.float32((source_length - 1) / max(target_length - 1, 1))
    target_positions = np.arange(target_length, dtype=np.float32) * target_step
    resampled = np.interp(target_positions, source_positions, waveform_f32)
    return resampled.astype(np.float32, copy=False)
