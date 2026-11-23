"""Resample a camera trace and the belt onto one 20 Hz grid and align them.

The camera and the belt are independent clocks. Both series are anchored at
their last sample, interpolated, z-scored, and shifted by the lag that
maximizes the cross-correlation. That is the preparation in section 3.3.3.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import zscore

from analysis.config import (
    BELT_BPM_COLUMN,
    BELT_SIGNAL_COLUMN,
    BELT_TIME_COLUMN,
    FS,
    MAX_LAG_SEC,
    MIN_ALIGNED_SEC,
    SENSOR_APNEA_COLUMN,
    SENSOR_BPM_COLUMN,
    SENSOR_TIME_COLUMN,
)


def _fill_nan(values):
    values = np.asarray(values, dtype=float)
    missing = np.isnan(values)
    if not missing.any():
        return values
    if missing.all():
        return np.zeros_like(values)
    known = np.flatnonzero(~missing)
    values = values.copy()
    values[missing] = np.interp(np.flatnonzero(missing), known, values[known])
    return values


def _shift(values, lag):
    if lag > 0:
        return values[lag:]
    if lag < 0:
        return values[:lag]
    return values


def synchronize(sensor_file, reference_file, signal_column, signal_sign, fs=FS):
    """Return the aligned frame and the bookkeeping needed to map belt time.

    ``signal_sign`` is -1 for both RGB optical flow and filtered ToF depth:
    that is the polarity that matches the archived synchronized files.
    ``IsApnea`` is carried through the same lag as the waveform. Samples that
    would be dropped by an extreme lag fall back to zero lag.
    """
    sensor = pd.read_csv(sensor_file)
    reference = pd.read_csv(reference_file)

    sensor_time = _fill_nan(sensor[SENSOR_TIME_COLUMN])
    reference_time = _fill_nan(reference[BELT_TIME_COLUMN])
    sensor_signal = signal_sign * _fill_nan(sensor[signal_column])
    reference_signal = _fill_nan(reference[BELT_SIGNAL_COLUMN])
    sensor_bpm = _fill_nan(sensor[SENSOR_BPM_COLUMN])
    reference_bpm = _fill_nan(reference[BELT_BPM_COLUMN])
    if SENSOR_APNEA_COLUMN in sensor.columns:
        sensor_apnea = _fill_nan(sensor[SENSOR_APNEA_COLUMN])
    else:
        sensor_apnea = np.zeros(len(sensor_time))

    sensor_zero = sensor_time - sensor_time[-1]
    reference_zero = reference_time - reference_time[-1]
    t_start = max(sensor_zero[0], reference_zero[0])
    t_end = min(sensor_zero[-1], reference_zero[-1])
    if t_end <= t_start:
        raise ValueError(f"No overlapping time in {sensor_file} and {reference_file}")
    common_time = np.arange(t_start, t_end, 1.0 / fs)

    def _interp(source_time, values):
        return interp1d(source_time, values, bounds_error=False, fill_value="extrapolate")(common_time)

    sensor_on_grid = _interp(sensor_zero, sensor_signal)
    reference_on_grid = _interp(reference_zero, reference_signal)
    sensor_bpm_on_grid = _interp(sensor_zero, sensor_bpm)
    reference_bpm_on_grid = _interp(reference_zero, reference_bpm)
    apnea_on_grid = _interp(sensor_zero, sensor_apnea)

    sensor_norm = np.asarray(zscore(sensor_on_grid), dtype=float)
    reference_norm = np.asarray(zscore(reference_on_grid), dtype=float)

    correlation = np.correlate(sensor_norm, reference_norm, mode="full")
    lags = np.arange(-len(reference_norm) + 1, len(sensor_norm))
    max_lag = int(MAX_LAG_SEC * fs)
    admissible = np.abs(lags) <= max_lag
    if not np.any(admissible):
        admissible = np.ones(len(lags), dtype=bool)
    best_lag = int(lags[admissible][np.argmax(correlation[admissible])])

    min_samples = int(MIN_ALIGNED_SEC * fs)
    if abs(best_lag) >= len(sensor_norm) - min_samples:
        best_lag = 0

    sensor_alg = _shift(sensor_norm, best_lag)
    sensor_bpm_alg = _shift(sensor_bpm_on_grid, best_lag)
    apnea_alg = _shift(apnea_on_grid, best_lag)
    if best_lag > 0:
        reference_alg = reference_norm[:-best_lag]
        reference_bpm_alg = reference_bpm_on_grid[:-best_lag]
        time_alg = common_time[best_lag:]
        gt_index_offset = 0
    elif best_lag < 0:
        abs_lag = abs(best_lag)
        reference_alg = reference_norm[abs_lag:]
        reference_bpm_alg = reference_bpm_on_grid[abs_lag:]
        time_alg = common_time[:-abs_lag]
        gt_index_offset = abs_lag
    else:
        reference_alg = reference_norm
        reference_bpm_alg = reference_bpm_on_grid
        time_alg = common_time
        gt_index_offset = 0

    length = min(len(sensor_alg), len(reference_alg), len(time_alg), len(sensor_bpm_alg), len(reference_bpm_alg))
    if length < min_samples:
        raise ValueError(f"Aligned series from {Path(sensor_file).name} is only {length} samples")

    relative_time = time_alg[:length] - time_alg[0]
    aligned = pd.DataFrame(
        {
            "Relative_Time_s": relative_time,
            "Estimated_Signal_ZScore": sensor_alg[:length],
            "GT_Signal_ZScore": reference_alg[:length],
            "Estimated_BPM": sensor_bpm_alg[:length],
            "GT_BPM": reference_bpm_alg[:length],
            "Estimated_Is_Apnea": (apnea_alg[:length] >= 0.5).astype(int),
        }
    )
    meta = {
        "t_gt_end": float(reference_time[-1]),
        "t_start": float(t_start),
        "best_lag": best_lag,
        "gt_index_offset": int(gt_index_offset),
        "fs": float(fs),
        "relative_time": relative_time,
    }
    return aligned, meta


def belt_time_to_relative(belt_time, meta):
    """Map a timestamp from the belt file onto the aligned relative clock."""
    try:
        value = float(belt_time)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(value):
        return np.nan
    zeroed = value - meta["t_gt_end"]
    index = int(np.rint((zeroed - meta["t_start"]) * meta["fs"]))
    output_index = index - meta["gt_index_offset"]
    relative = meta["relative_time"]
    if output_index < 0 or output_index >= len(relative):
        return np.nan
    return float(relative[output_index])
