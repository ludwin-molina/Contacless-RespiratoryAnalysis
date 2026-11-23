"""Recover the protocol phases from the reference respiratory rate.

The belt annotation files only contain a header. The manual clicks were
applied when the reference rate was estimated and were not stored again.
The breath-hold is the long stretch forced to 0 breaths per minute.
Hyperventilation is the tall rate bout before that stretch, not the longest
mild rise. A rise of a few breaths per minute is not treated as the protocol
bout.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.config import PHASES_PATH


def _runs(mask):
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    changes = np.diff(padded.astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts, ends))


def _bridge_short_gaps(mask, time, gap_s):
    filled = np.asarray(mask, dtype=bool).copy()
    for start, end in _runs(~filled):
        if start == 0 or end >= len(filled):
            continue
        if time[end - 1] - time[start] <= gap_s:
            filled[start:end] = True
    return filled


def _smooth_rate(time, bpm):
    step = float(np.median(np.diff(time)))
    width = max(5, int(round(6.0 / step)))
    positive = np.asarray(bpm, dtype=float).copy()
    positive[np.isclose(positive, 0.0)] = np.nan
    smooth = (
        pd.Series(positive)
        .interpolate(limit_direction="both")
        .rolling(width, center=True, min_periods=max(3, width // 3))
        .median()
        .to_numpy()
    )
    return smooth


def _hyperventilation_window(time, bpm, apnea_start):
    smooth = _smooth_rate(time, bpm)
    pre = (time >= 8.0) & (time < apnea_start - 2.0) & np.isfinite(smooth)
    if int(np.count_nonzero(pre)) < 20:
        return np.nan, np.nan
    baseline = float(np.nanpercentile(smooth[pre], 20))
    search = pre & (time >= 15.0)
    if int(np.count_nonzero(search)) < 10:
        return np.nan, np.nan
    peak = float(np.nanmax(smooth[search]))
    # The protocol bout is a clear step up. A drift of a few BPM is resting variation.
    if peak < baseline + 4.0:
        return np.nan, np.nan
    threshold = max(baseline + 4.0, baseline + 0.40 * (peak - baseline))
    elevated = _bridge_short_gaps(search & (smooth >= threshold), time, gap_s=3.0)
    candidates = []
    for start, end in _runs(elevated):
        if time[end - 1] - time[start] < 8.0:
            continue
        candidates.append((float(np.nanmedian(smooth[start:end])), start, end))
    if not candidates:
        return np.nan, np.nan
    _score, start, end = max(candidates, key=lambda item: (item[0], item[1]))
    return float(time[start]), float(time[end - 1])


def infer_phase_windows(reference):
    """Return apnea and hyperventilation bounds on the belt clock."""
    time = reference["Time"].to_numpy(dtype=float)
    bpm = reference["BPM"].to_numpy(dtype=float)
    zero_runs = []
    for start, end in _runs(np.isclose(bpm, 0.0)):
        if time[end - 1] - time[start] >= 10.0:
            zero_runs.append((start, end))
    if not zero_runs:
        return {
            "apnea_start_s": np.nan,
            "apnea_stop_s": np.nan,
            "hyperventilation_start_s": np.nan,
            "hyperventilation_stop_s": np.nan,
            "status": "segmented",
        }

    apnea_start_i, apnea_end_i = max(zero_runs, key=lambda run: (run[1] - run[0], run[0]))
    apnea_start = float(time[apnea_start_i])
    apnea_stop = float(time[apnea_end_i - 1])
    hyper_start, hyper_stop = _hyperventilation_window(time, bpm, apnea_start)

    if not np.isfinite(hyper_start) and "Filtered" in reference.columns:
        from analysis.ground_truth import estimate_bpm

        step = float(np.median(np.diff(time)))
        fast = estimate_bpm(
            time,
            reference["Filtered"].to_numpy(dtype=float),
            1.0 / step,
            fast_start=float(time[0]),
            fast_stop=apnea_start,
        )
        hyper_start, hyper_stop = _hyperventilation_window(time, fast, apnea_start)

    return {
        "apnea_start_s": apnea_start,
        "apnea_stop_s": apnea_stop,
        "hyperventilation_start_s": hyper_start,
        "hyperventilation_stop_s": hyper_stop,
        "status": "segmented",
    }


def ensure_phases(manifest, path: Path = PHASES_PATH):
    """Load the protocol windows published in ``Data/03_protocol-phases.csv``."""
    if not path.is_file():
        raise FileNotFoundError(f"Protocol phases are not in the dataset: {path}")
    phases = pd.read_csv(path)
    if "participant_id" in phases.columns:
        phases = phases.rename(columns={"participant_id": "subject"})
    needed = {(row.subject, row.modality) for row in manifest.itertuples(index=False)}
    present = {(row.subject, row.modality) for row in phases.itertuples(index=False)}
    missing = needed.difference(present)
    if missing:
        raise ValueError(f"Protocol phases are missing sessions: {sorted(missing)}")
    return phases
