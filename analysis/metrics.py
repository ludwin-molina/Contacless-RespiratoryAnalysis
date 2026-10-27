"""Waveform agreement, respiratory-rate error, and apnea detection.

The synced traces are sampled at 20 Hz. Every published rate statistic uses
one sample per second, which is every 20th row of that grid.

The MAE in Table 4 is the mean absolute error of those samples over the whole
recording, including the breath-hold. Bland–Altman bias and limits of agreement
drop samples whose reference rate is zero, because that phase is scored
separately as apnea detection.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.synchronize import belt_time_to_relative


def _finite_pair(left, right):
    mask = np.isfinite(left) & np.isfinite(right)
    return left[mask], right[mask]


def spearman_waveform(estimated, reference):
    left, right = _finite_pair(np.asarray(estimated, dtype=float), np.asarray(reference, dtype=float))
    if len(left) < 3 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return np.nan
    result = spearmanr(left, right)
    coefficient = result.correlation if hasattr(result, "correlation") else result[0]
    return float(coefficient)


def at_one_hertz(frame):
    """Return every 20th row of a trace that was resampled at 20 Hz."""
    return frame.iloc[::20]


def one_hertz_mae(estimated, reference):
    """Mean absolute error of one sample per second, breath-hold included."""
    estimated = np.asarray(estimated, dtype=float)[::20]
    reference = np.asarray(reference, dtype=float)[::20]
    estimated, reference = _finite_pair(estimated, reference)
    if len(estimated) == 0:
        return np.nan
    return float(np.mean(np.abs(estimated - reference)))


def mae_bpm(estimated, reference, keep):
    left = np.asarray(estimated, dtype=float)[keep]
    right = np.asarray(reference, dtype=float)[keep]
    left, right = _finite_pair(left, right)
    if len(left) == 0:
        return np.nan
    return float(np.mean(np.abs(left - right)))


def bland_altman(estimated, reference):
    """Bias and 95 percent limits of agreement of the signed rate error."""
    difference = np.asarray(estimated, dtype=float) - np.asarray(reference, dtype=float)
    difference = difference[np.isfinite(difference)]
    if len(difference) < 2:
        return {"bias": np.nan, "loa_low": np.nan, "loa_high": np.nan, "sd": np.nan}
    bias = float(np.mean(difference))
    sd = float(np.std(difference, ddof=1))
    return {"bias": bias, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd, "sd": sd}


def _nearest_indices(time, step=1.0):
    time = np.asarray(time, dtype=float)
    if len(time) == 0:
        return np.array([], dtype=int)
    grid = np.arange(time[0], time[-1] + 1e-9, step)
    positions = np.searchsorted(time, grid)
    positions = np.clip(positions, 1, len(time) - 1)
    left = positions - 1
    choose_left = np.abs(time[left] - grid) <= np.abs(time[positions] - grid)
    return np.where(choose_left, left, positions)


def apnea_scores(time, reference_bpm, estimated_apnea):
    """Sensitivity, specificity, and F1 of apnea labels at 1 Hz."""
    indices = _nearest_indices(time, step=1.0)
    if len(indices) == 0:
        return {
            "sensitivity": np.nan,
            "specificity": np.nan,
            "f1": np.nan,
            "n": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }
    reference = np.isclose(np.asarray(reference_bpm, dtype=float)[indices], 0.0)
    predicted = np.asarray(estimated_apnea)[indices].astype(bool)
    true_positive = int(np.count_nonzero(predicted & reference))
    true_negative = int(np.count_nonzero(~predicted & ~reference))
    false_positive = int(np.count_nonzero(predicted & ~reference))
    false_negative = int(np.count_nonzero(~predicted & reference))

    def _rate(numerator, denominator):
        if denominator == 0:
            return np.nan
        return numerator / denominator

    sensitivity = _rate(true_positive, true_positive + false_negative)
    specificity = _rate(true_negative, true_negative + false_positive)
    precision = _rate(true_positive, true_positive + false_positive)
    if np.isnan(precision) or np.isnan(sensitivity) or precision + sensitivity == 0:
        f1 = np.nan
    else:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "n": int(len(indices)),
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
    }


def _phase_mask(time, start, stop):
    if start is None or stop is None or not np.isfinite(start) or not np.isfinite(stop):
        return np.zeros(len(time), dtype=bool)
    return (time >= start) & (time <= stop)


def bland_altman_points(aligned):
    """Instantaneous rate pairs at 1 Hz, breath-hold samples removed."""
    samples = at_one_hertz(aligned)
    estimated = samples["Estimated_BPM"].to_numpy(dtype=float)
    reference = samples["GT_BPM"].to_numpy(dtype=float)
    keep = np.isfinite(estimated) & np.isfinite(reference) & ~np.isclose(reference, 0.0)
    return pd.DataFrame(
        {
            "mean_bpm": 0.5 * (estimated[keep] + reference[keep]),
            "difference": estimated[keep] - reference[keep],
        }
    )


def metrics_for_recording(aligned, meta, phase_row):
    samples = at_one_hertz(aligned)
    time = samples["Relative_Time_s"].to_numpy(dtype=float)
    reference_bpm = samples["GT_BPM"].to_numpy(dtype=float)
    estimated_bpm = samples["Estimated_BPM"].to_numpy(dtype=float)
    breathing = np.isfinite(reference_bpm) & ~np.isclose(reference_bpm, 0.0)

    hyper_start = belt_time_to_relative(phase_row["hyperventilation_start_s"], meta)
    hyper_stop = belt_time_to_relative(phase_row["hyperventilation_stop_s"], meta)
    if np.isfinite(hyper_start):
        normal = breathing & _phase_mask(time, time[0], hyper_start)
    else:
        normal = np.zeros(len(time), dtype=bool)
    hyper = breathing & _phase_mask(time, hyper_start, hyper_stop)

    difference = estimated_bpm[breathing] - reference_bpm[breathing]
    difference = difference[np.isfinite(difference)]
    apnea = apnea_scores(time, reference_bpm, samples["Estimated_Is_Apnea"].to_numpy())

    return {
        "spearman": spearman_waveform(aligned["Estimated_Signal_ZScore"], aligned["GT_Signal_ZScore"]),
        "mae": one_hertz_mae(aligned["Estimated_BPM"], aligned["GT_BPM"]),
        "mae_normal": mae_bpm(estimated_bpm, reference_bpm, normal),
        "mae_hyperventilation": mae_bpm(estimated_bpm, reference_bpm, hyper),
        "phase_status": phase_row["status"],
        "ba_n": int(len(difference)),
        "ba_sum": float(np.sum(difference)) if len(difference) else 0.0,
        "ba_sumsq": float(np.sum(difference ** 2)) if len(difference) else 0.0,
        **{f"apnea_{key}": value for key, value in apnea.items()},
    }


def summarize_methods(per_recording):
    """Median and mean of the per-subject metrics, in the paper's method order."""
    rows = []
    for method, group in per_recording.groupby("method", sort=False):
        def _median(column):
            return float(np.nanmedian(group[column]))

        def _mean_sd(column):
            values = group[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                return np.nan, np.nan
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            return float(np.mean(values)), sd

        mae_mean, mae_sd = _mean_sd("mae")
        spearman_mean, spearman_sd = _mean_sd("spearman")
        normal_mean, normal_sd = _mean_sd("mae_normal")
        hyper_mean, hyper_sd = _mean_sd("mae_hyperventilation")
        count = float(group["ba_n"].sum())
        total = float(group["ba_sum"].sum())
        total_sq = float(group["ba_sumsq"].sum())
        if count > 1:
            bias = total / count
            variance = (total_sq - (total ** 2) / count) / (count - 1)
            spread = float(np.sqrt(max(variance, 0.0)))
            loa_low = bias - 1.96 * spread
            loa_high = bias + 1.96 * spread
        else:
            bias = loa_low = loa_high = np.nan
        pooled = {
            "tp": int(group["apnea_tp"].sum()),
            "tn": int(group["apnea_tn"].sum()),
            "fp": int(group["apnea_fp"].sum()),
            "fn": int(group["apnea_fn"].sum()),
        }

        def _safe(numerator, denominator):
            return np.nan if denominator == 0 else numerator / denominator

        sensitivity = _safe(pooled["tp"], pooled["tp"] + pooled["fn"])
        specificity = _safe(pooled["tn"], pooled["tn"] + pooled["fp"])
        precision = _safe(pooled["tp"], pooled["tp"] + pooled["fp"])
        if np.isnan(precision) or np.isnan(sensitivity) or precision + sensitivity == 0:
            f1 = np.nan
        else:
            f1 = 2 * precision * sensitivity / (precision + sensitivity)

        rows.append(
            {
                "method": method,
                "n": int(len(group)),
                "spearman_median": _median("spearman"),
                "spearman_q1": float(np.nanpercentile(group["spearman"], 25)),
                "spearman_q3": float(np.nanpercentile(group["spearman"], 75)),
                "spearman_mean": spearman_mean,
                "spearman_sd": spearman_sd,
                "mae_median": _median("mae"),
                "mae_mean": mae_mean,
                "mae_sd": mae_sd,
                "mae_normal_mean": normal_mean,
                "mae_normal_sd": normal_sd,
                "mae_normal_median": _median("mae_normal"),
                "mae_hyperventilation_mean": hyper_mean,
                "mae_hyperventilation_sd": hyper_sd,
                "mae_hyperventilation_median": _median("mae_hyperventilation"),
                "bland_bias": bias,
                "bland_loa_low": loa_low,
                "bland_loa_high": loa_high,
                "apnea_sensitivity": sensitivity,
                "apnea_specificity": specificity,
                "apnea_f1": f1,
            }
        )
    summary = pd.DataFrame(rows)
    order = ["ToF", "Farneback", "Horn-Schunck", "Lucas-Kanade"]
    summary["method"] = pd.Categorical(summary["method"], categories=order, ordered=True)
    return summary.sort_values("method").reset_index(drop=True)
