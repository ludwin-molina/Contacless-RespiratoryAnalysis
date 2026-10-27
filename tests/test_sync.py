import numpy as np
import pandas as pd

from analysis.config import (
    LEGACY_SYNCED,
    RGB_SIGNAL_COLUMN,
    RGB_SIGNAL_SIGN,
    ROOT,
    TOF_SIGNAL_COLUMN,
    TOF_SIGNAL_SIGN,
)
from analysis.synchronize import belt_time_to_relative, synchronize


def _compare(aligned, legacy, estimated_column, bpm_column):
    assert len(aligned) == len(legacy)
    pairs = [
        ("Estimated_Signal_ZScore", estimated_column),
        ("GT_Signal_ZScore", "GT_Signal_ZScore"),
        ("Estimated_BPM", bpm_column),
        ("GT_BPM", "GT_BPM"),
        ("Relative_Time_s", "Relative_Time_s"),
    ]
    for new_name, old_name in pairs:
        new_values = aligned[new_name].to_numpy(dtype=float)
        old_values = legacy[old_name].to_numpy(dtype=float)
        assert np.allclose(new_values, old_values, rtol=1e-5, atol=1e-5), new_name
    assert set(np.unique(aligned["Estimated_Is_Apnea"])).issubset({0, 1})


def test_missing_phase_time_stays_missing():
    assert np.isnan(belt_time_to_relative(np.float64("nan"), {}))
    assert np.isnan(belt_time_to_relative(None, {}))


def test_rgb_sync_matches_archived_s01():
    aligned, _ = synchronize(
        ROOT / "data/rgb/20260503_132612_Farneback.csv",
        ROOT / "data/reference/processed_data-S01-rgb.csv",
        RGB_SIGNAL_COLUMN,
        RGB_SIGNAL_SIGN,
    )
    legacy = pd.read_csv(LEGACY_SYNCED / "synced_S01_Farneback.csv", sep=";")
    _compare(aligned, legacy, "RGB_Signal_ZScore", "RGB_BPM")


def test_tof_sync_matches_archived_s01():
    aligned, _ = synchronize(
        ROOT / "data/tof/20260503_131345_ToF.csv",
        ROOT / "data/reference/processed_data-S01-tof.csv",
        TOF_SIGNAL_COLUMN,
        TOF_SIGNAL_SIGN,
    )
    legacy = pd.read_csv(LEGACY_SYNCED / "synced_S01_ToF.csv", sep=";")
    _compare(aligned, legacy, "Lidar_Signal_ZScore", "Lidar_BPM")
