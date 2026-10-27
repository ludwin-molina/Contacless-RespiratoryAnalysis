"""Paths, cohort constants, and the summary figures published in the paper."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MANIFEST_PATH = DATA / "manifest.csv"
PHASES_PATH = DATA / "annotations" / "phases.csv"
LEGACY_SYNCED = DATA / "reference" / "legacy_synced"

FS = 20.0
# Cross-correlation lags outside this window are rejected so a spurious peak
# cannot wipe the aligned series. Recordings in this cohort stay well inside it.
MAX_LAG_SEC = 30.0
MIN_ALIGNED_SEC = 5.0

RGB_METHODS = ("Farneback", "Horn-Schunck", "Lucas-Kanade")
METHOD_ORDER = ("ToF", "Farneback", "Horn-Schunck", "Lucas-Kanade")

# Both waveforms are inverted so inspiration on the camera matches the belt.
# The sign is the one that reproduces the archived synchronized files.
RGB_SIGNAL_COLUMN = "Signal_Pos"
RGB_SIGNAL_SIGN = -1.0
TOF_SIGNAL_COLUMN = "Signal_Filtered"
TOF_SIGNAL_SIGN = -1.0

# Table 4 medians and Table 6 apnea sensitivity from the manuscript.
PUBLISHED_MEDIAN_MAE = {
    "ToF": 6.24,
    "Farneback": 4.24,
    "Horn-Schunck": 4.26,
    "Lucas-Kanade": 3.83,
}
PUBLISHED_APNEA_SENSITIVITY = {
    "ToF": 0.660,
    "Farneback": 0.530,
    "Horn-Schunck": 0.512,
    "Lucas-Kanade": 0.549,
}
PUBLISHED_APNEA_SPECIFICITY = {
    "ToF": 0.951,
    "Farneback": 0.983,
    "Horn-Schunck": 0.983,
    "Lucas-Kanade": 0.984,
}
PUBLISHED_APNEA_F1 = {
    "ToF": 0.662,
    "Farneback": 0.647,
    "Horn-Schunck": 0.632,
    "Lucas-Kanade": 0.667,
}


def load_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"subject", "modality", "method", "sensor_file", "reference_file", "belt_start"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return manifest
