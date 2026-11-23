"""Paths, cohort constants, and the summary figures published in the paper."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FILE_LIST_PATH = DATA / "04_file-list.csv"
RECORDINGS_TABLE_PATH = DATA / "02_recordings.csv"
PHASES_PATH = DATA / "03_protocol-phases.csv"


def archive_root() -> Path:
    """Local study archive. It is not the analysis input and is not in git."""
    for name in ("recordings", "Recordings"):
        path = ROOT / name
        if (path / "manifest.csv").is_file():
            return path
    return ROOT / "recordings"


LEGACY_SYNCED = archive_root() / "reference" / "legacy_synced"

FS = 20.0
# Cross-correlation lags outside this window are rejected so a spurious peak
# cannot wipe the aligned series. Recordings in this cohort stay well inside it.
MAX_LAG_SEC = 30.0
MIN_ALIGNED_SEC = 5.0

RGB_METHODS = ("Farneback", "Horn-Schunck", "Lucas-Kanade")
METHOD_ORDER = ("ToF", "Farneback", "Horn-Schunck", "Lucas-Kanade")

# Both waveforms are inverted so inspiration on the camera matches the belt.
# The sign is the one that reproduces the archived synchronized files.
RGB_SIGNAL_COLUMN = "OpticalFlow"
RGB_SIGNAL_SIGN = -1.0
TOF_SIGNAL_COLUMN = "DistanceFiltered"
TOF_SIGNAL_SIGN = -1.0
SENSOR_TIME_COLUMN = "Elapsed_s"
SENSOR_BPM_COLUMN = "RespiratoryRate_bpm"
SENSOR_APNEA_COLUMN = "IsApnea"
BELT_TIME_COLUMN = "Time_s"
BELT_SIGNAL_COLUMN = "Filtered"
BELT_BPM_COLUMN = "RespiratoryRate_bpm"

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


def load_manifest(
    file_list_path: Path = FILE_LIST_PATH,
    recordings_path: Path = RECORDINGS_TABLE_PATH,
) -> pd.DataFrame:
    """Build the cohort index from the RODBUK tables in ``data/``.

    ``04_file-list.csv`` pairs each sensor file with its belt file.
    ``02_recordings.csv`` supplies the belt start time. Paths are relative
    to the repository root. The ToF method token in the file list is ``Tof``;
    the analysis uses ``ToF``.
    """
    inventory = pd.read_csv(file_list_path, dtype=str, keep_default_na=False)
    sessions = pd.read_csv(recordings_path, dtype=str, keep_default_na=False)
    sensors = inventory[inventory["role"] == "sensor"][
        ["participant_id", "modality", "method", "file_name"]
    ].copy()
    belts = inventory[inventory["role"] == "belt"][
        ["participant_id", "modality", "file_name"]
    ].rename(columns={"file_name": "reference_name"})
    merged = sensors.merge(belts, on=["participant_id", "modality"], how="left")
    merged = merged.merge(
        sessions[["participant_id", "modality", "session_start"]],
        on=["participant_id", "modality"],
        how="left",
    )
    if merged["reference_name"].eq("").any() or merged["session_start"].eq("").any():
        raise ValueError("A sensor file has no belt file or no session start in data/")

    manifest = pd.DataFrame(
        {
            "subject": merged["participant_id"],
            "modality": merged["modality"],
            "method": merged["method"].replace({"Tof": "ToF"}),
            "sensor_file": merged["file_name"].map(lambda name: (DATA / name).relative_to(ROOT).as_posix()),
            "reference_file": merged["reference_name"].map(
                lambda name: (DATA / name).relative_to(ROOT).as_posix()
            ),
            "belt_start": merged["session_start"],
        }
    )
    modality_rank = {"rgb": 0, "tof": 1}
    method_rank = {name: index for index, name in enumerate(RGB_METHODS)}
    manifest["_modality_rank"] = manifest["modality"].map(modality_rank)
    manifest["_method_rank"] = manifest["method"].map(lambda name: method_rank.get(name, len(RGB_METHODS)))
    manifest = (
        manifest.sort_values(["subject", "_modality_rank", "_method_rank"])
        .drop(columns=["_modality_rank", "_method_rank"])
        .reset_index(drop=True)
    )
    required = {"subject", "modality", "method", "sensor_file", "reference_file", "belt_start"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return manifest
