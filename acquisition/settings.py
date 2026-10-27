"""Paths and network settings for the MaixSense apps.

RGB and ToF talk to the same sensor address, so only one app can stream at a time.
Override the defaults with environment variables before starting an app.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sensor_host() -> str:
    return os.environ.get("MAIX_HOST", "192.168.233.1")


def sensor_port() -> int:
    return int(os.environ.get("MAIX_PORT", "80"))


def rgb_http_port() -> int:
    return int(os.environ.get("RGB_PORT", "5000"))


def tof_http_port() -> int:
    return int(os.environ.get("TOF_PORT", "8080"))


def tof_grid_port() -> int:
    return int(os.environ.get("TOF_GRID_PORT", "8081"))


def rgb_recordings_dir() -> Path:
    path = Path(os.environ.get("RGB_RECORDINGS", ROOT / "acquisition" / "rgb" / "recordings"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def tof_recordings_dir() -> Path:
    path = Path(os.environ.get("TOF_RECORDINGS", ROOT / "acquisition" / "tof" / "recordings"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def pose_onnx_path() -> Path:
    return Path(
        os.environ.get(
            "POSE_ONNX",
            ROOT / "acquisition" / "rgb" / "lightweight_pose_estimation_201912.onnx",
        )
    )
