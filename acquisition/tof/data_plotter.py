"""Plot the latest ToF recording: filtered distance, rate, and raw motion."""

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from acquisition.settings import tof_recordings_dir

OUTPUT_DIR = str(tof_recordings_dir())


def get_latest_session_timestamp():
    if not os.path.exists(OUTPUT_DIR):
        return None
    files = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
    if not files:
        return None
    timestamps = set()
    for path in files:
        parts = os.path.basename(path).split("_")
        if len(parts) >= 2:
            timestamps.add(f"{parts[0]}_{parts[1]}")
    if not timestamps:
        return None
    return sorted(timestamps)[-1]


def generate_session_plot(target_timestamp=None):
    """Save a three-panel figure for one session. Returns the PNG path."""
    timestamp = target_timestamp if target_timestamp else get_latest_session_timestamp()
    if not timestamp:
        print("[PLOTTER] No CSV files in the ToF recordings folder.")
        return None

    print(f"[PLOTTER] Building ToF figure: {timestamp}...")
    session_files = glob.glob(os.path.join(OUTPUT_DIR, f"{timestamp}*.csv"))
    data = {}
    for filepath in session_files:
        algo_name = os.path.basename(filepath).replace(f"{timestamp}_", "").replace(".csv", "")
        try:
            data[algo_name] = pd.read_csv(filepath, delimiter=";")
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            print(f"[PLOTTER] Could not read {filepath}: {exc}")

    if not data:
        print(f"[PLOTTER] No readable samples for {timestamp}.")
        return None

    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except (OSError, ValueError):
        pass

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"ToF distance report: {timestamp}", fontsize=16, fontweight="bold")

    # Raw millimetres sit around the subject distance, so the breathing trace is the filtered column.
    ax1.set_title("1. Distance change (respiratory signal)")
    ax1.set_ylabel("Amplitude [mm]")
    for algo, frame in data.items():
        ax1.plot(frame["Elapsed_Sec"], frame["Signal_Filtered"], label=f"{algo} (filtered)", color="purple", linewidth=1.5)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("2. Estimated respiratory rate")
    ax2.set_ylabel("Breaths per minute")
    for algo, frame in data.items():
        ax2.plot(frame["Elapsed_Sec"], frame["BPM"], label=algo, color="blue", linewidth=2)
    ax2.grid(True, alpha=0.3)

    ax3.set_title("3. Raw distance, mean removed")
    ax3.set_ylabel("Distance [mm]")
    ax3.set_xlabel("Time [s]")
    for _algo, frame in data.items():
        raw_centered = frame["Signal_Raw_mm"] - frame["Signal_Raw_mm"].mean()
        ax3.plot(frame["Elapsed_Sec"], raw_centered, label="Raw, mean removed", color="gray", alpha=0.6, linewidth=1)
    ax3.legend(loc="upper right")

    plt.tight_layout()
    output_filename = os.path.join(OUTPUT_DIR, f"REPORT_TOF_{timestamp}.png")
    plt.savefig(output_filename, dpi=150)
    plt.close(fig)
    print(f"[PLOTTER] Figure saved: {output_filename}")
    return output_filename


if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_session_plot(sys.argv[1])
    else:
        generate_session_plot()
