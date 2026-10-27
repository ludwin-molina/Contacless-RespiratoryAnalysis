"""Plot the latest RGB session: position, rate, and signal strength for each flow method."""

import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

from acquisition.settings import rgb_recordings_dir

plt.rcParams.update({"font.size": 10})
OUTPUT_DIR = str(rgb_recordings_dir())


def get_latest_session_files():
    """Return the newest timestamp and the CSV files that belong to it."""
    if not os.path.exists(OUTPUT_DIR):
        print(f"Missing recordings folder: {OUTPUT_DIR}")
        return None

    files = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
    if not files:
        print("No CSV files in the RGB recordings folder.")
        return None

    timestamps = set()
    for path in files:
        parts = os.path.basename(path).split("_")
        if len(parts) >= 2:
            timestamps.add(f"{parts[0]}_{parts[1]}")
    if not timestamps:
        return None

    latest = sorted(timestamps)[-1]
    print(f"Latest RGB session: {latest}")
    session_files = glob.glob(os.path.join(OUTPUT_DIR, f"{latest}*.csv"))
    return latest, session_files


def plot_session(timestamp, file_list):
    """Save a three-panel comparison of the optical-flow methods."""
    data = {}
    for filepath in file_list:
        filename = os.path.basename(filepath)
        algo_name = filename.replace(f"{timestamp}_", "").replace(".csv", "")
        try:
            data[algo_name] = pd.read_csv(filepath, delimiter=";")
        except (OSError, pd.errors.ParserError, UnicodeError, ValueError) as exc:
            print(f"Could not read {filename}: {exc}")

    if not data:
        print("No readable samples.")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Optical-flow comparison, session {timestamp}", fontsize=14)
    colors = {"Farneback": "green", "Horn-Schunck": "orange", "Lucas-Kanade": "magenta"}

    ax1.set_title("1. Filtered respiratory position")
    ax1.set_ylabel("Amplitude [a.u.]")
    ax1.grid(True, alpha=0.3)
    for algo, frame in data.items():
        ax1.plot(frame["Elapsed_Sec"], frame["Signal_Pos"], label=algo, color=colors.get(algo, "blue"), linewidth=1.5, alpha=0.8)
    ax1.legend(loc="upper right")

    ax2.set_title("2. Estimated breaths per minute")
    ax2.set_ylabel("Breaths per minute")
    ax2.grid(True, alpha=0.3)
    for algo, frame in data.items():
        ax2.plot(frame["Elapsed_Sec"], frame["BPM"], label=algo, color=colors.get(algo, "blue"), linewidth=2)

    ax3.set_title("3. Signal strength (standard deviation)")
    ax3.set_ylabel("Standard deviation")
    ax3.set_xlabel("Time [s]")
    ax3.grid(True, alpha=0.3)
    for algo, frame in data.items():
        ax3.plot(frame["Elapsed_Sec"], frame["Signal_Std"], label=algo, color=colors.get(algo, "blue"), linewidth=1.5)

    output_img = os.path.join(OUTPUT_DIR, f"RESULT_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {output_img}")


def main():
    print("--- RGB session figure ---")
    found = get_latest_session_files()
    if not found:
        print("No session to plot.")
        return
    timestamp, files = found
    print(f"Reading {len(files)} files...")
    plot_session(timestamp, files)


if __name__ == "__main__":
    main()
