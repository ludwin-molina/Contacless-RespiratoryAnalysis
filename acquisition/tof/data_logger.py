"""Write one ToF sample per row and flush it before the next frame."""

import csv
import os
import time
from datetime import datetime


class BreathLogger:
    def __init__(self, output_dir="recordings"):
        self.output_dir = output_dir
        self.is_recording = False
        self.files = {}
        self.writers = {}
        self.start_time = 0.0

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def start(self, algo_names):
        """Open one CSV per method name."""
        if self.is_recording:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.files = {}
        self.writers = {}

        print(f"[LOGGER] ToF recording started: {timestamp}")

        for name in algo_names:
            clean_name = name.replace(" ", "_")
            filename = f"{self.output_dir}/{timestamp}_{clean_name}.csv"

            handle = open(filename, mode="w", newline="", encoding="utf-8")
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(
                [
                    "Timestamp_Abs",
                    "Elapsed_Sec",
                    "Signal_Raw_mm",
                    "Signal_Filtered",
                    "BPM",
                    "Is_Apnea",
                ]
            )

            self.files[name] = handle
            self.writers[name] = writer

        self.is_recording = True

    def log_sample(self, algo_name, raw_mm, filtered_val, bpm, is_apnea):
        """Append one sample and flush it so a crash does not drop the row."""
        if not self.is_recording or algo_name not in self.writers:
            return

        now = time.time()
        elapsed = now - self.start_time
        bpm_val = f"{bpm:.2f}" if bpm is not None else ""
        apnea_val = "1" if is_apnea else "0"
        self.writers[algo_name].writerow(
            [
                f"{now:.3f}",
                f"{elapsed:.3f}",
                f"{raw_mm:.4f}",
                f"{filtered_val:.6f}",
                bpm_val,
                apnea_val,
            ]
        )
        self.files[algo_name].flush()

    def stop(self):
        """Flush and close every open file."""
        if not self.is_recording:
            return

        print("[LOGGER] ToF recording stopped.")
        for name, handle in self.files.items():
            try:
                handle.flush()
                handle.close()
            except OSError as exc:
                print(f"[LOGGER] Could not close {name}: {exc}")

        self.files = {}
        self.writers = {}
        self.is_recording = False
