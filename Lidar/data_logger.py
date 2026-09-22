import csv
import time
import os
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
        """Start sesji Lidar."""
        if self.is_recording:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.files = {}
        self.writers = {}

        print(f"[LOGGER] Start nagrywania Lidar: {timestamp}")

        for name in algo_names:
            clean_name = name.replace(" ", "_")
            filename = f"{self.output_dir}/{timestamp}_{clean_name}.csv"
            
            f = open(filename, mode='w', newline='', encoding='utf-8')
            writer = csv.writer(f, delimiter=';')
            
            # Nagłówki specyficzne dla Lidara
            writer.writerow([
                "Timestamp_Abs", 
                "Elapsed_Sec", 
                "Signal_Raw_mm",    # Surowy odczyt z ROI (zmieniający się dystans)
                "Signal_Filtered",  # Sygnał po filtracji pasmowej
                "BPM", 
                "Is_Apnea"
            ])
            
            self.files[name] = f
            self.writers[name] = writer

        self.is_recording = True

    def log_sample(self, algo_name, raw_mm, filtered_val, bpm, is_apnea):
        """Zapisuje próbkę."""
        if not self.is_recording:
            return

        if algo_name in self.writers:
            t_now = time.time()
            elapsed = t_now - self.start_time
            
            bpm_val = f"{bpm:.2f}" if bpm is not None else ""
            apnea_val = "1" if is_apnea else "0"
            
            self.writers[algo_name].writerow([
                f"{t_now:.3f}", 
                f"{elapsed:.3f}", 
                f"{raw_mm:.4f}", 
                f"{filtered_val:.6f}", 
                bpm_val, 
                apnea_val
            ])

    def stop(self):
        """Stop."""
        if not self.is_recording:
            return

        print("[LOGGER] Stop Lidar.")
        for f in self.files.values():
            try: f.close()
            except: pass
        
        self.files = {}
        self.writers = {}
        self.is_recording = False