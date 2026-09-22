import csv
import time
import os
from datetime import datetime

class BreathLogger:
    def __init__(self, output_dir="recordings"):
        self.output_dir = output_dir
        self.is_recording = False
        self.files = {}     # Uchwyty do plików
        self.writers = {}   # Obiekty piszące CSV
        self.start_time = 0.0

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def start(self, algo_names):
        """Rozpoczyna nową sesję nagrywania."""
        if self.is_recording:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.files = {}
        self.writers = {}

        print(f"[LOGGER] Start nagrywania sesji: {timestamp}")

        for name in algo_names:
            # Tworzymy nazwę pliku: recordings/20260106_1230_Farneback.csv
            clean_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            filename = f"{self.output_dir}/{timestamp}_{clean_name}.csv"
            
            # Otwieramy plik w trybie zapisu
            # newline='' jest wymagane przez moduł csv w Pythonie 3
            f = open(filename, mode='w', newline='', encoding='utf-8')
            writer = csv.writer(f, delimiter=';') # Średnik do Excela
            
            # Nagłówki kolumn
            writer.writerow([
                "Timestamp_Abs",    # Czas systemowy
                "Elapsed_Sec",      # Czas od początku nagrania
                "Signal_Pos",       # Pozycja (Sygnał po całkowaniu)
                "Signal_Std",       # Odchylenie standardowe (Amplituda)
                "BPM",              # Obliczone oddechy na minutę
                "Is_Apnea"          # Flaga bezdechu (0 lub 1)
            ])
            
            self.files[name] = f
            self.writers[name] = writer

        self.is_recording = True

    def log_sample(self, algo_name, signal_pos, std_val, bpm, is_apnea):
        """Zapisuje pojedynczą próbkę dla danego algorytmu."""
        if not self.is_recording:
            return

        if algo_name in self.writers:
            t_now = time.time()
            elapsed = t_now - self.start_time
            
            # Obsługa wartości None i boolean
            bpm_val = f"{bpm:.2f}" if bpm is not None else ""
            apnea_val = "1" if is_apnea else "0"
            
            # Zapis wiersza
            self.writers[algo_name].writerow([
                f"{t_now:.3f}", 
                f"{elapsed:.3f}", 
                f"{signal_pos:.6f}", 
                f"{std_val:.6f}", 
                bpm_val, 
                apnea_val
            ])
            # Opcjonalnie: flush() co jakiś czas, jeśli boisz się utraty danych przy crashu
            # self.files[algo_name].flush()

    def stop(self):
        """Kończy nagrywanie i zamyka pliki."""
        if not self.is_recording:
            return

        print("[LOGGER] Stop nagrywania.")
        for f in self.files.values():
            try:
                f.close()
            except Exception as e:
                print(f"[LOGGER] Błąd przy zamykaniu pliku: {e}")
        
        self.files = {}
        self.writers = {}
        self.is_recording = False