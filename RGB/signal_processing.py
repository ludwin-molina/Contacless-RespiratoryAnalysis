import cv2
import numpy as np
import time
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks, detrend

# =================================================================
# SEKCJA 1: KONFIGURACJA I PARAMETRY SYGNAŁOWE
# =================================================================
# [PARAMETR]: Fizjologiczne granice częstotliwości oddechowej (BPM).
MIN_BPM = 4.0
MAX_BPM = 150.0
SAMPLING_RATE = 30.0

class BreathAnalysisEngine:
    """
    Zaawansowany silnik analityczny realizujący potok DSP (Digital Signal Processing).
    Odpowiada za filtrację pasmowoprzepustową, analizę rytmiczności (autokorelacja),
    detekcję bezdechów (ApneaGuard) oraz adaptacyjną estymację BPM.
    """
    def __init__(self, buffer_size_sec=30.0, fps=30.0):
        self.fps = fps
        # [PARAMETR]: Rozmiar bufora kołowego (w próbkach).
        self.buffer_size = int(buffer_size_sec * fps)

        # [PARAMETR - KALIBRACJA]: Projektowanie filtru Butterwortha 4. rzędu.
        # Pasmo 0.08 Hz - 2.5 Hz pokrywa pełen zakres ludzkiego oddechu.
        nyquist = 0.5 * fps
        self.b, self.a = butter(4, [0.08 / nyquist, 2.5 / nyquist], btype='band')

        # --- Parametry Integratora Bezdechu (Histereza) ---
        self.apnea_integrator = 0.0
        # [PARAMETR]: Szybkość wygaszania stanu alarmowego.
        self.integration_decay = 0.98  
        # [PARAMETR]: Próg aktywacji alarmu bezdechu w integratorze.
        self.apnea_threshold = 0.5
        # [PARAMETR]: Krok narastania stanu bezdechu na każdą próbkę.
        self.apnea_increment = 0.015 

        # Bufor historyczny dla debouncera gęstości występowania bezdechu
        self.apnea_history = []
        self.history_len = int(fps * 1.0)

        self.last_valid_bpm = 0.0
        self.raw_buffer = []
        self.times = []

    def process_sample(self, t_now, raw_val, breath_threshold):
        """
        Główna metoda procesora. Przetwarza pojedynczą próbkę wejściową i zwraca
        aktualny stan systemu: (BPM, flaga_bezdechu, sygnał_przefiltrowany, std_amplitudy).
        """
        self.times.append(t_now)
        self.raw_buffer.append(raw_val)

        # Zarządzanie rozmiarem okna przesuwnego
        if len(self.raw_buffer) > self.buffer_size:
            self.raw_buffer.pop(0)
            self.times.pop(0)

        # Wymagane minimum 2 sekundy danych do rozpoczęcia filtracji
        if len(self.raw_buffer) < self.fps * 2:
            return None, False, np.array(self.raw_buffer), 0.0

        raw_arr = np.array(self.raw_buffer)
        time_array = np.array(self.times)

        # =================================================================
        # KROK 1: FILTRACJA I DETRENDING (DSP)
        # =================================================================
        # Usuwanie trendu liniowego (dryftu niskoczęstotliwościowego)
        raw_detrended = detrend(raw_arr, type='linear')
        # Aplikacja filtru zerofazowego (filtfilt) w celu uniknięcia przesunięć grupowych
        filtered = filtfilt(self.b, self.a, raw_detrended)

        # =================================================================
        # KROK 2: ANALIZA STATYSTYCZNA (Amplituda i Rytm)
        # =================================================================
        # Analiza wariancji na krótkim oknie (3 sekundy) dla detekcji "ciszy" oddechowej
        tail_len = int(3.0 * self.fps)
        tail_data = filtered[-tail_len:] if len(filtered) > tail_len else filtered
        current_std = np.std(tail_data)

        # Analiza autokorelacji sygnału w celu wyznaczenia współczynnika rytmiczności
        rhythm_peak = 0.0
        n = len(filtered)
        if n > 0:
            acorr = np.correlate(filtered, filtered, mode='full')
            acorr = acorr[n - 1:] / (acorr[n - 1] + 1e-9)

            # [PARAMETR]: Zakres lagów (opóźnień) dla analizy okresowości (od 0.33s do 15s)
            min_lag = int(0.33 * self.fps)
            max_lag = min(int(15.0 * self.fps), n - 1)

            if max_lag > min_lag:
                valid_acorr = acorr[min_lag:max_lag]
                if len(valid_acorr) > 0:
                    rhythm_peak = np.max(valid_acorr)

        # =================================================================
        # KROK 3: LOGIKA APNEAGUARD (Wstępna klasyfikacja bezdechu)
        # =================================================================
        raw_apnea = False
        if current_std < breath_threshold:
            # Sytuacja A: Amplituda poniżej progu adaptacyjnego
            if current_std < (breath_threshold * 0.6):
                raw_apnea = True
            elif rhythm_peak < 0.55: # Brak rytmiczności przy niskiej amplitudzie
                raw_apnea = True
            else:
                raw_apnea = False
        else:
            # Sytuacja B: Amplituda poprawna, sprawdzamy spójność rytmu
            if rhythm_peak < 0.25:
                raw_apnea = True   
            else:
                raw_apnea = False  

        # --- Debouncer gęstości bezdechu (Filtracja zakłóceń klasyfikacji) ---
        self.apnea_history.append(1 if raw_apnea else 0)
        if len(self.apnea_history) > self.history_len:
            self.apnea_history.pop(0)

        local_apnea = raw_apnea
        if len(self.apnea_history) == self.history_len:
            apnea_density = sum(self.apnea_history) / self.history_len
            # [PARAMETR]: Wymagana gęstość próbek bezdechu w oknie 1s dla potwierdzenia stanu
            if apnea_density > 0.6:
                local_apnea = True

        # =================================================================
        # KROK 4: ESTYMACJA CZĘSTOTLIWOŚCI (BPM)
        # =================================================================
        current_bpm = None
        if not local_apnea:
            peak_vals = np.abs(filtered)
            
            # Dynamiczna estymacja poziomu szumu
            noise_floor = max(breath_threshold * 0.5, 0.05)
            max_val = np.percentile(peak_vals, 95) if len(peak_vals) > 0 else 1.0
            if max_val < noise_floor:
                max_val = noise_floor

            # Normalizacja sygnału dla algorytmu find_peaks
            norm_signal = np.clip(filtered / max_val, -2.0, 2.0)

            # [PARAMETR - KALIBRACJA]: Minimalny odstęp i minimalna wyrazistość (prominence) piku
            min_dist = int(self.fps * 0.33)
            peaks, _ = find_peaks(norm_signal, distance=min_dist, prominence=0.12)

            if len(peaks) >= 2:
                peak_times = time_array[peaks]
                intervals = np.diff(peak_times)

                # Walidacja interwałów między-szczytowych (fizjologiczna)
                valid_intervals = intervals[(intervals > 0.33) & (intervals < 15.0)]
                if len(valid_intervals) > 0:
                    
                    # --- Algorytm "Emergency Flush" (Reakcja na drastyczne zmiany tempa) ---
                    t_end = time_array[-1]
                    recent_peak_times = peak_times[peak_times > (t_end - 10.0)]
                    
                    recent_bpm_est = None
                    if len(recent_peak_times) >= 2:
                        recent_intervals = np.diff(recent_peak_times)
                        valid_recent = recent_intervals[(recent_intervals > 0.33) & (recent_intervals < 15.0)]
                        if len(valid_recent) > 0:
                            recent_bpm_est = 60.0 / np.median(valid_recent)
                            
                    global_bpm_est = 60.0 / np.median(valid_intervals)
                    
                    # [PARAMETR - KALIBRACJA]: Progi wyzwalające czyszczenie bufora (Emergency Flush)
                    if recent_bpm_est is not None:
                        bpm_diff = abs(recent_bpm_est - global_bpm_est)
                        # Wyzwalane, jeśli różnica > 12 BPM lub > 40% aktualnej wartości
                        if bpm_diff > 12.0 or (bpm_diff / max(global_bpm_est, 1.0)) > 0.4:
                            keep_len = int(10.0 * self.fps) 
                            if len(self.raw_buffer) > keep_len:
                                self.raw_buffer = self.raw_buffer[-keep_len:]
                                self.times = self.times[-keep_len:]
                                
                            current_bpm = recent_bpm_est 
                            
                    # --- Standardowa estymacja statystyczna (Stabilny oddech) ---
                    if current_bpm is None:
                        median_diff = np.median(valid_intervals)
                        # Odrzucenie wartości odstających (Outliers) powyżej 30% odchylenia od mediany
                        reliable_diffs = valid_intervals[
                            np.abs(valid_intervals - median_diff) < 0.3 * median_diff
                        ]

                        final_avg_diff = np.mean(reliable_diffs) if len(reliable_diffs) > 0 else median_diff

                        if final_avg_diff > 0:
                            current_bpm = 60.0 / final_avg_diff
                        
            # Filtracja artefaktów wywołanych gwałtownym powrotem oddechu po bezdechu
            if current_bpm is not None and self.apnea_integrator > 0.1:
                if self.last_valid_bpm > 0 and abs(current_bpm - self.last_valid_bpm) > 25.0:
                    current_bpm = None  
                elif self.last_valid_bpm == 0 and current_bpm > 50.0:
                    current_bpm = None  

        if current_bpm is None:
            local_apnea = True

        # =================================================================
        # KROK 5: INTEGRATOR BEZDECHU (Histereza alarmowa)
        # =================================================================
        if local_apnea:
            # Akceleracja narastania alarmu przy bardzo niskiej amplitudzie sygnału
            if current_std < (breath_threshold * 0.6):
                self.apnea_integrator += (self.apnea_increment * 1.5)
            else:
                self.apnea_integrator += self.apnea_increment
        else:
            # Wygaszanie stanu bezdechu przy wykryciu poprawnego rytmu
            self.apnea_integrator *= self.integration_decay

        self.apnea_integrator = max(0.0, min(self.apnea_integrator, 1.0))

        # --- Finalna decyzja o stanie pacjenta ---
        if self.apnea_integrator > self.apnea_threshold:
            self.last_valid_bpm = 0.0
            return 0.0, True, filtered, current_std
        else:
            if current_bpm is not None:
                self.last_valid_bpm = current_bpm
            return self.last_valid_bpm, False, filtered, current_std


def contrast_boost(gray):
    """
    Algorytm wzmocnienia kontrastu (Contrast Stretching) oparty na analizie 
    percentyli (1st-99th), optymalizujący obraz dla algorytmów Optical Flow.
    """
    lo, hi = np.percentile(gray, (1, 99))
    if hi <= lo:
        return gray
    g = np.clip((gray - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return g