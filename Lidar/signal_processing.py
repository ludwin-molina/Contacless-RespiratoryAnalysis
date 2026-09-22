import cv2
import numpy as np
import time
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks

# =================================================================
# SEKCJA 1: IMPORTY WEWNĘTRZNE
# =================================================================
from apnea_guard_plus import ApneaGuardPlus

class AdaptiveTrackingStrategy:
    """
    Klasa odpowiedzialna za przestrzenne śledzenie klatki piersiowej pacjenta.
    Wykorzystuje modele wariancji do inicjalizacji oraz dynamiczne śledzenie 
    w osi Z (głębia) w celu redukcji dryftu i szumów ruchowych (detrending przestrzenny).
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.calibration_frames = []
        self.tracking_active = False
        self.roi_mask = None
        self.last_centroid = None   # Współrzędne (x, y) w postaci float
        self.last_z_ref = None
        
        # [PARAMETR]: Promień poszukiwań (w pikselach) dla algorytmu śledzenia.
        self.search_radius = 60     
        
        # [PARAMETR]: Tolerancja głębi (w mm) - definiuje zakres akceptowalnych 
        # wahań odległości obiektu od kamery, ignorując tło i obiekty na pierwszym planie.
        self.depth_tolerance = 300 
        
        # [PARAMETR]: Rozmiar (w pikselach) boku kwadratowego regionu zainteresowania (ROI).
        self.ROI_BOX_SIZE = 60 

    def add_calibration_frame(self, frame):
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        self.calibration_frames.append(blurred)

    def _create_box_mask(self, shape, center):
        """Generuje binarną maskę dla prostokątnego regionu zainteresowania (ROI)."""
        mask = np.zeros(shape, dtype=np.uint8)
        cx, cy = int(center[0]), int(center[1])
        half = self.ROI_BOX_SIZE // 2
        
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(shape[1], cx + half)
        y2 = min(shape[0], cy + half)
        
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask, (x1, y1, x2, y2)

    def build_variance_model(self):
        """
        Buduje przestrzenny model statystyczny tła i obiektu na podstawie 
        fazy kalibracyjnej, izolując klatkę piersiową poprzez analizę wariancji.
        """
        if len(self.calibration_frames) < 30:
            print("Za mało klatek do kalibracji!")
            return False

        stack = np.array(self.calibration_frames, dtype=np.float32)
        mean_map = np.mean(stack, axis=0)
        std_map = np.std(stack, axis=0)
        
        # --- Filtracja dystansu i wariancji ---
        # [PARAMETR]: Maksymalna odległość pacjenta (w mm).
        MAX_DISTANCE_MM = 2000 
        dist_mask = (mean_map < MAX_DISTANCE_MM) & (mean_map > 100) 
        
        # [PARAMETR]: Przedział odchylenia standardowego charakterystyczny dla oddechu.
        MIN_VARIANCE_MM = 2.0
        MAX_VARIANCE_MM = 15.0
        variance_mask = (std_map > MIN_VARIANCE_MM) & (std_map < MAX_VARIANCE_MM)
        
        final_mask = (dist_mask & variance_mask).astype(np.uint8) * 255

        kernel = np.ones((7,7), np.uint8) 
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("Błąd: Nie znaleziono ciała.")
            return False

        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 200: 
            print("Obszar zbyt mały.")
            return False

        # --- Inicjalizacja Centroidu ---
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            self.last_centroid = (cx, cy)
        else:
            x, y, w, h = cv2.boundingRect(largest_contour)
            self.last_centroid = (float(x + w//2), float(y + h//2))

        # --- Utworzenie bazowego ROI ---
        self.roi_mask, _ = self._create_box_mask(final_mask.shape, self.last_centroid)
        self.last_z_ref = np.mean(mean_map[self.roi_mask > 0])
        
        print(f"MODEL ZBUDOWANY. Z-Ref: {self.last_z_ref:.1f} mm, Center: {self.last_centroid}")
        
        self.calibration_frames = []
        self.tracking_active = True
        return True

    def process(self, frame):
        """
        Główna pętla śledząca. Aktualizuje pozycję ROI w oparciu o ruch klatki
        piersiowej i zwraca uśredniony sygnał odległości dla danego momentu.
        """
        curr = cv2.GaussianBlur(frame, (5, 5), 0)
        vis_preview = np.clip(frame / 16, 0, 255).astype(np.uint8)
        vis_frame = cv2.cvtColor(vis_preview, cv2.COLOR_GRAY2BGR)

        if not self.tracking_active:
            return vis_frame, 0.0

        # --- Adaptacyjne Śledzenie (Tracking) ---
        min_z = int(max(0, self.last_z_ref - self.depth_tolerance))
        max_z = int(min(65535, self.last_z_ref + self.depth_tolerance))
        
        depth_mask = cv2.inRange(curr, min_z, max_z)
        contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        min_dist = float('inf')

        current_cx, current_cy = self.last_centroid

        if contours:
            for cnt in contours:
                if cv2.contourArea(cnt) < 200: continue
                
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])
                    
                    dist = np.sqrt((cx - current_cx)**2 + (cy - current_cy)**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_contour = cnt

        signal_val = 0.0
        
        if best_contour is not None and min_dist < self.search_radius:
            M = cv2.moments(best_contour)
            target_cx = float(M["m10"] / M["m00"])
            target_cy = float(M["m01"] / M["m00"])
            
            # [PARAMETR]: Współczynnik wygładzania trajektorii śledzenia (Filtr IIR).
            alpha = 0.1 
            new_cx = (1 - alpha) * current_cx + alpha * target_cx
            new_cy = (1 - alpha) * current_cy + alpha * target_cy
            
            self.last_centroid = (new_cx, new_cy)

            self.roi_mask, (x1,y1,x2,y2) = self._create_box_mask(depth_mask.shape, self.last_centroid)
            
            roi_pixels = curr[self.roi_mask > 0]
            if len(roi_pixels) > 0:
                current_z_avg = np.mean(roi_pixels)
                signal_val = current_z_avg
                # Dynamiczna aktualizacja referencyjnej głębi
                self.last_z_ref = 0.95 * self.last_z_ref + 0.05 * current_z_avg

            # Wizualizacja diagnostyczna
            cv2.drawContours(vis_frame, [best_contour], -1, (0, 100, 0), 1) 
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(vis_frame, (int(new_cx), int(new_cy)), 4, (0, 0, 255), -1)
            
        else:
            cv2.putText(vis_frame, "LOST TRACKING", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            signal_val = self.last_z_ref if self.last_z_ref else 0

        return vis_frame, signal_val


class SignalProcessor:
    """
    Klasa procesora sygnałowego. Realizuje cyfrową filtrację pasmowoprzepustową,
    algorytm detekcji szczytów (z interpolacją paraboliczną) oraz zaawansowaną
    logikę adaptacyjnego wygładzania wskaźnika BPM.
    """
    def __init__(self, buffer_size=600):
        # [PARAMETR]: Rozmiar bufora danych (600 próbek to ok. 40s dla 15 FPS).
        self.buffer_size = buffer_size
        self.fps = 15.0
        
        self.apnea_detector = ApneaGuardPlus(fps=self.fps)
        
        # [PARAMETR - KALIBRACJA]: Górna i dolna granica filtru pasmowoprzepustowego Butterwortha.
        # Górne pasmo na poziomie 1.50 Hz umożliwia detekcję szybkiego oddechu (do 90 BPM).
        nyquist = 0.5 * self.fps
        self.b, self.a = butter(4, [0.05 / nyquist, 1.50 / nyquist], btype='band')
        
        self.reset()

    def reset(self):
        self.buf = deque(maxlen=self.buffer_size)
        self.times = deque(maxlen=self.buffer_size)
        self.bpm = 0.0
        self.last_calc = 0
        self.filtered = []
        
        # Zmienne skali dla interfejsu graficznego
        self.scale_max = 1000 
        self.scale_min = 0
        
        # --- Zmienne stanowe integratora bezdechu ---
        self.apnea_integrator = 0.0
        # [PARAMETR - KLUCZOWY]: Zwiększona szybkość drenażu integratora w celu
        # wyższej odporności na dryft i fałszywe stany alarmowe w zaszumionym środowisku.
        self.integration_decay = 0.98  
        self.apnea_threshold = 0.5     

    def update(self, val, is_calibrating=False):
        """Dodaje nową próbkę do bufora i co ustalony interwał wyzwala analizę."""
        self.times.append(time.time())
        if val == 0 and len(self.buf) > 0:
            val = self.buf[-1]
        self.buf.append(val)
        
        if not is_calibrating:
            # Dynamiczne skalowanie osi Y dla wyświetlania wykresu
            if len(self.buf) > 50:
                recent = list(self.buf)[-150:]
                local_min = np.min(recent)
                local_max = np.max(recent)
                self.scale_min = 0.9 * self.scale_min + 0.1 * local_min
                self.scale_max = 0.9 * self.scale_max + 0.1 * local_max

            # [PARAMETR]: Interwał wyliczania BPM (w sekundach)
            if time.time() - self.last_calc > 0.2:
                self.calc_bpm()
                self.last_calc = time.time()

    def _get_subpixel_peak_time(self, y_data, peak_idx, times):
        """
        Implementacja interpolacji parabolicznej (Sub-sample Peak Detection) 
        dla precyzyjnego pozycjonowania szczytów fali pomiędzy fizycznymi próbkami.
        Znacznie redukuje błąd kwantyzacji wynikający z niskiego FPS.
        """
        if peak_idx <= 0 or peak_idx >= len(y_data) - 1:
            return times[peak_idx]
            
        y_left = y_data[peak_idx - 1]
        y_center = y_data[peak_idx]
        y_right = y_data[peak_idx + 1]
        
        denominator = (y_left - 2 * y_center + y_right)
        if denominator == 0:
            return times[peak_idx]
            
        delta = (y_left - y_right) / (2 * denominator)
        delta = np.clip(delta, -0.5, 0.5)
        
        t_center = times[peak_idx]
        dt_prev = t_center - times[peak_idx - 1]
        dt_next = times[peak_idx + 1] - t_center
        avg_dt = (dt_prev + dt_next) / 2.0
        
        return t_center + delta * avg_dt

    def calc_bpm(self):
        """
        Główny potok przetwarzania sygnału: detrending, filtracja, detekcja szczytów,
        obliczanie interwałów oraz adaptacyjne wygładzanie wyniku końcowego.
        """
        if len(self.buf) < int(self.fps * 15.0): 
            return
        
        y = np.array(self.buf)
        # Usunięcie składowej stałej (Detrending) przed nałożeniem filtru
        y_detrend = y - np.mean(y) 
        
        y_filt = filtfilt(self.b, self.a, y_detrend)
        self.filtered = y_filt 

        # --- Logika detekcji anomalii (ApneaGuard) ---
        local_apnea, purge_buffer = self.apnea_detector.detect(y_filt)
        
        if purge_buffer:
            # Czyszczenie bufora ze starych, niespójnych danych (np. po silnym szumie)
            keep_len = int(self.fps * 10.0)
            while len(self.buf) > keep_len:
                self.buf.popleft()
                self.times.popleft()
                
            self.apnea_integrator = 0.0
            
            y_new = np.array(self.buf)
            y_detrend_new = y_new - np.mean(y_new)
            y_filt = filtfilt(self.b, self.a, y_detrend_new)
            self.filtered = y_filt

        current_bpm = None
        
        if not local_apnea:
            peak_vals = np.abs(y_filt)
            max_val = np.max(peak_vals) if len(peak_vals) > 0 else 1.0
            if max_val < 1e-9: 
                max_val = 1.0
            
            norm_signal = y_filt / max_val
            
            # [PARAMETR - KALIBRACJA]: Minimalny odstęp między szczytami oraz ich wyraźność.
            min_dist = int(self.fps * 0.6) 
            peaks, _ = find_peaks(norm_signal, distance=min_dist, prominence=0.25)
            
            if len(peaks) >= 2:
                time_window = list(self.times)
                refined_times = []
                for p in peaks:
                    t_refined = self._get_subpixel_peak_time(norm_signal, p, time_window)
                    refined_times.append(t_refined)
                
                refined_times = np.array(refined_times)
                intervals = np.diff(refined_times)
                valid_intervals = intervals[(intervals > 0.6) & (intervals < 20.0)]
                
                if len(valid_intervals) > 0:
                    # =========================================================
                    # SEKCJA 2.1: ANALIZA INTERWAŁÓW (Algorytm Recent-Biased)
                    # =========================================================
                    # [PARAMETR]: Liczba ostatnich interwałów określających obecne tempo.
                    # Wartość 2 zapewnia natychmiastową responsywność na zmiany rytmu.
                    RECENT_INTERVALS_COUNT = 2
                    
                    recent_intervals = valid_intervals[-RECENT_INTERVALS_COUNT:]
                    
                    # Eliminacja błędów pomiarowych na poziomie pojedynczych interwałów
                    median_recent = np.median(recent_intervals)
                    reliable_recent = recent_intervals[np.abs(recent_intervals - median_recent) < 0.3 * median_recent]
                    
                    if len(reliable_recent) > 0:
                        final_avg_diff = np.mean(reliable_recent)
                    else:
                        final_avg_diff = median_recent

                    if final_avg_diff > 0:
                        current_bpm = 60.0 / final_avg_diff

        if current_bpm is None:
            local_apnea = True

        # =========================================================
        # SEKCJA 2.2: HISTEREZA BEZDECHU (Integrator z drenażem)
        # =========================================================
        if local_apnea:
            self.apnea_integrator += 0.35 
        else:
            self.apnea_integrator *= self.integration_decay 
            
        self.apnea_integrator = max(0.0, min(self.apnea_integrator, 1.0))

        # =========================================================
        # SEKCJA 2.3: ADAPTACYJNE WYGŁADZANIE SYGNAŁU (Adaptive Smoothing)
        # =========================================================
        if self.apnea_integrator > self.apnea_threshold:
            self.bpm = 0.0
        else:
            if current_bpm is not None:
                if self.bpm == 0.0:
                    self.bpm = current_bpm
                else:
                    # [PARAMETR]: Próg różnicy (w BPM), aktywujący szybką korektę wskazania.
                    BPM_DIFF_THRESHOLD = 2.0 
                    
                    # [PARAMETR]: Waga wygładzania podczas STABILNEGO oddechu (wysoka inercja).
                    SMOOTH_STRONG = 0.8
                    
                    # [PARAMETR]: Waga wygładzania podczas ZMIANY TEMPA (niska inercja, szybki skok).
                    SMOOTH_WEAK = 0.2
                    
                    diff = abs(self.bpm - current_bpm)
                    
                    if diff > BPM_DIFF_THRESHOLD:
                        # Gwałtowna zmiana rytmu - ufa nowemu pomiarowi
                        self.bpm = SMOOTH_WEAK * self.bpm + (1.0 - SMOOTH_WEAK) * current_bpm
                    else:
                        # Rytm ustabilizowany - agresywne wygładzanie szumów
                        self.bpm = SMOOTH_STRONG * self.bpm + (1.0 - SMOOTH_STRONG) * current_bpm