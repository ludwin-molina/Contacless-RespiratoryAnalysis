import numpy as np

class ApneaGuardPlus:
    """
    Klasa realizująca hybrydowe wykrywanie bezdechów oparte na 
    autokorelacji, analizie amplitudy oraz logice przejścia (histereza).
    """
    def __init__(self, fps=15.0):
        # [PARAMETR GLOBALNY]: Znamionowa liczba klatek na sekundę dla sygnału wejściowego.
        self.fps = fps
        
        # [PARAMETR KLUCZOWY - KALIBRACJA]: Próg akceptacji rytmu.
        # Wartość 0.45 jest rygorystycznie skalibrowana pod środowisko o wysokim szumie,
        # minimalizując podatność na fałszywe detekcje.
        self.rhythm_threshold = 0.45 

    def _check_rhythm(self, signal_window, min_lag_sec, max_lag_sec):
        """
        Funkcja pomocnicza: Oblicza znormalizowany szczyt autokorelacji 
        w celu detekcji cykliczności sygnału w zadanym oknie czasowym.
        """
        n = len(signal_window)
        if n < int(min_lag_sec * self.fps):
            return 0.0
        
        acorr = np.correlate(signal_window, signal_window, mode='full')
        acorr = acorr[n-1:] / (acorr[n-1] + 1e-9) 
        
        min_lag = int(min_lag_sec * self.fps)
        max_lag = min(int(max_lag_sec * self.fps), n - 1)
        
        if max_lag > min_lag:
            valid_acorr = acorr[min_lag:max_lag]
            if len(valid_acorr) > 0:
                return np.max(valid_acorr)
        return 0.0

    def detect(self, full_signal):
        """
        Główna pętla decyzyjna analizująca sygnał w poszukiwaniu anomalii oddechowych.
        Zwraca krotkę: (is_apnea, purge_buffer)
        """
        
        # =================================================================
        # SEKCJA 1: ANALIZA AMPLITUDY (Detekcja całkowitego braku ruchu)
        # =================================================================
        
        # [PARAMETR]: Długość ogona analizy w sekundach (okno dla amplitudy).
        tail_len = int(8.0 * self.fps) 
        tail_data = full_signal[-tail_len:] if len(full_signal) > tail_len else full_signal
        
        current_std = np.std(tail_data)
        
        # [PARAMETR]: Próg odchylenia standardowego - poniżej tej wartości 
        # system natychmiastowo diagnozuje bezdech (cisza w sygnale).
        if current_std < 0.8: 
            return True, False 

        # =================================================================
        # SEKCJA 2: ANALIZA RYTMU (Hybrydowe śledzenie w dwóch oknach)
        # =================================================================
        
        # [PARAMETR]: Zakres lagów (w sekundach) dla długiego okna.
        long_rhythm = self._check_rhythm(full_signal, 0.4, 20.0)
        
        # [PARAMETR]: Długość krótkiego okna w sekundach (do wyłapywania gwałtownych zmian).
        short_window_len = int(10.0 * self.fps)
        short_signal = full_signal[-short_window_len:] if len(full_signal) > short_window_len else full_signal
        
        # [PARAMETR]: Zakres lagów (w sekundach) dla krótkiego okna.
        short_rhythm = self._check_rhythm(short_signal, 0.4, 6.0)

        # =================================================================
        # SEKCJA 3: LOGIKA TRANZYCJI I PODEJMOWANIE DECYZJI
        # =================================================================
        
        is_apnea = False
        purge_buffer = False

        if long_rhythm < self.rhythm_threshold:
            if short_rhythm >= self.rhythm_threshold:
                # Sygnał odzyskał rytm w krótkim oknie - flaga do czyszczenia zaszumionego bufora.
                purge_buffer = True
                is_apnea = False
            else:
                # Brak rytmu w obu oknach analitycznych - potwierdzenie bezdechu.
                is_apnea = True 
                
        return is_apnea, purge_buffer