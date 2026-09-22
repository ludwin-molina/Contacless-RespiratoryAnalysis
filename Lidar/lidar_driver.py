import struct
import requests
import numpy as np
import time

class LidarDriver:
    """
    Klasa sterownika dla czujnika LiDAR (np. Sipeed Maix Sense).
    Zarządza komunikacją HTTP, utrzymuje stałą sesję dla wyższej wydajności
    oraz dekoduje binarny strumień danych do macierzy o wysokiej precyzji (16-bit).
    """
    def __init__(self, host='192.168.233.1', port=80):
        # =================================================================
        # SEKCJA 1: KONFIGURACJA SIECIOWA I OPTYMALIZACJA POŁĄCZENIA
        # =================================================================
        
        # [PARAMETR]: Adres IP i port modułu LiDAR w sieci lokalnej.
        self.host = host
        self.port = port
        
        # Utrzymywanie stałego połączenia TCP (Keep-Alive) drastycznie redukuje 
        # narzut sieciowy (overhead), stabilizując klatkaż (FPS) systemu.
        self.http_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0)
        self.http_session.mount('http://', adapter)

    # =================================================================
    # SEKCJA 2: KODOWANIE I DEKODOWANIE PROTOKOŁU BINARNEGO
    # =================================================================

    def _frame_config_decode(self, frame_config):
        """Dekoduje nagłówek konfiguracyjny ramki z formatu binarnego (C-struct)."""
        return struct.unpack("<BBBBBBBBi", frame_config)

    def _frame_config_encode(self, trigger_mode=1, deep_mode=1, deep_shift=255, ir_mode=1, status_mode=2, status_mask=7, rgb_mode=1, rgb_res=0, expose_time=0):
        """Pakuje parametry konfiguracyjne do binarnego nagłówka przesyłanego do urządzenia."""
        return struct.pack("<BBBBBBBBi", trigger_mode, deep_mode, deep_shift, ir_mode, status_mode, status_mask, rgb_mode, rgb_res, expose_time)

    def _frame_payload_decode(self, frame_data: bytes, with_config: tuple):
        """
        Główny parser surowych danych (Payload). 
        Dekoduje strumień bajtów na dwuwymiarową macierz numpy (głębia w mm).
        """
        deep_data_size, rgb_data_size = struct.unpack("<ii", frame_data[:8])
        frame_payload = frame_data[8:]
        
        # Obliczenie rozmiaru bufora dla trybu RAW 16-bit.
        # Wymiary 320x240, gdzie każdy piksel zajmuje 2 bajty (UINT16).
        deep_len = 320 * 240 * 2 
        
        depth_bytes = frame_payload[:deep_len]
        
        # Urządzenie wysyła dane w formacie Little Endian.
        # Rzutowanie na macierz 240x320, gdzie 1 jednostka = 1 milimetr (0 - 65535 mm).
        depth_numpy = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((240, 320))
        
        return depth_numpy

    # =================================================================
    # SEKCJA 3: ZARZĄDZANIE STANEM KAMERY I AKWIZYCJA
    # =================================================================

    def setup_camera(self):
        """
        Wysyła ładunek konfiguracyjny (POST) ustawiający kamerę w tryb
        najwyższej precyzji milimetrowej, niezbędnej do detekcji oddechu.
        """
        config = self._frame_config_encode(
            trigger_mode=1, 
            # [PARAMETR - KLUCZOWY]: Tryb 0 to "Raw Output" – wyłącza wewnętrzną kompresję
            # i mapowanie kolorów urządzenia, zwracając surową odległość.
            deep_mode=0,      
            
            # [PARAMETR - KLUCZOWY]: Brak przesunięcia bitowego. Gwarantuje pełną precyzję 16-bit.
            deep_shift=0,     
            ir_mode=0, 
            status_mode=2, 
            status_mask=7, 
            rgb_mode=1, 
            rgb_res=0, 
            
            # [PARAMETR - KALIBRACJA]: Czas naświetlania matrycy (Exposure Time). 
            # Wartość 80 stanowi kompromis: obniża szum tła kosztem maksymalnego klatkażu (stabilizuje na ok. 15 FPS).
            expose_time=80    
        )
        try:
            # [PARAMETR]: Timeout żądania POST ustawiony na 2 sekundy (tylko dla setupu)
            self.http_session.post(f'http://{self.host}:{self.port}/set_cfg', config, timeout=2)
        except requests.exceptions.RequestException:
            pass

    def get_frame(self):
        """
        Wykonuje szybkie odpytanie (GET) o najnowszą ramkę głębi.
        Zwraca sparsowaną macierz numpy lub None w przypadku błędu odczytu.
        """
        try:
            # [PARAMETR]: Bardzo krótki timeout (0.5s) zapobiega blokowaniu głównej pętli 
            # systemu (Thread-blocking) w przypadku chwilowej utraty pakietu.
            r = self.http_session.get(f'http://{self.host}:{self.port}/getdeep', timeout=0.5)
            
            if r.status_code == 200:
                deepimg = r.content
                config = self._frame_config_decode(deepimg[16:28])
                return self._frame_payload_decode(deepimg[28:], config)
                
        except requests.exceptions.RequestException:
            pass
            
        return None