import struct
import requests
import numpy as np
import cv2

# =================================================================
# SEKCJA 1: KONFIGURACJA SIECIOWA I ZARZĄDZANIE SESJĄ
# =================================================================
# [PARAMETR]: Adres IP oraz port serwera wizyjnego (czujnika).
HOST = '192.168.233.1'
PORT = 80
MAIX_URL_CFG = f'http://{HOST}:{PORT}/set_cfg'
MAIX_URL_GET = f'http://{HOST}:{PORT}/getdeep'

# Wykorzystanie trwałej sesji HTTP (Keep-Alive) do minimalizacji opóźnień 
# związanych z nawiązywaniem połączeń TCP dla każdej klatki (narzut sieciowy).
http_session = requests.Session()

# =================================================================
# SEKCJA 2: PROTOKÓŁ BINARNY I DESERIALIZACJA DANYCH
# =================================================================

def frame_config_decode(frame_config):
    """Odczytuje nagłówek konfiguracyjny z binarnego formatu C-struct."""
    return struct.unpack("<BBBBBBBBi", frame_config)

def frame_config_encode(trigger_mode=1, deep_mode=1, deep_shift=255, ir_mode=1, status_mode=2, status_mask=7, rgb_mode=1, rgb_res=0, expose_time=0):
    """Tworzy binarny ładunek (payload) konfigurujący tryby pracy matryc czujnika."""
    return struct.pack("<BBBBBBBBi", trigger_mode, deep_mode, deep_shift, ir_mode, status_mode, status_mask, rgb_mode, rgb_res, expose_time)

def frame_payload_decode(frame_data: bytes, with_config: tuple):
    """
    Zaawansowany parser strumienia binarnego. Dokonuje dynamicznej kalkulacji
    offsetów pamięci w celu odrzucenia zbędnych danych (głębia, IR, status) 
    i izolacji docelowego obrazu w formacie JPEG.
    """
    deep_data_size, rgb_data_size = struct.unpack("<ii", frame_data[:8])
    frame_payload = frame_data[8:]
    
    # Obliczanie i pomijanie bloku danych głębi (Depth)
    deepth_size = (320 * 240 * 2) >> with_config[1]
    frame_payload = frame_payload[deepth_size:]
    
    # Obliczanie i pomijanie bloku danych w podczerwieni (IR)
    ir_size = (320 * 240 * 2) >> with_config[3]
    frame_payload = frame_payload[ir_size:]
    
    # Obliczanie i pomijanie bloku statusu przestrzennego
    status_size = (320 * 240 // 8) * (16 if 0 == with_config[4] else 2 if 1 == with_config[4] else 8 if 2 == with_config[4] else 1)
    frame_payload = frame_payload[status_size:]
    
    # Ekstrakcja właściwego ładunku RGB
    rgb_size = len(frame_payload)
    rgb_img_bytes = struct.unpack("<%us" % rgb_size, frame_payload[:rgb_size])[0] if 0 != rgb_size else None
    
    rgb_numpy = None
    if (rgb_img_bytes is not None) and (1 == with_config[6]):
        # Dekompresja w locie (In-memory decoding) strumienia bajtów JPEG do macierzy BGR
        jpeg = cv2.imdecode(np.frombuffer(rgb_img_bytes, 'uint8'), cv2.IMREAD_COLOR)
        if jpeg is not None:
            rgb_numpy = jpeg
            
    return rgb_numpy

# =================================================================
# SEKCJA 3: AKWIZYCJA OBRAZU I STEROWANIE SPRZĘTEM
# =================================================================

def setup_camera():
    """
    Inicjalizuje strumień RGB urządzenia poprzez wysłanie odpowiedniej 
    konfiguracji bitowej przez protokół HTTP.
    """
    print(f"🔧 Konfiguracja kamery {HOST}...")
    
    # [PARAMETR - KLUCZOWY]: Ustawienie parametrów rejestracji.
    # Uaktywnienie trybu RGB (rgb_mode=1) przy domyślnej rozdzielczości (rgb_res=0).
    config = frame_config_encode(
        trigger_mode=1, 
        deep_mode=1, 
        deep_shift=255, 
        ir_mode=0, 
        status_mode=2, 
        status_mask=7, 
        rgb_mode=1, 
        rgb_res=0, 
        expose_time=0
    )
    
    try:
        # [PARAMETR]: Maksymalny czas oczekiwania (Timeout) na odpowiedź sprzętową.
        r = http_session.post(MAIX_URL_CFG, config, timeout=2)
        if r.status_code == requests.codes.ok:
            return True
    except requests.exceptions.RequestException:
        pass
        
    return False

def get_maix_frame():
    """
    Pobiera najnowszą klatkę (GET) i przekazuje do parsowania.
    Zwraca zdekodowaną macierz RGB (wymiary: 240x320x3) lub None w przypadku utraty pakietu.
    """
    try:
        # [PARAMETR]: Timeout ograniczony do 1 sekundy, zapobiegający blokowaniu 
        # wątku w razie desynchronizacji z czujnikiem.
        r = http_session.get(MAIX_URL_GET, timeout=1)
        if r.status_code == requests.codes.ok:
            deepimg = r.content
            
            # Wypakowanie nagłówka z konkretnego offsetu pamięci (bajty 16-28)
            config_bytes = deepimg[16:28]
            config = frame_config_decode(config_bytes)
            
            # Parsowanie części użytecznej (od bajtu 28)
            rgb_map = frame_payload_decode(deepimg[28:], config)
            return rgb_map
            
    except requests.exceptions.RequestException:
        pass
        
    return None