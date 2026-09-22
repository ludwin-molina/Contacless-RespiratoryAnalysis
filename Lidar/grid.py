import cv2
import numpy as np
import time
from collections import deque
from flask import Flask, Response, render_template_string
from lidar_driver import LidarDriver

# =================================================================
# SEKCJA 1: KONFIGURACJA I PARAMETRY GLOBALNE (16-bit / mm)
# =================================================================

# [PARAMETR]: Rozmiar komórki siatki przestrzennej (w pikselach). 
# Wpływa na rozdzielczość analizy ROI.
GRID_SIZE = 20          

# [PARAMETR]: Długość bufora kołowego (liczba klatek). 
# Definiuje okno czasowe do obliczania statystyk (odchylenia standardowego).
HISTORY_LEN = 40        

# [PARAMETR - KALIBRACJA]: Maksymalne odchylenie standardowe (w mm) 
# przyjmowane do normalizacji mapy ciepła. Oddechy generują wartości rzędu 5-15 mm.
MAX_STD_VIS = 20.0      

driver = LidarDriver()
app = Flask(__name__)

# Prekalkulacja wymiarów siatki na podstawie rozdzielczości bazowej (320x240)
grid_w = 320 // GRID_SIZE
grid_h = 240 // GRID_SIZE
history_buffer = deque(maxlen=HISTORY_LEN)

# =================================================================
# SEKCJA 2: PRZETWARZANIE PRZESTRZENNO-CZASOWE
# =================================================================

def calculate_grid_std_dev(depth_frame):
    """
    Agreguje dane z czujnika i wylicza odchylenie standardowe dla każdej 
    komórki siatki wzdłuż osi czasu, co pozwala na lokalizację mikroruchów (oddechu).
    """
    # Redukcja wymiarowości (Downscaling) - uśrednianie przestrzenne dla pojedynczej komórki
    small_view = cv2.resize(depth_frame, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    small_view = small_view.astype(np.float32)
    
    history_buffer.append(small_view)
    
    # Blokada obliczeń do momentu pełnego zainicjowania bufora
    if len(history_buffer) < HISTORY_LEN:
        return np.zeros((grid_h, grid_w), dtype=np.float32)
    
    stack = np.array(history_buffer)
    
    # Obliczenie odchylenia standardowego wzdłuż osi czasu (axis=0).
    # Wynik pozostaje w milimetrach (mm).
    std_map = np.std(stack, axis=0)
    
    return std_map

# =================================================================
# SEKCJA 3: GENERATOR STRUMIENIA I WIZUALIZACJA DIAGNOSTYCZNA
# =================================================================

def generate_frames():
    """
    Pętla główna pobierająca klatki z czujnika LiDAR, generująca mapę 
    odchyleń standardowych i nakładająca ją jako mapę ciepła na obraz surowy.
    """
    driver.setup_camera()
    
    while True:
        depth = driver.get_frame()
        if depth is None:
            time.sleep(0.01)
            continue

        std_map = calculate_grid_std_dev(depth)
        
        # --- RENDEROWANIE NAKŁADKI (Heatmap) ---
        
        # Normalizacja danych z zakresu [0, MAX_STD_VIS] do [0, 255]
        vis_grid = np.clip(std_map / MAX_STD_VIS * 255, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(vis_grid, cv2.COLORMAP_JET)
        heatmap_big = cv2.resize(heatmap, (320, 240), interpolation=cv2.INTER_NEAREST)
        
        # Przygotowanie obrazu w tle (normalizacja surowych danych 16-bit do podglądu)
        vis_preview = np.clip(depth / 16, 0, 255).astype(np.uint8)
        depth_vis = cv2.cvtColor(vis_preview, cv2.COLOR_GRAY2BGR)
        
        # Fuzja obrazów: 60% tło z kamery, 40% mapa ciepła
        final_vis = cv2.addWeighted(depth_vis, 0.6, heatmap_big, 0.4, 0)
        
        # --- RYSOWANIE SIATKI I WARTOŚCI ---
        for y in range(grid_h):
            for x in range(grid_w):
                px = x * GRID_SIZE
                py = y * GRID_SIZE
                
                cv2.rectangle(final_vis, (px, py), (px+GRID_SIZE, py+GRID_SIZE), (50, 50, 50), 1)
                
                val = std_map[y, x]
                
                # [PARAMETR]: Próg szumu tła (w mm). Wartości poniżej tego progu 
                # nie są renderowane tekstowo w celu zachowania czytelności interfejsu.
                if val > 2.0:
                    # Adaptacyjny kolor tekstu dla zachowania kontrastu na mapie ciepła
                    color = (255, 255, 255) if val < (MAX_STD_VIS/1.5) else (0, 0, 0)
                    cv2.putText(final_vis, f"{val:.1f}", (px+2, py+12), cv2.FONT_HERSHEY_PLAIN, 0.7, color, 1)

        ok, buf = cv2.imencode(".jpg", final_vis)
        if ok: yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

# =================================================================
# SEKCJA 4: INTERFEJS SIECIOWY (FLASK)
# =================================================================

@app.route("/")
def home():
    """Generuje prosty interfejs diagnostyczny w przeglądarce."""
    return render_template_string("""
        <body style='background:#111; color:#fff; text-align:center; font-family:sans-serif;'>
            <h1>Lidar Deviation Grid (Std Dev)</h1>
            <p>Wartości w kratkach to <b>Odchylenie Standardowe (mm)</b> z ostatnich 40 klatek.</p>
            <p>Niebieski = Stabilnie (Szum < 5mm), Czerwony = Ruch (Odchylenie > 20mm).</p>
            <img src='/stream' style='border: 2px solid #555;'>
        </body>
    """)

@app.route("/stream")
def stream():
    """Strumieniuje wygenerowany podgląd z mapą wariancji."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, threaded=True)