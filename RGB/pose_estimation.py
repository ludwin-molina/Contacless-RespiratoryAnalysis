import cv2
import numpy as np
import os
import math

# =================================================================
# SEKCJA 1: KONFIGURACJA MODELU I PARAMETRY GEOMETRYCZNE
# =================================================================
# [PARAMETR]: Ścieżka do wag sieci w formacie ONNX (Lightweight Pose Estimation).
ONNX_PATH = "lightweight_pose_estimation_201912.onnx"

# [PARAMETR]: Rozmiar wejściowy obrazu dla sieci (kwadrat 192x192).
POSE_INP_SIZE = 192

# --- ZKALIBROWANE PARAMETRY EKSTRAKCJI ROI ---
# [PARAMETR]: Przesunięcie pionowe środka ROI względem linii ramion (w pikselach).
ROI_LOWER_OFFSET = 35 

# [PARAMETR]: Mnożniki skalujące wymiary ROI względem rozstawu ramion i długości tułowia.
ROI_W_MULT = 0.30  
ROI_H_MULT = 2.94  

# [PARAMETR]: Ograniczenia proporcji (Aspect Ratio) dla stabilizacji obszaru analizy.
MIN_ASPECT_RATIO = 0.04 
MAX_ASPECT_RATIO = 1.20 
# ----------------------------------------------

# =================================================================
# SEKCJA 2: INICJALIZACJA SILNIKA INFERENCYJNEGO (OpenCV DNN)
# =================================================================
if not os.path.exists(ONNX_PATH):
    print(f"BRAK PLIKU SIECI: {ONNX_PATH}")
    net = None
else:
    print(f"Ładowanie sieci neuronowej...")
    try:
        # Inicjalizacja sieci przy użyciu modułu DNN OpenCV
        net = cv2.dnn.readNetFromONNX(ONNX_PATH)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Sieć gotowa.")
    except Exception as e:
        print(f" Błąd ładowania sieci: {e}")
        net = None

# =================================================================
# SEKCJA 3: LOGIKA EKSTRAKCJI PUNKTÓW KLUCZOWYCH I ROI
# =================================================================

def get_pose_roi(frame):
    """
    Realizuje pełny proces estymacji pozy pacjenta. Na podstawie map ciepła (heatmaps)
    wyznacza punkty kluczowe (keypoints), a następnie stosuje logikę geometryczną
    do automatycznej lokalizacji ROI na klatce piersiowej.
    """
    if net is None: 
        return None, None

    h, w = frame.shape[:2]
    
    # Preprocessing: Tworzenie blobu wejściowego (normalizacja i zmiana rozmiaru)
    blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (POSE_INP_SIZE, POSE_INP_SIZE), (128, 128, 128), False, False)
    net.setInput(blob)
    
    # Inferencja (Forward Pass)
    out = net.forward()
    out = out[0] 

    keypoints = []
    _, Hm, Wm = out.shape 

    # --- Ekstrakcja współrzędnych z map ciepła ---
    for i in range(18): 
        if i >= out.shape[0]: 
            keypoints.append(None)
            continue 
            
        heatmap = out[i, :, :] 
        # Lokalizacja globalnego maksimum na mapie prawdopodobieństwa dla danego stawu
        _, conf, _, point = cv2.minMaxLoc(heatmap)
        
        # [PARAMETR]: Próg ufności (Confidence) dla punktu kluczowego
        if conf > 0.1:
            x = (point[0] / (Wm - 1)) * w
            y = (point[1] / (Hm - 1)) * h
            keypoints.append((int(x), int(y)))
        else:
            keypoints.append(None)

    # --- Obliczanie punktów pochodnych (Wirtualne punkty kontrolne) ---
    # Punkt 18 i 19: Środki odcinków ramion (prawy/lewy względem mostka)
    mid_right = None
    if keypoints[1] is not None and keypoints[2] is not None:
        mx = int((keypoints[1][0] + keypoints[2][0]) / 2)
        my = int((keypoints[1][1] + keypoints[2][1]) / 2)
        mid_right = (mx, my)
    keypoints.append(mid_right) 

    mid_left = None
    if keypoints[1] is not None and keypoints[5] is not None:
        mx = int((keypoints[1][0] + keypoints[5][0]) / 2)
        my = int((keypoints[1][1] + keypoints[5][1]) / 2)
        mid_left = (mx, my)
    keypoints.append(mid_left) 
    
    # Wyznaczanie środka ciężkości klatki piersiowej (Indeksy 20 i 21)
    center_y_point = None
    lower_center_point = None
    if keypoints[1] is not None:
        valid_ys = []
        for idx in [1, 2, 5, 18, 19]:
            if idx < len(keypoints) and keypoints[idx] is not None:
                valid_ys.append(keypoints[idx][1])
                
        if len(valid_ys) > 0:
            avg_y = int(sum(valid_ys) / len(valid_ys))
            center_y_point = (keypoints[1][0], avg_y)
            lower_center_point = (keypoints[1][0], avg_y + ROI_LOWER_OFFSET)
            
    keypoints.append(center_y_point) 
    keypoints.append(lower_center_point) 
    
    # --- Wyznaczanie współrzędnych i wymiarów ROI ---
    roi_coords = None
    if lower_center_point is not None and keypoints[1] is not None and keypoints[2] is not None and keypoints[5] is not None:
        cx, cy = lower_center_point
        
        # Obliczanie euklidesowego rozstawu ramion (szerokość referencyjna)
        dx_shoulders = keypoints[2][0] - keypoints[5][0]
        dy_shoulders = keypoints[2][1] - keypoints[5][1]
        raw_w = math.sqrt(dx_shoulders**2 + dy_shoulders**2)
        
        # Obliczanie długości tułowia (wysokość referencyjna)
        dx_trunk = keypoints[1][0] - cx
        dy_trunk = keypoints[1][1] - cy
        raw_h = math.sqrt(dx_trunk**2 + dy_trunk**2)
        
        # Skalowanie wymiarów docelowych
        target_w = raw_w * ROI_W_MULT
        target_h = raw_h * ROI_H_MULT
        
        # Korekta proporcji (Aspect Ratio Constraint)
        if target_h > 0:
            current_ratio = target_w / target_h
            if current_ratio < MIN_ASPECT_RATIO:
                target_w = target_h * MIN_ASPECT_RATIO
            elif current_ratio > MAX_ASPECT_RATIO:
                target_h = target_w / MAX_ASPECT_RATIO
                
        target_w = int(target_w)
        target_h = int(target_h)
        
        # Transformacja do układu współrzędnych obrazu (lewy górny róg)
        x1 = int(cx - target_w / 2)
        y1 = int(cy - target_h / 2)
        
        # Zabezpieczenie przed wyjściem poza krawędzie obrazu
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        
        roi_w_safe = min(target_w, w - x1)
        roi_h_safe = min(target_h, h - y1)
        
        # Walidacja minimalnego rozmiaru obszaru analizy
        if roi_w_safe >= 10 and roi_h_safe >= 10:
            roi_coords = [x1, y1, roi_w_safe, roi_h_safe]
            
    return roi_coords, keypoints