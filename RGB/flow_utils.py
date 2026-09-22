import cv2
import numpy as np

# =================================================================
# SEKCJA 1: SYSTEM WIZUALIZACJI SYGNAŁU (Robust Plotter)
# =================================================================
class RealTimePlotter:
    """
    Klasa odpowiedzialna za renderowanie przebiegów czasowych sygnału.
    Wykorzystuje skalowanie odporne na wartości odstające (Robust Scaling) 
    poprzez analizę percentyli, co eliminuje wpływ gwałtownych artefaktów ruchowych.
    """
    def __init__(self, width=320, height=100, bg_color=(30, 30, 30), line_color=(0, 255, 0)):
        self.w = width
        self.h = height
        self.bg_color = bg_color
        self.line_color = line_color
        self.canvas = np.full((self.h, self.w, 3), self.bg_color, dtype=np.uint8)
        
        # [PARAMETR]: Minimalny zakres osi Y. Zabezpiecza przed ekstremalnym 
        # powiększaniem szumu tła (np. gdy pacjent wstrzymuje oddech).
        self.min_y_range = 0.5 

    def set_scale_reference(self, amplitude_std):
        """Aktualizuje bazową skalę wykresu na podstawie wariancji z fazy kalibracji."""
        if amplitude_std > 0:
            # [PARAMETR]: Mnożnik odchylenia standardowego definiujący optymalny 
            # margines wyświetlania (6-krotność odchylenia obejmuje >99% wariancji sygnału).
            self.min_y_range = amplitude_std * 6.0 

    def update(self, data_buffer):
        """Główna metoda aktualizująca stan wizualny wykresu."""
        self.canvas[:] = self.bg_color  
        if len(data_buffer) < 2: 
            return self.canvas

        data = np.array(list(data_buffer))
        
        # Usuwanie trendu przestrzennego (odcięcie składowej stałej)
        data = data - np.mean(data)
        
        # [PARAMETR - KLUCZOWY]: Skalowanie odporne (Robust Scaling) oparte na 98. percentylu.
        # Pozwala na ignorowanie 2% najbardziej skrajnych wartości (izolowane piki szumu).
        limit = np.percentile(np.abs(data), 98)
        
        display_limit = max(limit, self.min_y_range / 2.0)
        
        # [PARAMETR]: Margines pionowy (wykres renderowany jest na 90% dostępnej wysokości)
        draw_h = self.h * 0.9
        scale = (draw_h / 2.0) / display_limit if display_limit > 1e-5 else 1.0
        center_y = self.h / 2.0
        
        xs = np.linspace(0, self.w - 1, len(data)).astype(int)
        
        # Transformacja, rzutowanie i obcinanie (Clipping) punktów wykraczających poza płótno
        ys = center_y - (np.clip(data, -display_limit, display_limit) * scale)
        ys = np.clip(ys, 0, self.h - 1).astype(int)

        pts = np.column_stack((xs, ys))
        cv2.polylines(self.canvas, [pts], isClosed=False, color=self.line_color, thickness=1, lineType=cv2.LINE_AA)
        
        return self.canvas


# =================================================================
# SEKCJA 2: ALGORYTMY PRZEPŁYWU OPTYCZNEGO (Optical Flow)
# =================================================================

def sparse_lucas_kanade_features(prev_gray, curr_gray, max_points=50):
    """
    Metoda rzadkiego przepływu optycznego Lucas-Kanade.
    Opiera się na ekstrakcji lokalnych punktów charakterystycznych i śledzeniu ich
    w przestrzeni kolejnych klatek. Cechuje się wysoką wydajnością obliczeniową.
    """
    # [PARAMETR - KALIBRACJA]: Konfiguracja detektora Shi-Tomasi (Good Features to Track).
    # Parametry qualityLevel i minDistance odpowiadają za selekcję stabilnych tekstur na ciele pacjenta.
    feature_params = dict(maxCorners=max_points, qualityLevel=0.01, minDistance=5, blockSize=3)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    
    if p0 is None: 
        return 0.0
        
    p0 = p0.reshape(-1, 2)
    
    if len(p0) < 1: 
        return 0.0

    # [PARAMETR - KALIBRACJA]: Parametry algorytmu piramidalnego (Pyramidal Lucas-Kanade).
    # Rozmiar okna (winSize) ogranicza maksymalne wykrywalne przemieszczenie (w pikselach).
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
    
    if p1 is None: 
        return 0.0
        
    status = st.flatten() == 1
    good_new = p1[status]
    good_old = p0[status]
    
    if len(good_new) < 1: 
        return 0.0
    
    # Ekstrakcja ruchu: Mediana wektorów przemieszczenia w osi Y gwarantuje 
    # wysoką odporność na błędnie przypisane ("zgubione") punkty śledzenia.
    flow_y = good_new[:, 1] - good_old[:, 1]
    return float(np.median(flow_y))


def horn_schunck_numpy_dense(img1, img2, alpha=1.0, Niter=3):
    """
    Metoda gęstego przepływu optycznego Horn-Schuncka (wektoryzacja NumPy).
    Wykorzystuje założenie o globalnej gładkości pola wektorowego, co minimalizuje
    lokalne zakłócenia i dobrze odzwierciedla jednorodny ruch klatki piersiowej.
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Wektorowe wyznaczanie pochodnych czasoprzestrzennych (różnice skończone).
    # Operacje uwzględniają redukcję siatki o 1 wiersz i kolumnę, aby zachować zgodność wymiarową macierzy.
    
    # Ex: Gradient przestrzenny w osi X
    Ex = (img1[:-1, 1:] - img1[:-1, :-1] + img2[:-1, 1:] - img2[:-1, :-1]) / 2

    # Ey: Gradient przestrzenny w osi Y
    Ey = (img1[1:, :-1] - img1[:-1, :-1] + img2[1:, :-1] - img2[:-1, :-1]) / 2

    # Et: Pochodna czasowa (różnica natężenia pikseli między klatkami)
    Et = img2[:-1, :-1] - img1[:-1, :-1]

    u = np.zeros_like(Ex)
    v = np.zeros_like(Ex)

    # [PARAMETR]: Filtr usredniający (odwrócony operator Laplace'a) niezbędny do wyliczeń relaksacyjnych.
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32) / 4

    # [PARAMETR - KALIBRACJA]: Liczba iteracji relaksacyjnych Jacobiego (Niter). 
    # Definiuje stopień dyfuzji ruchu w obrazie. Większa liczba = lepsza gładkość kosztem obciążenia CPU.
    for _ in range(Niter):
        u_avg = cv2.filter2D(u, -1, kernel)
        v_avg = cv2.filter2D(v, -1, kernel)
        
        p = Ex * u_avg + Ey * v_avg + Et
        
        # Ograniczenie gładkości (Smoothness Constraint) kontrolowane współczynnikiem alpha
        d = alpha**2 + Ex**2 + Ey**2
        
        # Zabezpieczenie przed dzieleniem przez zero w obszarach płaskich (brak gradientu)
        d[d == 0] = 1e-6
        
        u = u_avg - Ex * (p / d)
        v = v_avg - Ey * (p / d)

    # Globalna estymacja ruchu odbywa się poprzez uśrednienie całej składowej pionowej pola wektorowego.
    return float(np.mean(v))