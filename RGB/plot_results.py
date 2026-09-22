import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import sys

plt.rcParams.update({'font.size': 10})
OUTPUT_DIR = "recordings"

def get_latest_session_files():
    """Znajduje pliki CSV z ostatniej nagranej sesji."""
    if not os.path.exists(OUTPUT_DIR):
        print(f"Błąd: Katalog '{OUTPUT_DIR}' nie istnieje.")
        return None

    # Pobieramy wszystkie pliki csv
    files = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
    if not files:
        print("Brak plików CSV w katalogu recordings.")
        return None

    # Wyciągamy unikalne sygnatury czasowe (YYYYMMDD_HHMMSS)
    timestamps = set()
    for f in files:
        # Nazwa pliku: recordings/20260106_123045_Algorytm.csv
        # Bierzemy samą nazwę pliku
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) >= 2:
            ts = f"{parts[0]}_{parts[1]}"
            timestamps.add(ts)
    
    if not timestamps:
        return None

    # Sortujemy i bierzemy najnowszą
    latest_ts = sorted(list(timestamps))[-1]
    print(f"znaleziono najnowszą sesję: {latest_ts}")

    # Znajdujemy pliki należące do tej sesji
    session_files = glob.glob(os.path.join(OUTPUT_DIR, f"{latest_ts}*.csv"))
    return latest_ts, session_files

def plot_session(timestamp, file_list):
    """Generuje wykres porównawczy."""
    data = {}
    
    # Wczytywanie danych
    for filepath in file_list:
        # Wyciągamy nazwę algorytmu z nazwy pliku
        filename = os.path.basename(filepath)
        # Format: timestamp_AlgorytmName.csv
        # Usuwamy timestamp i rozszerzenie
        algo_name = filename.replace(f"{timestamp}_", "").replace(".csv", "")
        
        try:
            # Wczytujemy CSV (używamy separatora ';' z data_logger.py)
            df = pd.read_csv(filepath, delimiter=';')
            data[algo_name] = df
        except Exception as e:
            print(f"Błąd odczytu pliku {filename}: {e}")

    if not data:
        print("Nie udało się wczytać danych.")
        return

    # --- RYSOWANIE WYKRESU ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Analiza Porównawcza Optical Flow - Sesja {timestamp}", fontsize=14)

    colors = {'Farneback': 'green', 'Horn-Schunck': 'orange', 'Lucas-Kanade': 'magenta'}

    # 1. Wykres Sygnału Oddechowego (Pozycja)
    ax1.set_title("1. Przefiltrowany Sygnał Oddechu (Pozycja Klatki)")
    ax1.set_ylabel("Amplituda [a.u.]")
    ax1.grid(True, alpha=0.3)

    for algo, df in data.items():
        col = colors.get(algo, 'blue')
        ax1.plot(df['Elapsed_Sec'], df['Signal_Pos'], label=algo, color=col, linewidth=1.5, alpha=0.8)
    ax1.legend(loc="upper right")

    # 2. Wykres BPM (Częstotliwość)
    ax2.set_title("2. Estymowane BPM (Oddechy na minutę)")
    ax2.set_ylabel("RPM")
    ax2.grid(True, alpha=0.3)

    for algo, df in data.items():
        col = colors.get(algo, 'blue')
        # BPM może mieć luki (None), pandas radzi sobie z tym
        ax2.plot(df['Elapsed_Sec'], df['BPM'], label=algo, color=col, linewidth=2)
    
    # 3. Wykres Stabilności (Signal_Std / Amplituda Chwilowa)
    # To pokaże, czy algorytm gubi sygnał (spadek do zera) lub szumi (skoki)
    ax3.set_title("3. Siła Sygnału (Energy/StdDev)")
    ax3.set_ylabel("Std Dev")
    ax3.set_xlabel("Czas [s]")
    ax3.grid(True, alpha=0.3)

    for algo, df in data.items():
        col = colors.get(algo, 'blue')
        ax3.plot(df['Elapsed_Sec'], df['Signal_Std'], label=algo, color=col, linewidth=1.5)

    # Zapisywanie do pliku
    output_img = os.path.join(OUTPUT_DIR, f"RESULT_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    print(f"\nWykres wygenerowany: {output_img}")
    print("Możesz go pobrać i wkleić do pracy magisterskiej.")

def main():
    print("--- AUTOMATYCZNY GENERATOR WYKRESÓW ---")
    res = get_latest_session_files()
    if res:
        ts, files = res
        print(f"Przetwarzanie {len(files)} plików...")
        plot_session(ts, files)
    else:
        print("Nie znaleziono sesji do przetworzenia.")

if __name__ == "__main__":
    main()