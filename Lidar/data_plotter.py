import os
import sys
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "recordings"

def get_latest_session_timestamp():
    if not os.path.exists(OUTPUT_DIR): return None
    files = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
    if not files: return None
    timestamps = set()
    for f in files:
        parts = os.path.basename(f).split('_')
        if len(parts) >= 2: timestamps.add(f"{parts[0]}_{parts[1]}")
    if not timestamps: return None
    return sorted(list(timestamps))[-1]

def generate_session_plot(target_timestamp=None):
    # Jeśli nie podano timestampu, bierzemy najnowszy
    timestamp = target_timestamp if target_timestamp else get_latest_session_timestamp()
    
    if not timestamp: 
        print("[PLOTTER] Brak danych w folderze 'recordings' do wygenerowania wykresu.")
        return None

    print(f"[PLOTTER] Generowanie wykresu Lidar: {timestamp}...")
    session_files = glob.glob(os.path.join(OUTPUT_DIR, f"{timestamp}*.csv"))
    data = {}

    for filepath in session_files:
        algo_name = os.path.basename(filepath).replace(f"{timestamp}_", "").replace(".csv", "")
        try: data[algo_name] = pd.read_csv(filepath, delimiter=';')
        except: pass

    if not data: 
        print(f"[PLOTTER] Nie udało się wczytać danych dla timestampu: {timestamp}")
        return None

    try: plt.style.use('seaborn-v0_8-darkgrid')
    except: pass

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Raport Lidar (ToF Distance): {timestamp}", fontsize=16, fontweight='bold')

    # 1. Sygnał Surowy (Dystans absolutny lub znormalizowany)
    # Rysujemy Signal_Filtered, bo Signal_Raw_mm może mieć duży offset (np. 1200mm)
    ax1.set_title("1. Zmiana Dystansu (Sygnał Oddechowy)", fontsize=12)
    ax1.set_ylabel("Amplituda [mm]") 
    for algo, df in data.items():
        # Lidar daje bardzo czysty sygnał
        ax1.plot(df['Elapsed_Sec'], df['Signal_Filtered'], label=f"{algo} (Filtered)", color='purple', linewidth=1.5)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # 2. BPM
    ax2.set_title("2. Estymacja BPM", fontsize=12)
    ax2.set_ylabel("RPM")
    for algo, df in data.items():
        ax2.plot(df['Elapsed_Sec'], df['BPM'], label=algo, color='blue', linewidth=2)
    ax2.grid(True, alpha=0.3)

    # 3. Analiza Surowego Ruchu (Raw Variance)
    # Tu możemy pokazać jak bardzo zmieniał się surowy odczyt w mm
    ax3.set_title("3. Surowa Dynamika (Raw Delta)", fontsize=12)
    ax3.set_ylabel("Dystans [mm]")
    ax3.set_xlabel("Czas [s]")
    for algo, df in data.items():
        # Odejmujemy średnią, żeby zobaczyć tylko zmiany wokół zera
        raw_centered = df['Signal_Raw_mm'] - df['Signal_Raw_mm'].mean()
        ax3.plot(df['Elapsed_Sec'], raw_centered, label="Raw Delta", color='gray', alpha=0.6, linewidth=1)
    
    plt.tight_layout()
    output_filename = os.path.join(OUTPUT_DIR, f"REPORT_LIDAR_{timestamp}.png")
    plt.savefig(output_filename, dpi=150)
    plt.close(fig)
    
    print(f"[PLOTTER] Wykres gotowy: {output_filename}")
    return output_filename

# --- Wywołanie ręczne z terminala ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        manual_timestamp = sys.argv[1]
        generate_session_plot(manual_timestamp)
    else:
        # W przeciwnym razie generujemy dla najnowszego pliku
        generate_session_plot()