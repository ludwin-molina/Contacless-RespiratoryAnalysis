import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ścieżka do Twojego folderu
FOLDER_PATH = r"C:\Users\ww\Desktop\pomiary_d\gt"

def process_mae():
    results = []
    
    # Szukamy wszystkich plików CSV w głównym folderze
    csv_files = glob.glob(os.path.join(FOLDER_PATH, "*.csv"))
    
    for file in csv_files:
        filename = os.path.basename(file)
        
        # Parsowanie nazwy pliku i przypisanie odpowiednich kolumn dla BPM
        if filename.startswith("synced_") and filename.endswith(".csv"):
            # RGB Format: synced_{imie}_{algorytm}.csv
            parts = filename.replace("synced_", "").replace(".csv", "").split("_")
            if len(parts) == 2:
                person, method = parts[0], parts[1]
                est_col = "RGB_BPM"
            else:
                continue
        elif filename.startswith("synchronized_output_zscore-") and filename.endswith("-lidar.csv"):
            # LiDAR Format: synchronized_output_zscore-{imie}-lidar.csv
            person = filename.split("-")[1]
            method = "LiDAR"
            est_col = "Lidar_BPM"
        else:
            continue
            
        try:
            # Wczytanie danych z odpowiednim separatorem
            df = pd.read_csv(file, sep=';')
            gt_col = "GT_BPM"
            
            # Usunięcie wierszy z brakującymi danymi, jeśli takie wystąpiły
            df_clean = df[[est_col, gt_col]].dropna()
            
            # Obliczenie MAE: średnia z wartości bezwzględnych różnicy między estymacją a GT
            mae = np.mean(np.abs(df_clean[est_col] - df_clean[gt_col]))
                
            results.append({
                "Person": person,
                "Method": method,
                "MAE": mae,
                "Filename": filename
            })
            
        except Exception as e:
            print(f"[!] Error processing file {filename}: {e}")

    # Utworzenie DataFrame z wynikami
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        print("No valid files found for analysis.")
        return

    # =========================================================
    # USUWANIE NAJWIĘKSZEGO OUTLIERA
    # =========================================================
    max_idx = df_results['MAE'].idxmax()
    outlier_row = df_results.loc[max_idx]
    
    print("\n" + "!"*80)
    print(f"UWAGA METODOLOGICZNA: Usunięto 1 największy outlier z danych do wykresu!")
    print(f"Usunięty plik: {outlier_row['Filename']}")
    print(f"Metoda: {outlier_row['Method']} | Osoba: {outlier_row['Person']} | MAE: {outlier_row['MAE']:.2f}")
    print("Warto opisać to odrzucenie w tekście pracy.")
    print("!"*80)
    
    # Usunięcie wiersza i zresetowanie indeksów
    df_results = df_results.drop(max_idx).reset_index(drop=True)
    # =========================================================

    # Wyświetlanie rankingu w terminalu (od najmniejszego błędu do największego)
    print("\n" + "="*80)
    print(f"{'MEAN ABSOLUTE ERROR (MAE) RANKING (Best to Worst - OUTLIER REMOVED)':^80}")
    print("="*80)
    print(f"{'Rank':<8} | {'MAE (BPM)':<10} | {'Method/Sensor':<18} | {'Person':<10} | {'Filename'}")
    print("-" * 80)
    
    # Sortujemy rosnąco, bo im mniejszy błąd MAE, tym lepiej
    df_sorted = df_results.sort_values(by="MAE", ascending=True).reset_index(drop=True)
    
    for index, row in df_sorted.iterrows():
        print(f"{index+1:<8d} | {row['MAE']:.4f}     | {row['Method']:<18s} | {row['Person']:<10s} | {row['Filename']}")

    # Generowanie i zapisywanie Boxplota
    plt.figure(figsize=(10, 6))
    
    # Używamy ładnego stylu naukowego z seaborn
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Wymuszamy kolejność, żeby LiDAR był pierwszy jako baza referencyjna
    order = ["LiDAR", "Farneback", "Horn-Schunck", "Lucas-Kanade"]
    
    # Boxplot + Swarmplot
    ax = sns.boxplot(x="Method", y="MAE", data=df_results, order=order, palette="Pastel1", width=0.5)
    sns.swarmplot(x="Method", y="MAE", data=df_results, order=order, color=".2", size=5, alpha=0.7)
    
    # Angielskie opisy wykresu
    plt.title("Distribution of Mean Absolute Error (MAE) for BPM Estimation\n(LiDAR vs. RGB Vision - Top Outlier Removed)")
    plt.ylabel("Mean Absolute Error [BPM]")
    plt.xlabel("Sensor / Tracking Method")
    
    # Zapis do PNG z wysoką rozdzielczością (dpi=300)
    plot_path = os.path.join(FOLDER_PATH, "mae_comparison_boxplot_clean.png")
    plt.tight_layout()
    plt.savefig(plot_path, format="png", dpi=300, bbox_inches="tight")
    
    print("="*80)
    print(f"Done! Cleaned MAE boxplot saved to:\n{plot_path}")

if __name__ == "__main__":
    process_mae()