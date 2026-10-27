import os
import glob
import pandas as pd
from scipy.stats import shapiro, pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Ścieżka do Twojego folderu
FOLDER_PATH = r"C:\Users\ww\Desktop\pomiary_d\gt"

def process_correlations():
    results = []
    
    # Szukamy wszystkich plików CSV w głównym folderze
    csv_files = glob.glob(os.path.join(FOLDER_PATH, "*.csv"))
    
    for file in csv_files:
        filename = os.path.basename(file)
        
        # Parsowanie nazwy pliku i przypisanie odpowiednich kolumn
        if filename.startswith("synced_") and filename.endswith(".csv"):
            # RGB Format: synced_{imie}_{algorytm}.csv
            parts = filename.replace("synced_", "").replace(".csv", "").split("_")
            if len(parts) == 2:
                person, method = parts[0], parts[1]
                signal_col = "RGB_Signal_ZScore"
            else:
                continue
        elif filename.startswith("synchronized_output_zscore-") and filename.endswith("-lidar.csv"):
            # LiDAR Format: synchronized_output_zscore-{imie}-lidar.csv
            person = filename.split("-")[1]
            method = "LiDAR"
            signal_col = "Lidar_Signal_ZScore"
        else:
            continue
            
        try:
            # Wczytanie danych z odpowiednim separatorem
            df = pd.read_csv(file, sep=';')
            gt_col = "GT_Signal_ZScore"
            
            # Usunięcie wierszy z brakującymi danymi, jeśli takie wystąpiły
            df_clean = df[[signal_col, gt_col]].dropna()
            sig1 = df_clean[signal_col].values
            sig2 = df_clean[gt_col].values
            
            # 1. Test normalności Shapiro-Wilka (alfa = 0.05)
            stat1, p1 = shapiro(sig1)
            stat2, p2 = shapiro(sig2)
            
            # 2. Wybór odpowiedniej korelacji
            if p1 > 0.05 and p2 > 0.05:
                corr, _ = pearsonr(sig1, sig2)
                corr_type = "Pearson"
            else:
                corr, _ = spearmanr(sig1, sig2)
                corr_type = "Spearman"
                
            results.append({
                "Person": person,
                "Method": method,
                "Correlation": corr,
                "Type": corr_type,
                "Filename": filename
            })
            
        except Exception as e:
            print(f"[!] Error processing file {filename}: {e}")

    # Utworzenie DataFrame z wynikami
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        print("No valid files found for analysis.")
        return

    # Wyświetlanie rankingu w terminalu (teraz po angielsku)
    print("\n" + "="*85)
    print(f"{'CORRELATION RANKING (Best to Worst)':^85}")
    print("="*85)
    print(f"{'Rank':<8} | {'Corr':<9} | {'Test Type':<9} | {'Method/Sensor':<18} | {'Person':<10} | {'Filename'}")
    print("-" * 85)
    
    df_sorted = df_results.sort_values(by="Correlation", ascending=False).reset_index(drop=True)
    
    for index, row in df_sorted.iterrows():
        print(f"{index+1:<8d} | {row['Correlation']:.4f}    | {row['Type']:<9s} | {row['Method']:<18s} | {row['Person']:<10s} | {row['Filename']}")

    # Generowanie i zapisywanie Boxplota
    plt.figure(figsize=(10, 6))
    
    # Używamy ładnego stylu naukowego z seaborn
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Wymuszamy kolejność, żeby LiDAR był pierwszy jako baza referencyjna
    order = ["LiDAR", "Farneback", "Horn-Schunck", "Lucas-Kanade"]
    
    # Boxplot (pudełka) + Swarmplot (pojedyncze kropki reprezentujące każde nagranie)
    ax = sns.boxplot(x="Method", y="Correlation", data=df_results, order=order, palette="Pastel1", width=0.5)
    sns.swarmplot(x="Method", y="Correlation", data=df_results, order=order, color=".2", size=5, alpha=0.7)
    
    # Angielskie opisy wykresu
    plt.title("Distribution of Correlation Coefficients vs. Ground Truth\n(LiDAR vs. RGB Vision-based Methods)")
    plt.ylabel("Correlation Coefficient")
    plt.xlabel("Sensor / Tracking Method")
    
    # Zapis do PNG z wysoką rozdzielczością (dpi=300)
    plot_path = os.path.join(FOLDER_PATH, "correlation_comparison_boxplot.png")
    plt.tight_layout()
    plt.savefig(plot_path, format="png", dpi=300, bbox_inches="tight")
    
    print("="*85)
    print(f"Done! Boxplot saved to:\n{plot_path}")

if __name__ == "__main__":
    process_correlations()