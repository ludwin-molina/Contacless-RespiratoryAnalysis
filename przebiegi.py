import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import shapiro, pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Ścieżka do Twojego folderu
FOLDER_PATH = r"C:\Users\ww\Desktop\pomiary_d\gt"
# Folder na zapisanie najlepszych wykresów
OUTPUT_DIR = os.path.join(FOLDER_PATH, "top_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_and_plot_top_signals():
    results = []
    
    # KROK 1: Skanowanie wszystkich plików i wyliczenie metryk
    csv_files = glob.glob(os.path.join(FOLDER_PATH, "*.csv"))
    
    for file in csv_files:
        filename = os.path.basename(file)
        
        # Rozpoznanie typu czujnika i przypisanie kolumn
        if filename.startswith("synced_") and filename.endswith(".csv"):
            parts = filename.replace("synced_", "").replace(".csv", "").split("_")
            if len(parts) == 2:
                person, method = parts[0], parts[1]
                sensor = "RGB"
                signal_col = "RGB_Signal_ZScore"
                bpm_col = "RGB_BPM"
            else:
                continue
        elif filename.startswith("synchronized_output_zscore-") and filename.endswith("-lidar.csv"):
            person = filename.split("-")[1]
            sensor = "LiDAR"
            method = "LiDAR"
            signal_col = "Lidar_Signal_ZScore"
            bpm_col = "Lidar_BPM"
        else:
            continue
            
        try:
            df = pd.read_csv(file, sep=';')
            gt_sig_col = "GT_Signal_ZScore"
            gt_bpm_col = "GT_BPM"
            time_col = "Relative_Time_s"
            
            df_clean = df[[time_col, signal_col, gt_sig_col, bpm_col, gt_bpm_col]].dropna()
            
            sig_est = df_clean[signal_col].values
            sig_gt = df_clean[gt_sig_col].values
            
            # Korelacja
            _, p1 = shapiro(sig_est)
            _, p2 = shapiro(sig_gt)
            
            if p1 > 0.05 and p2 > 0.05:
                corr, _ = pearsonr(sig_est, sig_gt)
                corr_type = "Pearson"
            else:
                corr, _ = spearmanr(sig_est, sig_gt)
                corr_type = "Spearman"
                
            # MAE
            mae = np.mean(np.abs(df_clean[bpm_col] - df_clean[gt_bpm_col]))
            
            results.append({
                "Person": person,
                "Method": method,
                "Sensor": sensor,
                "Correlation": corr,
                "Corr_Type": corr_type,
                "MAE": mae,
                "Filename": file,
                "Signal_Col": signal_col,
                "BPM_Col": bpm_col
            })
            
        except Exception as e:
            print(f"[!] Error processing file {filename}: {e}")

    df_results = pd.DataFrame(results)
    if df_results.empty:
        print("No valid files found.")
        return

    # KROK 2: Wybór TOP 5 dla LiDAR i RGB
    top_lidar = df_results[df_results["Sensor"] == "LiDAR"].sort_values(by="Correlation", ascending=False).head(5)
    top_rgb = df_results[df_results["Sensor"] == "RGB"].sort_values(by="Correlation", ascending=False).head(5)
    
    top_files = pd.concat([top_lidar, top_rgb])
    
    print("\nGenerowanie wykresów dla TOP 10 nagrań...")
    
    # Konfiguracja stylu wykresów
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # KROK 3: Generowanie wykresów jakościowych
    for rank, (index, row) in enumerate(top_files.iterrows()):
        df = pd.read_csv(row["Filename"], sep=';')
        
        time = df["Relative_Time_s"]
        sig_est = df[row["Signal_Col"]]
        sig_gt = df["GT_Signal_ZScore"]
        bpm_est = df[row["BPM_Col"]]
        bpm_gt = df["GT_BPM"]
        
        # Tworzenie figury z 2 podwykresami (Z-score góra, BPM dół)
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # --- Górny wykres (Sygnał Z-Score) ---
        axs[0].plot(time, sig_gt, label="Reference Signal (GT)", color='#1f77b4', linewidth=2, alpha=0.8)
        axs[0].plot(time, sig_est, label=f"Estimated Signal ({row['Method']})", color='#d62728', linewidth=1.5, alpha=0.9, linestyle='--')
        axs[0].set_ylabel("Z-Score Amplitude")
        axs[0].set_title(f"Respiratory Signal Comparison ({row['Sensor']} vs GT)", fontweight='bold')
        axs[0].legend(loc="upper right")
        
        # Dodanie okienka z metrykami (jak na Twoim screenie)
        metrics_text = (
            f"Subject: {row['Person'].capitalize()}\n"
            f"Method: {row['Method']}\n"
            f"Correlation: {row['Correlation']:.3f} ({row['Corr_Type']})\n"
            f"MAE: {row['MAE']:.2f} BPM"
        )
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray')
        axs[0].text(0.02, 0.95, metrics_text, transform=axs[0].transAxes, fontsize=11,
                    verticalalignment='top', bbox=props)

        # --- Dolny wykres (BPM) ---
        axs[1].plot(time, bpm_gt, label="Reference BPM (GT)", color='#1f77b4', linewidth=2, alpha=0.8)
        axs[1].plot(time, bpm_est, label=f"Estimated BPM ({row['Method']})", color='#d62728', linewidth=1.5, alpha=0.9, linestyle='--')
        axs[1].set_ylabel("Heart / Breathing Rate [BPM]")
        axs[1].set_xlabel("Relative Time [s]")
        axs[1].set_title("BPM Estimation over Time", fontweight='bold')
        axs[1].legend(loc="upper right")
        
        # Zapisywanie
        sensor_prefix = row['Sensor'].lower()
        # Np. "TOP_1_lidar_kacper.png" lub "TOP_3_rgb_adrian_Farneback.png"
        rank_in_group = list(top_lidar.index).index(index) + 1 if row['Sensor'] == 'LiDAR' else list(top_rgb.index).index(index) + 1
        
        safe_method = row['Method'].replace("-", "_")
        out_name = f"TOP_{rank_in_group:02d}_{sensor_prefix}_{row['Person']}_{safe_method}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        plt.tight_layout()
        plt.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig) # Zamknięcie figury zwalnia pamięć RAM!
        
        print(f"Zapisano: {out_name} (Korelacja: {row['Correlation']:.3f})")

    print("="*80)
    print(f"Wszystkie {len(top_files)} wykresów zapisano w folderze:\n{OUTPUT_DIR}")

if __name__ == "__main__":
    process_and_plot_top_signals()