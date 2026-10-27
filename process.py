import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import zscore
import matplotlib.pyplot as plt
import os
import glob

FS = 20.0
METHODS = ['Farneback', 'Horn-Schunck', 'Lucas-Kanade']
NAMES = [
    'dominik', 'adrian', 'szymon', 'bartek', 'pawel', 'paulina', 
    'olaw', 'olaa', 'matylda', 'kacper', 'agnieszka', 'grzegorz'
]

def synchronize_and_save(rgb_file, gt_file, person_name, method_name):
    try:
        df_rgb = pd.read_csv(rgb_file, sep=';')
        df_gt = pd.read_csv(gt_file, sep=',')
    except Exception as e:
        print(f"Błąd odczytu plików dla {person_name} ({method_name}): {e}")
        return False

    t_rgb = df_rgb['Elapsed_Sec'].values
    t_gt = df_gt['Time'].values

    t_rgb_zero = t_rgb - t_rgb[-1]
    t_gt_zero = t_gt - t_gt[-1]

    t_start = max(t_rgb_zero[0], t_gt_zero[0])
    t_end = min(t_rgb_zero[-1], t_gt_zero[-1])
    common_time = np.arange(t_start, t_end, 1.0 / FS)

    interp_rgb_filt = interp1d(t_rgb_zero, df_rgb['Signal_Pos'], bounds_error=False, fill_value="extrapolate")
    interp_rgb_bpm = interp1d(t_rgb_zero, df_rgb['BPM'], bounds_error=False, fill_value="extrapolate")
    interp_gt_filt = interp1d(t_gt_zero, df_gt['Filtered'], bounds_error=False, fill_value="extrapolate")
    interp_gt_bpm = interp1d(t_gt_zero, df_gt['BPM'], bounds_error=False, fill_value="extrapolate")

    rgb_filt_res = interp_rgb_filt(common_time)
    rgb_bpm_res = interp_rgb_bpm(common_time)
    gt_filt_res = interp_gt_filt(common_time)
    gt_bpm_res = interp_gt_bpm(common_time)

    rgb_norm = zscore(-1.0 * rgb_filt_res)
    gt_norm = zscore(gt_filt_res)

    correlation = np.correlate(rgb_norm, gt_norm, mode='full')
    lags = np.arange(-len(gt_norm) + 1, len(rgb_norm))
    best_lag = lags[np.argmax(correlation)]

    if best_lag > 0:
        rgb_alg = rgb_norm[best_lag:]
        rgb_bpm_alg = rgb_bpm_res[best_lag:]
        gt_alg = gt_norm[:-best_lag]
        gt_bpm_alg = gt_bpm_res[:-best_lag]
        time_alg = common_time[best_lag:]
    elif best_lag < 0:
        abs_lag = abs(best_lag)
        rgb_alg = rgb_norm[:-abs_lag]
        rgb_bpm_alg = rgb_bpm_res[:-abs_lag]
        gt_alg = gt_norm[abs_lag:]
        gt_bpm_alg = gt_bpm_res[abs_lag:]
        time_alg = common_time[:-abs_lag]
    else:
        rgb_alg, rgb_bpm_alg = rgb_norm, rgb_bpm_res
        gt_alg, gt_bpm_alg = gt_norm, gt_bpm_res
        time_alg = common_time

    relative_time = time_alg - time_alg[0]

    out_csv = f'synced_{person_name}_{method_name}.csv'
    df_synced = pd.DataFrame({
        'Relative_Time_s': relative_time,
        'RGB_Signal_ZScore': rgb_alg,
        'GT_Signal_ZScore': gt_alg,
        'RGB_BPM': rgb_bpm_alg,
        'GT_BPM': gt_bpm_alg
    })
    df_synced.to_csv(out_csv, sep=';', index=False)

    out_plot = f'plot_{person_name}_{method_name}.png'
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axs[0].plot(relative_time, rgb_alg, label=f'RGB {method_name} (Z-Score)', color='blue', alpha=0.7)
    axs[0].plot(relative_time, gt_alg, label='Ground Truth (Z-Score)', color='orange', alpha=0.7)
    axs[0].set_title(f'Sygnały: {person_name} - {method_name} (Synchronizacja)')
    axs[0].set_ylabel('Amplituda znormalizowana')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    axs[1].plot(relative_time, rgb_bpm_alg, label='RGB BPM', color='red', linewidth=1.5)
    axs[1].plot(relative_time, gt_bpm_alg, label='GT BPM', color='green', linewidth=1.5)
    axs[1].set_title('Estymacja BPM')
    axs[1].set_xlabel('Czas [s]')
    axs[1].set_ylabel('BPM')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.close()
    
    return True

def main():
    rgb_files = glob.glob("*_Farneback.csv")
    timestamps = sorted([f.split('_Farneback')[0] for f in rgb_files])
    
    sync_log = []

    if len(timestamps) != len(NAMES):
        count = min(len(timestamps), len(NAMES))
    else:
        count = len(NAMES)

    for i in range(count):
        timestamp = timestamps[i]
        person = NAMES[i]
        gt_file = f'processed_data-{person}-rgb.csv'
        
        for method in METHODS:
            rgb_file = f"{timestamp}_{method}.csv"
            
            if os.path.exists(rgb_file) and os.path.exists(gt_file):
                success = synchronize_and_save(rgb_file, gt_file, person, method)
                if success:
                    sync_log.append(f"GT: {gt_file} <---> RGB: {rgb_file}")

    print("\n[RAPORT SYNCHRONIZACJI]")
    for entry in sync_log:
        print(entry)

if __name__ == "__main__":
    main()