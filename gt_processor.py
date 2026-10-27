import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, detrend

LOWCUT = 0.05
HIGHCUT = 1.5

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def filter_signal(data, fs):
    detrended = detrend(data)
    b, a = butter_bandpass(LOWCUT, HIGHCUT, fs)
    return filtfilt(b, a, detrended)

def estimate_bpm(time, signal, fs, apnea_start=-1, apnea_stop=-1, fast_start=-1, fast_stop=-1):
    p95 = np.percentile(signal, 95)
    p5 = np.percentile(signal, 5)
    amplitude_range = p95 - p5
    
    # Parametry standardowe
    prominence_std = 0.15 * amplitude_range
    distance_std = int(0.6 * fs)
    
    # Parametry dla oddechu szybkiego (płytka amplituda, wysoka częstotliwość do 240 BPM)
    prominence_fast = 0.05 * amplitude_range
    distance_fast = int(0.25 * fs)
    
    peaks_std, _ = find_peaks(signal, distance=distance_std, prominence=prominence_std)
    peaks_fast, _ = find_peaks(signal, distance=distance_fast, prominence=prominence_fast)
    
    valid_peaks = []
    for p in np.unique(np.concatenate((peaks_std, peaks_fast))):
        t_p = time[p]
        
        if apnea_start != -1 and apnea_stop != -1 and (apnea_start <= t_p <= apnea_stop):
            continue
            
        is_fast_region = (fast_start != -1 and fast_stop != -1) and (fast_start <= t_p <= fast_stop)
        
        if is_fast_region:
            if p in peaks_fast:
                valid_peaks.append(p)
        else:
            if p in peaks_std:
                valid_peaks.append(p)
                
    peaks = np.sort(np.array(valid_peaks))
    
    if len(peaks) < 2:
        return np.zeros_like(time)
    
    peak_times = time[peaks]
    rr_intervals = np.diff(peak_times)
    instant_bpm = 60.0 / rr_intervals
    
    continuous_bpm = np.zeros_like(time)
    current_peak_idx = 0
    
    for i, t in enumerate(time):
        if current_peak_idx < len(peak_times) - 1:
            if t >= peak_times[current_peak_idx + 1]:
                current_peak_idx += 1
                
        safe_idx = min(current_peak_idx, len(instant_bpm) - 1)
        time_since_last_peak = t - peak_times[current_peak_idx]
        
        is_fast_breathing = (fast_start != -1 and fast_stop != -1) and (fast_start <= t <= fast_stop)
        
        if time_since_last_peak > 6.0 and not is_fast_breathing:
            decay_factor = max(0, 1.0 - (time_since_last_peak - 6.0) * 0.25)
            continuous_bpm[i] = instant_bpm[safe_idx] * decay_factor
        else:
            continuous_bpm[i] = instant_bpm[safe_idx]
            
    early_mask = time < peak_times[0]
    if np.any(early_mask):
        continuous_bpm[early_mask] = instant_bpm[0]
        
    if apnea_start != -1 and apnea_stop != -1:
        apnea_mask = (time >= apnea_start) & (time <= apnea_stop)
        continuous_bpm[apnea_mask] = 0.0
            
    return continuous_bpm

def process_ground_truth_file(file_path, output_path):
    df = pd.read_csv(file_path, skiprows=1, names=['Time', 'Raw'])
    
    times = df['Time'].values
    fs = 1.0 / np.mean(np.diff(times))
    
    filtered = filter_signal(df['Raw'].values, fs)
    df['Filtered'] = filtered
    
    clicks = []
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times, filtered, color='royalblue')
    ax.set_title("1. Wskaż bezdech (2 kliknięcia, czerwone) | 2. Wskaż szybki oddech (2 kliknięcia, zielone)")
    ax.set_xlabel("Czas [s]")
    ax.grid(True)
    
    def onclick(event):
        if event.xdata is not None and event.button == 1:
            clicks.append(event.xdata)
            
            if len(clicks) <= 2:
                color = 'red'
            else:
                color = 'green'
                
            ax.axvline(x=event.xdata, color=color, linestyle='--')
            fig.canvas.draw()
            
            if len(clicks) == 4:
                plt.pause(0.5)
                plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)
    
    apnea_start, apnea_stop = -1, -1
    fast_start, fast_stop = -1, -1
    
    if len(clicks) >= 2:
        apnea_start, apnea_stop = min(clicks[0:2]), max(clicks[0:2])
    if len(clicks) == 4:
        fast_start, fast_stop = min(clicks[2:4]), max(clicks[2:4])
    
    df['BPM'] = estimate_bpm(times, filtered, fs, apnea_start, apnea_stop, fast_start, fast_stop)
    
    df.to_csv(output_path, index=False, float_format='%.6f')

if __name__ == "__main__":
    INPUT_FILE = r"C:\Users\ww\Desktop\pomiary_d\gt\record-agnieszka-rgb_data.txt"
    OUTPUT_FILE = r"C:\Users\ww\Desktop\pomiary_d\gt\processed_data-agnieszka-rgb.csv"
    
    process_ground_truth_file(INPUT_FILE, OUTPUT_FILE)