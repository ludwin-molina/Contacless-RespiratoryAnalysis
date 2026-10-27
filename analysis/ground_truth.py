"""Belt filtering and reference respiratory-rate estimation.

The study's reference BPM is the archived ``processed_data-*.csv``. Those files
already contain the manual apnea and hyperventilation clicks, which were not
stored as separate intervals. ``estimate_bpm`` reproduces that estimator when
the windows are known.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, detrend

LOWCUT = 0.05
HIGHCUT = 1.5


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype="band")


def filter_signal(data, fs, lowcut=LOWCUT, highcut=HIGHCUT):
    detrended = detrend(np.asarray(data, dtype=float))
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, detrended)


def estimate_bpm(time, signal, fs, apnea_start=-1, apnea_stop=-1, fast_start=-1, fast_stop=-1):
    """Instantaneous breaths per minute, with apnea forced to zero.

    Fast breathing uses a lower prominence and a shorter minimum peak distance
    so hyperventilation (up to about 240 BPM) is not missed. Outside that
    window the detector stays conservative. A gap longer than 6 s, outside
    fast breathing, decays the last rate toward zero.
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    amplitude_range = np.percentile(signal, 95) - np.percentile(signal, 5)

    prominence_std = 0.15 * amplitude_range
    distance_std = int(0.6 * fs)
    prominence_fast = 0.05 * amplitude_range
    distance_fast = int(0.25 * fs)

    peaks_std, _ = find_peaks(signal, distance=distance_std, prominence=prominence_std)
    peaks_fast, _ = find_peaks(signal, distance=distance_fast, prominence=prominence_fast)

    valid_peaks = []
    for peak in np.unique(np.concatenate((peaks_std, peaks_fast))):
        peak_time = time[peak]
        if apnea_start != -1 and apnea_stop != -1 and apnea_start <= peak_time <= apnea_stop:
            continue
        in_fast = fast_start != -1 and fast_stop != -1 and fast_start <= peak_time <= fast_stop
        if in_fast and peak in peaks_fast:
            valid_peaks.append(peak)
        elif not in_fast and peak in peaks_std:
            valid_peaks.append(peak)

    peaks = np.sort(np.asarray(valid_peaks))
    if len(peaks) < 2:
        return np.zeros_like(time)

    peak_times = time[peaks]
    instant_bpm = 60.0 / np.diff(peak_times)
    continuous_bpm = np.zeros_like(time)
    current_peak_idx = 0

    for i, sample_time in enumerate(time):
        if current_peak_idx < len(peak_times) - 1 and sample_time >= peak_times[current_peak_idx + 1]:
            current_peak_idx += 1
        safe_idx = min(current_peak_idx, len(instant_bpm) - 1)
        since_peak = sample_time - peak_times[current_peak_idx]
        in_fast = fast_start != -1 and fast_stop != -1 and fast_start <= sample_time <= fast_stop
        if since_peak > 6.0 and not in_fast:
            decay = max(0.0, 1.0 - (since_peak - 6.0) * 0.25)
            continuous_bpm[i] = instant_bpm[safe_idx] * decay
        else:
            continuous_bpm[i] = instant_bpm[safe_idx]

    continuous_bpm[time < peak_times[0]] = instant_bpm[0]
    if apnea_start != -1 and apnea_stop != -1:
        continuous_bpm[(time >= apnea_start) & (time <= apnea_stop)] = 0.0
    return continuous_bpm
