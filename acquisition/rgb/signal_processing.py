import cv2
import numpy as np
import time
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks, detrend

# Physiological limits of the rate estimate, in breaths per minute.
MIN_BPM = 4.0
MAX_BPM = 150.0
SAMPLING_RATE = 30.0

class BreathAnalysisEngine:
    """Band-pass the motion trace, score its rhythm, and estimate breaths per minute."""
    def __init__(self, buffer_size_sec=30.0, fps=30.0):
        self.fps = fps
        self.buffer_size = int(buffer_size_sec * fps)

        # Order-4 Butterworth, 0.08 to 2.5 Hz, covers the respiratory band at this frame rate.
        nyquist = 0.5 * fps
        self.b, self.a = butter(4, [0.08 / nyquist, 2.5 / nyquist], btype='band')

        self.apnea_integrator = 0.0
        self.integration_decay = 0.98
        self.apnea_threshold = 0.5
        self.apnea_increment = 0.015
        self.apnea_history = []
        self.history_len = int(fps * 1.0)

        self.last_valid_bpm = 0.0
        self.raw_buffer = []
        self.times = []

    def process_sample(self, t_now, raw_val, breath_threshold):
        """Return (bpm, is_apnea, filtered_signal, amplitude_std) for one new sample."""
        self.times.append(t_now)
        self.raw_buffer.append(raw_val)

        if len(self.raw_buffer) > self.buffer_size:
            self.raw_buffer.pop(0)
            self.times.pop(0)

        # Filtering needs at least two seconds of samples.
        if len(self.raw_buffer) < self.fps * 2:
            return None, False, np.array(self.raw_buffer), 0.0

        raw_arr = np.array(self.raw_buffer)
        time_array = np.array(self.times)

        # Remove the slow drift, then apply the zero-phase band-pass.
        raw_detrended = detrend(raw_arr, type='linear')
        filtered = filtfilt(self.b, self.a, raw_detrended)

        # Quiet breathing is a low standard deviation on the last three seconds.
        tail_len = int(3.0 * self.fps)
        tail_data = filtered[-tail_len:] if len(filtered) > tail_len else filtered
        current_std = np.std(tail_data)

        # Peak of the normalized autocorrelation between 0.33 s and 15 s.
        rhythm_peak = 0.0
        n = len(filtered)
        if n > 0:
            acorr = np.correlate(filtered, filtered, mode='full')
            acorr = acorr[n - 1:] / (acorr[n - 1] + 1e-9)

            min_lag = int(0.33 * self.fps)
            max_lag = min(int(15.0 * self.fps), n - 1)

            if max_lag > min_lag:
                valid_acorr = acorr[min_lag:max_lag]
                if len(valid_acorr) > 0:
                    rhythm_peak = np.max(valid_acorr)

        raw_apnea = False
        if current_std < breath_threshold:
            if current_std < (breath_threshold * 0.6):
                raw_apnea = True
            elif rhythm_peak < 0.55:
                raw_apnea = True
            else:
                raw_apnea = False
        else:
            if rhythm_peak < 0.25:
                raw_apnea = True
            else:
                raw_apnea = False

        self.apnea_history.append(1 if raw_apnea else 0)
        if len(self.apnea_history) > self.history_len:
            self.apnea_history.pop(0)

        local_apnea = raw_apnea
        if len(self.apnea_history) == self.history_len:
            apnea_density = sum(self.apnea_history) / self.history_len
            # Confirm apnea only when most of the last second agrees.
            if apnea_density > 0.6:
                local_apnea = True

        current_bpm = None
        if not local_apnea:
            peak_vals = np.abs(filtered)
            
            noise_floor = max(breath_threshold * 0.5, 0.05)
            max_val = np.percentile(peak_vals, 95) if len(peak_vals) > 0 else 1.0
            if max_val < noise_floor:
                max_val = noise_floor

            norm_signal = np.clip(filtered / max_val, -2.0, 2.0)

            # Peaks must be at least 0.33 s apart and stand out from the neighbours.
            min_dist = int(self.fps * 0.33)
            peaks, _ = find_peaks(norm_signal, distance=min_dist, prominence=0.12)

            if len(peaks) >= 2:
                peak_times = time_array[peaks]
                intervals = np.diff(peak_times)

                valid_intervals = intervals[(intervals > 0.33) & (intervals < 15.0)]
                if len(valid_intervals) > 0:
                    
                    # If the last 10 s disagree with the whole buffer, keep only the recent peaks.
                    t_end = time_array[-1]
                    recent_peak_times = peak_times[peak_times > (t_end - 10.0)]
                    
                    recent_bpm_est = None
                    if len(recent_peak_times) >= 2:
                        recent_intervals = np.diff(recent_peak_times)
                        valid_recent = recent_intervals[(recent_intervals > 0.33) & (recent_intervals < 15.0)]
                        if len(valid_recent) > 0:
                            recent_bpm_est = 60.0 / np.median(valid_recent)
                            
                    global_bpm_est = 60.0 / np.median(valid_intervals)
                    
                    if recent_bpm_est is not None:
                        bpm_diff = abs(recent_bpm_est - global_bpm_est)
                        # Flush when the recent rate is more than 12 BPM, or 40 percent, away.
                        if bpm_diff > 12.0 or (bpm_diff / max(global_bpm_est, 1.0)) > 0.4:
                            keep_len = int(10.0 * self.fps) 
                            if len(self.raw_buffer) > keep_len:
                                self.raw_buffer = self.raw_buffer[-keep_len:]
                                self.times = self.times[-keep_len:]
                                
                            current_bpm = recent_bpm_est 
                            
                    if current_bpm is None:
                        median_diff = np.median(valid_intervals)
                        reliable_diffs = valid_intervals[
                            np.abs(valid_intervals - median_diff) < 0.3 * median_diff
                        ]

                        final_avg_diff = np.mean(reliable_diffs) if len(reliable_diffs) > 0 else median_diff

                        if final_avg_diff > 0:
                            current_bpm = 60.0 / final_avg_diff
                        
            # Ignore a wild rate in the first moments after a breath-hold.
            if current_bpm is not None and self.apnea_integrator > 0.1:
                if self.last_valid_bpm > 0 and abs(current_bpm - self.last_valid_bpm) > 25.0:
                    current_bpm = None  
                elif self.last_valid_bpm == 0 and current_bpm > 50.0:
                    current_bpm = None  

        if current_bpm is None:
            local_apnea = True

        if local_apnea:
            if current_std < (breath_threshold * 0.6):
                self.apnea_integrator += (self.apnea_increment * 1.5)
            else:
                self.apnea_integrator += self.apnea_increment
        else:
            self.apnea_integrator *= self.integration_decay

        self.apnea_integrator = max(0.0, min(self.apnea_integrator, 1.0))

        if self.apnea_integrator > self.apnea_threshold:
            self.last_valid_bpm = 0.0
            return 0.0, True, filtered, current_std
        else:
            if current_bpm is not None:
                self.last_valid_bpm = current_bpm
            return self.last_valid_bpm, False, filtered, current_std


def contrast_boost(gray):
    """Stretch contrast between the 1st and 99th percentile so optical flow sees the chest."""
    lo, hi = np.percentile(gray, (1, 99))
    if hi <= lo:
        return gray
    g = np.clip((gray - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return g