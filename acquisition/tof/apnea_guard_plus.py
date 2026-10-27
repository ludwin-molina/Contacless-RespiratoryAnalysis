"""Apnea decision from amplitude and from autocorrelation of the depth trace."""

import numpy as np


class ApneaGuardPlus:
    """Flag apnea when the recent trace is quiet or has lost its rhythm."""

    def __init__(self, fps=15.0):
        self.fps = fps
        # Autocorrelation peak required before a window counts as rhythmic.
        # 0.45 rejects noisy stretches that are not a breath cycle.
        self.rhythm_threshold = 0.45

    def _check_rhythm(self, signal_window, min_lag_sec, max_lag_sec):
        """Largest normalized autocorrelation peak between the two lags."""
        count = len(signal_window)
        if count < int(min_lag_sec * self.fps):
            return 0.0

        correlation = np.correlate(signal_window, signal_window, mode="full")
        correlation = correlation[count - 1 :] / (correlation[count - 1] + 1e-9)

        min_lag = int(min_lag_sec * self.fps)
        max_lag = min(int(max_lag_sec * self.fps), count - 1)
        if max_lag > min_lag:
            window = correlation[min_lag:max_lag]
            if len(window) > 0:
                return np.max(window)
        return 0.0

    def detect(self, full_signal):
        """Return (is_apnea, purge_buffer). purge_buffer means rhythm just returned."""
        tail_len = int(8.0 * self.fps)
        tail = full_signal[-tail_len:] if len(full_signal) > tail_len else full_signal
        # A standard deviation under 0.8 mm is treated as no chest motion.
        if np.std(tail) < 0.8:
            return True, False

        long_rhythm = self._check_rhythm(full_signal, 0.4, 20.0)
        short_len = int(10.0 * self.fps)
        short_signal = full_signal[-short_len:] if len(full_signal) > short_len else full_signal
        short_rhythm = self._check_rhythm(short_signal, 0.4, 6.0)

        is_apnea = False
        purge_buffer = False
        if long_rhythm < self.rhythm_threshold:
            if short_rhythm >= self.rhythm_threshold:
                # The short window recovered a cycle, so the long buffer is stale noise.
                purge_buffer = True
            else:
                is_apnea = True
        return is_apnea, purge_buffer
