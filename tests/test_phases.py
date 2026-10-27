import numpy as np
import pandas as pd

from analysis.phases import infer_phase_windows


def _series(bpm, step=0.1):
    time = np.arange(0.0, len(bpm) * step, step)[: len(bpm)]
    return pd.DataFrame({"Time": time, "BPM": bpm})


def test_hyperventilation_is_the_tall_bout_before_apnea():
    bpm = np.concatenate(
        [
            np.full(400, 12.0),
            np.full(300, 16.0),
            np.full(400, 12.0),
            np.full(250, 45.0),
            np.full(300, 12.0),
            np.zeros(300),
        ]
    )
    windows = infer_phase_windows(_series(bpm))
    assert windows["apnea_start_s"] == np.float64(165.0) or windows["apnea_start_s"] == 165.0
    assert 100.0 < windows["hyperventilation_start_s"] < 115.0
    assert 130.0 < windows["hyperventilation_stop_s"] <= 140.0


def test_a_small_rate_drift_is_not_hyperventilation():
    bpm = np.concatenate([np.full(1200, 12.0), np.full(200, 14.0), np.full(200, 12.0), np.zeros(300)])
    windows = infer_phase_windows(_series(bpm))
    assert np.isnan(windows["hyperventilation_start_s"])
    assert windows["apnea_stop_s"] > windows["apnea_start_s"]
