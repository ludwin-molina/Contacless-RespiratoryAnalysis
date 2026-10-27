import numpy as np
import pytest

from analysis.ground_truth import estimate_bpm, filter_signal


def test_filtered_rate_is_zero_during_apnea():
    time = np.arange(0.0, 80.0, 0.05)
    raw = np.sin(2 * np.pi * 0.25 * time) + 0.01 * time
    filtered = filter_signal(raw, fs=20.0)
    bpm = estimate_bpm(time, filtered, fs=20.0, apnea_start=40.0, apnea_stop=55.0)

    assert np.isfinite(filtered).all()
    apnea = (time >= 40.0) & (time <= 55.0)
    assert np.all(bpm[apnea] == 0.0)
    resting = (time > 10.0) & (time < 35.0)
    assert np.median(bpm[resting]) == pytest.approx(15.0, abs=2.0)
