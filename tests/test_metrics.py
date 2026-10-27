import numpy as np
import pytest

from analysis.metrics import apnea_scores, bland_altman, mae_bpm, spearman_waveform


def test_spearman_of_identical_series_is_one():
    values = np.array([1.0, 2.0, 4.0, 8.0, 3.0])
    assert spearman_waveform(values, values) == pytest.approx(1.0)


def test_masked_mae_uses_only_the_selected_samples():
    estimated = np.array([10.0, 10.0, 10.0, 12.0, 50.0])
    reference = np.array([10.0, 12.0, 11.0, 14.0, 0.0])
    keep = reference != 0.0
    assert mae_bpm(estimated, reference, keep) == pytest.approx(1.25)


def test_one_hertz_mae_includes_the_breath_hold():
    from analysis.metrics import one_hertz_mae

    estimated = np.zeros(40)
    reference = np.zeros(40)
    estimated[0] = 10.0
    estimated[20] = 4.0
    assert one_hertz_mae(estimated, reference) == pytest.approx(7.0)


def test_bland_altman_bias_of_constant_offset():
    reference = np.array([10.0, 12.0, 14.0, 16.0])
    estimated = reference + 2.0
    agreement = bland_altman(estimated, reference)
    assert agreement["bias"] == pytest.approx(2.0)
    assert agreement["loa_low"] == pytest.approx(2.0)
    assert agreement["loa_high"] == pytest.approx(2.0)


def test_apnea_confusion_at_one_hertz():
    time = np.arange(0.0, 4.0, 1.0)
    reference = np.array([10.0, 0.0, 0.0, 12.0])
    predicted = np.array([0, 0, 1, 1])
    scores = apnea_scores(time, reference, predicted)
    assert scores["sensitivity"] == pytest.approx(0.5)
    assert scores["specificity"] == pytest.approx(0.5)
    assert scores["tp"] == 1 and scores["fn"] == 1
    assert scores["tn"] == 1 and scores["fp"] == 1
