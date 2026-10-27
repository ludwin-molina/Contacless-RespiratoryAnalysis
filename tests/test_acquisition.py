import numpy as np


def test_apps_import_without_contacting_the_sensor():
    import acquisition.rgb.app
    import acquisition.tof.app
    import acquisition.tof.grid

    assert acquisition.rgb.app.app is not None
    assert acquisition.tof.app.app is not None
    assert acquisition.tof.grid.app is not None


def test_pose_without_weights_returns_no_roi():
    from acquisition.rgb.pose_estimation import get_pose_roi

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    roi, keypoints = get_pose_roi(frame)
    assert roi is None
    assert keypoints is None


def test_rgb_engine_filters_a_synthetic_breath():
    from acquisition.rgb.signal_processing import BreathAnalysisEngine

    engine = BreathAnalysisEngine(buffer_size_sec=10.0, fps=30.0)
    last = None
    sample_time = 0.0
    for _ in range(30 * 8):
        sample_time += 1.0 / 30.0
        sample = np.sin(2 * np.pi * 0.25 * sample_time)
        last = engine.process_sample(sample_time, sample, breath_threshold=0.05)
    bpm, is_apnea, filtered, _std = last
    assert filtered is not None and len(filtered) > 10
    assert np.isfinite(filtered).all()
    assert bpm is not None


def test_tof_apnea_detector_flags_a_silent_trace():
    from acquisition.tof.apnea_guard_plus import ApneaGuardPlus

    is_apnea, _purge = ApneaGuardPlus(fps=15.0).detect(np.zeros(200))
    assert is_apnea is True


def test_tof_processor_filters_a_synthetic_breath():
    from acquisition.tof.signal_processing import SignalProcessor

    processor = SignalProcessor(buffer_size=600)
    for index in range(300):
        processor.buf.append(np.sin(2 * np.pi * 0.3 * index / 15.0))
        processor.times.append(index / 15.0)
    processor.calc_bpm()
    assert len(processor.filtered) == 300
    assert np.isfinite(processor.filtered).all()


def test_logger_keeps_the_sample_after_stop(tmp_path):
    from acquisition.rgb.data_logger import BreathLogger

    logger = BreathLogger(output_dir=tmp_path)
    logger.start(["Farneback"])
    logger.log_sample("Farneback", 1.25, 0.1, 15.0, False)
    logger.stop()
    written = list(tmp_path.glob("*.csv"))
    assert len(written) == 1
    text = written[0].read_text(encoding="utf-8")
    assert "1.250000" in text
    assert "Is_Apnea" in text
