from analysis.config import RGB_METHODS, ROOT, load_manifest


def test_manifest_covers_the_cohort():
    manifest = load_manifest()
    subjects = set(manifest["subject"])
    assert len(subjects) == 12

    rgb = manifest[manifest["modality"] == "rgb"]
    tof = manifest[manifest["modality"] == "tof"]
    assert set(rgb["method"]) == set(RGB_METHODS)
    assert set(tof["method"]) == {"ToF"}
    assert tof["subject"].nunique() == 12
    for _, group in rgb.groupby("subject"):
        assert set(group["method"]) == set(RGB_METHODS)

    for record in manifest.itertuples(index=False):
        assert (ROOT / record.sensor_file).is_file(), record.sensor_file
        assert (ROOT / record.reference_file).is_file(), record.reference_file
