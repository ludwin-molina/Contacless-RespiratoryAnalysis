# Contactless respiratory rate: RGB, ToF, and a chest belt

Twelve volunteers were recorded with a MaixSense A075V, once with the RGB camera and once with the ToF depth sensor. Each session has normal breathing, voluntary hyperventilation, and a breath-hold. The reference is a Neurobit piezoresistive belt. The manuscript is [docs/Computers_in_Biology_and_Medicine.pdf](docs/Computers_in_Biology_and_Medicine.pdf).

## Layout

- `acquisition/rgb` and `acquisition/tof`: live apps. `acquisition/tof/grid.py` only shows the depth variance. It is not a fourth method.
- `data/belt`: belt export. The analysis reads the text files. The original Neurobit EDF is kept beside them.
- `data/rgb` and `data/tof`: signals already computed on the device (Farneback, Horn-Schunck, Lucas-Kanade, and ToF depth).
- `data/reference`: reference rate. A breath-hold is stored as 0 breaths per minute. The original click times were not kept.
- `data/manifest.csv`: which camera file belongs to which person. Pairing follows the belt header time.
- `data/annotations/phases.csv`: apnea and hyperventilation windows segmented from the reference rate. The belt annotation files are empty.
- `deposit`: flat file list for the RODBUK Files tab, with subject codes S01–S12 and `00_readme.txt`. Rebuild it with `python -m analysis.export_deposit`.
- `data/reference/legacy_synced`: archived synchronization, used to check that the new pipeline has not drifted.
- `analysis`: 20 Hz synchronization, Spearman correlation, respiratory-rate error, Bland–Altman, phase error, and apnea at 1 Hz.
- `results`: tables and figures. The folder is not versioned. Running the analysis recreates it.

## Install and repeat the analysis

Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
python -m analysis.run
```

`python -m analysis.run` writes `results/report.md`, `results/summary.csv`, `results/per_recording.csv`, and the figures in `results/figures`. It also opens the figure windows. Close them to let the process exit. Thresholds are not tuned to force the published numbers.

With the recordings in this repository the waveform Spearman correlation matches Table 3 (median ToF 0.634, Farneback 0.630, Horn-Schunck 0.634, Lucas-Kanade 0.644). The respiratory-rate error is the mean absolute error of one sample per second over the whole recording, including the breath-hold, which is how Table 4 was computed. Bland–Altman bias drops the samples whose reference rate is zero. Apnea sensitivity stays within about one hundredth of Table 6.

## Phases

Every row in `data/annotations/phases.csv` is `segmented`. The belt annotation files contain only a header, so the original clicks are not in the repository. Apnea is the longest stretch where the reference rate is zero and lasts at least 10 s. Hyperventilation is the tallest smoothed bout before that stretch, at least 8 s long and at least 4 breaths per minute above the resting level. The RGB recording of S08 has no such bout, so those cells stay empty. These windows are not the manual annotation used for Table 5. An existing phase file is not overwritten.

## Apps, with the sensor connected

RGB and ToF use the same MaixSense address, so they cannot stream at the same time. The pose network is not in the repository. Without `acquisition/rgb/lightweight_pose_estimation_201912.onnx` the RGB app still starts, but it does not place the region of interest. `POSE_ONNX` points at the weights.

```bash
python -m acquisition.rgb.app
python -m acquisition.tof.app
python -m acquisition.tof.grid
```

The sensor defaults to `192.168.233.1` port 80. The RGB page listens on 5000, the ToF page on 8080, and the variance view on 8081. Override them with `MAIX_HOST`, `MAIX_PORT`, `RGB_PORT`, `TOF_PORT`, and `TOF_GRID_PORT`. Recordings go to `acquisition/rgb/recordings` and `acquisition/tof/recordings`.

Redraw the latest session:

```bash
python -m acquisition.rgb.plot_results
python -m acquisition.tof.data_plotter
```
