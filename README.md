# Contactless respiratory rate: RGB optical flow and ToF depth

## Objective

Compare a low-cost contactless respiratory-rate estimate with a chest belt. The camera signals are the waveforms already computed on the device. This repository does not contain video or raw image frames.

The related manuscript is [docs/Computers_in_Biology_and_Medicine.pdf](docs/Computers_in_Biology_and_Medicine.pdf): *A Real-Time Low-Cost Platform for Non-Contact Respiratory Rate Monitoring*.

## Study

Twelve adults took part between 3 and 9 May 2026. Each person completed the protocol twice with one MaixSense A075V. The RGB camera and the time-of-flight depth sensor share one network address, so they cannot stream together. The two sessions are minutes apart, and which one came first differs by person. Within a session the chest belt runs at the same time as the active sensor.

Participants are coded S01–S12 in acquisition order. The laboratory names are not in this repository. The only name key is `data/private/participant_key.csv`, which is local, ignored by git, and must not be uploaded to RODBUK.

### Protocol

Each recording starts in resting breathing. The volunteer then hyperventilates, returns toward the resting rate, holds the breath for about 30 s to simulate apnea, and breathes again. Recovery after the hold is not a protocol phase.

The belt recorder exported annotation files with only the header `Onset,Duration,Annotation`. Those empty files are not kept. The clicks used for the manuscript were applied when the reference rate was saved and were not stored again. `data/annotations/phases.csv` recovers the windows from that reference rate:

- apnea is the longest stretch at 0 breaths per minute lasting at least 10 s
- hyperventilation is the tallest smoothed bout before that stretch, at least 8 s long and at least 4 breaths per minute above the resting level
- the gap between those two windows is resting breathing

The RGB session of S08 has no such hyperventilation bout, so those cells are empty. These windows are not the manual clicks behind Table 5 of the manuscript. An existing phase file is not overwritten.

### Devices and signals

| Role | Instrument | What is stored |
| --- | --- | --- |
| Contactless, RGB session | MaixSense A075V | Chest signal from Farnebäck, Horn–Schunck, and Lucas–Kanade optical flow, plus the rate estimated on the device |
| Contactless, ToF session | Same sensor, depth mode | Respiratory signal derived from distance, plus the rate estimated on the device |
| Reference | Neurobit Optima+ 4 BLE piezoresistive belt | Resistance at 15.625 Hz (16 samples in each 1.024 s record) |

The reference rate in `data/reference/processed_data-*.csv` is the archived estimate: a Butterworth band-pass of order 3 from 0.05 to 1.5 Hz, then peak detection. During the breath-hold that rate is 0. The raw resistance is the `data/belt` text export. The vendor EDF is the same resistance in a proprietary file; it stays on the recording machine and is not part of git or of the RODBUK package.

## Analysis

`analysis` repeats the comparison in the manuscript.

1. Anchor the camera trace and the belt at their last sample, interpolate both onto one 20 Hz grid, z-score the waveforms, and shift them by the lag of maximum cross-correlation. The search is limited to 30 s.
2. Spearman correlation uses every aligned 20 Hz sample of the waveforms.
3. Respiratory-rate error is the mean absolute error of one sample per second over the whole recording, including the breath-hold.
4. Bland–Altman bias and limits of agreement drop samples whose reference rate is zero.
5. Apnea detection is scored at 1 Hz with the device flag `Is_Apnea`.
6. Phase error uses the segmented windows above and excludes samples whose reference rate is zero.

Thresholds are not tuned to force the published numbers. With these recordings the waveform correlation matches Table 3: median ToF 0.634, Farnebäck 0.630, Horn–Schunck 0.634, and Lucas–Kanade 0.644. The whole-recording rate error matches Table 4: median ToF 6.24, Farnebäck 4.24, Horn–Schunck 4.26, and Lucas–Kanade 3.83. Bland–Altman bias for ToF is +3.37 against the published +3.38. The RGB biases stay within a few tenths of a breath per minute of the published values. Apnea sensitivity stays within about one hundredth of Table 6. Phase error is not expected to match Table 5, because the original clicks were not saved. One ToF session, S04, keeps a device rate near 54 breaths per minute after the opening seconds. That point is inside Table 4 and sits above the axis of Figure 7 in the manuscript.

## Layout

```text
acquisition/     live RGB and ToF apps
data/            study files used by the analysis
  manifest.csv   sensor file, belt file, and session start for each recording
  annotations/   segmented phase windows
  belt/          raw resistance, header, and channel description
  rgb/           optical-flow signals
  tof/           depth signals
  reference/     processed belt rate, plus legacy_synced/ for the sync check
analysis/        synchronization, metrics, figures, and the RODBUK export
tests/           offline checks
docs/            manuscript PDF
results/         tables and figures, recreated by the analysis, not versioned
```

`acquisition/tof/grid.py` only shows the depth variance. It is not a fourth method. `data/reference/legacy_synced` is the archived alignment used to confirm that synchronization has not drifted. It is not a second copy of the raw study.

## What stays off GitHub

- `deposit/` is the flat file list for the RODBUK Files tab. It repeats the signals in comma-separated form, so it is not versioned. Rebuild it with `python -m analysis.export_deposit` and upload those files, not this git repository.
- `data/private/` holds the S01–S12 name key.
- `*.edf` are the vendor belt files.
- `results/` is recreated by the analysis.

## Install and run

Python 3.10 or newer.

```powershell
cd C:\Users\ludwi\Documents\GitHub\Contacless-RespiratoryAnalysis
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
.\.venv\Scripts\python.exe -m analysis.run
```

`python -m analysis.run` writes `results/report.md`, `results/summary.csv`, `results/per_recording.csv`, and the figures in `results/figures`. It also opens the figure windows. Close them to let the process exit.

## Apps, with the sensor connected

```powershell
python -m acquisition.rgb.app
python -m acquisition.tof.app
python -m acquisition.tof.grid
```

The sensor defaults to `192.168.233.1` port 80. The RGB page listens on 5000, the ToF page on 8080, and the variance view on 8081. Override them with `MAIX_HOST`, `MAIX_PORT`, `RGB_PORT`, `TOF_PORT`, and `TOF_GRID_PORT`. New recordings go to `acquisition/rgb/recordings` and `acquisition/tof/recordings`, which are not versioned.

The pose network is not in the repository. Without `acquisition/rgb/lightweight_pose_estimation_201912.onnx` the RGB app still starts, but it does not place the region of interest. `POSE_ONNX` points at the weights.

Redraw the latest session:

```powershell
python -m acquisition.rgb.plot_results
python -m acquisition.tof.data_plotter
```
