# A real-time low-cost platform for contactless respiratory-rate monitoring

This repository is the platform and the study recordings behind *A Real-Time Low-Cost Platform for Non-Contact Respiratory Rate Monitoring* (Computers in Biology and Medicine).

The platform estimates respiratory rate from a chest signal computed on the device. The files in `data/` are those respiratory signals, together with a piezoresistive chest belt recorded at the same time. They are not camera video.

## Objective

Build a real-time, low-cost platform that monitors respiratory rate without contact, and compare its estimates with a chest belt during resting breathing, voluntary hyperventilation, and a breath-hold.

## Hardware

One MaixSense A075V provides both the RGB camera and the time-of-flight depth sensor. It uses the address `192.168.233.1`, port 80. The sensor streams one modality at a time, so each person is recorded twice: once with RGB and once with depth. In this study the pipelines ran on a Raspberry Pi 5, and the respiratory signal was computed on the device.

The reference is a Neurobit Optima+ 4 BLE piezoresistive belt on the chest. It records resistance in kilohms, 16 samples in each 1.024 s record (15.625 Hz), with a high-pass at 0 Hz and a low-pass at 5.78125 Hz. The belt runs together with whichever camera modality is active.

### RGB session

A pose network places a region of interest on the chest. Optical flow inside that region produces three respiratory signals, one per method: Farnebäck, Horn–Schunck, and Lucas–Kanade. Each RGB file stores:

| Column | Meaning |
| --- | --- |
| `Timestamp_Abs` | Device clock, seconds since 1970-01-01 |
| `Elapsed_Sec` | Seconds from the start of the file |
| `Signal_Pos` | Chest motion from optical flow |
| `Signal_Std` | Spread of that motion inside the frame |
| `BPM` | Respiratory rate estimated on the device, breaths per minute |
| `Is_Apnea` | Device apnea flag, 1 or 0 |

Point `POSE_ONNX` at the pose weights, or place them at `acquisition/rgb/lightweight_pose_estimation_201912.onnx`, so the app can set the chest region.

### ToF session

The depth pipeline tracks a region on the subject and uses the mean distance in that region as the respiratory signal. Each ToF file stores:

| Column | Meaning |
| --- | --- |
| `Timestamp_Abs` | Device clock, seconds since 1970-01-01 |
| `Elapsed_Sec` | Seconds from the start of the file |
| `Signal_Raw_mm` | Mean distance in the region, millimetres |
| `Signal_Filtered` | Filtered distance used for the rate |
| `BPM` | Respiratory rate estimated on the device, breaths per minute |
| `Is_Apnea` | Device apnea flag, 1 or 0 |

## Cohort and protocol

Twelve adults, coded S01–S12, were recorded from 3 to 9 May 2026. Each person completed two sessions. Inside a session the order is resting breathing, voluntary hyperventilation, a return toward the resting rate, and a breath-hold of about 30 s. The breath-hold is the apnea interval.

`data/annotations/phases.csv` gives the apnea and hyperventilation windows for each session, in seconds from the first belt sample. `data/manifest.csv` links every sensor file to its belt recording and session start.

## Recordings

| Path | Contents |
| --- | --- |
| `data/rgb/` | Respiratory signal from Farnebäck, Horn–Schunck, or Lucas–Kanade. Semicolon-separated. |
| `data/tof/` | Respiratory signal from ToF depth. Semicolon-separated. |
| `data/belt/` | Belt resistance (`*_data.txt`), recorder header, and channel description. Comma-separated. |
| `data/reference/processed_data-*.csv` | Belt resistance, band-pass filtered resistance (Butterworth, order 3, 0.05–1.5 Hz), and reference rate in breaths per minute. The rate is 0 during the breath-hold. |
| `data/reference/legacy_synced/` | Camera and belt waveforms already aligned at 20 Hz, used to check the synchronization. |
| `data/annotations/phases.csv` | Apnea and hyperventilation windows. |
| `data/manifest.csv` | Which sensor file belongs to which session and belt. |

## Acquire a new recording

Connect the MaixSense, then start one app. RGB and ToF cannot stream together.

```powershell
python -m acquisition.rgb.app
python -m acquisition.tof.app
python -m acquisition.tof.grid
```

The RGB page listens on port 5000, the ToF page on 8080, and the depth-variance view on 8081. Override the defaults with `MAIX_HOST`, `MAIX_PORT`, `RGB_PORT`, `TOF_PORT`, and `TOF_GRID_PORT`. Recordings are written to `acquisition/rgb/recordings` and `acquisition/tof/recordings`. Start the Neurobit belt for the same session.

Redraw the latest session:

```powershell
python -m acquisition.rgb.plot_results
python -m acquisition.tof.data_plotter
```

## Reproduce the comparison

Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
python -m analysis.run
```

The analysis places the camera signal and the belt on one 20 Hz grid, z-scores both waveforms, and aligns them by cross-correlation. It then reports:

- Spearman correlation of the waveforms
- mean absolute error of respiratory rate, one sample per second over the whole recording
- Bland–Altman bias and limits of agreement, using samples whose reference rate is above zero
- apnea detection at 1 Hz from `Is_Apnea`

`python -m analysis.run` writes `results/report.md`, `results/summary.csv`, `results/per_recording.csv`, and the figures in `results/figures`. It opens the figure windows. Close them to let the process finish.

On these recordings the correlation medians are ToF 0.634, Farnebäck 0.630, Horn–Schunck 0.634, and Lucas–Kanade 0.644. The rate-error medians are ToF 6.24, Farnebäck 4.24, Horn–Schunck 4.26, and Lucas–Kanade 3.83 breaths per minute.

## Layout

```text
acquisition/    RGB and ToF apps for a live MaixSense
analysis/       alignment, metrics, and figures
data/           respiratory signals, belt, and phase windows
docs/           manuscript PDF
results/        tables and figures from the analysis
tests/          offline checks
```

## Cite

If you use the platform or these recordings, cite the paper:

Jasiówka, D., Molina Arias, L., Ciężarek, P., & Żyźnawska, J. (2026). A real-time low-cost platform for non-contact respiratory rate monitoring. *Computers in Biology and Medicine*.

Cite the recordings as:

Jasiówka, D., Molina Arias, L., Ciężarek, P., & Żyźnawska, J. (2026). *Contactless respiratory rate from RGB optical flow and ToF depth* [Data set]. AGH University of Krakow; Jagiellonian University Medical College.

Contact: Ludwin Molina Arias, AGH University of Krakow, arias@agh.edu.pl.

## License

The acquisition and analysis code is released under the MIT License. See [LICENSE](LICENSE).

The recordings in `data/` and the manuscript in `docs/` are released under Creative Commons Attribution 4.0 International. See [LICENSE-DATA](LICENSE-DATA).
