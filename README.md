# A Real-Time Low-Cost Platform for Non-Contact Respiratory Rate Monitoring

This repository contains the platform developed for non-contact respiratory rate monitoring using RGB and time-of-flight (ToF) cameras. Respiratory signals are extracted from chest motion and processed on the device to estimate respiratory rate in real time.

The repository also contains the analysis used to validate the platform against a piezoresistive chest belt.

## Overview

The platform uses a MaixSense A075V, which integrates an RGB camera and a ToF depth sensor.

Two independent approaches are implemented:

- **RGB:** chest motion is estimated using optical flow within a chest region identified by pose estimation. Farnebäck, Horn–Schunck, and Lucas–Kanade optical-flow methods are available.
- **ToF:** chest motion is estimated from changes in the mean depth of a tracked chest region.

A Neurobit Optima+ 4 BLE piezoresistive chest belt was used as the reference signal during validation.

## Study data

The platform was evaluated in 12 participants (S01–S12). Each participant completed separate RGB and ToF sessions including:

- resting breathing,
- voluntary hyperventilation,
- recovery toward the resting respiratory rate, and
- a breath-hold of approximately 30 s.

The study data and simultaneously recorded reference belt signals are available in `data/`.

The main data files are:

| Path | Contents |
| --- | --- |
| `data/RR_Belt_Sxx_Rgb_YYYYMMDD.csv` | Reference belt signal for the RGB session |
| `data/RR_Belt_Sxx_Tof_YYYYMMDD.csv` | Reference belt signal for the ToF session |
| `data/RR_Rgb_*_Sxx_YYYYMMDD.csv` | Respiratory signals obtained with the RGB optical-flow methods |
| `data/RR_Tof_Sxx_YYYYMMDD.csv` | Respiratory signal obtained from ToF depth |
| `data/01_participants.csv` | Participant codes and session information |
| `data/02_recordings.csv` | Recording information and belt settings |
| `data/03_protocol-phases.csv` | Hyperventilation and breath-hold intervals |
| `data/04_file-list.csv` | Complete list of dataset files |

See `data/00_readme.txt` for a detailed description of the dataset and variables.

### Dataset repository

The study dataset is available through RODBUK:

**RODBUK:** <!-- add the dataset URL here -->

## Acquire a new recording

Connect the MaixSense A075V and run the application corresponding to the desired sensing modality:

```powershell
python -m acquisition.rgb.app
python -m acquisition.tof.app
```

RGB and ToF recordings are acquired separately because the sensor streams one modality at a time.

For RGB acquisition, the pose-estimation model should be available at:

```text
acquisition/rgb/lightweight_pose_estimation_201912.onnx
```

Alternatively, its location can be specified using `POSE_ONNX`.

The acquisition applications estimate the respiratory signal and respiratory rate in real time and store the recorded signals for subsequent analysis.

## Reproduce the analysis

Python 3.10 or newer is required.

Place the study dataset in `data/`, then create the environment and run the analysis:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
python -m analysis.run
```

The analysis synchronizes the camera-derived respiratory signals with the reference chest-belt signal and evaluates the performance of each method using:

- Spearman correlation between respiratory signals,
- mean absolute error (MAE) of respiratory rate,
- Bland–Altman bias and limits of agreement, and
- breath-hold detection performance.

Generated tables and figures are saved in `results/`.

## Repository structure

```text
acquisition/    RGB and ToF acquisition applications
analysis/       Signal processing, synchronization, metrics, and figures
data/           Study dataset
docs/           Manuscript
results/        Analysis results and figures
tests/          Offline tests
```

## Citation

If you use the platform or code, please cite:

Jasiówka, D., Molina Arias, L., Ciężarek, P., & Żyźnawska, J. (2026). *A real-time low-cost platform for non-contact respiratory rate monitoring*. Computers in Biology and Medicine.

If you use the study data, please also cite:

Jasiówka, D., Molina Arias, L., Ciężarek, P., & Żyźnawska, J. (2026). *Contactless respiratory rate from RGB optical flow and ToF depth* [Data set]. AGH University of Krakow; Jagiellonian University Medical College.

**Contact:** Ludwin Molina Arias, AGH University of Krakow  
arias@agh.edu.pl

## License

The acquisition and analysis code is released under the MIT License. See [LICENSE](LICENSE).

The dataset in `data/` and the manuscript in `docs/` are released under the Creative Commons Attribution 4.0 International License. See [LICENSE-DATA](LICENSE-DATA).
