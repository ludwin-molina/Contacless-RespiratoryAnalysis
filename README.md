# A Real-Time Low-Cost Platform for Non-Contact Respiratory Rate Monitoring

This repository contains the platform developed for non-contact respiratory rate monitoring using RGB and time-of-flight (ToF) cameras. Respiratory signals are extracted from chest motion and processed on the device to estimate respiratory rate in real time.

The repository also contains the data and analysis used to validate the platform against a piezoresistive chest belt.

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

## Acquisition

Connect the MaixSense A075V and run the application corresponding to the desired modality:

```powershell
python -m acquisition.rgb.app
python -m acquisition.tof.app
