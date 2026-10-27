Dataset title
Contactless respiratory rate from RGB optical flow and ToF depth, with a piezoresistive chest belt

Description
Simultaneous belt and camera recordings from 12 adult volunteers. Each person completed two separate sessions with a MaixSense A075V: one with the RGB camera (three optical-flow methods) and one with the time-of-flight depth sensor. The reference in every session is a Neurobit Optima+ 4 BLE resistance belt on the chest.

Research objective
Compare contactless respiratory-rate estimates with the belt during resting breathing, voluntary hyperventilation, and a breath-hold used to simulate apnea.

Related manuscript
A Real-Time Low-Cost Platform for Non-Contact Respiratory Rate Monitoring.
Authors: Dominik Jasiówka, Ludwin Molina Arias, Patryk Ciężarek, Joanna Żyźnawska.
Affiliations: AGH University of Krakow; Jagiellonian University Medical College.
The manuscript PDF is not part of this package.

Contact
Ludwin Molina Arias, AGH University of Krakow, arias@agh.edu.pl

Timeframe
2026-05-03 to 2026-05-09. Session start times are in 01_metadata/recordings.csv.

License
Proposed for deposit: Creative Commons Attribution 4.0 International (CC BY 4.0). Confirm the license in the RODBUK form before publishing.

Participant codes
Signals use codes S01 to S12 in acquisition order: S01, S02, S03, S04, S05, S06, S07, S08, S09, S10, S11, S12.
The package does not contain names. 01_metadata/participants.csv lists the same codes with the session date and time only.

Protocol
Each recording starts with resting breathing. The volunteer then hyperventilates, returns toward the resting rate, holds the breath for about 30 s, and breathes again. The breath-hold is the apnea interval. Recovery after the hold is not a protocol phase.

The belt annotation files exported by the recorder contain only the header Onset,Duration,Annotation. The clicks used in the manuscript were not stored. Intervals in 01_metadata/protocol_phases.csv were segmented from the reference respiratory rate, on the belt clock (seconds from the first belt sample):

- apnea is the longest stretch at 0 breaths per minute that lasts at least 10 s
- hyperventilation is the tallest smoothed bout before that stretch
- the bout must last at least 8 s and rise at least 4 breaths per minute above the resting level
- a mild early rise is not labeled as hyperventilation
- resting breathing is the time before hyperventilation and the gap between hyperventilation and apnea

The RGB session of S08 has no bout that meets the hyperventilation rule. Those two cells are empty. The breath-hold for that session is still labeled. Status in the table is segmented. These windows are not the manual clicks of the manuscript.

One ToF session, S04, has a device respiratory rate that stays near 54 breaths per minute after the opening seconds, while the belt stays near the resting rate. Its whole-recording error is about 37 breaths per minute. That session is part of the cohort behind Table 4. The article figure draws the axis only up to about 12 breaths per minute, so the point lies above the frame. The first 20 seconds of that recording agree with the belt. The same check on the other ToF sessions shows that the opening and closing 20 seconds are not systematically worse than the middle of the recording.

Folder structure
00_readme.txt
    This file. It describes the whole dataset.
01_metadata/
    participants.csv, recordings.csv, protocol_phases.csv, files.csv
02_reference-belt/
    Raw belt resistance and the processed reference rate. One file per session.
03_rgb-optical-flow/
    Farneback, Horn-Schunck, and Lucas-Kanade signals from the RGB session.
04_tof-depth/
    Depth signal from the separate ToF session.

File names
RR_<source>_<method>_Sxx_YYYYMMDD.csv
Examples:
    02_reference-belt/RR_BeltRaw_S05_Rgb_20260505.csv
    02_reference-belt/RR_BeltProcessed_S05_Rgb_20260505.csv
    03_rgb-optical-flow/RR_Rgb_Farneback_S05_20260505.csv
    03_rgb-optical-flow/RR_Rgb_Horn-Schunck_S05_20260505.csv
    03_rgb-optical-flow/RR_Rgb_Lucas-Kanade_S05_20260505.csv
    04_tof-depth/RR_Tof_S05_20260505.csv
YYYYMMDD is the date in the sensor file name. Sxx is the participant code. Names use digits, letters, hyphens, and underscores.

Formats
All tables are UTF-8 comma-separated values with a header row. Open them in any spreadsheet or with pandas.read_csv. No EDF files are included: the Neurobit text export is the open copy of the same resistance samples. Empty annotation files are not included.

Belt raw columns
Time_s: seconds from the first sample
Resistance_kOhm: chest-belt resistance in kilohms
Sample period 0.064 s (16 samples in each 1.024 s record, 15.625 Hz). Hardware filters are in recordings.csv (high-pass 0 Hz, low-pass 5.78125 Hz).

Belt processed columns
Time_s: same clock as the raw file
Resistance_kOhm: same resistance samples
Filtered: band-pass of the resistance, Butterworth order 3, 0.05 to 1.5 Hz
RespiratoryRate_bpm: reference rate in breaths per minute. During the breath-hold this column is 0.

RGB optical-flow columns
TimeUnix_s: device clock, seconds since 1970-01-01
Elapsed_s: seconds from the start of the sensor file
OpticalFlow: chest motion from the named optical-flow method, arbitrary units
OpticalFlow_sd: spread of that motion inside the frame
RespiratoryRate_bpm: rate estimated on the device
IsApnea: device flag, 1 or 0. This is not the protocol breath-hold. Use protocol_phases.csv for the protocol interval.

ToF columns
TimeUnix_s, Elapsed_s: same meaning as the RGB files
Distance_mm: distance from the depth sensor, millimetres
DistanceFiltered: filtered distance used for the rate
RespiratoryRate_bpm, IsApnea: device estimates, as in the RGB files

The three RGB files of one participant share one session. The ToF file is the other session and has its own belt file and its own phase windows. Do not treat RGB and ToF rows with the same code as one simultaneous recording.

Quality control
Pairing of each sensor file with its belt uses the belt start time. Phase windows were checked against the reference rate: the labeled hyperventilation bout sits above the resting rate, and the labeled breath-hold has rate 0. The S08 RGB session is the exception described above.

Change log
2026-10-27: package prepared for RODBUK. Phase windows segmented from the reference rate because the annotation files were empty.

Recommended software
A text editor or spreadsheet for the CSV files. Python with pandas is enough to reload them. No proprietary viewer is required.
