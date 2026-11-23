Dataset title
Contactless respiratory rate from RGB optical flow and ToF depth, with a piezoresistive chest belt

Description
Respiratory signals from 12 adult volunteers, stored as comma-separated tables and coded S01-S12. The package does not contain video or raw image frames. Each person completed two separate sessions with a MaixSense A075V. In the RGB session the device stored the chest signal from Farnebäck, Horn–Schunck, and Lucas–Kanade optical flow. In the time-of-flight session it stored the respiratory signal derived from depth. A Neurobit Optima+ 4 BLE resistance belt was recorded at the same time and is the reference. The protocol is resting breathing, voluntary hyperventilation, and a breath-hold used to simulate apnea. Column definitions and the protocol windows are in 00_readme.txt.

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
2026-05-03 to 2026-05-09. Session start times are in 02_recordings.csv.

License
RODBUK assigns CC0 when a dataset is saved. Change that license in the form if another licence should apply. The licence proposed for this dataset is Creative Commons Attribution 4.0 International (CC BY 4.0).

Participant codes
Signals use codes S01 to S12 in acquisition order: S01, S02, S03, S04, S05, S06, S07, S08, S09, S10, S11, S12.
The package does not contain names. 01_participants.csv lists the same codes with the session date and time only.

Protocol
Each recording starts with resting breathing. The volunteer then hyperventilates, returns toward the resting rate, holds the breath for about 30 s, and breathes again. The breath-hold is the apnea interval. Recovery after the hold is not a protocol phase.

The belt annotation files exported by the recorder contain only the header Onset,Duration,Annotation. The clicks used in the manuscript were not stored. Intervals in 03_protocol-phases.csv were segmented from the reference respiratory rate, on the belt clock (seconds from the first belt sample):

- apnea is the longest stretch at 0 breaths per minute that lasts at least 10 s
- hyperventilation is the tallest smoothed bout before that stretch
- the bout must last at least 8 s and rise at least 4 breaths per minute above the resting level
- a mild early rise is not labeled as hyperventilation
- resting breathing is the time before hyperventilation and the gap between hyperventilation and apnea

The RGB session of S08 has no bout that meets the hyperventilation rule. Those two cells are empty. The breath-hold for that session is still labeled. Status in the table is segmented. These windows are not the manual clicks of the manuscript.

One ToF session, S04, has a device respiratory rate that stays near 54 breaths per minute after the opening seconds, while the belt stays near the resting rate. Its whole-recording error is about 37 breaths per minute. That session is part of the cohort behind Table 4. The article figure draws the axis only up to about 12 breaths per minute, so the point lies above the frame. The first 20 seconds of that recording agree with the belt. The same check on the other ToF sessions shows that the opening and closing 20 seconds are not systematically worse than the middle of the recording.

Files in this package
The RODBUK form accepts files. Upload every file in this folder. Do not upload the folder as a directory. The list sorts alphabetically, so the documentation comes first.

00_readme.txt
    This file.
01_participants.csv
    Codes S01-S12 and the start of each session. No names.
02_recordings.csv
    Belt settings and sample timing.
03_protocol-phases.csv
    Segmented apnea and hyperventilation windows on the belt clock.
04_file-list.csv
    One row per file, with the description and the tag to type into the Files tab.

Signal files use these patterns:
    RR_Belt_Sxx_Rgb_YYYYMMDD.csv
    RR_Belt_Sxx_Tof_YYYYMMDD.csv
    RR_Rgb_<method>_Sxx_YYYYMMDD.csv
    RR_Tof_Sxx_YYYYMMDD.csv
Examples:
    RR_Belt_S05_Rgb_20260505.csv
    RR_Rgb_Farneback_S05_20260505.csv
    RR_Rgb_Horn-Schunck_S05_20260505.csv
    RR_Rgb_Lucas-Kanade_S05_20260505.csv
    RR_Tof_S05_20260505.csv
YYYYMMDD is the date in the sensor file name. Sxx is the participant code. Rgb and Tof in a belt name are the two separate sessions. Names use digits, letters, hyphens, and underscores.

How to fill the RODBUK form
Keep the metadata language set to English. Changing the language clears the form.

Title
Contactless respiratory rate from RGB optical flow and ToF depth, with a piezoresistive chest belt

Authors
Dominik Jasiówka. AGH University of Krakow.
Ludwin Molina Arias. AGH University of Krakow. Contact: arias@agh.edu.pl.
Patryk Ciężarek. Jagiellonian University Medical College.
Joanna Żyźnawska. Jagiellonian University Medical College.
Add each ORCID in the form where you have it. This package does not record ORCID identifiers.

Summary of the project
This project evaluates a low-cost, contactless platform for respiratory-rate monitoring. Twelve adult volunteers each completed two sessions with a MaixSense A075V, because the RGB camera and the time-of-flight depth sensor cannot stream together. The deposited files are the respiratory signals already computed on the device, not the camera video. For RGB, those signals come from Farnebäck, Horn–Schunck, and Lucas–Kanade optical flow. For time-of-flight, the signal is the depth-derived chest motion. A Neurobit Optima+ 4 BLE piezoresistive chest belt was recorded at the same time as the reference. The protocol included resting breathing, voluntary hyperventilation, and breath-hold periods used to simulate apnea, so respiratory-rate estimation and apnea detection can be compared with the belt for each modality. Data were collected from 3 to 9 May 2026 by the Department of Biocybernetics and Biomedical Engineering at AGH University of Krakow and the Institute of Physiotherapy at Jagiellonian University Medical College.

Description
Copy the Description and Research objective sections above.

Subject
Medicine, Health and Life Sciences

Keywords
respiratory rate; contactless monitoring; optical flow; time-of-flight; apnea; hyperventilation; chest belt

Related publication
A Real-Time Low-Cost Platform for Non-Contact Respiratory Rate Monitoring. Computers in Biology and Medicine. Add the DOI in the form when it exists.

Time period covered
2026-05-03 to 2026-05-09

Files tab
Select all files from this folder. For each file, copy description and rodbuk_tag from 04_file-list.csv. Use Documentation for the readme and the four tables. Use Data for every RR_ file. None of the files exceeds 4 GB. Save the dataset. It remains a draft, with an inactive DOI, until a data steward publishes it.

Formats
All tables are UTF-8 comma-separated values with a header row. Open them in any spreadsheet or with pandas.read_csv. No EDF files are included: the Neurobit text export is the open copy of the same resistance samples. Empty annotation files are not included.

Belt columns
Time_s: seconds from the first sample
Resistance_kOhm: chest-belt resistance in kilohms
Filtered: band-pass of the resistance, Butterworth order 3, 0.05 to 1.5 Hz
RespiratoryRate_bpm: reference rate in breaths per minute. During the breath-hold this column is 0.
Sample period 0.064 s (16 samples in each 1.024 s record, 15.625 Hz). Hardware filters are in 02_recordings.csv (high-pass 0 Hz, low-pass 5.78125 Hz). The raw resistance and the filtered reference share this clock, so they are stored in the same file.

RGB optical-flow columns
TimeUnix_s: device clock, seconds since 1970-01-01
Elapsed_s: seconds from the start of the sensor file
OpticalFlow: chest motion from the named optical-flow method, arbitrary units
OpticalFlow_sd: spread of that motion inside the frame
RespiratoryRate_bpm: rate estimated on the device
IsApnea: device flag, 1 or 0. This is not the protocol breath-hold. Use 03_protocol-phases.csv for the protocol interval.

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
2026-11-23: raw resistance and the filtered reference are stored in one belt file per session.

Recommended software
A text editor or spreadsheet for the CSV files. Python with pandas is enough to reload them. No proprietary viewer is required.
