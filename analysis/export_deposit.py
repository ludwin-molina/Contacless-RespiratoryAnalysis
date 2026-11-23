"""Build the RODBUK package in ``Data/`` from the local ``recordings/`` archive.

The web form stores files, not folders. Names sort the documentation ahead of
the signals. Tables are comma-separated, dates are ISO, and participant codes
are S01-S12. Each session has one belt file with the raw resistance and the
filtered reference. Vendor EDF files and empty annotation files are left out.
"""

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.config import DATA, archive_root

PACKAGE = DATA
SENSOR_STAMP = re.compile(r"(\d{8})_(\d{6})")
PREFILTER = re.compile(r"HP:([0-9.]+)Hz LP:([0-9.]+)Hz")
FORBIDDEN = set(" !?*&#~@$%^&()`;?,'\"[]{}")


def participant_codes(manifest):
    codes = {}
    for subject in manifest["subject"]:
        if subject not in codes:
            codes[subject] = f"S{len(codes) + 1:02d}"
    return codes


def _table(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    header = [cell.strip() for cell in rows[0]]
    values = [cell.strip() for cell in rows[1]]
    return dict(zip(header, values))


def _stamp(sensor_file):
    match = SENSOR_STAMP.search(Path(sensor_file).name)
    if match is None:
        raise ValueError(f"Sensor file has no YYYYMMDD_HHMMSS stamp: {sensor_file}")
    return match.group(1), match.group(2)


def _write_exact(destination, header, body_lines):
    payload = header + "\n" + "\n".join(body_lines) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8", newline="\n")
    relative = destination.relative_to(PACKAGE).as_posix()
    if any(character in FORBIDDEN or ord(character) > 127 for character in relative):
        raise ValueError(f"Deposit path is not RODBUK-safe: {relative}")
    return relative


def _archive_path(archive: Path, relative) -> Path:
    text = Path(str(relative)).as_posix()
    if text.startswith("data/"):
        text = text[len("data/") :]
    return archive / text


def _load_archive(archive: Path):
    manifest_path = archive / "manifest.csv"
    phases_path = archive / "annotations" / "phases.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "The local recordings archive is not present. "
            f"Expected {manifest_path} so Data/ can be rebuilt."
        )
    if not phases_path.is_file():
        raise FileNotFoundError(f"The archive has no phase table: {phases_path}")
    manifest = pd.read_csv(manifest_path)
    phases = pd.read_csv(phases_path)
    return manifest, phases


def _body(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line for line in lines[1:] if line.strip()]


def _sensor_period(path):
    elapsed = pd.read_csv(path, sep=";", usecols=["Elapsed_Sec"])["Elapsed_Sec"].to_numpy(dtype=float)
    return int(len(elapsed)), float(np.median(np.diff(elapsed)))


def _readme(codes):
    order = ", ".join(f"{code}" for code in codes.values())
    return f"""Dataset title
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
Signals use codes S01 to S12 in acquisition order: {order}.
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
"""


def _describe(role, code, modality, method, date):
    iso = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    session = "RGB" if modality == "rgb" else "ToF"
    if role == "belt":
        text = (
            f"Chest-belt resistance, filtered signal, and reference respiratory rate "
            f"for participant {code}, {session} session, {iso}."
        )
    elif modality == "rgb":
        text = f"{method} optical-flow respiratory signal for participant {code}, RGB session, {iso}."
    else:
        text = f"ToF depth respiratory signal for participant {code}, {iso}."
    return text


def _reset(directory: Path):
    if directory.exists():
        for path in sorted(directory.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    directory.mkdir(parents=True, exist_ok=True)


def export_deposit():
    archive = archive_root()
    manifest, phases = _load_archive(archive)
    codes = participant_codes(manifest)
    _reset(PACKAGE)

    file_rows = []
    recording_rows = []
    recording_index = {}
    participant_rows = []
    seen_sessions = set()

    for subject, code in codes.items():
        subject_rows = manifest[manifest["subject"] == subject]
        row = {"participant_id": code}
        for modality in ("rgb", "tof"):
            session = subject_rows[subject_rows["modality"] == modality].iloc[0]
            date, clock = _stamp(session.sensor_file)
            row[f"{modality}_session_start"] = session.belt_start
            row[f"{modality}_sensor_date"] = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
            row[f"{modality}_sensor_time"] = f"{clock[:2]}:{clock[2:4]}:{clock[4:6]}"
        participant_rows.append(row)

    for record in manifest.itertuples(index=False):
        code = codes[record.subject]
        date, _clock = _stamp(record.sensor_file)
        modality = record.modality
        session_key = (record.subject, modality)
        if session_key not in seen_sessions:
            seen_sessions.add(session_key)
            belt_dir = archive / "belt" / modality
            header = _table(belt_dir / f"record-{record.subject}-{modality}_header.txt")
            signals = _table(belt_dir / f"record-{record.subject}-{modality}_signals.txt")
            prefilter = PREFILTER.search(signals["PreFilter"])
            if prefilter is None:
                raise ValueError(f"Unparsed belt filter for {record.subject} {modality}")
            sample_period = float(header["Duration"]) / float(signals["Smp/Rec"])
            raw_lines = _body(belt_dir / f"record-{record.subject}-{modality}_data.txt")
            processed_lines = _body(_archive_path(archive, record.reference_file))
            if len(raw_lines) != len(processed_lines):
                raise ValueError(
                    f"Belt raw and processed lengths differ for {record.subject} {modality}: "
                    f"{len(raw_lines)} vs {len(processed_lines)}"
                )
            belt_name = f"RR_Belt_{code}_{modality.capitalize()}_{date}.csv"
            belt_columns = "Time_s,Resistance_kOhm,Filtered,RespiratoryRate_bpm"
            belt_rel = _write_exact(PACKAGE / belt_name, belt_columns, processed_lines)
            file_rows.append(
                {
                    "file_name": belt_rel,
                    "participant_id": code,
                    "modality": modality,
                    "method": "",
                    "role": "belt",
                    "rows": len(processed_lines),
                    "columns": belt_columns,
                    "description": _describe("belt", code, modality, "", date),
                    "rodbuk_tag": "Data",
                }
            )
            recording_rows.append(
                {
                    "participant_id": code,
                    "modality": modality,
                    "session_start": record.belt_start,
                    "sensor_date": date,
                    "belt_n_records": int(header["NumRec"]),
                    "belt_record_duration_s": float(header["Duration"]),
                    "belt_samples_per_record": int(signals["Smp/Rec"]),
                    "belt_sample_period_s": sample_period,
                    "belt_unit": signals["Units"],
                    "belt_highpass_hz": float(prefilter.group(1)),
                    "belt_lowpass_hz": float(prefilter.group(2)),
                    "belt_transducer": signals["Transducer"],
                    "recorder": "Neurobit Optima+ 4 BLE",
                }
            )
            recording_index[session_key] = len(recording_rows) - 1

        sensor_path = _archive_path(archive, record.sensor_file)
        n_samples, period = _sensor_period(sensor_path)
        sensor_lines = _body(sensor_path)
        sensor_lines = [line.replace(";", ",") for line in sensor_lines]
        if len(sensor_lines) != n_samples:
            raise ValueError(f"Row count mismatch in {record.sensor_file}")
        if modality == "rgb":
            method_token = record.method
            header_line = "TimeUnix_s,Elapsed_s,OpticalFlow,OpticalFlow_sd,RespiratoryRate_bpm,IsApnea"
            file_name = f"RR_Rgb_{method_token}_{code}_{date}.csv"
        else:
            method_token = "Tof"
            header_line = "TimeUnix_s,Elapsed_s,Distance_mm,DistanceFiltered,RespiratoryRate_bpm,IsApnea"
            file_name = f"RR_Tof_{code}_{date}.csv"
        relative = _write_exact(PACKAGE / file_name, header_line, sensor_lines)
        file_rows.append(
            {
                "file_name": relative,
                "participant_id": code,
                "modality": modality,
                "method": method_token,
                "role": "sensor",
                "rows": n_samples,
                "columns": header_line,
                "description": _describe("sensor", code, modality, method_token, date),
                "rodbuk_tag": "Data",
            }
        )
        session = recording_rows[recording_index[session_key]]
        session[f"sensor_n_samples_{method_token}"] = n_samples
        session[f"sensor_median_period_s_{method_token}"] = round(period, 6)

    phase_out = phases.copy()
    phase_out.insert(0, "participant_id", phase_out["subject"].map(codes))
    phase_out = phase_out.drop(columns=["subject"])
    phase_out.insert(2, "clock", "belt_seconds_from_first_sample")
    phase_out["note"] = ""
    missing = phase_out["hyperventilation_start_s"].isna()
    phase_out.loc[missing, "note"] = "no hyperventilation bout met the 4 BPM rule"

    participants = pd.DataFrame(participant_rows)
    recordings = pd.DataFrame(recording_rows)
    inventory_columns = "file_name,participant_id,modality,method,role,rows,columns,description,rodbuk_tag"
    documentation = [
        {
            "file_name": "00_readme.txt",
            "participant_id": "",
            "modality": "",
            "method": "",
            "role": "documentation",
            "rows": "",
            "columns": "",
            "description": "Dataset description, protocol, column definitions, and the text for the RODBUK metadata form.",
            "rodbuk_tag": "Documentation",
        },
        {
            "file_name": "01_participants.csv",
            "participant_id": "",
            "modality": "",
            "method": "",
            "role": "documentation",
            "rows": len(participants),
            "columns": ",".join(participants.columns),
            "description": "Anonymous participant codes S01-S12 and the start time of each session.",
            "rodbuk_tag": "Documentation",
        },
        {
            "file_name": "02_recordings.csv",
            "participant_id": "",
            "modality": "",
            "method": "",
            "role": "documentation",
            "rows": len(recordings),
            "columns": ",".join(recordings.columns),
            "description": "Belt recorder settings and sample timing for each session.",
            "rodbuk_tag": "Documentation",
        },
        {
            "file_name": "03_protocol-phases.csv",
            "participant_id": "",
            "modality": "",
            "method": "",
            "role": "documentation",
            "rows": len(phase_out),
            "columns": ",".join(phase_out.columns),
            "description": "Segmented apnea and hyperventilation windows on the belt clock.",
            "rodbuk_tag": "Documentation",
        },
        {
            "file_name": "04_file-list.csv",
            "participant_id": "",
            "modality": "",
            "method": "",
            "role": "documentation",
            "rows": "",
            "columns": inventory_columns,
            "description": "Inventory of every file, with the description and tag to enter in the RODBUK Files tab.",
            "rodbuk_tag": "Documentation",
        },
    ]
    inventory = documentation + sorted(file_rows, key=lambda row: row["file_name"])
    documentation[-1]["rows"] = len(inventory)

    participants.to_csv(PACKAGE / "01_participants.csv", index=False)
    recordings.to_csv(PACKAGE / "02_recordings.csv", index=False)
    phase_out.to_csv(PACKAGE / "03_protocol-phases.csv", index=False)
    pd.DataFrame(inventory).to_csv(PACKAGE / "04_file-list.csv", index=False)
    (PACKAGE / "00_readme.txt").write_text(_readme(codes), encoding="utf-8", newline="\n")
    print(f"Wrote {len(inventory)} files to {PACKAGE}")


if __name__ == "__main__":
    export_deposit()
