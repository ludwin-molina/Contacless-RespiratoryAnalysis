"""Build the RODBUK deposit package.

The package follows the RODBUK data-preparation guide: content-based folder
names, comma-separated tables, ISO dates, multi-digit participant codes, and
a 00_readme.txt. Vendor EDF files and empty annotation files are left out.
"""

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.config import PHASES_PATH, ROOT, load_manifest

DEPOSIT = ROOT / "deposit"
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
    relative = destination.relative_to(DEPOSIT).as_posix()
    if any(character in FORBIDDEN or ord(character) > 127 for character in relative):
        raise ValueError(f"Deposit path is not RODBUK-safe: {relative}")
    return relative


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
Signals use codes S01 to S12 in acquisition order: {order}.
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
"""


def _folder_readme(title, body):
    return f"{title}\n\n{body}\n\nSee ../00_readme.txt for the dataset description, columns, and protocol.\n"


def export_deposit():
    manifest = load_manifest()
    codes = participant_codes(manifest)
    phases = pd.read_csv(PHASES_PATH)
    if DEPOSIT.exists():
        for path in sorted(DEPOSIT.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
        for path in sorted(DEPOSIT.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
    DEPOSIT.mkdir(parents=True, exist_ok=True)

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
            belt_dir = ROOT / "data" / "belt" / modality
            header = _table(belt_dir / f"record-{record.subject}-{modality}_header.txt")
            signals = _table(belt_dir / f"record-{record.subject}-{modality}_signals.txt")
            prefilter = PREFILTER.search(signals["PreFilter"])
            if prefilter is None:
                raise ValueError(f"Unparsed belt filter for {record.subject} {modality}")
            sample_period = float(header["Duration"]) / float(signals["Smp/Rec"])
            raw_lines = _body(belt_dir / f"record-{record.subject}-{modality}_data.txt")
            raw_name = f"RR_BeltRaw_{code}_{modality.capitalize()}_{date}.csv"
            raw_rel = _write_exact(
                DEPOSIT / "02_reference-belt" / raw_name,
                "Time_s,Resistance_kOhm",
                raw_lines,
            )
            file_rows.append(
                {
                    "file_name": raw_rel,
                    "participant_id": code,
                    "modality": modality,
                    "method": "",
                    "role": "belt-raw",
                    "rows": len(raw_lines),
                    "columns": "Time_s,Resistance_kOhm",
                }
            )
            processed_lines = _body(ROOT / record.reference_file)
            processed_name = f"RR_BeltProcessed_{code}_{modality.capitalize()}_{date}.csv"
            processed_rel = _write_exact(
                DEPOSIT / "02_reference-belt" / processed_name,
                "Time_s,Resistance_kOhm,Filtered,RespiratoryRate_bpm",
                processed_lines,
            )
            file_rows.append(
                {
                    "file_name": processed_rel,
                    "participant_id": code,
                    "modality": modality,
                    "method": "",
                    "role": "belt-processed",
                    "rows": len(processed_lines),
                    "columns": "Time_s,Resistance_kOhm,Filtered,RespiratoryRate_bpm",
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

        n_samples, period = _sensor_period(ROOT / record.sensor_file)
        sensor_lines = _body(ROOT / record.sensor_file)
        sensor_lines = [line.replace(";", ",") for line in sensor_lines]
        if len(sensor_lines) != n_samples:
            raise ValueError(f"Row count mismatch in {record.sensor_file}")
        if modality == "rgb":
            folder = "03_rgb-optical-flow"
            method_token = record.method
            header_line = "TimeUnix_s,Elapsed_s,OpticalFlow,OpticalFlow_sd,RespiratoryRate_bpm,IsApnea"
            file_name = f"RR_Rgb_{method_token}_{code}_{date}.csv"
        else:
            folder = "04_tof-depth"
            method_token = "Tof"
            header_line = "TimeUnix_s,Elapsed_s,Distance_mm,DistanceFiltered,RespiratoryRate_bpm,IsApnea"
            file_name = f"RR_Tof_{code}_{date}.csv"
        relative = _write_exact(DEPOSIT / folder / file_name, header_line, sensor_lines)
        file_rows.append(
            {
                "file_name": relative,
                "participant_id": code,
                "modality": modality,
                "method": method_token,
                "role": "sensor",
                "rows": n_samples,
                "columns": header_line,
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

    metadata = DEPOSIT / "01_metadata"
    metadata.mkdir(parents=True)
    pd.DataFrame(participant_rows).to_csv(metadata / "participants.csv", index=False)
    pd.DataFrame(recording_rows).to_csv(metadata / "recordings.csv", index=False)
    phase_out.to_csv(metadata / "protocol_phases.csv", index=False)
    pd.DataFrame(file_rows).to_csv(metadata / "files.csv", index=False)

    (DEPOSIT / "00_readme.txt").write_text(_readme(codes), encoding="utf-8", newline="\n")
    (metadata / "00_readme.txt").write_text(
        _folder_readme(
            "Metadata",
            "participants.csv lists S01-S12 with session date and time. It has no names.\n"
            "recordings.csv holds session times and belt settings.\n"
            "protocol_phases.csv holds segmented apnea and hyperventilation windows on the belt clock.\n"
            "files.csv lists every data file in this package.",
        ),
        encoding="utf-8",
        newline="\n",
    )
    (DEPOSIT / "02_reference-belt" / "00_readme.txt").write_text(
        _folder_readme(
            "Reference belt",
            "RR_BeltRaw is the resistance export.\n"
            "RR_BeltProcessed adds the filtered resistance and the reference respiratory rate.\n"
            "Rgb and Tof in the file name are the two separate sessions of the same participant.",
        ),
        encoding="utf-8",
        newline="\n",
    )
    (DEPOSIT / "03_rgb-optical-flow" / "00_readme.txt").write_text(
        _folder_readme(
            "RGB optical flow",
            "Three files per participant, one per method: Farneback, Horn-Schunck, Lucas-Kanade.\n"
            "They share one RGB session and one belt file.",
        ),
        encoding="utf-8",
        newline="\n",
    )
    (DEPOSIT / "04_tof-depth" / "00_readme.txt").write_text(
        _folder_readme(
            "ToF depth",
            "One depth file per participant.\n"
            "This session is not simultaneous with the RGB session.",
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {len(file_rows)} data files to {DEPOSIT}")


if __name__ == "__main__":
    export_deposit()
