"""Run the offline cohort analysis and write the paper's tables.

Usage from the repository root::

    python -m analysis.run
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.config import (
    PUBLISHED_APNEA_F1,
    PUBLISHED_APNEA_SENSITIVITY,
    PUBLISHED_APNEA_SPECIFICITY,
    PUBLISHED_MEDIAN_MAE,
    RESULTS,
    RGB_SIGNAL_COLUMN,
    RGB_SIGNAL_SIGN,
    ROOT,
    TOF_SIGNAL_COLUMN,
    TOF_SIGNAL_SIGN,
    load_manifest,
)
from analysis.metrics import bland_altman_points, metrics_for_recording, summarize_methods
from analysis.phases import ensure_phases
from analysis.plots import (
    save_bland_altman,
    save_distribution,
    save_phase_mae,
    save_signal_comparison,
    show_figures,
)
from analysis.synchronize import belt_time_to_relative, synchronize


def _signal_setup(modality):
    if modality == "rgb":
        return RGB_SIGNAL_COLUMN, RGB_SIGNAL_SIGN
    if modality == "tof":
        return TOF_SIGNAL_COLUMN, TOF_SIGNAL_SIGN
    raise ValueError(f"Unknown modality: {modality}")


def _fmt(value, digits=3):
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def write_report(summary, phases, path: Path):
    missing_hyper = phases[phases["hyperventilation_start_s"].isna()]
    missing_text = "Every recording has a segmented hyperventilation window."
    if len(missing_hyper):
        names = ", ".join(f"{row.subject} ({row.modality})" for row in missing_hyper.itertuples(index=False))
        missing_text = (
            "No hyperventilation bout met the rule in: "
            f"{names}. Those cells are empty because the reference rate does not rise clearly."
        )
    lines = [
        "# Comparison with the article",
        "",
        "Respiratory-rate error uses one sample per second across the whole recording, including the breath-hold.",
        "Bland–Altman bias drops samples whose reference rate is zero.",
        "Differences from the published numbers are reported as computed. Thresholds are not adjusted to force a match.",
        "",
        "Phase windows in `Data/03_protocol-phases.csv` are segmented from the reference rate.",
        "The belt annotation files contain only the header `Onset,Duration,Annotation`, so the original clicks were not available.",
        "Apnea is the longest stretch at 0 breaths per minute (at least 10 s).",
        "Hyperventilation is the tallest smoothed bout before that stretch: at least 8 s, and at least 4 breaths per minute above the resting level.",
        "A mild early rise is not taken as the protocol bout. The gap between hyperventilation and the breath-hold is resting breathing.",
        "These windows are not the manual clicks used in the article, so the phase MAE is not expected to match Table 5.",
        missing_text,
        "",
        "Figure 7 in the article draws the MAE axis only up to about 12 BPM.",
        "The point near 37 BPM is the ToF session S04. After the first 20 s the device rate stays near 54 BPM while the belt stays near the resting rate.",
        "That session is inside Table 4: the published mean and SD, 8.80 ± 9.41, are the values obtained with it.",
        "The first and last 20 s of the ToF recordings are not the source of this point, and they are not worse than the middle of the recording. The MAE stays the whole-recording value used in the article.",
        "",
        "## Respiratory-rate MAE, whole recording at 1 Hz",
        "",
        "| Method | Median | Published median | Difference | Mean ± SD | Published mean ± SD |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    published_spread = {
        "ToF": "8.80 ± 9.41",
        "Farneback": "4.39 ± 1.43",
        "Horn-Schunck": "4.79 ± 1.65",
        "Lucas-Kanade": "4.29 ± 1.47",
    }
    for row in summary.itertuples(index=False):
        method = str(row.method)
        published = PUBLISHED_MEDIAN_MAE.get(method, np.nan)
        delta = row.mae_median - published
        lines.append(
            f"| {method} | {_fmt(row.mae_median, 2)} | {_fmt(published, 2)} | {_fmt(delta, 2)} | {_fmt(row.mae_mean, 2)} ± {_fmt(row.mae_sd, 2)} | {published_spread.get(method, '—')} |"
        )

    published_bias = {
        "ToF": (3.38, -27.35, 34.10),
        "Farneback": (1.71, -12.55, 15.98),
        "Horn-Schunck": (2.11, -12.79, 17.00),
        "Lucas-Kanade": (1.60, -12.69, 15.90),
    }
    lines.extend(
        [
            "",
            "## Bland–Altman bias at 1 Hz, breath-hold removed",
            "",
            "| Method | Bias | Published bias | Limits of agreement | Published limits |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary.itertuples(index=False):
        published = published_bias.get(str(row.method))
        if published is None:
            published_bias_text = "—"
            published_limits = "—"
        else:
            published_bias_text = f"{published[0]:+.2f}"
            published_limits = f"{published[1]:+.2f} to {published[2]:+.2f}"
        lines.append(
            f"| {row.method} | {_fmt(row.bland_bias, 2)} | {published_bias_text} | {_fmt(row.bland_loa_low, 2)} to {_fmt(row.bland_loa_high, 2)} | {published_limits} |"
        )

    lines.extend(
        [
            "",
            "## Apnea at 1 Hz",
            "",
            "| Method | Sensitivity | Published | Specificity | Published | F1 | Published |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary.itertuples(index=False):
        method = str(row.method)
        lines.append(
            "| {method} | {sens} | {sens_pub} | {spec} | {spec_pub} | {f1} | {f1_pub} |".format(
                method=method,
                sens=_fmt(row.apnea_sensitivity),
                sens_pub=_fmt(PUBLISHED_APNEA_SENSITIVITY.get(method, np.nan)),
                spec=_fmt(row.apnea_specificity),
                spec_pub=_fmt(PUBLISHED_APNEA_SPECIFICITY.get(method, np.nan)),
                f1=_fmt(row.apnea_f1),
                f1_pub=_fmt(PUBLISHED_APNEA_F1.get(method, np.nan)),
            )
        )

    lines.extend(
        [
            "",
            "## Waveform Spearman correlation",
            "",
            "| Method | Median | Q1 | Q3 | Mean ± SD |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.method} | {_fmt(row.spearman_median)} | {_fmt(row.spearman_q1)} | {_fmt(row.spearman_q3)} | {_fmt(row.spearman_mean)} ± {_fmt(row.spearman_sd)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(manifest=None, show_plots=False):
    RESULTS.mkdir(parents=True, exist_ok=True)
    synced_dir = RESULTS / "synced"
    synced_dir.mkdir(parents=True, exist_ok=True)
    if manifest is None:
        manifest = load_manifest()
    phases = ensure_phases(manifest)

    rows = []
    clouds = []
    phase_windows = {}
    phase_columns = (
        "apnea_start_s",
        "apnea_stop_s",
        "hyperventilation_start_s",
        "hyperventilation_stop_s",
    )
    for record in manifest.itertuples(index=False):
        column, sign = _signal_setup(record.modality)
        aligned, meta = synchronize(ROOT / record.sensor_file, ROOT / record.reference_file, column, sign)
        out_name = f"synced_{record.subject}_{record.method}.csv"
        aligned.to_csv(synced_dir / out_name, sep=";", index=False)
        phase_row = phases[(phases["subject"] == record.subject) & (phases["modality"] == record.modality)].iloc[0]
        metrics = metrics_for_recording(aligned, meta, phase_row)
        cloud = bland_altman_points(aligned)
        cloud.insert(0, "method", record.method)
        clouds.append(cloud)
        if record.method == "Lucas-Kanade":
            phase_windows[record.subject] = {
                column: belt_time_to_relative(phase_row[column], meta) for column in phase_columns
            }
        rows.append(
            {
                "subject": record.subject,
                "modality": record.modality,
                "method": record.method,
                "file": out_name,
                **metrics,
            }
        )
        print(f"{record.subject:12} {record.method:16} spearman={metrics['spearman']:.3f} mae={metrics['mae']:.2f}")

    per_recording = pd.DataFrame(rows)
    per_recording.to_csv(RESULTS / "per_recording.csv", index=False)
    summary = summarize_methods(per_recording)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    summary.to_csv(RESULTS / "table_spearman_mae_apnea.csv", index=False)

    figure_dir = RESULTS / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_distribution(
        per_recording,
        "spearman",
        "Spearman correlation",
        "Waveform agreement with the respiratory belt",
        figure_dir / "spearman.png",
        show=show_plots,
    )
    save_distribution(
        per_recording,
        "mae",
        "Mean absolute error [BPM]",
        "Respiratory-rate error over the whole recording",
        figure_dir / "mae.png",
        show=show_plots,
    )
    save_bland_altman(pd.concat(clouds, ignore_index=True), summary, figure_dir / "bland_altman.png", show=show_plots)
    save_phase_mae(per_recording, figure_dir / "phase_mae.png", show=show_plots)
    save_signal_comparison(
        synced_dir,
        per_recording,
        figure_dir / "signal_comparison.png",
        phase_windows=phase_windows,
        show=show_plots,
    )
    write_report(summary, phases, RESULTS / "report.md")
    print(f"Results written to {RESULTS}")
    if show_plots:
        print("Close the figure windows to finish.")
        show_figures()
    return per_recording, summary


def main():
    run(show_plots=True)


if __name__ == "__main__":
    main()
