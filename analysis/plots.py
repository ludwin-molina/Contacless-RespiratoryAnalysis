"""Result figures matching the paper: distributions, traces, and Bland–Altman."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.config import METHOD_ORDER

METHOD_COLOR = {
    "ToF": "#E69F00",
    "Farneback": "#56B4E9",
    "Horn-Schunck": "#009E73",
    "Lucas-Kanade": "#CC79A7",
}
METHOD_LABEL = {
    "ToF": "ToF",
    "Farneback": "Farnebäck",
    "Horn-Schunck": "Horn–Schunck",
    "Lucas-Kanade": "Lucas–Kanade",
}
PHASE_COLOR = {
    "normal": "#b7e1c1",
    "hyperventilation": "#f6e3a1",
    "apnea": "#f4c1c1",
}
TRACE_COLOR = {
    "GT": "#222222",
    **METHOD_COLOR,
}


def _order(frame):
    return [method for method in METHOD_ORDER if method in set(frame["method"])]


def _apply_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update(
        {
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.frameon": True,
            "legend.fontsize": 8,
        }
    )


def _finish(fig, path, show):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    if show:
        manager = getattr(fig.canvas, "manager", None)
        if manager is not None and fig._suptitle is not None:
            manager.set_window_title(fig._suptitle.get_text())
        elif manager is not None and fig.axes:
            manager.set_window_title(fig.axes[0].get_title())
    else:
        plt.close(fig)


def _labeled(frame):
    labeled = frame.copy()
    labeled["method_label"] = labeled["method"].astype(str).map(METHOD_LABEL)
    return labeled


def save_distribution(frame, value, ylabel, title, path: Path, show=False):
    """Box plot with one filled point per subject. Outlier glyphs are not drawn twice."""
    _apply_style()
    labeled = _labeled(frame)
    order = [METHOD_LABEL[method] for method in _order(frame)]
    palette = {METHOD_LABEL[method]: METHOD_COLOR[method] for method in _order(frame)}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sns.boxplot(
        data=labeled,
        x="method_label",
        y=value,
        order=order,
        hue="method_label",
        palette=palette,
        legend=False,
        showfliers=False,
        width=0.55,
        linewidth=0.8,
        ax=ax,
    )
    sns.stripplot(
        data=labeled,
        x="method_label",
        y=value,
        order=order,
        hue="method_label",
        palette=palette,
        legend=False,
        jitter=0.15,
        size=5.5,
        linewidth=0,
        marker="o",
        ax=ax,
        zorder=3,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    _finish(fig, path, show)


def save_phase_mae(per_recording, path: Path, show=False):
    """Mean per-subject MAE by phase, with the standard deviation as the only whisker."""
    _apply_style()
    methods = _order(per_recording)
    labels = [METHOD_LABEL[method] for method in methods]
    series = [
        ("mae_normal", PHASE_COLOR["normal"], "Normal breathing"),
        ("mae_hyperventilation", "#E69F00", "Hyperventilation"),
    ]
    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for offset, (column, color, name) in zip((-width / 2, width / 2), series):
        means = []
        spreads = []
        for method in methods:
            values = per_recording.loc[per_recording["method"] == method, column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            means.append(float(np.mean(values)) if len(values) else np.nan)
            spreads.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=spreads,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=name,
            capsize=3,
            error_kw={"ecolor": "#333333", "elinewidth": 0.9, "capthick": 0.9},
            zorder=2,
        )
    ax.axhline(0.0, color="#888888", linewidth=0.6)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean absolute error [BPM]")
    ax.set_xlabel("")
    ax.set_title("Per-subject respiratory-rate error by protocol phase")
    ax.legend(title="Protocol phase", loc="upper right")
    fig.tight_layout()
    _finish(fig, path, show)


def save_bland_altman(points, summary, path: Path, show=False):
    """Instantaneous rate error against the mean of the estimate and the reference."""
    _apply_style()
    methods = [method for method in METHOD_ORDER if method in set(points["method"])]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, method in zip(axes.ravel(), methods):
        group = points[points["method"] == method]
        row = summary[summary["method"].astype(str) == method].iloc[0]
        ax.scatter(
            group["mean_bpm"],
            group["difference"],
            s=8,
            alpha=0.35,
            color=METHOD_COLOR[method],
            linewidths=0,
            rasterized=True,
        )
        bias = float(row["bland_bias"])
        low = float(row["bland_loa_low"])
        high = float(row["bland_loa_high"])
        ax.axhline(bias, color="#333333", linewidth=1.0, label=f"bias = {bias:+.2f}")
        ax.axhline(high, color="#333333", linewidth=0.8, linestyle="--", label=f"+1.96 SD = {high:+.2f}")
        ax.axhline(low, color="#333333", linewidth=0.8, linestyle=":", label=f"−1.96 SD = {low:+.2f}")
        ax.set_title(METHOD_LABEL[method])
        ax.set_xlabel("Mean of estimated and reference BPM")
        ax.set_ylabel("Estimated − reference [BPM]")
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle("Bland–Altman plots of instantaneous respiratory rate", fontweight="bold")
    fig.tight_layout()
    _finish(fig, path, show)


def _shade_phases(ax, windows, x_start, x_limit):
    hyper_start = windows["hyperventilation_start_s"]
    hyper_stop = windows["hyperventilation_stop_s"]
    apnea_start = windows["apnea_start_s"]
    apnea_stop = windows["apnea_stop_s"]

    def _span(start, stop, color, alpha):
        if not (np.isfinite(start) and np.isfinite(stop)):
            return
        left = max(float(start), x_start)
        right = min(float(stop), x_limit)
        if right > left:
            ax.axvspan(left, right, color=color, alpha=alpha, lw=0)

    if np.isfinite(apnea_start):
        _span(x_start, hyper_start if np.isfinite(hyper_start) else apnea_start, PHASE_COLOR["normal"], 0.45)
        if np.isfinite(hyper_stop) and hyper_stop < apnea_start:
            _span(hyper_stop, apnea_start, PHASE_COLOR["normal"], 0.45)
    _span(hyper_start, hyper_stop, PHASE_COLOR["hyperventilation"], 0.55)
    _span(apnea_start, apnea_stop, PHASE_COLOR["apnea"], 0.45)


def _load_trace(synced_dir, subject, method):
    path = Path(synced_dir) / f"synced_{subject}_{method}.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path, sep=";")


def save_signal_comparison(synced_dir, per_recording, path: Path, phase_windows=None, show=False):
    """Typical and challenging recordings, chosen from the Lucas–Kanade correlation."""
    _apply_style()
    lucas = per_recording[per_recording["method"] == "Lucas-Kanade"].copy()
    median = float(lucas["spearman"].median())
    lucas["distance"] = (lucas["spearman"] - median).abs()
    typical = lucas.sort_values(["distance", "subject"]).iloc[0]
    challenging = lucas.sort_values(["spearman", "subject"]).iloc[0]
    cases = [
        (typical, "Typical recording"),
        (challenging, "Challenging recording"),
    ]

    fig = plt.figure(figsize=(10, 11))
    grid = fig.add_gridspec(4, 1, hspace=0.55, top=0.93, bottom=0.05)
    axes = [
        fig.add_subplot(grid[0]),
        fig.add_subplot(grid[1]),
        fig.add_subplot(grid[2]),
        fig.add_subplot(grid[3]),
    ]
    axes[1].sharex(axes[0])
    axes[3].sharex(axes[2])

    for case_index, (case, case_name) in enumerate(cases):
        subject = case["subject"]
        waveform_ax = axes[case_index * 2]
        rate_ax = axes[case_index * 2 + 1]
        reference = _load_trace(synced_dir, subject, "Lucas-Kanade")
        windows = (phase_windows or {}).get(subject, {})
        x_start = float(reference["Relative_Time_s"].iloc[0])
        x_limit = float(reference["Relative_Time_s"].iloc[-1])
        for ax in (waveform_ax, rate_ax):
            _shade_phases(ax, windows, x_start, x_limit)
        waveform_ax.plot(
            reference["Relative_Time_s"],
            reference["GT_Signal_ZScore"],
            color=TRACE_COLOR["GT"],
            lw=1.3,
            label="GT (belt)",
            zorder=3,
        )
        rate_ax.plot(
            reference["Relative_Time_s"],
            reference["GT_BPM"],
            color=TRACE_COLOR["GT"],
            lw=1.3,
            label="GT BPM",
            zorder=3,
        )
        for method in METHOD_ORDER:
            trace = reference if method == "Lucas-Kanade" else _load_trace(synced_dir, subject, method)
            if trace is None:
                continue
            waveform_ax.plot(
                trace["Relative_Time_s"],
                trace["Estimated_Signal_ZScore"],
                color=TRACE_COLOR[method],
                lw=0.9,
                alpha=0.9,
                label=METHOD_LABEL[method],
            )
            rate_ax.plot(
                trace["Relative_Time_s"],
                trace["Estimated_BPM"],
                color=TRACE_COLOR[method],
                lw=0.9,
                alpha=0.9,
                label=METHOD_LABEL[method],
            )
        waveform_ax.set_ylabel("Z-score amplitude")
        rate_ax.set_ylabel("Breathing rate [BPM]")
        waveform_ax.tick_params(labelbottom=False)
        rate_ax.set_xlabel("Time [s]", labelpad=6)
        waveform_ax.set_title(
            f"{case_name}: {subject}  (Lucas–Kanade r = {case['spearman']:.3f})",
            pad=10,
        )
        waveform_ax.legend(loc="upper right", ncol=3, fontsize=7)
        rate_ax.legend(loc="upper right", ncol=3, fontsize=7)
        panel = "abcd"[case_index * 2 : case_index * 2 + 2]
        waveform_ax.text(0.01, 0.92, f"({panel[0]})", transform=waveform_ax.transAxes, fontweight="bold")
        rate_ax.text(0.01, 0.92, f"({panel[1]})", transform=rate_ax.transAxes, fontweight="bold")

    fig.suptitle(
        "Estimated and reference signals for a typical and a challenging recording",
        fontweight="bold",
    )
    _finish(fig, path, show)


def show_figures():
    """Block until every figure window is closed."""
    plt.show()
