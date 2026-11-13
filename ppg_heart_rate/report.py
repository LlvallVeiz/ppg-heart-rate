"""Markdown report generation from an analysis summary."""

from __future__ import annotations

from pathlib import Path


def _fmt(value, ndigits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        if float(value) != float(value):  # NaN
            return "n/a"
    except (TypeError, ValueError):
        return str(value)
    return f"{float(value):.{ndigits}f}"


def build_report(summary: dict, source_name: str = "recording") -> str:
    """Render a Markdown report for one analysed recording."""
    hrv = summary["hrv"]
    sqi = summary["sqi"]
    lines = [
        f"# PPG Heart-Rate Report — {source_name}",
        "",
        "## Recording",
        "",
        f"- Duration: {_fmt(summary['duration_s'], 1)} s "
        f"({summary['n_samples']} samples @ {_fmt(summary['fs_hz'], 1)} Hz)",
        f"- Detected beats: {summary['n_peaks']} "
        f"({summary['n_nn_intervals']} NN intervals after correction; "
        f"{summary['n_removed_range']} removed out of range, "
        f"{summary['n_removed_deviation']} removed as ectopic/deviant)",
        "",
        "## Heart rate",
        "",
        f"- BPM (mean interval): **{_fmt(summary['bpm_mean'])}**",
        f"- BPM (median interval): **{_fmt(summary['bpm_median'])}**",
        f"- BPM (spectral, Welch PSD peak): **{_fmt(summary['bpm_spectral'])}**",
        "",
        "## Heart-rate variability (Task Force 1996, time domain)",
        "",
        f"- Mean NN: {_fmt(hrv['mean_nn_ms'], 1)} ms",
        f"- SDNN: {_fmt(hrv['sdnn_ms'])} ms",
        f"- RMSSD: {_fmt(hrv['rmssd_ms'])} ms",
        f"- NN50: {hrv['nn50']}  |  pNN50: {_fmt(hrv['pnn50_pct'], 1)}%",
        f"- Coefficient of variation: {_fmt(hrv['cv_pct'], 1)}%",
        "",
        "## Signal quality",
        "",
        f"- SQI (0-1): **{_fmt(sqi['sqi'])}**",
        f"- Beats matching the median template (r > 0.8): "
        f"{_fmt(100 * sqi['frac_good_beats'], 1)}%",
        f"- Mean template correlation: {_fmt(sqi['mean_template_corr'])}",
        f"- SNR (0.5-8 Hz vs out-of-band): "
        f"{_fmt(sqi['snr_db'], 1) if sqi['snr_db'] is not None else 'n/a'} dB",
        f"- Interval regularity: {_fmt(100 * sqi['regularity'], 1)}%",
        f"- Unreliable segments: {summary['unreliable_segments']} "
        f"of {summary['n_segments']} "
        f"(10 s windows, SQI < 0.6)",
        "",
    ]
    bad = [s for s in summary["segments"] if not s["reliable"]]
    if bad:
        lines += ["### Unreliable segments", ""]
        for s in bad:
            lines.append(
                f"- {s['start_s']:.0f}-{s['end_s']:.0f} s: "
                f"SQI {_fmt(s['sqi'])}, {s['n_peaks']} peaks"
            )
        lines.append("")
    lines += [
        "## Method",
        "",
        "Bandpass 0.5-8 Hz (zero-phase Butterworth) -> Elgendi-style peak "
        "detection (two moving averages on the squared signal, adaptive "
        "threshold) -> interval correction (300-2000 ms range, 20% median "
        "deviation rule) -> BPM/HRV per Task Force (1996). See "
        "`docs/RESEARCH.md` for sources and limitations.",
        "",
        "_This report is computed from the supplied recording only; it is "
        "not a medical diagnosis._",
        "",
    ]
    return "\n".join(lines)


def write_report(path: str | Path, summary: dict, source_name: str) -> Path:
    """Write the Markdown report to ``path`` and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(summary, source_name), encoding="utf-8")
    return path
