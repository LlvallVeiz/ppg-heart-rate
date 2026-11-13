"""Plots for PPG analysis (Matplotlib, Agg backend -- no display needed)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .filtering import bandpass_filter  # noqa: E402


def plot_analysis(
    time: np.ndarray,
    signal: np.ndarray,
    fs: float,
    peak_idx: np.ndarray,
    out_path: str | Path,
    title: str = "PPG analysis",
) -> Path:
    """Two-panel figure: raw signal with detected peaks + IBI tachogram."""
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    peak_idx = np.asarray(peak_idx, dtype=int)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    axes[0].plot(time, signal, color="#1f77b4", linewidth=0.8,
                 label="raw PPG")
    axes[0].plot(time, bandpass_filter(signal, fs), color="#d62728",
                 linewidth=0.8, alpha=0.7, label="bandpassed 0.5-8 Hz")
    if len(peak_idx):
        axes[0].plot(time[peak_idx], signal[peak_idx], "o", color="#2ca02c",
                     markersize=4, label="detected systolic peaks")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("PPG (a.u.)")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    if len(peak_idx) >= 2:
        ibis = np.diff(peak_idx) / fs * 1000.0
        mid_times = (time[peak_idx[1:]] + time[peak_idx[:-1]]) / 2
        axes[1].plot(mid_times, ibis, "-o", color="#9467bd", markersize=3,
                     linewidth=0.8)
        axes[1].axhline(60000.0 / 60.0, color="grey", linestyle="--",
                        linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Inter-beat interval (ms)")
    axes[1].set_title("Inter-beat interval tachogram")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_waveform(
    time: np.ndarray,
    signal: np.ndarray,
    out_path: str | Path,
    title: str = "PPG waveform",
    window_s: float | None = None,
) -> Path:
    """Single-panel waveform plot, optionally limited to the first window."""
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if window_s is not None:
        mask = time <= window_s
        time, signal = time[mask], signal[mask]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(time, signal, color="#1f77b4", linewidth=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PPG (a.u.)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
