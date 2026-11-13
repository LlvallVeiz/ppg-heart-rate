"""End-to-end analysis pipeline: filter -> detect -> correct -> measure.

``analyze_signal`` is the single entry point used by the CLI and the
examples. It returns a plain, JSON-serialisable dict (all NumPy scalars
converted) with the recording's peaks, intervals, BPM estimates (mean,
median, spectral), HRV metrics, signal quality and per-segment quality.
"""

from __future__ import annotations

import numpy as np

from .filtering import bandpass_filter
from .intervals import correct_intervals, peak_intervals_ms
from .metrics import bpm_from_intervals, hrv_metrics, spectral_bpm
from .peaks import detect_peaks_elgendi
from .quality import segment_quality, signal_quality_index


def analyze_signal(
    signal: np.ndarray,
    fs: float,
    *,
    segment_s: float = 10.0,
    sqi_threshold: float = 0.6,
    **detector_kwargs,
) -> dict:
    """Run the full pipeline on one recording and return a summary dict."""
    signal = np.asarray(signal, dtype=float)
    if len(signal) == 0:
        raise ValueError("empty signal")
    peak_idx = detect_peaks_elgendi(signal, fs, **detector_kwargs)
    peak_times = peak_idx / fs
    ibis = peak_intervals_ms(peak_times)
    corrected = correct_intervals(ibis)
    nn_ms = corrected["nn_ms"]
    bpm = bpm_from_intervals(nn_ms)
    hrv = hrv_metrics(nn_ms)
    spec_bpm = spectral_bpm(signal, fs)
    sqi = signal_quality_index(signal, fs, peak_idx)
    segments = segment_quality(
        signal, fs, peak_idx, window_s=segment_s, threshold=sqi_threshold
    )
    n_unreliable = sum(1 for s in segments if not s["reliable"])
    filtered = bandpass_filter(signal, fs)
    return {
        "duration_s": float(len(signal) / fs),
        "fs_hz": float(fs),
        "n_samples": int(len(signal)),
        "n_peaks": int(len(peak_idx)),
        "peak_times_s": [float(t) for t in peak_times],
        "n_intervals": int(len(ibis)),
        "n_nn_intervals": int(len(nn_ms)),
        "n_removed_range": int(corrected["n_removed_range"]),
        "n_removed_deviation": int(corrected["n_removed_deviation"]),
        "bpm_mean": bpm["bpm_mean"],
        "bpm_median": bpm["bpm_median"],
        "bpm_spectral": float(spec_bpm),
        "hrv": hrv,
        "sqi": sqi,
        "unreliable_segments": n_unreliable,
        "n_segments": len(segments),
        "segments": segments,
        "filtered_rms": float(np.sqrt(np.mean(filtered ** 2))),
        "raw_mean": float(np.mean(signal)),
        "raw_std": float(np.std(signal)),
    }
