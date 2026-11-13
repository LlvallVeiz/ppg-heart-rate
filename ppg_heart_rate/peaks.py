"""Systolic-peak detection for PPG signals.

Primary detector: an Elgendi-style algorithm (Elgendi et al. 2013, PLoS ONE
8(10):e76585), adapted from the acceleration-photoplethysmogram detector to
the raw PPG waveform. The published method has three stages -- bandpass
pre-processing, feature extraction with two event-related moving averages,
and threshold classification -- and reached 99.84% sensitivity /
99.89% positive predictivity on 40 heat-stressed recordings. Here the
same structure is kept:

1. bandpass filter (0.5-8 Hz) and square the signal, which emphasises the
   tall systolic peaks over the smaller diastolic wave;
2. two moving averages of the squared signal: a fast one (~0.12 s, tracks
   pulse energy) and a slow one (~0.6 s, tracks the local energy level);
3. a block of interest opens where the fast average exceeds the slow
   average by a relative offset ``alpha``; the systolic peak is the maximum
   of the filtered signal inside that block.

A refractory period (default 0.3 s, i.e. at most 200 BPM) and an adaptive
amplitude gate (a fraction of the median peak amplitude) remove duplicate
and noise detections. The exact window lengths and ``alpha`` are tunable
parameters, not the paper's constants -- the paper detects peaks in the
second-derivative (APG) signal, while this implementation works on the
raw PPG, so its parameters were tuned on synthetic ground truth (see
``docs/RESEARCH.md``).

``detect_peaks_findpeaks`` (SciPy ``find_peaks`` with distance + prominence)
is provided as an independent second implementation for cross-checking.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks as _scipy_find_peaks

from .filtering import bandpass_filter, moving_average


def _blocks(above: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True-runs as (start, end-exclusive) index pairs."""
    blocks: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            blocks.append((start, i))
            start = None
    if start is not None:
        blocks.append((start, len(above)))
    return blocks


def detect_peaks_elgendi(
    signal: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 8.0,
    w_fast_s: float = 0.12,
    w_slow_s: float = 0.6,
    alpha: float = 0.5,
    refractory_s: float = 0.3,
    amp_frac: float = 0.3,
) -> np.ndarray:
    """Detect systolic peaks, Elgendi-style (two moving averages).

    Returns sample indices of the detected systolic peaks, sorted.
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) == 0:
        return np.array([], dtype=int)
    filtered = bandpass_filter(signal, fs, low, high)
    energy = filtered ** 2
    w_fast = max(3, int(round(w_fast_s * fs)))
    w_slow = max(w_fast + 1, int(round(w_slow_s * fs)))
    ma_fast = moving_average(energy, w_fast)
    ma_slow = moving_average(energy, w_slow)
    threshold = ma_slow * (1.0 + alpha)
    peaks: list[int] = []
    for start, end in _blocks(ma_fast > threshold):
        lo = max(0, start - w_fast // 2)
        hi = min(len(filtered), end + w_fast // 2)
        peaks.append(lo + int(np.argmax(filtered[lo:hi])))
    if not peaks:
        return np.array([], dtype=int)
    peaks = np.asarray(sorted(peaks), dtype=int)

    # Refractory period: keep the taller peak when two are too close.
    min_dist = int(round(refractory_s * fs))
    kept: list[int] = [int(peaks[0])]
    for idx in peaks[1:]:
        if idx - kept[-1] >= min_dist:
            kept.append(int(idx))
        elif filtered[idx] > filtered[kept[-1]]:
            kept[-1] = int(idx)
    peaks = np.asarray(kept, dtype=int)

    # Adaptive amplitude gate: drop peaks far below the median amplitude.
    if len(peaks) > 2:
        amps = filtered[peaks]
        gate = amp_frac * float(np.median(amps))
        peaks = peaks[amps >= gate]
    return peaks


def detect_peaks_findpeaks(
    signal: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 8.0,
    refractory_s: float = 0.3,
    prominence_frac: float = 0.25,
) -> np.ndarray:
    """Independent detector: SciPy ``find_peaks`` on the bandpassed signal.

    Used as a cross-check for the Elgendi-style detector; the prominence
    threshold is a fraction of the signal's peak-to-peak range.
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) == 0:
        return np.array([], dtype=int)
    filtered = bandpass_filter(signal, fs, low, high)
    distance = max(1, int(round(refractory_s * fs)))
    prominence = prominence_frac * (float(np.max(filtered)) - float(np.min(filtered)))
    peaks, _ = _scipy_find_peaks(
        filtered, distance=distance, prominence=prominence
    )
    return np.asarray(peaks, dtype=int)


def evaluate_detection(
    detected_s: np.ndarray,
    true_s: np.ndarray,
    tolerance_s: float = 0.10,
) -> dict:
    """Score detected peaks against ground truth (greedy nearest match).

    Returns counts plus sensitivity, positive predictivity, F1 and the
    mean absolute timing error of matched peaks, in seconds.
    """
    detected = np.asarray(detected_s, dtype=float)
    true = np.asarray(true_s, dtype=float)
    used = np.zeros(len(detected), dtype=bool)
    errors: list[float] = []
    tp = 0
    for t in true:
        if len(detected) == 0:
            break
        dist = np.abs(detected - t)
        j = int(np.argmin(dist))
        if dist[j] <= tolerance_s and not used[j]:
            used[j] = True
            tp += 1
            errors.append(float(dist[j]))
    fp = int(len(detected) - tp)
    fn = int(len(true) - tp)
    sensitivity = tp / len(true) if len(true) else 0.0
    ppv = tp / len(detected) if len(detected) else 0.0
    f1 = (
        2 * sensitivity * ppv / (sensitivity + ppv)
        if (sensitivity + ppv) > 0
        else 0.0
    )
    return {
        "n_true": int(len(true)),
        "n_detected": int(len(detected)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sensitivity": float(sensitivity),
        "ppv": float(ppv),
        "f1": float(f1),
        "mean_abs_error_s": float(np.mean(errors)) if errors else float("nan"),
    }
