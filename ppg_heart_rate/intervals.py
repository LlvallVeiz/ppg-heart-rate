"""Inter-beat intervals (IBIs) from detected peaks, with artefact correction.

Intervals outside the physiological range (default 300-2000 ms, i.e.
30-200 BPM) are dropped, and intervals deviating more than 20% from the
running median -- the usual ectopic / missed-beat rule -- are dropped too.
Only the surviving ("NN") intervals feed the HRV metrics, following the
Task Force (1996) definition of NN intervals as intervals between normal
beats.
"""

from __future__ import annotations

import numpy as np


def peak_intervals_ms(peak_times_s: np.ndarray) -> np.ndarray:
    """Inter-beat intervals in milliseconds from peak times in seconds."""
    peaks = np.asarray(peak_times_s, dtype=float)
    if len(peaks) < 2:
        return np.array([], dtype=float)
    return np.diff(peaks) * 1000.0


def correct_intervals(
    ibis_ms: np.ndarray,
    min_ms: float = 300.0,
    max_ms: float = 2000.0,
    deviation: float = 0.20,
) -> dict:
    """Filter IBIs to normal-to-normal (NN) intervals.

    Returns a dict with the kept intervals (``nn_ms``), a boolean mask of
    kept entries, and counts of removed intervals by cause.
    """
    ibis = np.asarray(ibis_ms, dtype=float)
    if len(ibis) == 0:
        return {
            "nn_ms": ibis,
            "kept": np.array([], dtype=bool),
            "n_removed_range": 0,
            "n_removed_deviation": 0,
        }
    kept = np.ones(len(ibis), dtype=bool)
    in_range = (ibis >= min_ms) & (ibis <= max_ms)
    n_range = int(np.count_nonzero(~in_range))
    kept &= in_range
    if np.count_nonzero(kept) >= 3:
        median_ibi = float(np.median(ibis[kept]))
        deviant = np.abs(ibis - median_ibi) > deviation * median_ibi
        n_dev = int(np.count_nonzero(deviant & kept))
        kept &= ~deviant
    else:
        n_dev = 0
    return {
        "nn_ms": ibis[kept],
        "kept": kept,
        "n_removed_range": n_range,
        "n_removed_deviation": n_dev,
    }
