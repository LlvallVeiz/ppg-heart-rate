"""Heart-rate and heart-rate-variability (HRV) metrics.

Definitions follow the Task Force of the European Society of Cardiology
and the North American Society of Pacing and Electrophysiology (1996),
"Heart rate variability: standards of measurement, physiological
interpretation, and clinical use", Eur. Heart J. 17(3):354-381:

- SDNN: standard deviation of NN intervals (sample std, ddof=1).
- RMSSD: square root of the mean of squared successive differences.
- NN50 / pNN50: count / percentage of successive NN differences > 50 ms.

BPM is reported three ways: from the mean interval, from the median
interval (robust to a few missed/extra beats), and spectrally, from the
dominant frequency of the bandpassed PPG (Welch PSD peak in 0.5-3 Hz,
i.e. 30-180 BPM). Agreement between the three is a useful sanity check.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .filtering import bandpass_filter


def bpm_from_intervals(ibis_ms: np.ndarray) -> dict:
    """BPM from NN intervals: mean-based and median-based."""
    ibis = np.asarray(ibis_ms, dtype=float)
    if len(ibis) == 0:
        return {"bpm_mean": float("nan"), "bpm_median": float("nan"), "n": 0}
    return {
        "bpm_mean": float(60000.0 / np.mean(ibis)),
        "bpm_median": float(60000.0 / np.median(ibis)),
        "n": int(len(ibis)),
    }


def spectral_bpm(
    signal: np.ndarray, fs: float, fmin: float = 0.5, fmax: float = 3.0
) -> float:
    """BPM from the dominant frequency of the bandpassed PPG (Welch PSD).

    Returns NaN when no spectral peak can be found in the cardiac band.
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) < int(fs):
        return float("nan")
    filtered = bandpass_filter(signal, fs)
    freqs, psd = welch(filtered, fs=fs, nperseg=min(2048, len(filtered)))
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask) or float(np.max(psd[mask])) <= 0:
        return float("nan")
    peak_freq = float(freqs[mask][int(np.argmax(psd[mask]))])
    return 60.0 * peak_freq


def hrv_metrics(ibis_ms: np.ndarray) -> dict:
    """Time-domain HRV metrics (Task Force 1996) from NN intervals in ms."""
    ibis = np.asarray(ibis_ms, dtype=float)
    if len(ibis) < 2:
        return {
            "mean_nn_ms": float(np.mean(ibis)) if len(ibis) else float("nan"),
            "sdnn_ms": float("nan"),
            "rmssd_ms": float("nan"),
            "nn50": 0,
            "pnn50_pct": float("nan"),
            "cv_pct": float("nan"),
            "n": int(len(ibis)),
        }
    diffs = np.diff(ibis)
    sdnn = float(np.std(ibis, ddof=1)) if len(ibis) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    nn50 = int(np.count_nonzero(np.abs(diffs) > 50.0))
    mean_nn = float(np.mean(ibis))
    return {
        "mean_nn_ms": mean_nn,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "nn50": nn50,
        "pnn50_pct": float(100.0 * nn50 / len(diffs)) if len(diffs) else 0.0,
        "cv_pct": float(100.0 * sdnn / mean_nn) if mean_nn else float("nan"),
        "n": int(len(ibis)),
    }
