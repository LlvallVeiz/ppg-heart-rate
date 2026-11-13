"""Filtering helpers for PPG signals.

The standard preprocessing chain for PPG (Allen 2007; Elgendi et al. 2013)
is a bandpass filter that keeps the cardiac band (about 0.5-8 Hz) while
removing baseline wander below ~0.5 Hz and high-frequency noise above
~8 Hz. Filtering here is zero-phase (``filtfilt``) so peak times are not
shifted -- important when peak timing feeds HRV metrics.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 8.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter (default 0.5-8 Hz).

    Raises ``ValueError`` for invalid cutoffs or signals too short for
    the filter padding.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if not 0 < low < high < fs / 2:
        raise ValueError(
            f"need 0 < low < high < fs/2, got low={low}, high={high}, fs={fs}"
        )
    if len(signal) < 3 * order + 1:
        raise ValueError("signal too short for filtering")
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average (window in samples, odd or even).

    Edges are handled by shrinking the window (``mode='same'`` with a
    normalised boxcar kernel); the output has the same length as the input.
    """
    signal = np.asarray(signal, dtype=float)
    if window < 1:
        raise ValueError("window must be >= 1")
    if window == 1:
        return signal.copy()
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


def remove_baseline(
    signal: np.ndarray, fs: float, cutoff: float = 0.5, order: int = 4
) -> np.ndarray:
    """High-pass filter that removes baseline wander below ``cutoff`` Hz."""
    signal = np.asarray(signal, dtype=float)
    if not 0 < cutoff < fs / 2:
        raise ValueError(f"need 0 < cutoff < fs/2, got cutoff={cutoff}, fs={fs}")
    sos = butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def band_power(
    signal: np.ndarray, fs: float, low: float, high: float
) -> float:
    """Power of ``signal`` inside [low, high] Hz (Welch PSD, trapezoidal)."""
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    mask = (freqs >= low) & (freqs <= high)
    return float(np.trapezoid(psd[mask], freqs[mask]))


def snr_db(signal: np.ndarray, fs: float, low: float = 0.5, high: float = 8.0) -> float:
    """Signal-to-noise ratio in dB: in-band (0.5-8 Hz) vs out-of-band power."""
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    in_band = (freqs >= low) & (freqs <= high)
    p_signal = float(np.trapezoid(psd[in_band], freqs[in_band]))
    p_noise = float(np.trapezoid(psd[~in_band], freqs[~in_band]))
    if p_noise <= 0:
        return float("inf")
    if p_signal <= 0:
        return float("-inf")
    return float(10.0 * np.log10(p_signal / p_noise))
