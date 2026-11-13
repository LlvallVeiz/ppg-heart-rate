"""Tests for filtering helpers."""

import numpy as np
import pytest

from ppg_heart_rate.filtering import (
    band_power,
    bandpass_filter,
    moving_average,
    remove_baseline,
    snr_db,
)

FS = 100.0


def _sine(freq, duration=10.0, amp=1.0):
    t = np.arange(int(duration * FS)) / FS
    return t, amp * np.sin(2 * np.pi * freq * t)


def test_bandpass_removes_dc():
    t = np.arange(1000) / FS
    sig = 2.0 + np.sin(2 * np.pi * 1.0 * t)
    out = bandpass_filter(sig, FS)
    assert abs(out.mean()) < 0.05
    assert out.std() > 0.3  # 1 Hz component survives


def test_bandpass_attenuates_high_frequency():
    _, sig = _sine(20.0)  # far above the 8 Hz cutoff
    out = bandpass_filter(sig, FS)
    # Steady state (edges excluded: zero-phase filtering has transients).
    assert out[200:-200].std() < 0.05 * sig.std()


def test_bandpass_preserves_cardiac_frequency():
    _, sig = _sine(1.2)
    out = bandpass_filter(sig, FS)
    # 1.2 Hz is well inside 0.5-8 Hz: amplitude preserved within 10%.
    assert abs(out.std() - sig.std()) < 0.10 * sig.std()


def test_bandpass_zero_phase_no_peak_shift():
    """A pulse stays at the same sample after zero-phase filtering."""
    sig = np.zeros(1000)
    sig[500] = 1.0
    sig[499] = 0.5
    sig[501] = 0.5
    out = bandpass_filter(sig, FS)
    assert int(np.argmax(np.abs(out))) == 500


def test_bandpass_invalid_cutoffs_raise():
    sig = np.zeros(500)
    with pytest.raises(ValueError):
        bandpass_filter(sig, FS, low=8.0, high=0.5)
    with pytest.raises(ValueError):
        bandpass_filter(sig, FS, low=0.5, high=60.0)  # above Nyquist
    with pytest.raises(ValueError):
        bandpass_filter(np.zeros(5), FS)  # too short


def test_moving_average_smooths_noise():
    rng = np.random.default_rng(0)
    noisy = np.sin(np.linspace(0, 20, 1000)) + rng.normal(0, 0.5, 1000)
    smooth = moving_average(noisy, 21)
    assert len(smooth) == len(noisy)
    assert smooth.std() < noisy.std()
    assert moving_average(noisy, 1).tolist() == noisy.tolist()


def test_remove_baseline_removes_slow_drift():
    t = np.arange(1000) / FS
    sig = 1.0 + 0.5 * t / 10.0 + np.sin(2 * np.pi * 1.0 * t)
    out = remove_baseline(sig, FS)
    assert abs(out.mean()) < 0.1
    assert out.std() > 0.4


def test_band_power_and_snr():
    _, sig = _sine(1.2)
    assert band_power(sig, FS, 0.5, 8.0) > 0.9 * band_power(sig, FS, 0.0, 50.0)
    assert snr_db(sig, FS) > 10.0  # pure tone: almost all power in band
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1.0, 1000)
    assert snr_db(noise, FS) < snr_db(sig, FS)
