"""Tests for BPM and HRV metrics, with hand-computed values.

Hand-computed reference (intervals in ms [1000, 1020, 980, 1010]):
- mean NN = 1002.5 ms -> BPM mean = 60000 / 1002.5 = 59.850...
- median NN = 1005 ms -> BPM median = 60000 / 1005 = 59.701...
- SDNN (sample, ddof=1): deviations -2.5, 17.5, -22.5, 7.5;
  squares sum to 875; 875 / 3 = 291.6667; sqrt = 17.078...
- RMSSD: diffs 20, -40, 30; squares 400 + 1600 + 900 = 2900;
  2900 / 3 = 966.6667; sqrt = 31.091...
- pNN50 on [1000, 1060, 1000, 1000]: diffs 60, -60, 0 -> NN50 = 2 of 3
  -> 66.666...%.
"""

import numpy as np

from ppg_heart_rate.metrics import (
    bpm_from_intervals,
    hrv_metrics,
    spectral_bpm,
)
from ppg_heart_rate.synthesis import synthesize


def test_bpm_hand_computed():
    ibis = np.array([1000.0, 1020.0, 980.0, 1010.0])
    out = bpm_from_intervals(ibis)
    assert abs(out["bpm_mean"] - 60000.0 / 1002.5) < 1e-9
    assert abs(out["bpm_median"] - 60000.0 / 1005.0) < 1e-9
    assert out["n"] == 4


def test_bpm_constant_intervals():
    assert bpm_from_intervals(np.array([1000.0, 1000.0]))["bpm_mean"] == 60.0
    assert bpm_from_intervals(np.array([500.0, 500.0]))["bpm_mean"] == 120.0


def test_bpm_empty():
    out = bpm_from_intervals(np.array([]))
    assert np.isnan(out["bpm_mean"]) and out["n"] == 0


def test_sdnn_hand_computed():
    ibis = np.array([1000.0, 1020.0, 980.0, 1010.0])
    out = hrv_metrics(ibis)
    assert abs(out["sdnn_ms"] - 17.078) < 0.01
    assert abs(out["mean_nn_ms"] - 1002.5) < 1e-9


def test_rmssd_hand_computed():
    ibis = np.array([1000.0, 1020.0, 980.0, 1010.0])
    out = hrv_metrics(ibis)
    assert abs(out["rmssd_ms"] - 31.091) < 0.01


def test_pnn50_hand_computed():
    ibis = np.array([1000.0, 1060.0, 1000.0, 1000.0])
    out = hrv_metrics(ibis)
    assert out["nn50"] == 2
    assert abs(out["pnn50_pct"] - 200.0 / 3.0) < 1e-9


def test_hrv_constant_intervals_zero():
    out = hrv_metrics(np.array([1000.0, 1000.0, 1000.0, 1000.0]))
    assert out["sdnn_ms"] == 0.0
    assert out["rmssd_ms"] == 0.0
    assert out["pnn50_pct"] == 0.0


def test_hrv_too_short():
    out = hrv_metrics(np.array([1000.0]))
    assert np.isnan(out["sdnn_ms"])
    assert out["n"] == 1


def test_spectral_bpm_pure_tone():
    """Second, independent BPM route: a 1.2 Hz sine must read ~72 BPM."""
    fs = 100.0
    t = np.arange(6000) / fs
    sig = np.sin(2 * np.pi * 1.2 * t)
    assert abs(spectral_bpm(sig, fs) - 72.0) < 3.0


def test_spectral_bpm_synthetic_rest():
    rec = synthesize(condition="rest", duration=60.0, seed=7)
    assert abs(spectral_bpm(rec.signal, rec.fs) - 60.0) < 3.0


def test_spectral_bpm_short_signal_nan():
    assert np.isnan(spectral_bpm(np.zeros(50), 100.0))


def test_interval_bpm_agrees_with_spectral_on_synthetic():
    """Two independent BPM implementations must agree on the same signal."""
    for cond, nominal in (("rest", 60.0), ("exercise", 130.0)):
        rec = synthesize(condition=cond, duration=60.0, seed=9)
        from ppg_heart_rate.analysis import analyze_signal
        s = analyze_signal(rec.signal, rec.fs)
        assert abs(s["bpm_mean"] - nominal) < 2.0
        assert abs(s["bpm_spectral"] - nominal) < 4.0
        assert abs(s["bpm_mean"] - s["bpm_spectral"]) < 4.0
