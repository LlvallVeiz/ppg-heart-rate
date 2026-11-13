"""Tests for synthetic PPG generation."""

import numpy as np
import pytest

from ppg_heart_rate.synthesis import CONDITIONS, synthesize


def test_rest_beat_count_matches_bpm():
    rec = synthesize(condition="rest", duration=60.0, seed=1)
    # 60 BPM over 60 s -> about 60 beats (allow edge losses of one beat)
    assert 58 <= len(rec.true_peaks) <= 62
    assert rec.true_bpm == 60.0


def test_exercise_beat_count_matches_bpm():
    rec = synthesize(condition="exercise", duration=60.0, seed=1)
    assert 128 <= len(rec.true_peaks) <= 133
    assert rec.true_bpm == 130.0


def test_signal_length_and_time_axis():
    rec = synthesize(condition="rest", duration=30.0, fs=100.0, seed=0)
    assert len(rec.signal) == 3000
    assert len(rec.time) == 3000
    assert rec.time[0] == 0.0
    assert abs(rec.time[-1] - 29.99) < 1e-9
    assert rec.fs == 100.0


def test_seed_reproducibility():
    a = synthesize(condition="rest", duration=20.0, seed=42)
    b = synthesize(condition="rest", duration=20.0, seed=42)
    c = synthesize(condition="rest", duration=20.0, seed=43)
    assert np.array_equal(a.signal, b.signal)
    assert np.array_equal(a.true_peaks, b.true_peaks)
    assert not np.array_equal(a.signal, c.signal)


def test_true_peaks_sorted_and_inside_recording():
    for cond in CONDITIONS:
        rec = synthesize(condition=cond, duration=30.0, seed=2)
        assert np.all(np.diff(rec.true_peaks) > 0)
        assert rec.true_peaks[0] >= 0
        assert rec.true_peaks[-1] <= rec.duration


def test_waveform_has_dicrotic_notch():
    """One beat at rest: systolic peak, a dip, then a diastolic peak."""
    rec = synthesize(condition="rest", duration=10.0, seed=3, noise_std=0.0)
    # First full beat: onset 0, IBI ~1 s; systolic ~0.18 s, diastolic ~0.55 s.
    seg = rec.signal[(rec.time >= 0.05) & (rec.time <= 0.95)]
    t = rec.time[(rec.time >= 0.05) & (rec.time <= 0.95)]
    sys_i = int(np.argmax(seg))
    # Diastolic peak: max after the notch region (0.35-0.8 s after onset).
    dia_mask = (t >= 0.35) & (t <= 0.80)
    dia_i = int(np.argmax(seg[dia_mask]))
    dia_t = t[dia_mask][dia_i]
    notch_mask = (t > t[sys_i]) & (t < dia_t)
    notch_min = float(np.min(seg[notch_mask]))
    assert seg[sys_i] > notch_min  # dip below systolic peak
    assert seg[dia_mask][dia_i] > notch_min  # dip below diastolic peak
    assert seg[sys_i] > seg[dia_mask][dia_i]  # systolic taller than diastolic


def test_baseline_wander_present():
    """Mean of 5 s windows drifts (baseline wander), unlike a flat signal."""
    rec = synthesize(condition="rest", duration=30.0, seed=4)
    means = [rec.signal[int(i * 500): int((i + 1) * 500)].mean()
             for i in range(6)]
    assert max(means) - min(means) > 0.05


def test_noise_level_increases_std():
    clean = synthesize(condition="rest", duration=20.0, seed=5, noise_std=0.0)
    noisy = synthesize(condition="rest", duration=20.0, seed=5, noise_std=0.10)
    # Same underlying beats (same seed); extra noise raises residual std.
    resid = noisy.signal - clean.signal
    assert 0.08 < resid.std() < 0.13


def test_arrhythmia_more_variable_than_rest():
    rest = synthesize(condition="rest", duration=60.0, seed=6)
    arr = synthesize(condition="arrhythmia", duration=60.0, seed=6)
    cv_rest = np.std(rest.true_ibis_ms) / np.mean(rest.true_ibis_ms)
    cv_arr = np.std(arr.true_ibis_ms) / np.mean(arr.true_ibis_ms)
    assert cv_arr > 2 * cv_rest


def test_low_perfusion_smaller_amplitude():
    rest = synthesize(condition="rest", duration=20.0, seed=7)
    low = synthesize(condition="low-perfusion", duration=20.0, seed=7)
    assert low.signal.std() < 0.6 * rest.signal.std()


def test_unknown_condition_raises():
    with pytest.raises(ValueError, match="unknown condition"):
        synthesize(condition="sleep")


def test_invalid_duration_and_bpm_raise():
    with pytest.raises(ValueError):
        synthesize(duration=0)
    with pytest.raises(ValueError):
        synthesize(bpm=0)
    with pytest.raises(ValueError):
        synthesize(fs=-5)


def test_bpm_override():
    rec = synthesize(condition="rest", duration=30.0, seed=1, bpm=80.0)
    assert rec.true_bpm == 80.0
    assert 38 <= len(rec.true_peaks) <= 42
