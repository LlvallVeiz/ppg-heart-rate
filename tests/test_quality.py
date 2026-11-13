"""Tests for the signal quality index and segment flagging."""

import numpy as np

from ppg_heart_rate.peaks import detect_peaks_elgendi
from ppg_heart_rate.quality import (
    beat_template_correlations,
    extract_beats,
    segment_quality,
    signal_quality_index,
)
from ppg_heart_rate.synthesis import synthesize


def _peaks(rec):
    return detect_peaks_elgendi(rec.signal, rec.fs)


def test_clean_signal_high_sqi():
    rec = synthesize(condition="rest", duration=60.0, seed=7)
    q = signal_quality_index(rec.signal, rec.fs, _peaks(rec))
    assert q["sqi"] > 0.75
    assert q["frac_good_beats"] > 0.9
    assert q["mean_template_corr"] > 0.9


def test_sqi_degrades_with_noise():
    clean = synthesize(condition="rest", duration=60.0, seed=7)
    noisy = synthesize(condition="rest", duration=60.0, seed=7, noise_std=0.25)
    q_clean = signal_quality_index(clean.signal, clean.fs, _peaks(clean))
    q_noisy = signal_quality_index(noisy.signal, noisy.fs, _peaks(noisy))
    assert q_noisy["sqi"] < q_clean["sqi"] - 0.2
    assert q_noisy["sqi"] < 0.6


def test_sqi_degrades_with_extreme_motion():
    rec = synthesize(condition="motion", duration=60.0, seed=3,
                     motion_strength=2.0, noise_std=0.20)
    q = signal_quality_index(rec.signal, rec.fs, _peaks(rec))
    assert q["sqi"] < 0.6


def test_sqi_bounds_zero_to_one():
    for cond in ("rest", "exercise", "motion", "low-perfusion"):
        rec = synthesize(condition=cond, duration=30.0, seed=5)
        q = signal_quality_index(rec.signal, rec.fs, _peaks(rec))
        assert 0.0 <= q["sqi"] <= 1.0


def test_template_correlation_high_on_clean():
    rec = synthesize(condition="rest", duration=30.0, seed=7, noise_std=0.0)
    _, corrs = beat_template_correlations(rec.signal, rec.fs, _peaks(rec))
    assert len(corrs) > 20
    assert np.mean(corrs) > 0.95


def test_extract_beats_shape():
    rec = synthesize(condition="rest", duration=20.0, seed=7)
    beats = extract_beats(rec.signal, rec.fs, _peaks(rec))
    assert beats.shape[1] == 64
    assert 15 <= beats.shape[0] <= 22


def test_segments_cover_recording_and_flag_unreliable():
    rec = synthesize(condition="motion", duration=60.0, seed=3,
                     motion_strength=2.0, noise_std=0.20)
    segs = segment_quality(rec.signal, rec.fs, _peaks(rec))
    assert len(segs) == 6  # 60 s / 10 s windows
    assert segs[0]["start_s"] == 0.0 and segs[-1]["end_s"] == 60.0
    assert any(not s["reliable"] for s in segs)
    # Clean recording: all segments reliable.
    clean = synthesize(condition="rest", duration=60.0, seed=7)
    clean_segs = segment_quality(clean.signal, clean.fs, _peaks(clean))
    assert all(s["reliable"] for s in clean_segs)


def test_segments_flag_consistent_with_threshold():
    rec = synthesize(condition="rest", duration=30.0, seed=7)
    for s in segment_quality(rec.signal, rec.fs, _peaks(rec), threshold=0.6):
        assert s["reliable"] == (s["sqi"] >= 0.6)
