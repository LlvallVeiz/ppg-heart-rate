"""Tests for systolic-peak detection."""

import numpy as np

from ppg_heart_rate.peaks import (
    detect_peaks_elgendi,
    detect_peaks_findpeaks,
    evaluate_detection,
)
from ppg_heart_rate.synthesis import synthesize


def _score(condition, seed=7, **kw):
    rec = synthesize(condition=condition, duration=60.0, seed=seed, **kw)
    det = detect_peaks_elgendi(rec.signal, rec.fs)
    return evaluate_detection(det / rec.fs, rec.true_peaks), rec


def test_rest_detection_near_perfect():
    ev, rec = _score("rest")
    assert ev["sensitivity"] >= 0.98
    assert ev["ppv"] >= 0.98
    assert ev["mean_abs_error_s"] < 0.02  # timing error under 20 ms


def test_exercise_detection_high():
    ev, rec = _score("exercise")
    assert ev["sensitivity"] >= 0.95
    assert ev["ppv"] >= 0.95


def test_arrhythmia_detection_high():
    ev, rec = _score("arrhythmia")
    assert ev["sensitivity"] >= 0.95
    assert ev["ppv"] >= 0.95


def test_motion_detection_decent():
    ev, rec = _score("motion")
    assert ev["sensitivity"] >= 0.90
    assert ev["ppv"] >= 0.90


def test_low_perfusion_detection_high():
    ev, rec = _score("low-perfusion")
    assert ev["sensitivity"] >= 0.95
    assert ev["ppv"] >= 0.95


def test_detection_stable_across_seeds():
    for seed in (0, 1, 2, 3, 11):
        ev, _ = _score("rest", seed=seed)
        assert ev["f1"] >= 0.98, f"seed {seed}: F1 {ev['f1']:.3f}"


def test_refractory_period_enforced():
    rec = synthesize(condition="exercise", duration=30.0, seed=7)
    det = detect_peaks_elgendi(rec.signal, rec.fs, refractory_s=0.3)
    gaps = np.diff(det) / rec.fs
    assert np.all(gaps >= 0.29)


def test_empty_signal_returns_empty():
    assert len(detect_peaks_elgendi(np.array([]), 100.0)) == 0
    assert len(detect_peaks_findpeaks(np.array([]), 100.0)) == 0


def test_independent_detector_agrees():
    """Second implementation (SciPy find_peaks) finds the same beats."""
    rec = synthesize(condition="rest", duration=60.0, seed=7)
    a = detect_peaks_elgendi(rec.signal, rec.fs)
    b = detect_peaks_findpeaks(rec.signal, rec.fs)
    assert abs(len(a) - len(b)) <= 2
    ev_a = evaluate_detection(a / rec.fs, rec.true_peaks)
    ev_b = evaluate_detection(b / rec.fs, rec.true_peaks)
    assert ev_a["f1"] >= 0.95 and ev_b["f1"] >= 0.95


def test_evaluate_detection_counts():
    ev = evaluate_detection(
        np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0])
    )
    assert ev["tp"] == 3 and ev["fp"] == 0 and ev["fn"] == 1
    assert ev["sensitivity"] == 0.75
    assert ev["ppv"] == 1.0
    ev2 = evaluate_detection(
        np.array([1.0, 2.0, 5.0]), np.array([1.0, 2.0, 3.0])
    )
    assert ev2["tp"] == 2 and ev2["fp"] == 1 and ev2["fn"] == 1
    assert abs(ev2["f1"] - 2 / 3) < 1e-9
