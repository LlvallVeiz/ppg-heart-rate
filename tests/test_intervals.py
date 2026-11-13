"""Tests for inter-beat intervals and their correction."""

import numpy as np

from ppg_heart_rate.intervals import correct_intervals, peak_intervals_ms


def test_intervals_hand_computed():
    peaks = np.array([0.0, 1.0, 2.02, 3.0])
    ibis = peak_intervals_ms(peaks)
    assert np.allclose(ibis, [1000.0, 1020.0, 980.0])


def test_intervals_need_two_peaks():
    assert len(peak_intervals_ms(np.array([1.0]))) == 0
    assert len(peak_intervals_ms(np.array([]))) == 0


def test_correction_removes_out_of_range():
    ibis = np.array([1000.0, 250.0, 1000.0, 2500.0, 990.0])
    out = correct_intervals(ibis)
    assert out["n_removed_range"] == 2
    assert np.allclose(out["nn_ms"], [1000.0, 1000.0, 990.0])


def test_correction_removes_deviant_ectopic():
    # One interval 30% shorter than the median -> ectopic-style outlier.
    ibis = np.array([1000.0, 1000.0, 700.0, 1000.0, 1000.0, 1010.0])
    out = correct_intervals(ibis)
    assert out["n_removed_deviation"] == 1
    assert 700.0 not in out["nn_ms"]


def test_correction_keeps_normal_variation():
    ibis = np.array([980.0, 1010.0, 995.0, 1020.0, 990.0])
    out = correct_intervals(ibis)
    assert out["n_removed_range"] == 0
    assert out["n_removed_deviation"] == 0
    assert len(out["nn_ms"]) == 5


def test_correction_empty_input():
    out = correct_intervals(np.array([]))
    assert len(out["nn_ms"]) == 0
    assert out["n_removed_range"] == 0


def test_kept_mask_aligns_with_intervals():
    ibis = np.array([1000.0, 250.0, 1000.0])
    out = correct_intervals(ibis)
    assert out["kept"].tolist() == [True, False, True]
    assert len(out["nn_ms"]) == int(out["kept"].sum())
