"""Tests for the Markdown report and the analysis pipeline."""

import numpy as np

from ppg_heart_rate.analysis import analyze_signal
from ppg_heart_rate.report import build_report, write_report
from ppg_heart_rate.synthesis import synthesize


def _summary(**kw):
    kw.setdefault("seed", 7)
    rec = synthesize(duration=60.0, **kw)
    return analyze_signal(rec.signal, rec.fs), rec


def test_report_contains_key_metrics():
    summary, _ = _summary(condition="rest")
    text = build_report(summary, "rest_60bpm.csv")
    assert "PPG Heart-Rate Report" in text
    assert "BPM (mean interval)" in text
    assert "SDNN" in text and "RMSSD" in text and "pNN50" in text
    assert "SQI" in text
    assert "60." in text  # BPM near 60 appears in the report


def test_report_lists_unreliable_segments_when_present():
    summary, _ = _summary(condition="motion", motion_strength=2.0,
                          noise_std=0.20, seed=3)
    text = build_report(summary, "motion.csv")
    assert "Unreliable segments" in text


def test_write_report_creates_file(tmp_path):
    summary, _ = _summary(condition="rest")
    path = write_report(tmp_path / "out" / "report.md", summary, "rest.csv")
    assert path.exists()
    assert "Task Force" in path.read_text()


def test_analysis_summary_keys_and_types():
    summary, rec = _summary(condition="rest")
    for key in ("bpm_mean", "bpm_median", "bpm_spectral", "hrv", "sqi",
                "n_peaks", "segments", "duration_s"):
        assert key in summary
    assert summary["duration_s"] == 60.0
    assert summary["n_peaks"] == len(rec.true_peaks)
    assert isinstance(summary["peak_times_s"], list)
    assert all(isinstance(v, float) for v in summary["peak_times_s"])


def test_analysis_bpm_close_to_truth_rest_and_exercise():
    for cond, nominal in (("rest", 60.0), ("exercise", 130.0)):
        summary, rec = _summary(condition=cond)
        assert abs(summary["bpm_mean"] - nominal) < 2.0
        assert abs(summary["bpm_median"] - nominal) < 2.0
        assert abs(summary["hrv"]["mean_nn_ms"] - 60000.0 / nominal) < 40.0


def test_analysis_arrhythmia_high_hrv():
    summary, _ = _summary(condition="arrhythmia")
    rest, _ = _summary(condition="rest")
    assert summary["hrv"]["sdnn_ms"] > 2 * rest["hrv"]["sdnn_ms"]
    assert summary["hrv"]["rmssd_ms"] > 2 * rest["hrv"]["rmssd_ms"]


def test_analysis_empty_signal_raises():
    import pytest
    with pytest.raises(ValueError):
        analyze_signal(np.array([]), 100.0)
