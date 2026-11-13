"""Tests for plotting (PNG output)."""

import numpy as np

from ppg_heart_rate.peaks import detect_peaks_elgendi
from ppg_heart_rate.plotting import plot_analysis, plot_waveform
from ppg_heart_rate.synthesis import synthesize


def test_plot_analysis_creates_png(tmp_path):
    rec = synthesize(condition="rest", duration=30.0, seed=7)
    peaks = detect_peaks_elgendi(rec.signal, rec.fs)
    path = plot_analysis(rec.time, rec.signal, rec.fs, peaks,
                         tmp_path / "plot.png", title="test")
    assert path.exists()
    assert path.stat().st_size > 10_000  # a real figure, not empty
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_waveform_window(tmp_path):
    rec = synthesize(condition="rest", duration=30.0, seed=7)
    path = plot_waveform(rec.time, rec.signal, tmp_path / "wave.png",
                         window_s=10.0)
    assert path.exists() and path.stat().st_size > 5_000
