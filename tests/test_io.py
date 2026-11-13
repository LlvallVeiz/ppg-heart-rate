"""Tests for CSV / JSON I/O."""

import json

import numpy as np
import pytest

from ppg_heart_rate.io import (
    load_ppg_csv,
    save_peaks_csv,
    save_ppg_csv,
    save_summary_json,
)
from ppg_heart_rate.synthesis import synthesize


def test_save_load_roundtrip(tmp_path):
    rec = synthesize(condition="rest", duration=10.0, seed=1)
    path = tmp_path / "rec.csv"
    save_ppg_csv(path, rec.time, rec.signal, rec.true_peaks)
    loaded = load_ppg_csv(path)
    assert np.allclose(loaded["time"], rec.time, atol=1e-4)
    assert np.allclose(loaded["signal"], rec.signal, atol=1e-6)
    assert abs(loaded["fs"] - 100.0) < 0.01
    assert len(loaded["true_peaks"]) == len(rec.true_peaks)
    assert np.allclose(loaded["true_peaks"], rec.true_peaks, atol=0.01)


def test_save_load_without_truth(tmp_path):
    rec = synthesize(condition="rest", duration=5.0, seed=1)
    path = tmp_path / "rec.csv"
    save_ppg_csv(path, rec.time, rec.signal)
    loaded = load_ppg_csv(path)
    assert loaded["true_peaks"] is None


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ppg_csv(tmp_path / "nope.csv")


def test_load_malformed_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="expected columns"):
        load_ppg_csv(bad)
    empty = tmp_path / "empty.csv"
    empty.write_text("time_s,ppg\n")
    with pytest.raises(ValueError, match="at least 2 samples"):
        load_ppg_csv(empty)
    nonmono = tmp_path / "nonmono.csv"
    nonmono.write_text("time_s,ppg\n0.0,1.0\n0.0,1.0\n0.02,1.0\n")
    with pytest.raises(ValueError, match="strictly increasing"):
        load_ppg_csv(nonmono)


def test_save_summary_json_roundtrip(tmp_path):
    summary = {"bpm_mean": 60.5, "nested": {"sdnn_ms": 25.0}}
    path = save_summary_json(tmp_path / "s.json", summary)
    assert json.loads(path.read_text()) == summary


def test_save_peaks_csv(tmp_path):
    path = tmp_path / "peaks.csv"
    save_peaks_csv(path, np.array([1.0, 2.0, 3.0]),
                   np.array([1000.0, 1000.0]))
    lines = path.read_text().splitlines()
    assert lines[0] == "peak_time_s,ibi_after_ms"
    assert len(lines) == 4
    assert lines[1].startswith("1.0000,1000.00")
