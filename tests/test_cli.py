"""End-to-end CLI tests (subprocess, as a user would run it)."""

import csv
import json
import os
import subprocess
import sys

import pytest

from ppg_heart_rate.synthesis import synthesize
from ppg_heart_rate.io import save_ppg_csv

PY = [sys.executable, "-m", "ppg_heart_rate"]
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(args, cwd):
    env = dict(os.environ)
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(PY + args, cwd=cwd, capture_output=True, text=True,
                          env=env)


def test_cli_simulate(tmp_path):
    out = tmp_path / "sim.csv"
    r = _run(["simulate", "--condition", "rest", "--duration", "10",
              "--seed", "1", "--output", str(out)], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert out.exists()
    assert "10" in r.stdout and "true beats" in r.stdout


def test_cli_simulate_unknown_condition_fails(tmp_path):
    r = _run(["simulate", "--condition", "sleep"], cwd=tmp_path)
    assert r.returncode != 0


def test_cli_analyze_end_to_end(tmp_path):
    rec = synthesize(condition="rest", duration=30.0, seed=7)
    src = tmp_path / "rest.csv"
    save_ppg_csv(src, rec.time, rec.signal, rec.true_peaks)
    r = _run(["analyze", "--input", str(src), "--output-dir",
              str(tmp_path / "out")], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    summary = json.loads((tmp_path / "out" / "rest_summary.json").read_text())
    assert abs(summary["bpm_mean"] - 60.0) < 2.0
    assert "detection_vs_truth" in summary  # truth column was present
    assert summary["detection_vs_truth"]["sensitivity"] >= 0.98
    assert (tmp_path / "out" / "rest_peaks.csv").exists()
    assert (tmp_path / "out" / "rest_plot.png").exists()
    assert "BPM mean" in r.stdout


def test_cli_analyze_missing_input_fails(tmp_path):
    r = _run(["analyze", "--input", str(tmp_path / "nope.csv")],
             cwd=tmp_path)
    assert r.returncode != 0


def test_cli_batch(tmp_path):
    for name, cond in (("a_rest", "rest"), ("b_exercise", "exercise")):
        rec = synthesize(condition=cond, duration=20.0, seed=7)
        save_ppg_csv(tmp_path / f"{name}.csv", rec.time, rec.signal)
    out = tmp_path / "summary.csv"
    r = _run(["batch", "--input-dir", str(tmp_path), "--output", str(out)],
             cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    with open(out) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]["status"] == "ok" and rows[1]["status"] == "ok"
    assert abs(float(rows[0]["bpm_mean"]) - 60.0) < 2.0
    assert abs(float(rows[1]["bpm_mean"]) - 130.0) < 2.0


def test_cli_batch_empty_dir_fails(tmp_path):
    r = _run(["batch", "--input-dir", str(tmp_path), "--output",
              str(tmp_path / "s.csv")], cwd=tmp_path)
    assert r.returncode != 0


def test_cli_report(tmp_path):
    rec = synthesize(condition="rest", duration=30.0, seed=7)
    src = tmp_path / "rest.csv"
    save_ppg_csv(src, rec.time, rec.signal)
    out = tmp_path / "report.md"
    r = _run(["report", "--input", str(src), "--output", str(out)],
             cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "# PPG Heart-Rate Report" in text
    assert "SDNN" in text


def test_cli_no_args_fails():
    r = _run([], cwd=PROJECT_ROOT)
    assert r.returncode != 0
