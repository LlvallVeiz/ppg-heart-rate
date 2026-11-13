"""CSV / JSON input-output for PPG recordings.

Recording format (CSV, header row)::

    time_s,ppg[,true_peak]

``true_peak`` is an optional 0/1 column written by ``simulate`` for
synthetic ground truth; ``analyze`` ignores it. Times are seconds,
``ppg`` is the (arbitrary-unit) optical signal.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def save_ppg_csv(
    path: str | Path,
    time: np.ndarray,
    signal: np.ndarray,
    true_peaks: np.ndarray | None = None,
) -> Path:
    """Write a recording to CSV; include a ``true_peak`` column when given."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if true_peaks is not None:
        truth = np.zeros(len(time), dtype=int)
        idx = np.searchsorted(time, np.asarray(true_peaks, dtype=float))
        idx = idx[idx < len(time)]
        truth[idx] = 1
        data = np.column_stack([time, signal, truth])
        header = "time_s,ppg,true_peak"
    else:
        data = np.column_stack([time, signal])
        header = "time_s,ppg"
    np.savetxt(path, data, delimiter=",", header=header, comments="",
               fmt=["%.4f", "%.6f"] + (["%d"] if true_peaks is not None else []))
    return path


def load_ppg_csv(path: str | Path) -> dict:
    """Load a recording CSV; returns time, signal, fs and optional truth.

    Raises ``FileNotFoundError`` for a missing file and ``ValueError`` for
    a malformed one (missing columns, non-monotonic time, < 2 samples).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"empty file: {path}")
        header = [h.strip() for h in header]
        if "time_s" not in header or "ppg" not in header:
            raise ValueError(
                f"{path}: expected columns 'time_s' and 'ppg', got {header}"
            )
        ti, pi = header.index("time_s"), header.index("ppg")
        has_truth = "true_peak" in header
        qi = header.index("true_peak") if has_truth else None
        rows = [row for row in reader if row]
    if len(rows) < 2:
        raise ValueError(f"{path}: need at least 2 samples")
    try:
        arr = np.asarray(rows, dtype=float)
    except ValueError as exc:
        raise ValueError(f"{path}: non-numeric value in data rows") from exc
    time = arr[:, ti]
    signal = arr[:, pi]
    if not np.all(np.diff(time) > 0):
        raise ValueError(f"{path}: time_s must be strictly increasing")
    dt = float(np.median(np.diff(time)))
    # Round: CSV times are written to 4 decimals, so 1/dt carries float
    # noise (e.g. 100.00000000000213); 6 decimals is plenty for real data.
    fs = round(1.0 / dt, 6)
    truth = None
    if has_truth:
        truth_idx = np.flatnonzero(arr[:, qi] > 0.5)
        truth = time[truth_idx]
    return {"time": time, "signal": signal, "fs": fs, "true_peaks": truth}


def save_summary_json(path: str | Path, summary: dict) -> Path:
    """Write an analysis summary dict as indented JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def save_peaks_csv(path: str | Path, peak_times_s: np.ndarray,
                   ibis_ms: np.ndarray) -> Path:
    """Write detected peaks and the interval after each peak (NaN: last)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    peaks = np.asarray(peak_times_s, dtype=float)
    ibis = np.asarray(ibis_ms, dtype=float)
    padded = np.concatenate([ibis, [np.nan]]) if len(ibis) else np.array([np.nan])
    if len(padded) < len(peaks):
        padded = np.concatenate([padded, np.full(len(peaks) - len(padded), np.nan)])
    data = np.column_stack([peaks, padded[: len(peaks)]])
    np.savetxt(path, data, delimiter=",", header="peak_time_s,ibi_after_ms",
               comments="", fmt=["%.4f", "%.2f"])
    return path
