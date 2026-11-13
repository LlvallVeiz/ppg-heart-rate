"""ppg-heart-rate: synthetic PPG generation and heart-rate / HRV analysis."""

from .analysis import analyze_signal
from .io import load_ppg_csv, save_ppg_csv
from .metrics import bpm_from_intervals, hrv_metrics, spectral_bpm
from .peaks import detect_peaks_elgendi, evaluate_detection
from .quality import signal_quality_index
from .synthesis import CONDITIONS, SyntheticPPG, synthesize

__version__ = "1.0.0"

__all__ = [
    "CONDITIONS",
    "SyntheticPPG",
    "__version__",
    "analyze_signal",
    "bpm_from_intervals",
    "detect_peaks_elgendi",
    "evaluate_detection",
    "hrv_metrics",
    "load_ppg_csv",
    "save_ppg_csv",
    "signal_quality_index",
    "spectral_bpm",
    "synthesize",
]
