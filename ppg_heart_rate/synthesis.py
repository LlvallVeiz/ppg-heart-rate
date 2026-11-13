"""Realistic synthetic photoplethysmogram (PPG) signal generation.

Each heartbeat is modelled as the sum of two Gaussian pulses (Allen 2007;
Elgendi 2012): a tall, narrow *systolic* peak followed by a smaller, wider
*diastolic* peak. The dip between them is the dicrotic notch, linked to
aortic-valve closure and reflected pressure waves. Beat-to-beat timing
carries heart-rate variability (a slow respiratory modulation plus random
jitter), and the signal is degraded by baseline wander, white noise and
optional motion-artefact bursts, as in real wearable recordings
(Charlton et al. 2023).

All ground truth (the true systolic-peak times) is returned alongside the
signal, so detection algorithms can be scored exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Nominal condition presets. Values are typical textbook / literature
# ranges, not measurements from any individual.
CONDITIONS = {
    "rest": {
        "bpm": 60.0,
        "sdnn_ms": 35.0,
        "noise_std": 0.02,
        "amplitude": 1.0,
        "motion": False,
        "motion_strength": 0.0,
        "arrhythmia": False,
    },
    "exercise": {
        "bpm": 130.0,
        "sdnn_ms": 15.0,
        "noise_std": 0.05,
        "amplitude": 0.9,
        "motion": True,
        "motion_strength": 0.45,
        "arrhythmia": False,
    },
    "arrhythmia": {
        "bpm": 72.0,
        "sdnn_ms": 80.0,
        "noise_std": 0.03,
        "amplitude": 1.0,
        "motion": False,
        "motion_strength": 0.0,
        "arrhythmia": True,
    },
    "motion": {
        "bpm": 75.0,
        "sdnn_ms": 30.0,
        "noise_std": 0.05,
        "amplitude": 1.0,
        "motion": True,
        "motion_strength": 0.8,
        "arrhythmia": False,
    },
    "low-perfusion": {
        "bpm": 65.0,
        "sdnn_ms": 30.0,
        "noise_std": 0.03,
        "amplitude": 0.35,
        "motion": False,
        "motion_strength": 0.0,
        "arrhythmia": False,
    },
}


@dataclass
class SyntheticPPG:
    """A synthetic PPG recording with known ground truth."""

    time: np.ndarray
    signal: np.ndarray
    fs: float
    true_peaks: np.ndarray  # seconds, systolic-peak times
    true_bpm: float
    condition: str
    params: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.time[-1]) if len(self.time) else 0.0

    @property
    def true_ibis_ms(self) -> np.ndarray:
        """Ground-truth inter-beat intervals in milliseconds."""
        return np.diff(self.true_peaks) * 1000.0


def _pulse_shape(t_rel: np.ndarray, ibi: float, amplitude: float) -> np.ndarray:
    """One heartbeat: systolic Gaussian + diastolic Gaussian.

    Peak positions and widths scale with the beat length so the shape stays
    physiological from 40 to 180 BPM.
    """
    t_sys = 0.18 * ibi
    t_dia = 0.55 * ibi
    s_sys = 0.09 * ibi
    s_dia = 0.14 * ibi
    systolic = np.exp(-0.5 * ((t_rel - t_sys) / s_sys) ** 2)
    diastolic = 0.38 * np.exp(-0.5 * ((t_rel - t_dia) / s_dia) ** 2)
    return amplitude * (systolic + diastolic)


def _generate_ibis(
    bpm: float,
    duration: float,
    sdnn_ms: float,
    rng: np.random.Generator,
    arrhythmia: bool,
) -> np.ndarray:
    """Inter-beat intervals (seconds) with HRV and optional ectopy."""
    base = 60.0 / bpm
    n_beats = int(duration / base) + 3
    ibis = np.full(n_beats, base)
    # Respiratory sinus arrhythmia: slow sinusoidal modulation of the
    # beat interval (~0.2 Hz, +/-25 ms), as described in Task Force (1996).
    beat_times = np.cumsum(ibis)
    ibis = ibis + 0.025 * np.sin(2.0 * np.pi * 0.20 * beat_times)
    # Random beat-to-beat jitter (part of the target SDNN).
    ibis = ibis + rng.normal(0.0, (sdnn_ms / 1000.0) * 0.8, size=n_beats)
    if arrhythmia:
        # Premature beats: ~10% of beats shortened by 25-40%, followed by
        # a compensatory pause 15-30% longer than normal.
        for i in range(1, n_beats - 1):
            if rng.random() < 0.10:
                ibis[i] *= rng.uniform(0.60, 0.75)
                ibis[i + 1] *= rng.uniform(1.15, 1.30)
    return np.clip(ibis, 0.30, 2.00)


def _motion_bursts(
    time: np.ndarray,
    duration: float,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Low-frequency motion artefact bursts with smooth (Hann) envelopes."""
    artefact = np.zeros_like(time)
    n_bursts = int(rng.integers(2, 5))
    for _ in range(n_bursts):
        start = rng.uniform(0.0, max(0.1, duration - 5.0))
        burst_len = rng.uniform(2.0, 5.0)
        mask = (time >= start) & (time <= start + burst_len)
        if not np.any(mask):
            continue
        local_t = time[mask] - start
        envelope = np.hanning(mask.sum())
        freq = rng.uniform(0.7, 1.6)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        artefact[mask] += (
            strength * np.sin(2.0 * np.pi * freq * local_t + phase) * envelope
        )
    return artefact


def synthesize(
    condition: str = "rest",
    duration: float = 60.0,
    fs: float = 100.0,
    bpm: float | None = None,
    seed: int = 0,
    **overrides,
) -> SyntheticPPG:
    """Generate a synthetic PPG recording.

    Parameters
    ----------
    condition:
        One of ``rest``, ``exercise``, ``arrhythmia``, ``motion``,
        ``low-perfusion`` (see :data:`CONDITIONS`).
    duration:
        Recording length in seconds.
    fs:
        Sampling rate in Hz.
    bpm:
        Override the condition's nominal heart rate.
    seed:
        Random seed; identical seeds give identical signals.
    **overrides:
        Any condition parameter (``sdnn_ms``, ``noise_std``,
        ``amplitude``, ``motion_strength``, ``arrhythmia``) to override.

    Returns
    -------
    SyntheticPPG
        Signal, time axis, true systolic-peak times and parameters used.
    """
    if condition not in CONDITIONS:
        raise ValueError(
            f"unknown condition {condition!r}; choose from {sorted(CONDITIONS)}"
        )
    if duration <= 0:
        raise ValueError("duration must be positive")
    if fs <= 0:
        raise ValueError("fs must be positive")
    params = dict(CONDITIONS[condition])
    params.update(overrides)
    if bpm is not None:
        params["bpm"] = float(bpm)
    if params["bpm"] <= 0:
        raise ValueError("bpm must be positive")

    rng = np.random.default_rng(seed)
    n = int(round(duration * fs))
    time = np.arange(n) / fs

    ibis = _generate_ibis(
        params["bpm"], duration, params["sdnn_ms"], rng, params["arrhythmia"]
    )
    onsets = np.concatenate([[0.0], np.cumsum(ibis)[:-1]])
    # Keep only beats whose systolic peak falls inside the recording.
    onsets = onsets[onsets < duration]

    signal = np.zeros(n)
    true_peaks = []
    for k, onset in enumerate(onsets):
        ibi = ibis[k]
        # Respiratory amplitude modulation (+/-12%).
        amp = params["amplitude"] * (
            1.0 + 0.12 * np.sin(2.0 * np.pi * 0.18 * onset)
        )
        t_sys = onset + 0.18 * ibi
        if t_sys > duration:
            break
        true_peaks.append(t_sys)
        # Add this beat over a window slightly longer than one IBI so the
        # diastolic tail is not cut off abruptly.
        lo = int(onset * fs)
        hi = min(n, int((onset + 1.1 * ibi) * fs) + 1)
        if hi <= lo:
            continue
        t_rel = time[lo:hi] - onset
        signal[lo:hi] += _pulse_shape(t_rel, ibi, amp)

    # Baseline wander: slow drift (respiration ~0.12 Hz + Mayer waves).
    phase1 = rng.uniform(0.0, 2.0 * np.pi)
    phase2 = rng.uniform(0.0, 2.0 * np.pi)
    baseline = (
        0.12 * np.sin(2.0 * np.pi * 0.12 * time + phase1)
        + 0.06 * np.sin(2.0 * np.pi * 0.27 * time + phase2)
    )
    signal = signal + 0.6 + baseline

    # White measurement noise.
    signal = signal + rng.normal(0.0, params["noise_std"], size=n)

    # Motion artefact bursts.
    if params["motion"]:
        signal = signal + _motion_bursts(
            time, duration, params["motion_strength"], rng
        )
        # Extra broadband noise during movement.
        signal = signal + rng.normal(
            0.0, params["noise_std"] * 0.8, size=n
        )

    return SyntheticPPG(
        time=time,
        signal=signal,
        fs=float(fs),
        true_peaks=np.asarray(true_peaks, dtype=float),
        true_bpm=float(params["bpm"]),
        condition=condition,
        params={k: float(v) if isinstance(v, (int, float)) else v
                for k, v in params.items()},
    )
