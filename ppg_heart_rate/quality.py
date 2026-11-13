"""Signal quality index (SQI) for PPG recordings.

No single number captures "is this PPG usable", so the SQI here combines
three independent, interpretable components (each 0-1):

1. **Template agreement** -- every beat is resampled to a fixed-length
   shape and correlated with the median beat template; the fraction of
   beats with Pearson r > 0.8. Clean PPG beats look alike; motion
   artefacts distort them.
2. **SNR** -- in-band (0.5-8 Hz) vs out-of-band power, mapped from dB to
   0-1 (``-5 dB -> 0``, ``20 dB -> 1``).
3. **Interval regularity** -- fraction of IBIs within +/-30% of the median
   IBI; irregular detection usually means missed or false peaks.

``signal_quality_index`` weights them 0.5 / 0.3 / 0.2. ``segment_quality``
scores 10-second windows the same way (template correlation against the
global template plus local SNR) and flags windows below threshold as
unreliable, so downstream metrics can exclude them.
"""

from __future__ import annotations

import numpy as np

from .filtering import snr_db
from .intervals import peak_intervals_ms


def _resample_beat(beat: np.ndarray, n_points: int) -> np.ndarray:
    """Resample one beat segment to ``n_points`` samples (linear)."""
    if len(beat) == n_points:
        return beat.astype(float)
    x_old = np.linspace(0.0, 1.0, len(beat))
    x_new = np.linspace(0.0, 1.0, n_points)
    return np.interp(x_new, x_old, beat)


def extract_beats(
    signal: np.ndarray,
    fs: float,
    peak_idx: np.ndarray,
    pre_s: float = 0.25,
    post_s: float = 0.55,
    n_points: int = 64,
) -> np.ndarray:
    """Beat waveforms around each peak, resampled to ``n_points`` samples.

    Beats too close to the recording edges are skipped. Returns an array
    of shape (n_beats, n_points); empty when fewer than one full beat fits.
    """
    signal = np.asarray(signal, dtype=float)
    peaks = np.asarray(peak_idx, dtype=int)
    pre = int(round(pre_s * fs))
    post = int(round(post_s * fs))
    beats = []
    for p in peaks:
        lo, hi = p - pre, p + post
        if lo < 0 or hi > len(signal):
            continue
        beats.append(_resample_beat(signal[lo:hi], n_points))
    if not beats:
        return np.empty((0, n_points))
    return np.asarray(beats)


def beat_template_correlations(
    signal: np.ndarray, fs: float, peak_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Median beat template and per-beat Pearson correlation with it."""
    beats = extract_beats(signal, fs, peak_idx)
    if len(beats) < 2:
        return np.empty(0), np.empty(0)
    template = np.median(beats, axis=0)
    template = template - template.mean()
    t_std = float(np.std(template))
    if t_std == 0:
        return template, np.zeros(len(beats))
    corrs = []
    for beat in beats:
        b = beat - beat.mean()
        b_std = float(np.std(b))
        if b_std == 0:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(b, template)[0, 1]))
    return template, np.asarray(corrs)


def signal_quality_index(
    signal: np.ndarray, fs: float, peak_idx: np.ndarray
) -> dict:
    """Overall SQI (0-1) plus its three components."""
    signal = np.asarray(signal, dtype=float)
    peaks = np.asarray(peak_idx, dtype=int)
    _, corrs = beat_template_correlations(signal, fs, peaks)
    if len(corrs):
        frac_good = float(np.mean(corrs > 0.8))
        mean_corr = float(np.mean(corrs))
    else:
        frac_good, mean_corr = 0.0, 0.0
    try:
        snr = snr_db(signal, fs)
    except Exception:
        snr = float("-inf")
    if np.isfinite(snr):
        snr_mapped = float(np.clip((snr + 5.0) / 25.0, 0.0, 1.0))
    else:
        snr_mapped = 0.0
    ibis = peak_intervals_ms(peaks / fs)
    if len(ibis) >= 3:
        median_ibi = float(np.median(ibis))
        regularity = float(
            np.mean(np.abs(ibis - median_ibi) <= 0.30 * median_ibi)
        )
    else:
        regularity = 0.0
    sqi = 0.5 * frac_good + 0.3 * snr_mapped + 0.2 * regularity
    return {
        "sqi": float(np.clip(sqi, 0.0, 1.0)),
        "frac_good_beats": frac_good,
        "mean_template_corr": mean_corr,
        "snr_db": float(snr) if np.isfinite(snr) else None,
        "snr_mapped": snr_mapped,
        "regularity": regularity,
        "n_beats": int(len(corrs)),
    }


def segment_quality(
    signal: np.ndarray,
    fs: float,
    peak_idx: np.ndarray,
    window_s: float = 10.0,
    threshold: float = 0.6,
) -> list[dict]:
    """Per-window SQI; windows below ``threshold`` are unreliable."""
    signal = np.asarray(signal, dtype=float)
    peaks = np.asarray(peak_idx, dtype=int)
    template, _ = beat_template_correlations(signal, fs, peaks)
    duration = len(signal) / fs
    n_windows = int(np.ceil(duration / window_s))
    segments = []
    for w in range(n_windows):
        t0, t1 = w * window_s, min((w + 1) * window_s, duration)
        i0, i1 = int(t0 * fs), int(t1 * fs)
        seg = signal[i0:i1]
        seg_peaks = peaks[(peaks >= i0) & (peaks < i1)] - i0
        if len(seg_peaks) >= 2 and len(template):
            beats = extract_beats(seg, fs, seg_peaks)
            corrs = []
            for beat in beats:
                b = beat - beat.mean()
                t = template - template.mean()
                if np.std(b) > 0 and np.std(t) > 0:
                    corrs.append(float(np.corrcoef(b, t)[0, 1]))
            mean_corr = float(np.mean(corrs)) if corrs else 0.0
        else:
            mean_corr = 0.0
        try:
            seg_snr = snr_db(seg, fs) if len(seg) >= int(fs) else float("-inf")
        except Exception:
            seg_snr = float("-inf")
        snr_mapped = (
            float(np.clip((seg_snr + 5.0) / 25.0, 0.0, 1.0))
            if np.isfinite(seg_snr)
            else 0.0
        )
        sqi = 0.6 * float(np.clip(mean_corr, 0.0, 1.0)) + 0.4 * snr_mapped
        segments.append(
            {
                "start_s": float(t0),
                "end_s": float(t1),
                "sqi": float(np.clip(sqi, 0.0, 1.0)),
                "mean_template_corr": mean_corr,
                "snr_db": float(seg_snr) if np.isfinite(seg_snr) else None,
                "n_peaks": int(len(seg_peaks)),
                "reliable": bool(sqi >= threshold),
            }
        )
    return segments
