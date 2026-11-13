# PPG Heart-Rate Report — rest_60bpm.csv

## Recording

- Duration: 60.0 s (6000 samples @ 100.0 Hz)
- Detected beats: 61 (60 NN intervals after correction; 0 removed out of range, 0 removed as ectopic/deviant)

## Heart rate

- BPM (mean interval): **60.34**
- BPM (median interval): **60.30**
- BPM (spectral, Welch PSD peak): **61.52**

## Heart-rate variability (Task Force 1996, time domain)

- Mean NN: 994.3 ms
- SDNN: 25.93 ms
- RMSSD: 31.92 ms
- NN50: 6  |  pNN50: 10.2%
- Coefficient of variation: 2.6%

## Signal quality

- SQI (0-1): **0.84**
- Beats matching the median template (r > 0.8): 100.0%
- Mean template correlation: 0.99
- SNR (0.5-8 Hz vs out-of-band): 6.3 dB
- Interval regularity: 100.0%
- Unreliable segments: 0 of 6 (10 s windows, SQI < 0.6)

## Method

Bandpass 0.5-8 Hz (zero-phase Butterworth) -> Elgendi-style peak detection (two moving averages on the squared signal, adaptive threshold) -> interval correction (300-2000 ms range, 20% median deviation rule) -> BPM/HRV per Task Force (1996). See `docs/RESEARCH.md` for sources and limitations.

_This report is computed from the supplied recording only; it is not a medical diagnosis._
