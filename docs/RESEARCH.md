# Research notes — PPG heart-rate estimation

How the physiology, the signal processing and the metrics in this project
map to the published literature. Every claim below is traceable to one of
the references at the end; nothing here is a measurement from a real
person — all signals analysed in this repo are synthetic (see §6).

## 1. What a PPG signal is

Photoplethysmography (PPG) is a simple, low-cost optical technique that
detects blood-volume changes in the microvascular bed of tissue: an LED
illuminates the skin and a photodiode measures the light transmitted or
reflected back. The waveform has a pulsatile ("AC") component synchronous
with each heartbeat, superimposed on a slowly varying ("DC") baseline
attributed to respiration, sympathetic activity and thermoregulation
(Allen 2007). PPG entered clinical use in the 1980s as pulse oximetry and
is now the sensing principle in most smartwatches and fitness trackers
(Charlton et al. 2023).

## 2. Anatomy of one pulse wave

A single PPG pulse (Elgendi 2012) consists of:

- a **systolic wave** — fast rise to the systolic peak as the arterial
  pulse arrives;
- a **dicrotic notch** — a small dip after the systolic peak, linked to
  aortic-valve closure and reflected pressure waves;
- a **diastolic wave** — a smaller, wider second peak.

The second derivative of the PPG (acceleration plethysmogram, APG) turns
these bends into the characteristic *a, b, c, d, e* waves used in vascular
ageing studies (Elgendi 2012). This project synthesises each beat as a
tall narrow Gaussian (systolic) plus a smaller wider Gaussian (diastolic,
38% amplitude); the dip between them is the dicrotic notch. Beat positions
and widths scale with the inter-beat interval so the shape stays
physiological from 40 to 180 BPM.

## 3. Heart-rate variability and the Task Force (1996) standard

The Task Force of the European Society of Cardiology and the North
American Society of Pacing and Electrophysiology (1996) standardised HRV
measurement. The time-domain metrics used here are defined exactly as in
that report:

| Metric | Definition (Task Force 1996) |
| ------ | ---------------------------- |
| SDNN | standard deviation of NN intervals (square root of variance) |
| RMSSD | square root of the mean of squared successive differences |
| NN50 / pNN50 | count / percentage of successive NN differences > 50 ms |

"NN" means normal-to-normal: intervals between normal beats, after
ectopic/missed beats are removed. This project applies the usual
correction — drop intervals outside 300–2000 ms (30–200 BPM) and
intervals deviating more than 20% from the median — before computing
these metrics. The Task Force recommends 250–500 Hz sampling for precise
R-wave timing and at least 5-minute recordings for short-term HRV; the
synthetic recordings here are 60 s at 100 Hz, which is adequate for
demonstrating the pipeline but **not** a substitute for a 5-minute
clinical recording — see §7.

## 4. Peak detection: Elgendi-style (two moving averages)

Accurate systolic-peak detection is the first step of HRV analysis from
PPG (Elgendi et al. 2013). The method proposed in that paper has three
stages — bandpass pre-processing, feature extraction with two
event-related moving averages, and threshold classification — and reached
99.84% sensitivity / 99.89% positive predictivity on 40 heat-stressed
recordings (5,071 beats). The pipeline in this project follows the same
structure, adapted to the raw PPG waveform:

1. **Bandpass 0.5–8 Hz** (zero-phase Butterworth, `filtfilt`-equivalent)
   removes baseline wander and high-frequency noise without shifting peak
   times; squaring the filtered signal emphasises the tall systolic peaks
   over the smaller diastolic wave.
2. **Two moving averages** of the squared signal: a fast one (~0.12 s,
   tracks pulse energy) and a slow one (~0.6 s, tracks the local energy
   level). A block of interest opens where the fast average exceeds the
   slow average by a relative offset (default 50%).
3. **Classification**: the systolic peak is the maximum of the filtered
   signal inside each block; a 0.3 s refractory period (max 200 BPM) and
   an adaptive amplitude gate (30% of the median peak amplitude) remove
   duplicates and noise detections.

Honest difference from the paper: Elgendi et al. (2013) detect peaks in
the **acceleration** photoplethysmogram (second derivative) of
heat-stressed emergency responders; this implementation works on the raw
PPG, so its window lengths and offset were tuned on synthetic ground
truth rather than copied from the paper. The method is therefore
described here as "Elgendi-style", not as a reproduction. On the
synthetic conditions in this repo it scores sensitivity ≥ 0.985 and PPV
1.000 (see `examples/*_summary.json`, field `detection_vs_truth`).

## 5. Signal quality (SQI)

Motion artefact is the limiting factor for wearable PPG, especially
during exercise (Charlton et al. 2023). No single number captures
usability, so this project combines three interpretable components
(weights 0.5 / 0.3 / 0.2):

1. **Template agreement** — fraction of beats whose resampled shape
   correlates with the median beat template at r > 0.8;
2. **SNR** — in-band (0.5–8 Hz) vs out-of-band power, mapped from dB;
3. **Interval regularity** — fraction of IBIs within ±30% of the median.

The same idea is applied per 10-second window to flag unreliable
segments. Measured on synthetic data (seed 7, 60 s): clean rest SQI
0.835; heavy motion + noise 0.762; extreme motion + noise 0.504 with
5 of 6 windows flagged unreliable; pure heavy noise 0.480 with 6 of 6
flagged. The ordering (clean > degraded) is the property the tests
assert; the absolute numbers are specific to this synthetic generator.

## 6. How the synthetic signals are built

Each beat is two Gaussians (see §2). Beat timing carries HRV: a slow
respiratory modulation (~0.2 Hz, ±25 ms, respiratory sinus arrhythmia)
plus Gaussian jitter, and — for the `arrhythmia` condition — premature
beats shortened 25–40% followed by a compensatory pause. Degradations:
baseline wander (0.12 Hz + 0.27 Hz sinusoids), white noise, respiratory
amplitude modulation (±12%), and motion-artefact bursts (0.7–1.6 Hz
sinusoids with Hann envelopes). Presets: `rest` 60 BPM, `exercise`
130 BPM, `arrhythmia` 72 BPM, `motion` 75 BPM, `low-perfusion` 65 BPM
at 35% amplitude. Ground-truth systolic-peak times are stored in the
`true_peak` column of simulated CSVs.

## 7. Limitations

- **Synthetic only.** All validation is against synthetic ground truth.
  Real PPG has sensor, skin-tone, contact-pressure and perfusion effects
  this generator does not model; do not quote the detection scores as
  clinical accuracy.
- **Not a medical device.** BPM/HRV outputs here are for education and
  algorithm development, not diagnosis.
- **Short recordings.** Examples are 60 s; the Task Force (1996)
  recommends ≥ 5 minutes for short-term HRV.
- **100 Hz sampling.** Adequate for peak timing at ~10 ms resolution;
  the Task Force recommends 250–500 Hz for precise HRV work.
- **Pulse-rate variability ≠ HRV.** Intervals from PPG (pulse-rate
  variability, PRV) approximate ECG-derived HRV but differ under
  motion, arrhythmia and vascular disease; the metrics are labelled
  HRV here only because the pipeline follows the Task Force definitions
  on NN intervals.
- **Spectral BPM resolution.** Welch PSD with 2048-sample segments at
  100 Hz gives ~2.9 BPM bins; the spectral estimate is a cross-check,
  not the primary BPM.

## References

1. Allen, J. (2007). Photoplethysmography and its application in clinical
   physiological measurement. *Physiological Measurement*, 28(3), R1–R39.
   doi:10.1088/0967-3334/28/3/R01 — https://pubmed.ncbi.nlm.nih.gov/17322588/
2. Elgendi, M., Norton, I., Brearley, M., Abbott, D., & Schuurmans, D.
   (2013). Systolic peak detection in acceleration photoplethysmograms
   measured from emergency responders in tropical conditions. *PLoS ONE*,
   8(10), e76585. doi:10.1371/journal.pone.0076585 —
   https://pmc.ncbi.nlm.nih.gov/articles/PMC3805543/
3. Task Force of the European Society of Cardiology and the North American
   Society of Pacing and Electrophysiology (1996). Heart rate variability:
   standards of measurement, physiological interpretation, and clinical
   use. *European Heart Journal*, 17(3), 354–381 —
   https://www.academia.edu/57757907/Heart_Rate_Variability_Standards_of_Measurement_Physiological_Interpretation_and_Clinical_Use_Task_Force_of_The_European_Society_of_Cardiology_and_the_North_American_Society_for_Pacing_and_Electrophysiology
4. Elgendi, M. (2012). On the analysis of fingertip photoplethysmogram
   signals. *Current Cardiology Reviews*, 8(1), 14–25.
   doi:10.2174/157340312801215782 —
   https://pmc.ncbi.nlm.nih.gov/articles/PMC3394104/
5. Elgendi, M. (2012). Standard terminologies for photoplethysmogram
   signals. *Current Cardiology Reviews*, 8(3), 215–219.
   doi:10.2174/157340312803217184 —
   https://pmc.ncbi.nlm.nih.gov/articles/PMC3465827/
6. Charlton, P. H., Allen, J., Bailón, R., et al. (2023). The 2023 wearable
   photoplethysmography roadmap. *Physiological Measurement*, 44(11),
   111001. doi:10.1088/1361-6579/acead2 —
   https://pubmed.ncbi.nlm.nih.gov/37494945/
