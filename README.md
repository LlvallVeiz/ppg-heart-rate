# ppg-heart-rate

Estimate heart rate, heart-rate variability (HRV) and signal quality from
photoplethysmogram (PPG) signals — with a realistic synthetic PPG
generator so every result can be scored against known ground truth.

- **Synthesis**: realistic PPG pulses (systolic peak + dicrotic notch +
  diastolic peak), HRV, baseline wander, noise and motion artefacts, in
  five conditions: rest, exercise, arrhythmia, motion, low perfusion.
- **Processing**: zero-phase bandpass 0.5–8 Hz, Elgendi-style systolic-peak
  detection (two moving averages + adaptive threshold), interval
  correction to NN intervals.
- **Estimation**: BPM three ways (mean interval, median interval,
  spectral), HRV metrics per Task Force (1996) — SDNN, RMSSD, pNN50 —
  plus a signal quality index (SQI) and unreliable-segment flagging.
- **CLI**: `simulate`, `analyze`, `batch`, `report` — CSV/JSON output and
  PNG plots.

## Background

Photoplethysmography (PPG) is an optical technique that detects
blood-volume changes in tissue: an LED shines into the skin and a
photodiode records the returning light, which rises and falls with each
heartbeat (Allen 2007). One pulse has a fast rise to the **systolic
peak**, a small dip — the **dicrotic notch**, linked to aortic-valve
closure — and a smaller **diastolic peak** (Elgendi 2012). Because the
timing between successive systolic peaks tracks the heartbeat, PPG gives
heart rate and pulse-rate variability, the optical analogue of HRV
(Task Force 1996; Charlton et al. 2023). Motion artefact is the main
limiting factor for wearables, especially during exercise.

The full research notes — waveform model, the Elgendi-style detector,
the Task Force metric definitions, SQI design and limitations — are in
[`docs/RESEARCH.md`](docs/RESEARCH.md).

## Project structure

```text
ppg-heart-rate/
├── ppg_heart_rate/
│   ├── synthesis.py   # realistic synthetic PPG + ground-truth peaks
│   ├── filtering.py   # bandpass, moving average, SNR
│   ├── peaks.py       # Elgendi-style detection + scoring vs truth
│   ├── intervals.py   # IBI computation + NN correction
│   ├── metrics.py     # BPM (mean/median/spectral), HRV (SDNN/RMSSD/pNN50)
│   ├── quality.py     # SQI + per-segment quality
│   ├── analysis.py    # end-to-end pipeline
│   ├── io.py          # CSV / JSON load & save
│   ├── report.py      # Markdown report
│   ├── plotting.py    # Matplotlib plots
│   └── cli.py         # simulate / analyze / batch / report
├── tests/             # 81 pytest tests (hand-computed values included)
├── examples/          # real outputs from actual runs (see below)
├── docs/RESEARCH.md   # literature notes + references
├── requirements.txt
└── pyproject.toml
```

## Installation

```bash
pip install -r requirements.txt   # numpy, scipy, matplotlib, pytest
```

No installation step is needed to run it: `python -m ppg_heart_rate ...`
works from the repository root.

## Usage

```bash
# 1. Simulate a 60 s resting recording (with ground-truth peaks)
python -m ppg_heart_rate simulate --condition rest --duration 60 \
    --seed 7 --output rest.csv

# 2. Analyse one recording -> summary JSON + peaks CSV + PNG plot
python -m ppg_heart_rate analyze --input rest.csv --output-dir out/

# 3. Analyse every CSV in a directory -> one summary CSV
python -m ppg_heart_rate batch --input-dir recordings/ --output summary.csv

# 4. Markdown report for one recording
python -m ppg_heart_rate report --input rest.csv --output report.md
```

Recording CSV format: `time_s,ppg[,true_peak]` — the optional `true_peak`
column (0/1) is written by `simulate` and, when present, `analyze`
scores the detection against it in the summary JSON
(`detection_vs_truth`).

## Example results (real runs)

Regenerate everything with `bash examples/run_examples.sh`. Both examples
are 60 s at 100 Hz, seed 7.

### Rest — 60 BPM nominal (`examples/rest_60bpm.*`)

| Quantity | Measured | Ground truth |
| -------- | -------- | ------------ |
| Beats detected | 61 | 61 |
| BPM (mean / median / spectral) | 60.34 / 60.30 / 61.52 | 60 |
| SDNN / RMSSD / pNN50 | 25.93 ms / 31.92 ms / 10.17% | — |
| SQI (0–1) | 0.835 (0/6 segments unreliable) | — |
| Detection | sensitivity 1.000, PPV 1.000, timing MAE 2.6 ms | — |

### Exercise — 130 BPM nominal (`examples/exercise_130bpm.*`)

| Quantity | Measured | Ground truth |
| -------- | -------- | ------------ |
| Beats detected | 129 | 131 |
| BPM (mean / median / spectral) | 130.64 / 130.43 / 131.84 | 130 |
| SDNN / RMSSD / pNN50 | 21.20 ms / 18.88 ms / 1.59% | — |
| SQI (0–1) | 0.782 (0/6 segments unreliable) | — |
| Detection | sensitivity 0.985, PPV 1.000, timing MAE 3.5 ms | — |

Batch summary over both recordings is in
[`examples/batch_summary.csv`](examples/batch_summary.csv). SQI ordering
on degraded data (separate check, seed 3): clean rest 0.835 → heavy
motion + noise 0.762 → extreme motion + noise 0.504 (5/6 windows
flagged unreliable).

## Tests

```bash
python -m pytest tests/ -q
# 81 passed
```

The suite includes hand-computed values (e.g. RMSSD = 31.09 ms and
SDNN = 17.08 ms for intervals [1000, 1020, 980, 1010] ms), detection
scoring against synthetic ground truth for all five conditions, and an
independent cross-check: the SciPy `find_peaks` detector must agree with
the Elgendi-style detector, and interval-based BPM must agree with
spectral BPM.

## Design decisions

- **Zero-phase filtering** (`sosfiltfilt`): peak times must not shift,
  because HRV metrics are computed from peak timing.
- **Elgendi-style, not a copy**: the paper detects peaks in the second
  derivative (APG) of heat-stressed recordings; this implementation
  works on the raw PPG, so window lengths and the offset were tuned on
  synthetic ground truth (see `docs/RESEARCH.md` §4).
- **Three BPM estimates**: mean-interval, median-interval (robust to a
  few missed/extra beats) and spectral (Welch PSD peak). Agreement
  between them is a built-in sanity check.
- **SQI as three components** (template agreement / SNR / interval
  regularity) rather than one opaque number, so a low score is
  diagnosable.
- **100 Hz default sampling**: adequate for ~10 ms peak timing; the Task
  Force recommends 250–500 Hz for precise HRV work.

## Limitations

- **Synthetic only** — all validation is against synthetic ground truth.
  Real PPG adds sensor, skin-tone, contact-pressure and perfusion
  effects this generator does not model; detection scores here are not
  clinical accuracy claims.
- **Not a medical device** — outputs are for education and algorithm
  development, not diagnosis.
- **Short recordings** — examples are 60 s; the Task Force recommends
  ≥ 5 minutes for short-term HRV.
- **Pulse-rate variability ≠ HRV** — PPG-derived intervals approximate
  ECG-derived HRV but differ under motion, arrhythmia and vascular
  disease.

## References

1. Allen, J. (2007). *Physiological Measurement*, 28(3), R1–R39.
   doi:10.1088/0967-3334/28/3/R01
2. Elgendi, M., Norton, I., Brearley, M., Abbott, D., & Schuurmans, D.
   (2013). *PLoS ONE*, 8(10), e76585. doi:10.1371/journal.pone.0076585
3. Task Force of the ESC and NASPE (1996). *European Heart Journal*,
   17(3), 354–381.
4. Elgendi, M. (2012). *Current Cardiology Reviews*, 8(1), 14–25.
5. Elgendi, M. (2012). *Current Cardiology Reviews*, 8(3), 215–219.
6. Charlton, P. H., et al. (2023). *Physiological Measurement*, 44(11),
   111001. doi:10.1088/1361-6579/acead2

## License

MIT — see [LICENSE](LICENSE).
