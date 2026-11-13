#!/usr/bin/env bash
# Regenerate every example artefact in this directory from scratch.
# Run from the repository root:  bash examples/run_examples.sh
# Requires: numpy, scipy, matplotlib (see requirements.txt).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:$(cd .. && pwd)"

echo "== simulate: rest (60 BPM) and exercise (130 BPM), 60 s each =="
python3 -m ppg_heart_rate simulate --condition rest \
    --duration 60 --seed 7 --output rest_60bpm.csv
python3 -m ppg_heart_rate simulate --condition exercise \
    --duration 60 --seed 7 --output exercise_130bpm.csv

echo "== analyze both recordings =="
python3 -m ppg_heart_rate analyze --input rest_60bpm.csv --output-dir .
python3 -m ppg_heart_rate analyze --input exercise_130bpm.csv --output-dir .

echo "== Markdown report for the rest recording =="
python3 -m ppg_heart_rate report --input rest_60bpm.csv \
    --output rest_60bpm_report.md

echo "== batch summary over both recordings =="
BATCH_DIR="$(mktemp -d)"
cp rest_60bpm.csv exercise_130bpm.csv "$BATCH_DIR/"
python3 -m ppg_heart_rate batch --input-dir "$BATCH_DIR" \
    --output "$BATCH_DIR/batch_summary.csv"
cp "$BATCH_DIR/batch_summary.csv" .
rm -rf "$BATCH_DIR"

echo "== done: see rest_60bpm_summary.json, exercise_130bpm_summary.json,"
echo "   *_plot.png, rest_60bpm_report.md and batch_summary.csv =="
