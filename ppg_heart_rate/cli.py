"""Command-line interface for ppg-heart-rate.

Subcommands
-----------
simulate : generate a synthetic PPG recording (CSV, with ground truth).
analyze  : detect peaks and compute BPM / HRV / SQI for one recording.
batch    : analyse every CSV in a directory and write a summary CSV.
report   : analyse one recording and write a Markdown report.

Run ``python -m ppg_heart_rate --help`` for usage.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .analysis import analyze_signal
from .io import load_ppg_csv, save_peaks_csv, save_ppg_csv, save_summary_json
from .peaks import detect_peaks_elgendi, evaluate_detection
from .plotting import plot_analysis
from .report import write_report
from .synthesis import CONDITIONS, synthesize


def _add_common_analyze(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", "-o", default="output",
                        help="directory for outputs (default: output)")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip the PNG plot")


def _run_analyze(input_path: Path, output_dir: Path, make_plot: bool) -> dict:
    """Analyse one file; write JSON + peaks CSV (+ PNG) into output_dir."""
    rec = load_ppg_csv(input_path)
    summary = analyze_signal(rec["signal"], rec["fs"])
    summary["source"] = input_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    save_summary_json(output_dir / f"{stem}_summary.json", summary)
    peak_times = np.asarray(summary["peak_times_s"])
    ibis = np.diff(peak_times) * 1000.0 if len(peak_times) > 1 else np.array([])
    save_peaks_csv(output_dir / f"{stem}_peaks.csv", peak_times, ibis)
    if make_plot:
        peaks_idx = (peak_times * rec["fs"]).astype(int)
        plot_analysis(rec["time"], rec["signal"], rec["fs"], peaks_idx,
                      output_dir / f"{stem}_plot.png",
                      title=f"PPG analysis — {input_path.name}")
    if rec["true_peaks"] is not None:
        score = evaluate_detection(peak_times, rec["true_peaks"])
        summary["detection_vs_truth"] = score
        save_summary_json(output_dir / f"{stem}_summary.json", summary)
    return summary


def _print_summary(summary: dict) -> None:
    hrv = summary["hrv"]
    print(f"beats detected : {summary['n_peaks']}")
    print(f"BPM mean       : {summary['bpm_mean']:.2f}")
    print(f"BPM median     : {summary['bpm_median']:.2f}")
    print(f"BPM spectral   : {summary['bpm_spectral']:.2f}")
    print(f"SDNN           : {hrv['sdnn_ms']:.2f} ms")
    print(f"RMSSD          : {hrv['rmssd_ms']:.2f} ms")
    print(f"pNN50          : {hrv['pnn50_pct']:.2f} %")
    print(f"SQI            : {summary['sqi']['sqi']:.3f}")
    print(f"unreliable seg.: {summary['unreliable_segments']}"
          f" / {summary['n_segments']}")
    if "detection_vs_truth" in summary:
        d = summary["detection_vs_truth"]
        print(f"detection      : sensitivity {d['sensitivity']:.3f}, "
              f"PPV {d['ppv']:.3f}, F1 {d['f1']:.3f}, "
              f"timing MAE {d['mean_abs_error_s']*1000:.1f} ms")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ppg-heart-rate",
        description="Synthesise and analyse PPG signals: heart rate, "
                    "HRV and signal quality.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("simulate", help="generate a synthetic recording")
    p_sim.add_argument("--condition", default="rest",
                       choices=sorted(CONDITIONS),
                       help="recording condition (default: rest)")
    p_sim.add_argument("--duration", type=float, default=60.0)
    p_sim.add_argument("--bpm", type=float, default=None,
                       help="override the condition's nominal BPM")
    p_sim.add_argument("--fs", type=float, default=100.0)
    p_sim.add_argument("--seed", type=int, default=0)
    p_sim.add_argument("--output", "-o", default="ppg.csv")

    p_an = sub.add_parser("analyze", help="analyse one recording")
    p_an.add_argument("--input", "-i", required=True)
    _add_common_analyze(p_an)

    p_ba = sub.add_parser("batch", help="analyse all CSVs in a directory")
    p_ba.add_argument("--input-dir", "-i", required=True)
    p_ba.add_argument("--output", "-o", default="batch_summary.csv",
                      help="summary CSV path")
    p_ba.add_argument("--plots-dir", default=None,
                      help="optional directory for per-file PNG plots")

    p_rep = sub.add_parser("report", help="write a Markdown report")
    p_rep.add_argument("--input", "-i", required=True)
    p_rep.add_argument("--output", "-o", default="report.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "simulate":
        rec = synthesize(condition=args.condition, duration=args.duration,
                         fs=args.fs, bpm=args.bpm, seed=args.seed)
        save_ppg_csv(args.output, rec.time, rec.signal, rec.true_peaks)
        print(f"wrote {args.output}: {len(rec.time)} samples, "
              f"{len(rec.true_peaks)} true beats, "
              f"nominal {rec.true_bpm:.0f} BPM ({args.condition})")
        return 0

    if args.command == "analyze":
        summary = _run_analyze(Path(args.input), Path(args.output_dir),
                               make_plot=not args.no_plot)
        _print_summary(summary)
        print(f"outputs in {args.output_dir}/")
        return 0

    if args.command == "batch":
        input_dir = Path(args.input_dir)
        files = sorted(input_dir.glob("*.csv"))
        if not files:
            print(f"no CSV files in {input_dir}", file=sys.stderr)
            return 1
        rows = []
        for f in files:
            try:
                summary = _run_analyze(
                    f, Path(args.output).parent / f.stem,
                    make_plot=args.plots_dir is not None,
                )
                if args.plots_dir:
                    pass  # plots already written next to the summary
                row = {
                    "file": f.name,
                    "duration_s": round(summary["duration_s"], 1),
                    "n_peaks": summary["n_peaks"],
                    "bpm_mean": round(summary["bpm_mean"], 2),
                    "bpm_median": round(summary["bpm_median"], 2),
                    "bpm_spectral": round(summary["bpm_spectral"], 2),
                    "sdnn_ms": round(summary["hrv"]["sdnn_ms"], 2),
                    "rmssd_ms": round(summary["hrv"]["rmssd_ms"], 2),
                    "pnn50_pct": round(summary["hrv"]["pnn50_pct"], 2),
                    "sqi": round(summary["sqi"]["sqi"], 3),
                    "unreliable_segments": summary["unreliable_segments"],
                    "status": "ok",
                }
            except Exception as exc:  # keep the batch going, record failure
                row = {"file": f.name, "status": f"error: {exc}"}
            rows.append(row)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = ["file", "duration_s", "n_peaks", "bpm_mean", "bpm_median",
                  "bpm_spectral", "sdnn_ms", "rmssd_ms", "pnn50_pct", "sqi",
                  "unreliable_segments", "status"]
        with open(out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        ok = sum(1 for r in rows if r["status"] == "ok")
        print(f"batch: {ok}/{len(rows)} files analysed -> {out}")
        return 0

    if args.command == "report":
        rec = load_ppg_csv(args.input)
        summary = analyze_signal(rec["signal"], rec["fs"])
        write_report(args.output, summary, Path(args.input).name)
        _print_summary(summary)
        print(f"report written to {args.output}")
        return 0

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
