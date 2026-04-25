"""
sweep_weights.py

Sweeps a grid of (sasrec, audio, lyric) fusion weights for weighted_sum
and prints a ranked summary table.

Usage
-----
# Dry run — just print the combos that would be evaluated:
    python sweep_weights.py --dry_run

# Full sweep (submits a subprocess per combo, waits for each to finish):
    python sweep_weights.py

# Custom grid as JSON (overrides the built-in grid):
    python sweep_weights.py --grid='[[1,0,0],[2,1,1],[1,2,1]]'

# Point at a different eval script or results dir:
    python sweep_weights.py --eval_script=zeroshot_audio_lyric.py --results_dir=results
"""

import os
import re
import sys
import json
import subprocess
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
#  Default weight grid                                                 #
#  Each entry is [sasrec_w, audio_w, lyric_w].                        #
#  Values are raw importance scores — the eval script normalises them. #
# ------------------------------------------------------------------ #
DEFAULT_GRID = [
    # ---- ablations: single modality only ----
    [1, 0, 0],   # SASRec only  (CF baseline)
    [0, 1, 0],   # audio only
    [0, 0, 1],   # lyric only
    # ---- equal weights ----
    [1, 1, 1],
    # ---- SASRec dominant ----
    [2, 1, 1],
    [3, 1, 1],
    [4, 1, 1],
    # ---- audio dominant ----
    [1, 2, 1],
    [1, 3, 1],
    # ---- lyric dominant ----
    [1, 1, 2],
    [1, 1, 3],
    # ---- audio + lyric, no CF ----
    [0, 1, 1],
    [0, 2, 1],
    [0, 1, 2],
]

# Metric used for sorting the summary table
SORT_METRIC = "NDCG@10"

# Map "NDCG@10" → column name in the results txt ("NDCG") + row ("Top-10")
METRIC_COL_MAP = {
    "Recall@5":  ("Recall", "Top-5"),
    "Recall@10": ("Recall", "Top-10"),
    "Recall@20": ("Recall", "Top-20"),
    "NDCG@5":    ("NDCG",   "Top-5"),
    "NDCG@10":   ("NDCG",   "Top-10"),
    "NDCG@20":   ("NDCG",   "Top-20"),
}


def parse_results_file(path: str) -> dict:
    """Extract the metric table from a zeroshot_*.txt result file."""
    text = Path(path).read_text()

    # The table looks like:
    #          Precision  Recall   MRR   MAP  NDCG
    # Top-1        0.006   0.006 0.006 0.006 0.006
    # ...
    header_re = re.compile(r"Precision\s+Recall\s+MRR\s+MAP\s+NDCG")
    row_re    = re.compile(
        r"(Top-\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    )

    if not header_re.search(text):
        return {}

    rows = {}
    for m in row_re.finditer(text):
        label, prec, rec, mrr, mapp, ndcg = m.groups()
        rows[label] = {
            "Precision": float(prec),
            "Recall":    float(rec),
            "MRR":       float(mrr),
            "MAP":       float(mapp),
            "NDCG":      float(ndcg),
        }
    return rows


def weight_tag(weights):
    return ",".join(str(w) for w in weights)


def result_filename(weights, results_dir):
    """Reconstruct the filename the eval script would write."""
    norm = [w / sum(weights) for w in weights]
    tag = "-".join(f"{w:.1f}" for w in norm)          # e.g. 0.5-0.25-0.25
    # The eval script uses parsed_weights (already floats) for the tag:
    tag = "-".join(str(float(w)) for w in weights)    # simpler: raw floats
    fname = f"zeroshot_SASRec_audio_lyric_weighted_sum_{tag}.txt"
    return os.path.join(results_dir, fname)


def run_combo(weights, eval_script, python_bin, extra_args, dry_run):
    """Run the eval script for one weight combo."""
    w_str = ",".join(str(w) for w in weights)
    cmd = [
        python_bin, eval_script,
        "--fusion_strategy=weighted_sum",
        f"--fusion_weights={w_str}",
    ] + extra_args

    print(f"\n{'='*60}")
    print(f"  Running weights [{w_str}]")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print("  [dry_run] Skipping execution.")
        return 0

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  WARNING: process exited with code {result.returncode}")
    return result.returncode


def build_summary(grid, results_dir):
    """Parse all result files and return a ranked DataFrame."""
    records = []
    for weights in grid:
        path = result_filename(weights, results_dir)
        if not os.path.exists(path):
            print(f"  [missing] {path}")
            continue
        rows = parse_results_file(path)
        if not rows:
            print(f"  [empty]   {path}")
            continue

        rec = {"weights": weight_tag(weights)}
        for metric_name, (col, row_label) in METRIC_COL_MAP.items():
            rec[metric_name] = rows.get(row_label, {}).get(col, float("nan"))
        records.append(rec)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values(SORT_METRIC, ascending=False).reset_index(drop=True)
    df.index += 1   # rank from 1
    df.index.name = "rank"
    return df


def main():
    parser = argparse.ArgumentParser(description="Sweep fusion weights for weighted_sum")
    parser.add_argument("--eval_script",  default="zeroshot_audio_lyric.py")
    parser.add_argument("--results_dir",  default="results")
    parser.add_argument("--python_bin",   default=sys.executable,
                        help="Full path to the Python interpreter to use")
    parser.add_argument("--grid",         default=None,
                        help="JSON array of [sasrec,audio,lyric] weight lists")
    parser.add_argument("--dry_run",      action="store_true",
                        help="Print commands without running them")
    parser.add_argument("--summary_only", action="store_true",
                        help="Skip running evals, just parse existing result files")
    parser.add_argument("--extra_args",   default="",
                        help="Extra args forwarded verbatim to the eval script")
    args = parser.parse_args()

    grid = json.loads(args.grid) if args.grid else DEFAULT_GRID
    extra = args.extra_args.split() if args.extra_args.strip() else []

    print(f"\n{'='*60}")
    print(f"  Weight sweep — {len(grid)} combinations")
    print(f"  Eval script : {args.eval_script}")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Sort metric : {SORT_METRIC}")
    print(f"{'='*60}\n")

    if not args.summary_only:
        for weights in grid:
            run_combo(weights, args.eval_script, args.python_bin, extra, args.dry_run)

    if args.dry_run:
        print("\n[dry_run] Skipping summary (no result files written).")
        return

    print(f"\n\n{'='*60}")
    print(f"  SUMMARY — sorted by {SORT_METRIC} (descending)")
    print(f"{'='*60}\n")

    df = build_summary(grid, args.results_dir)
    if df.empty:
        print("No result files found to summarise. Run the sweep first.")
        return

    print(df.to_string(float_format=lambda x: f"{x:.4f}"))

    summary_path = os.path.join(args.results_dir, "weight_sweep_summary.csv")
    df.to_csv(summary_path)
    print(f"\nSummary saved to {summary_path}")

    best = df.iloc[0]
    print(f"\n  Best weights : [{best['weights']}]")
    print(f"  {SORT_METRIC}         : {best[SORT_METRIC]:.4f}")


if __name__ == "__main__":
    main()