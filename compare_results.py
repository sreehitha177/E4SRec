"""
compare_results.py
Reads all results/*.txt files and prints comparison tables:
  - Audio embeddings table  (CLAP, MERT, MUSIC2VEC, ENCODEC, MFCC)
  - Lyric embeddings table  (MINILM, BGEM3, MPNET, MULTILINGUAL, BERT)
  - Overall summary table   (all models)

Each table shows all Top-K rows per model, with * marking the best
value per column (per Top-K level across models in that group).
"""

import re
from pathlib import Path

RESULTS_DIR = Path("/home/snarayana_umass_edu/E4SRec-1/results")
TOPK        = [1, 5, 10, 20, 100]
METRICS     = ['Recall', 'NDCG', 'MRR', 'Precision']

AUDIO_MODELS  = ['CLAP', 'MERT', 'MUSIC2VEC', 'ENCODEC', 'MFCC']
LYRIC_MODELS  = ['MINILM', 'BGEM3', 'MPNET', 'MULTILINGUAL', 'BERT']

# ── Parse one result file ─────────────────────────────────────────────────
def parse_file(path):
    text = path.read_text()
    data = {}
    for metric in ['Precision', 'Recall', 'MRR', 'MAP', 'NDCG']:
        m = re.search(rf'^{metric}:\s*\[([^\]]+)\]', text, re.MULTILINE)
        if m:
            vals = [float(x) for x in m.group(1).split()]
            data[metric] = vals   # index: 0→@1, 1→@5, 2→@10, 3→@20, 4→@100
    return data

def model_key(path):
    name = path.stem
    if name.startswith("zeroshot_results_"):
        return "SASRec-only"
    return name.replace("zeroshot_", "")  # e.g. "CLAP"

# ── Load all results ──────────────────────────────────────────────────────
all_data = {}   # key → data dict
for f in RESULTS_DIR.glob("*.txt"):
    key  = model_key(f)
    data = parse_file(f)
    if data:
        all_data[key] = data

# ── Print one group table ─────────────────────────────────────────────────
def print_group(title, model_keys):
    col_w   = 9   # width per metric value
    name_w  = 16  # model name column
    topk_w  = 6   # Top-k label

    # Build column header
    metric_header = "".join(f"{m:>{col_w}}" for m in METRICS)
    sep_width = name_w + topk_w + len(METRICS) * col_w

    print(f"\n{'='*sep_width}")
    print(f"  {title}")
    print(f"  Baseline: SASRec-only  |  * = best in group per row")
    print(f"{'='*sep_width}")
    print(f"{'Model':<{name_w}}{'Top-K':<{topk_w}}" + metric_header)
    print(f"{'-'*sep_width}")

    # For each Top-K level, find the best value per metric across this group
    # (include SASRec-only as baseline for reference but mark * only within group)
    group_keys_with_baseline = ["SASRec-only"] + [k for k in model_keys if k in all_data]

    for ki, k in enumerate(TOPK):
        best = {}
        for m in METRICS:
            vals = [all_data[key][m][ki] for key in group_keys_with_baseline
                    if key in all_data and m in all_data[key]]
            best[m] = max(vals) if vals else float('nan')

        for i, key in enumerate(group_keys_with_baseline):
            if key not in all_data:
                continue
            data  = all_data[key]
            label = key if key != "SASRec-only" else "SASRec-only"
            topk_label = f"@{k}"

            row = f"{label:<{name_w}}{topk_label:<{topk_w}}"
            for m in METRICS:
                v    = data[m][ki] if m in data else float('nan')
                mark = "*" if abs(v - best[m]) < 1e-9 else " "
                row += f"{v:>{col_w-1}.4f}{mark}"
            print(row)

        # Separator between Top-K blocks
        if ki < len(TOPK) - 1:
            print(f"{'·'*sep_width}")

    print(f"{'='*sep_width}")

# ── Print overall summary at Top-10 ──────────────────────────────────────
def print_summary():
    ki      = TOPK.index(10)
    col_w   = 9
    name_w  = 18
    cat_w   = 8
    sep_width = name_w + cat_w + len(METRICS) * col_w

    metric_header = "".join(f"{m:>{col_w}}" for m in METRICS)

    ordered = (
        [("Baseline", "SASRec-only")] +
        [("Audio",    k) for k in AUDIO_MODELS if k in all_data] +
        [("Lyrics",   k) for k in LYRIC_MODELS if k in all_data]
    )

    all_vals = {m: [all_data[key][m][ki] for _, key in ordered
                    if key in all_data and m in all_data[key]]
                for m in METRICS}
    best = {m: max(v) for m, v in all_vals.items() if v}

    # Per-group best (Audio / Lyrics / Baseline)
    group_best = {}
    for cat in set(c for c, _ in ordered):
        keys_in_group = [k for c, k in ordered if c == cat and k in all_data]
        group_best[cat] = {
            m: max(all_data[k][m][ki] for k in keys_in_group if m in all_data[k])
            for m in METRICS
        }

    print(f"\n{'='*sep_width}")
    print(f"  OVERALL SUMMARY @ Top-10  (* = best overall, + = best in group)")
    print(f"{'='*sep_width}")
    print(f"{'Model':<{name_w}}{'Type':<{cat_w}}" + metric_header)
    print(f"{'-'*sep_width}")

    prev_cat = None
    for cat, key in ordered:
        if key not in all_data:
            continue
        if cat != prev_cat and prev_cat is not None:
            print(f"{'·'*sep_width}")
        prev_cat = cat
        data = all_data[key]
        row  = f"{key:<{name_w}}{cat:<{cat_w}}"
        for m in METRICS:
            v    = data[m][ki] if m in data else float('nan')
            if abs(v - best.get(m, -1)) < 1e-9:
                mark = "*"
            elif abs(v - group_best.get(cat, {}).get(m, -1)) < 1e-9:
                mark = "*"
            else:
                mark = " "
            row += f"{v:>{col_w-1}.4f}{mark}"
        print(row)
    print(f"{'='*sep_width}")

# ── Run ───────────────────────────────────────────────────────────────────
import sys

OUT_FILE = RESULTS_DIR / "comparison_table.txt"

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

with open(OUT_FILE, "w") as f:
    sys.stdout = Tee(sys.__stdout__, f)
    print_group("AUDIO EMBEDDINGS  (SASRec + Audio model vs SASRec-only)",  AUDIO_MODELS)
    print_group("LYRIC EMBEDDINGS  (SASRec + Lyrics model vs SASRec-only)", LYRIC_MODELS)
    print_summary()
    sys.stdout = sys.__stdout__

print(f"\nResults saved to {OUT_FILE}")

