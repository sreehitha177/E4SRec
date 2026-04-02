#!/usr/bin/env python3
"""
Annotate validation tracks for all composition features, then evaluate vs MGPHot.
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from composition_common import (
    COMPOSITION_FEATURES,
    COMPOSITION_FEATURE_INDICES,
    annotate_df_parallel,
)

_PKG_DIR = Path(__file__).resolve().parent
_METADATA_ROOT = _PKG_DIR.parent


def evaluate_annotations(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        idx = COMPOSITION_FEATURE_INDICES[feature]
        gt = df["gene_values"].apply(lambda x: x[idx])
        pred = df[feature]
        mask = pred.notna()
        gt, pred = gt[mask], pred[mask]
        if len(gt) < 10:
            continue
        mae = (gt - pred).abs().mean()
        rho, _ = spearmanr(gt, pred)
        rows.append(
            {
                "feature": feature,
                "n": len(gt),
                "MAE": round(mae, 4),
                "Spearman_rho": round(rho, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("Spearman_rho", ascending=False)


def print_bias(validation_with_preds: pd.DataFrame, features: list[str]) -> None:
    print(f"{'Feature':<35} {'GT mean':>8} {'Pred mean':>10} {'Bias':>8}")
    print("-" * 70)
    for feature in features:
        idx = COMPOSITION_FEATURE_INDICES[feature]
        gt = validation_with_preds["gene_values"].apply(lambda x: x[idx])
        pred = validation_with_preds[feature].dropna()
        bias = pred.mean() - gt.mean()
        print(f"{feature:<35} {gt.mean():>8.3f} {pred.mean():>10.3f} {bias:>8.3f}")


def plot_heatmap(eval_df: pd.DataFrame, out_dir: str) -> None:
    print(f"Average MAE: {eval_df['MAE'].mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    ax_rho, ax_mae = axes
    sns.heatmap(
        eval_df.set_index("feature")[["Spearman_rho"]],
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        ax=ax_rho,
    )
    ax_rho.set_title("Validation Spearman rho")

    sns.heatmap(
        eval_df.set_index("feature")[["MAE"]],
        annot=True,
        fmt=".3f",
        cmap="YlOrRd_r",
        vmin=0,
        vmax=max(0.5, float(eval_df["MAE"].max())),
        ax=ax_mae,
    )
    ax_mae.set_title("Validation MAE")

    plt.tight_layout()
    path = os.path.join(out_dir, "validation_composition_metrics_heatmap.png")
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()


def plot_distributions(validation_with_preds: pd.DataFrame, features: list[str], out_dir: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        idx = COMPOSITION_FEATURE_INDICES[feature]
        gt = validation_with_preds["gene_values"].apply(lambda x: x[idx])
        pred = validation_with_preds[feature].dropna()
        axes[i].hist(gt, bins=20, alpha=0.5, label="GT", density=True)
        axes[i].hist(pred, bins=20, alpha=0.5, label="Pred", density=True)
        axes[i].set_title(feature, fontsize=8)
        axes[i].legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, "validation_composition_distributions.png")
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-csv",
        default=str(_METADATA_ROOT / "shared" / "validation_df.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(_PKG_DIR / "output" / "validation_composition_annotations.csv"),
    )
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--subset-size", type=int, default=None, help="Optional limit for quick tests")
    parser.add_argument("--api-key", default=None, help="LLM API key (or set LLM_API_KEY)")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip LLM calls; load annotations from --output-csv and regenerate metrics + plots only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.plots_only:
        api_key = args.api_key or os.environ.get("LLM_API_KEY")
        if not api_key:
            raise ValueError("Provide --api-key or set LLM_API_KEY environment variable")

    validation_df = pd.read_csv(args.validation_csv)
    if args.subset_size:
        validation_df = validation_df.head(args.subset_size).copy()
    validation_df["gene_values"] = validation_df["gene_values"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    if args.plots_only:
        ann_path = Path(args.output_csv)
        if not ann_path.is_file():
            raise FileNotFoundError(f"--plots-only requires existing annotations at {ann_path}")
        annotations = pd.read_csv(ann_path)
        print(f"Loaded annotations from {ann_path} ({len(annotations)} rows)")
    else:
        tracks = validation_df[["artist_name", "track_name"]].drop_duplicates().reset_index(drop=True)
        print(f"Validation tracks: {len(tracks)}")

        annotations = annotate_df_parallel(
            tracks,
            args.api_key or os.environ.get("LLM_API_KEY"),
            selected_features=COMPOSITION_FEATURES,
            max_workers=args.max_workers,
        )
        Path(args.output_csv).resolve().parent.mkdir(parents=True, exist_ok=True)
        annotations.to_csv(args.output_csv, index=False)
        print(f"Saved annotations: {args.output_csv}")

    validation_with_preds = validation_df.merge(
        annotations,
        on=["artist_name", "track_name"],
        how="inner",
    )
    print(f"Matched {len(validation_with_preds)} / {len(validation_df)} validation tracks")

    eval_results = evaluate_annotations(validation_with_preds, COMPOSITION_FEATURES)
    print("\n=== Evaluation Results ===")
    print(eval_results.to_string(index=False))
    print("\n=== MAE Per Feature ===")
    print(eval_results[["feature", "MAE"]].sort_values("MAE").to_string(index=False))

    print("\n=== Bias Check ===")
    print_bias(validation_with_preds, COMPOSITION_FEATURES)

    plots_dir = _PKG_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap(eval_results, out_dir=str(plots_dir))
    plot_distributions(validation_with_preds, COMPOSITION_FEATURES, out_dir=str(plots_dir))


if __name__ == "__main__":
    main()
