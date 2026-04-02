#!/usr/bin/env python3
"""
Annotate selected composition features for all songs in ordered list.

Subcommands:
  - annotate: annotate one shard (for SLURM job arrays)
  - merge: combine shard outputs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from composition_common import (
    COMPOSITION_FEATURES,
    annotate_df_parallel,
    load_song_list,
    normalize_feature_subset,
)

_PKG_DIR = Path(__file__).resolve().parent
_METADATA_ROOT = _PKG_DIR.parent
_PROJECT_ROOT = _METADATA_ROOT.parent


def shard_path(output_csv: str, rank: int) -> str:
    p = Path(output_csv)
    return str(p.parent / f"{p.stem}_shard_{rank}{p.suffix}")


def parse_feature_args(raw_features: list[str] | None) -> list[str]:
    if not raw_features:
        return COMPOSITION_FEATURES.copy()

    if len(raw_features) == 1 and raw_features[0].lower().strip() == "all":
        return COMPOSITION_FEATURES.copy()

    flattened = []
    for token in raw_features:
        flattened.extend(part.strip() for part in token.split(","))
    return normalize_feature_subset(flattened)


def load_seed_annotations(seed_csv: str | None, selected_features: list[str]) -> pd.DataFrame:
    if not seed_csv:
        return pd.DataFrame(columns=["artist_name", "track_name", *selected_features])
    if not os.path.exists(seed_csv):
        raise FileNotFoundError(f"Seed annotations file not found: {seed_csv}")

    seed_df = pd.read_csv(seed_csv)
    required_cols = {"artist_name", "track_name"}
    missing_keys = required_cols - set(seed_df.columns)
    if missing_keys:
        raise ValueError(f"Seed annotations missing required key columns: {sorted(missing_keys)}")

    out = seed_df[["artist_name", "track_name"]].copy()
    for feature in selected_features:
        out[feature] = seed_df[feature] if feature in seed_df.columns else pd.NA
    return out.drop_duplicates(subset=["artist_name", "track_name"]).reset_index(drop=True)


def cmd_annotate(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.environ.get("LLM_API_KEY")
    if not api_key:
        raise ValueError("Provide --api-key or set LLM_API_KEY environment variable")

    selected_features = parse_feature_args(args.features)
    print(f"Selected features ({len(selected_features)}): {selected_features}")
    seed_annotations = load_seed_annotations(args.seed_annotations_csv, selected_features)
    if not seed_annotations.empty:
        print(f"Seed annotations loaded: {len(seed_annotations)} tracks (will be skipped in full generation)")

    node_rank = args.node_rank
    num_nodes = args.num_nodes
    if node_rank is None:
        node_rank = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if num_nodes is None:
        num_nodes = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    print(f"Node {node_rank}/{num_nodes}")

    unique_tracks = load_song_list(args.song_list_path)
    if not seed_annotations.empty:
        unique_tracks = unique_tracks.merge(
            seed_annotations[["artist_name", "track_name"]],
            on=["artist_name", "track_name"],
            how="left",
            indicator=True,
        )
        unique_tracks = unique_tracks[unique_tracks["_merge"] == "left_only"][
            ["artist_name", "track_name"]
        ].reset_index(drop=True)
    total = len(unique_tracks)
    print(f"Total unique tracks to generate: {total}")

    shard_size = (total + num_nodes - 1) // num_nodes
    start = node_rank * shard_size
    end = min(start + shard_size, total)
    shard = unique_tracks.iloc[start:end].copy()
    print(f"Shard [{start}:{end}] - {len(shard)} tracks")

    if shard.empty:
        print("Empty shard, nothing to do.")
        return

    annotations = annotate_df_parallel(
        shard,
        api_key,
        selected_features=selected_features,
        max_workers=args.max_workers,
    )

    out = shard_path(args.output_csv, node_rank)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    annotations.to_csv(out, index=False)
    print(f"Saved {len(annotations)} tracks to {out}")


def cmd_merge(args: argparse.Namespace) -> None:
    selected_features = parse_feature_args(args.features)
    seed_annotations = load_seed_annotations(args.seed_annotations_csv, selected_features)

    num_nodes = args.num_nodes
    if num_nodes is None:
        num_nodes = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    parts = []
    for rank in range(num_nodes):
        path = shard_path(args.output_csv, rank)
        if not os.path.exists(path):
            print(f"WARNING: missing shard {path}")
            continue
        df = pd.read_csv(path)
        parts.append(df)
        print(f"  shard {rank}: {len(df)} tracks")

    if not parts:
        raise RuntimeError("No shard files found to merge.")

    merged_new = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=["artist_name", "track_name"],
    )
    if not seed_annotations.empty:
        merged = pd.concat([merged_new, seed_annotations], ignore_index=True).drop_duplicates(
            subset=["artist_name", "track_name"],
        )
    else:
        merged = merged_new

    merged.to_csv(args.output_csv, index=False)
    print(
        f"Merged {len(merged)} unique tracks -> {args.output_csv} "
        f"(generated={len(merged_new)}, seed={len(seed_annotations)})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--song-list-path",
        default=str(_PROJECT_ROOT / "top_50k_songs.csv"),
    )
    common.add_argument(
        "--output-csv",
        default=str(_PKG_DIR / "output" / "composition_annotations_selected_features.csv"),
    )
    common.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help="Total number of nodes (default: SLURM_ARRAY_TASK_COUNT or 1)",
    )

    p_ann = sub.add_parser("annotate", parents=[common], help="Annotate one shard")
    p_ann.add_argument(
        "--features",
        nargs="+",
        default=None,
        help='Optional feature names (comma-separated); omit to annotate all features',
    )
    p_ann.add_argument(
        "--seed-annotations-csv",
        default=None,
        help="Existing annotations (e.g. validation set) to skip regenerating",
    )
    p_ann.add_argument("--api-key", default=None, help="LLM API key (or set LLM_API_KEY)")
    p_ann.add_argument("--node-rank", type=int, default=None, help="Node rank")
    p_ann.add_argument("--max-workers", type=int, default=10, help="Threads per node")

    p_merge = sub.add_parser("merge", parents=[common], help="Merge per-shard outputs")
    p_merge.add_argument(
        "--features",
        nargs="+",
        default=None,
        help='Optional feature names (comma-separated); omit to merge all features',
    )
    p_merge.add_argument(
        "--seed-annotations-csv",
        default=None,
        help="Existing annotations (e.g. validation set) to append into final merged output",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    {"annotate": cmd_annotate, "merge": cmd_merge}[args.command](args)


if __name__ == "__main__":
    main()
