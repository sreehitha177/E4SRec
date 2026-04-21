#!/usr/bin/env python3
"""Compute listening completion ratios from preprocessed LastFM session data."""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from datetime import datetime
from statistics import mean, median, pstdev
from typing import Dict, List, Optional, Tuple

DEFAULT_PREPROCESSED = "data_preproc/user_sessions_lastfm1k_minuser1000_minitem7_sessgap1200_minsesslen10_minhist50.csv"
DEFAULT_OUTPUT_INTERACTIONS = "data_preproc/user_sessions_with_completion.csv"
DEFAULT_OUTPUT_SUMMARY = "data_preproc/song_completion_summary.csv"
DEFAULT_LYRICS_DIRS = [
    "/scratch3/workspace/skandagatla_umass_edu-dolby/lyrics/batch_1",
    "/scratch3/workspace/skandagatla_umass_edu-dolby/lyrics/batch_2",
]

SESSION_FIELDS = [
    "user_id",
    "timestamp",
    "artist_id",
    "artist_name",
    "track_id",
    "track_name",
    "session_id",
]


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    return text


def read_lyrics_song_map(directories: List[str]) -> Tuple[Dict[str, int], Dict[int, Optional[float]]]:
    song_map: Dict[str, int] = {}
    duration_map: Dict[int, Optional[float]] = {}
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        paths = sorted(glob.glob(os.path.join(directory, "master_lyrics_node_*.csv")))
        for path in paths:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                if not {"track_index", "artist_name", "track_name", "duration_seconds"}.issubset(reader.fieldnames or []):
                    raise ValueError(f"Expected track_index, artist_name, track_name, and duration_seconds in {path}")
                for row in reader:
                    artist_name = normalize_text(row.get("artist_name", ""))
                    track_name = normalize_text(row.get("track_name", ""))
                    key = artist_name + "||" + track_name
                    try:
                        track_index = int(row["track_index"])
                    except (TypeError, ValueError):
                        continue
                    duration = None
                    if row.get("duration_seconds"):
                        try:
                            duration = float(row["duration_seconds"])
                        except ValueError:
                            duration = None
                    if key not in song_map:
                        song_map[key] = track_index
                    duration_map.setdefault(track_index, duration)
    return song_map, duration_map


def parse_timestamp(ts: str) -> datetime:
    if ts is None:
        raise ValueError("Missing timestamp")
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1]
    return datetime.fromisoformat(ts)


def read_session_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessed session file not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for line in reader:
            if not line or len(line) < len(SESSION_FIELDS):
                continue
            rows.append({
                "user_id": line[0],
                "timestamp": line[1],
                "artist_id": line[2],
                "artist_name": line[3],
                "track_id": line[4],
                "track_name": line[5],
                "session_id": line[6],
            })
    return rows


def sort_session_rows(rows: List[Dict[str, str]]):
    def key_fn(row: Dict[str, str]):
        return (
            row["user_id"],
            int(row["session_id"]),
            parse_timestamp(row["timestamp"]),
        )
    return sorted(rows, key=key_fn)


def compute_interaction_completion(rows, ordered_song_map, duration_map):
    output_rows = []
    song_ratios = defaultdict(list)
    sorted_rows = sort_session_rows(rows)
    total = len(sorted_rows)
    for idx, row in enumerate(sorted_rows):
        next_row = sorted_rows[idx + 1] if idx + 1 < total else None
        listening_duration = None
        if (
            next_row is not None
            and row["user_id"] == next_row["user_id"]
            and row["session_id"] == next_row["session_id"]
        ):
            start_ts = parse_timestamp(row["timestamp"])
            end_ts = parse_timestamp(next_row["timestamp"])
            delta = (end_ts - start_ts).total_seconds()
            listening_duration = max(0.0, delta)

        artist_name = normalize_text(row["artist_name"])
        track_name = normalize_text(row["track_name"])
        key = artist_name + "||" + track_name
        track_index = ordered_song_map.get(key)
        song_duration = duration_map.get(track_index) if track_index is not None else None
        completion_ratio = None
        if listening_duration is not None and song_duration is not None and song_duration > 0:
            completion_ratio = listening_duration / song_duration
            completion_ratio = max(0.0, min(1.0, completion_ratio))
            if track_index is not None:
                song_ratios[track_index].append(completion_ratio)

        output_rows.append({
            "user_id": row["user_id"],
            "timestamp": row["timestamp"],
            "session_id": row["session_id"],
            "artist_name": row["artist_name"],
            "track_name": row["track_name"],
            "track_index": "" if track_index is None else str(track_index),
            "song_duration_seconds": "" if song_duration is None else f"{song_duration:.3f}",
            "user_listening_duration_seconds": "" if listening_duration is None else f"{listening_duration:.3f}",
            "completion_ratio": "" if completion_ratio is None else f"{completion_ratio:.6f}",
        })
    return output_rows, song_ratios


def aggregate_song_summary(song_ratios):
    summary = []
    for track_index in sorted(song_ratios):
        ratios = song_ratios[track_index]
        if not ratios:
            continue
        summary.append({
            "track_index": track_index,
            "num_users": len(ratios),
            "avg_completion_ratio": f"{mean(ratios):.6f}",
            "median_completion_ratio": f"{median(ratios):.6f}",
            "std_completion_ratio": f"{pstdev(ratios):.6f}" if len(ratios) > 1 else "0.000000",
        })
    return summary


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Compute completion ratios for preprocessed LastFM sessions.")
    parser.add_argument("--preprocessed-file", default=DEFAULT_PREPROCESSED, help="Path to the preprocessed session CSV file.")
    parser.add_argument("--lyrics-dirs", nargs="+", default=DEFAULT_LYRICS_DIRS, help="Directories containing master_lyrics_node_*.csv duration files.")
    parser.add_argument("--output-interactions", default=DEFAULT_OUTPUT_INTERACTIONS, help="Output CSV for interaction-level completion ratios.")
    parser.add_argument("--output-summary", default=DEFAULT_OUTPUT_SUMMARY, help="Output CSV for per-song completion ratio summary.")
    args = parser.parse_args()

    ordered_song_map, duration_map = read_lyrics_song_map(args.lyrics_dirs)

    print(f"Loaded {len(ordered_song_map)} lyrics song keys.")
    print(f"Loaded duration map entries for {len(duration_map)} track_index values.")

    session_rows = read_session_data(args.preprocessed_file)
    print(f"Loaded {len(session_rows)} preprocessed session records.")

    output_rows, song_ratios = compute_interaction_completion(session_rows, ordered_song_map, duration_map)

    interaction_fields = [
        "user_id",
        "timestamp",
        "session_id",
        "artist_name",
        "track_name",
        "track_index",
        "song_duration_seconds",
        "user_listening_duration_seconds",
        "completion_ratio",
    ]
    write_csv(args.output_interactions, output_rows, interaction_fields)
    print(f"Saved interaction-level completion ratios to {args.output_interactions}")

    summary_rows = aggregate_song_summary(song_ratios)
    summary_fields = [
        "track_index",
        "num_users",
        "avg_completion_ratio",
        "median_completion_ratio",
        "std_completion_ratio",
    ]
    write_csv(args.output_summary, summary_rows, summary_fields)
    print(f"Saved song-level completion summary to {args.output_summary}")
    print(f"Computed completion ratio for {len(output_rows)} interactions and {len(summary_rows)} songs.")


if __name__ == "__main__":
    main()
