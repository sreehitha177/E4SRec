#!/usr/bin/env python3
"""Compute listening completion ratios from preprocessed LastFM session data.

Also produces interaction_completion_ratios.pkl — a {user_id_int: [float, ...]}
dict aligned with each user's training sequence in train.txt, used by finetune.py
for per-interaction completion ratio injection (prompt or embedding concat).
"""

import argparse
import csv
import glob
import math
import os
import pickle
import re
from collections import defaultdict
from datetime import datetime
from statistics import mean, median, pstdev
from typing import Dict, List, Optional, Tuple

DEFAULT_PREPROCESSED = "data_preproc/user_sessions_lastfm1k_minuser1000_minitem7_sessgap1200_minsesslen10_minhist50.csv"
DEFAULT_OUTPUT_INTERACTIONS = "data_preproc/user_sessions_with_completion.csv"
DEFAULT_OUTPUT_SUMMARY = "data_preproc/song_completion_summary.csv"
DEFAULT_OUTPUT_PKL = "datasets/sequential/LastFM/interaction_completion_ratios.pkl"
DEFAULT_TRAIN_TXT = "datasets/sequential/LastFM/train.txt"
DEFAULT_MASTER_MAP = "datasets/sequential/LastFM/item_id_master_map.csv"
DEFAULT_MIN_SEQ_LEN = 5
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
            "track_id": row.get("track_id", ""),   # MBID — used as reliable join key
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


def build_training_completion_pkl(
    output_rows: List[Dict],
    master_map_path: str,
    train_txt_path: str,
    min_seq_len: int,
) -> Dict[int, List[float]]:
    """Align per-interaction completion ratios with each user's train.txt sequence.

    output_rows is the already-sorted interaction list from compute_interaction_completion.
    Returns {user_id_int: [ratio_or_nan, ...]} with one value per position in train.txt.
    """
    if not os.path.exists(master_map_path):
        print(f"  [pkl] master_map not found at {master_map_path}, skipping pkl build.")
        return {}
    if not os.path.exists(train_txt_path):
        print(f"  [pkl] train.txt not found at {train_txt_path}, skipping pkl build.")
        return {}

    # Build two lookup dicts from item_id_master_map.csv:
    #   track_id  (MBID)       -> item_id   (primary,   reliable)
    #   artist||track (normed) -> item_id   (fallback,  less reliable)
    trackid_to_item: Dict[str, int] = {}
    name_to_item:    Dict[str, int] = {}
    with open(master_map_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = int(row["item_id"])
            tid = row.get("track_id", "").strip()
            if tid and tid not in trackid_to_item:
                trackid_to_item[tid] = iid
            key = normalize_text(row.get("artist_name", "")) + "||" + normalize_text(row.get("track_name", ""))
            if key not in name_to_item:
                name_to_item[key] = iid

    print(f"  [pkl] master_map: {len(trackid_to_item)} track_id entries, "
          f"{len(name_to_item)} name entries")

    # Build per-user ordered (item_id, completion_ratio) from already-sorted output_rows.
    user_events: Dict[str, List[tuple]] = defaultdict(list)
    n_rows = len(output_rows)
    n_id_hit, n_ratio_valid = 0, 0
    for row in output_rows:
        # Primary join: track_id (MBID) — stable across different CSV sources.
        tid = row.get("track_id", "").strip()
        item_id = trackid_to_item.get(tid)
        # Fallback: normalized artist||track name.
        if item_id is None:
            key = normalize_text(row.get("artist_name", "")) + "||" + normalize_text(row.get("track_name", ""))
            item_id = name_to_item.get(key)

        ratio_str = row.get("completion_ratio", "")
        ratio = float("nan")
        if ratio_str:
            try:
                ratio = float(ratio_str)
                n_ratio_valid += 1
            except ValueError:
                pass

        if item_id is not None:
            n_id_hit += 1
        user_events[row["user_id"]].append((item_id, ratio))

    print(f"  [pkl] Diagnostic over {n_rows} rows:")
    print(f"         item_id resolved : {n_id_hit:,} ({100*n_id_hit/max(n_rows,1):.1f}%)")
    print(f"         completion ratio  : {n_ratio_valid:,} ({100*n_ratio_valid/max(n_rows,1):.1f}%)")

    # Load train.txt: {user_id_int: [item_id, ...]}
    train_seqs: Dict[int, List[int]] = {}
    with open(train_txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            train_seqs[int(parts[0])] = [int(x) for x in parts[1:]]

    # Match user_id_int (from train.txt) to user_id_str (from output_rows) by item-set
    # overlap.  Sorting-based reconstruction fails when the two preprocessing runs have
    # different user sets — a shared sort order doesn't imply a shared identity.
    # With ~600–900 users on each side, 638×814 ≈ 520k comparisons is negligible.
    str_user_items: Dict[str, set] = {
        uid: {item_id for item_id, _ in events if item_id is not None}
        for uid, events in user_events.items()
        if sum(1 for item_id, _ in events if item_id is not None) >= min_seq_len
    }
    print(f"  [pkl] Valid string users (≥{min_seq_len} resolved items): {len(str_user_items)}")
    print(f"  [pkl] Users in train.txt: {len(train_seqs)}")

    # Greedy best-first bipartite matching: assign each int user to the str user with
    # the largest item-set intersection, then remove that str user from the pool.
    int_to_str: Dict[int, str] = {}
    available = dict(str_user_items)  # uid_str -> item_set (shrinks as matches are made)
    for uid_int, train_seq in sorted(train_seqs.items()):
        train_set = set(train_seq)
        best_str, best_overlap = None, 0
        for uid_str, item_set in available.items():
            overlap = len(train_set & item_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_str = uid_str
        if best_str is not None and best_overlap > 0:
            int_to_str[uid_int] = best_str
            del available[best_str]

    print(f"  [pkl] Matched {len(int_to_str)} int→str user pairs "
          f"(avg overlap: "
          f"{sum(len(set(train_seqs[i]) & str_user_items[s]) for i,s in int_to_str.items()) // max(len(int_to_str),1)} items)")

    # Greedy subsequence alignment per user.
    result: Dict[int, List[float]] = {}
    aligned, total_items, covered_items = 0, 0, 0
    for uid_int, uid_str in int_to_str.items():
        train_seq = train_seqs[uid_int]
        sess_events = [(item_id, ratio) for item_id, ratio in user_events[uid_str] if item_id is not None]

        ratios: List[float] = []
        ptr = 0
        for train_item in train_seq:
            found = False
            while ptr < len(sess_events):
                if sess_events[ptr][0] == train_item:
                    ratios.append(sess_events[ptr][1])
                    ptr += 1
                    found = True
                    break
                ptr += 1
            if not found:
                ratios.append(float("nan"))

        result[uid_int] = ratios
        aligned += 1
        total_items += len(ratios)
        covered_items += sum(1 for r in ratios if not math.isnan(r))

    print(f"  [pkl] Aligned {aligned} users, "
          f"{covered_items}/{total_items} ({100*covered_items/max(total_items,1):.1f}%) "
          "positions have a valid completion ratio.")
    return result


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Compute completion ratios for preprocessed LastFM sessions.")
    parser.add_argument("--preprocessed-file", default=DEFAULT_PREPROCESSED)
    parser.add_argument("--lyrics-dirs", nargs="+", default=DEFAULT_LYRICS_DIRS)
    parser.add_argument("--output-interactions", default=DEFAULT_OUTPUT_INTERACTIONS)
    parser.add_argument("--output-summary", default=DEFAULT_OUTPUT_SUMMARY)
    parser.add_argument("--output-pkl", default=DEFAULT_OUTPUT_PKL,
                        help="Output pkl mapping user_id_int -> [per-interaction ratio, ...]")
    parser.add_argument("--train-txt", default=DEFAULT_TRAIN_TXT,
                        help="train.txt produced by preprocess_lastfm_loo.py")
    parser.add_argument("--master-map", default=DEFAULT_MASTER_MAP,
                        help="item_id_master_map.csv produced by preprocess_lastfm_loo.py")
    parser.add_argument("--min-seq-len", type=int, default=DEFAULT_MIN_SEQ_LEN)
    args = parser.parse_args()

    ordered_song_map, duration_map = read_lyrics_song_map(args.lyrics_dirs)
    print(f"Loaded {len(ordered_song_map)} lyrics song keys.")
    print(f"Loaded duration map entries for {len(duration_map)} track_index values.")

    session_rows = read_session_data(args.preprocessed_file)
    print(f"Loaded {len(session_rows)} preprocessed session records.")

    output_rows, song_ratios = compute_interaction_completion(session_rows, ordered_song_map, duration_map)

    interaction_fields = [
        "user_id", "timestamp", "session_id", "artist_name", "track_name",
        "track_id", "track_index", "song_duration_seconds",
        "user_listening_duration_seconds", "completion_ratio",
    ]
    write_csv(args.output_interactions, output_rows, interaction_fields)
    print(f"Saved interaction-level completion ratios to {args.output_interactions}")

    summary_rows = aggregate_song_summary(song_ratios)
    summary_fields = [
        "track_index", "num_users", "avg_completion_ratio",
        "median_completion_ratio", "std_completion_ratio",
    ]
    write_csv(args.output_summary, summary_rows, summary_fields)
    print(f"Saved song-level completion summary to {args.output_summary}")
    print(f"Computed completion ratio for {len(output_rows)} interactions and {len(summary_rows)} songs.")

    print("Building per-interaction training pkl ...")
    pkl_data = build_training_completion_pkl(
        output_rows, args.master_map, args.train_txt, args.min_seq_len
    )
    if pkl_data:
        os.makedirs(os.path.dirname(args.output_pkl) or ".", exist_ok=True)
        with open(args.output_pkl, "wb") as f:
            pickle.dump(pkl_data, f)
        print(f"Saved training completion pkl to {args.output_pkl}")


if __name__ == "__main__":
    main()
