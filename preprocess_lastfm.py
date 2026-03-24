"""
Build LastFM sequential files with global temporal split.

Outputs:
  datasets/sequential/LastFM/train.txt
  datasets/sequential/LastFM/val.txt
  datasets/sequential/LastFM/test.txt
  datasets/sequential/LastFM/test_sample.txt
  datasets/sequential/LastFM/val_sample.txt
  datasets/sequential/LastFM/item_id_master_map.csv
"""

import os
import random
from typing import Dict, List, Tuple

import polars as pl

PREPROCESSED_CSV = (
    "/work/pi_dagarwal_umass_edu/project_7/hmagapu/"
    "user_sessions_lastfm1k_minuser1000_minitem7_sessgap1200_minsesslen10_minhist50.csv"
)

OUTPUT_DIR = "datasets/sequential/LastFM"
MIN_SEQ_LEN = 5
TEST_QUANTILE = 0.9
VAL_QUANTILE_PRETEST = 0.9
VAL_USER_RATIO = 0.1
NUM_CANDIDATES = 200
RANDOM_SEED = 42


def read_source() -> pl.DataFrame:
    print(f"Loading {PREPROCESSED_CSV} ...")
    data = pl.read_csv(
        PREPROCESSED_CSV,
        has_header=False,
        new_columns=[
            "user_id",
            "ts",
            "art_id",
            "artist_name",
            "track_id",
            "track_name",
            "session_id",
        ],
    )
    print(f"  Loaded {data.height} rows")
    data = (
        data.with_columns(
            pl.col("ts")
            .str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False)
            .alias("ts_dt")
        )
        .filter(pl.col("ts_dt").is_not_null())
        .sort(["ts_dt", "user_id"])
    )
    print(f"  Rows with valid timestamps: {data.height}")
    return data


def build_item_map(data: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, int]]:
    unique_tracks = (
        data.select(["track_id", "artist_name", "track_name"])
        .unique(subset=["track_id"])
        .sort("track_id")
    )
    item_map = {
        tid: idx + 1 for idx, tid in enumerate(unique_tracks["track_id"].to_list())
    }
    print(f"  Unique items: {len(item_map)}")
    return unique_tracks, item_map


def choose_cutoffs(data: pl.DataFrame) -> Tuple[object, object]:
    test_cutoff = data.select(pl.col("ts_dt").quantile(TEST_QUANTILE)).item()
    pretest_data = data.filter(pl.col("ts_dt") <= test_cutoff)
    val_cutoff = pretest_data.select(pl.col("ts_dt").quantile(VAL_QUANTILE_PRETEST)).item()
    print(f"  Test cutoff (q={TEST_QUANTILE}): {test_cutoff}")
    print(f"  Val cutoff in pre-test (q={VAL_QUANTILE_PRETEST}): {val_cutoff}")
    return test_cutoff, val_cutoff


def write_split(path: str, rows: Dict[int, List[int]]) -> None:
    with open(path, "w") as f:
        for uid in sorted(rows):
            items = rows[uid]
            if items:
                f.write(f"{uid} " + " ".join(map(str, items)) + "\n")


def sample_candidates(
    users: List[int],
    positive_map: Dict[int, List[int]],
    full_hist_map: Dict[int, List[int]],
    total_items: int,
) -> Dict[int, List[int]]:
    sampled: Dict[int, List[int]] = {}
    for uid in users:
        positives = positive_map.get(uid, [])
        if not positives:
            continue
        first_pos = positives[0]
        used = set(full_hist_map[uid])
        row = [first_pos]
        while len(row) < NUM_CANDIDATES:
            neg = random.randint(1, total_items)
            if neg not in used and neg not in row:
                row.append(neg)
        sampled[uid] = row
    return sampled


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)

    data = read_source()
    unique_tracks, item_map = build_item_map(data)
    total_items = len(item_map)
    test_cutoff, val_cutoff = choose_cutoffs(data)

    data = data.with_columns(pl.col("track_id").replace_strict(item_map).alias("item_idx"))

    grouped: Dict[str, List[Tuple[object, int]]] = {}
    for uid, ts_dt, item_idx in data.select(["user_id", "ts_dt", "item_idx"]).iter_rows():
        grouped.setdefault(uid, []).append((ts_dt, int(item_idx)))

    # Keep users with enough total interactions.
    candidate_users = [uid for uid in sorted(grouped) if len(grouped[uid]) >= MIN_SEQ_LEN]
    print(f"  Users with >= {MIN_SEQ_LEN} interactions: {len(candidate_users)}")

    val_user_count = max(1, int(round(len(candidate_users) * VAL_USER_RATIO)))
    val_users = set(random.sample(candidate_users, val_user_count))
    print(f"  Validation users selected: {len(val_users)}")

    # Build split by timestamp rules while remapping users contiguously.
    train_rows: Dict[int, List[int]] = {}
    val_rows: Dict[int, List[int]] = {}
    test_rows: Dict[int, List[int]] = {}
    full_hist_rows: Dict[int, List[int]] = {}

    new_uid = 1
    dropped_users = 0
    for uid in candidate_users:
        records = grouped[uid]
        is_val_user = uid in val_users

        if is_val_user:
            train_items = [item for ts, item in records if ts <= val_cutoff]
            val_items = [item for ts, item in records if val_cutoff < ts <= test_cutoff]
            test_items = [item for ts, item in records if ts > test_cutoff]
        else:
            train_items = [item for ts, item in records if ts <= test_cutoff]
            val_items = []
            test_items = [item for ts, item in records if ts > test_cutoff]

        if len(train_items) < 1 or len(test_items) < 1:
            dropped_users += 1
            continue

        train_rows[new_uid] = train_items
        if val_items:
            val_rows[new_uid] = val_items
        test_rows[new_uid] = test_items
        full_hist_rows[new_uid] = [item for _, item in records]
        new_uid += 1

    total_users = new_uid - 1
    print(f"  Kept users after split filter: {total_users}")
    print(f"  Dropped users (no train/test): {dropped_users}")

    # Write mapping.
    master_rows = []
    for tid, artist, track in zip(
        unique_tracks["track_id"].to_list(),
        unique_tracks["artist_name"].to_list(),
        unique_tracks["track_name"].to_list(),
    ):
        master_rows.append(
            {
                "item_id": item_map[tid],
                "track_id": tid,
                "artist_name": artist,
                "track_name": track,
            }
        )
    pl.DataFrame(master_rows).sort("item_id").write_csv(
        os.path.join(OUTPUT_DIR, "item_id_master_map.csv")
    )

    # Write split files.
    write_split(os.path.join(OUTPUT_DIR, "train.txt"), train_rows)
    write_split(os.path.join(OUTPUT_DIR, "val.txt"), val_rows)
    write_split(os.path.join(OUTPUT_DIR, "test.txt"), test_rows)

    # Write sampled candidate files.
    test_sample_rows = sample_candidates(
        users=list(test_rows.keys()),
        positive_map=test_rows,
        full_hist_map=full_hist_rows,
        total_items=total_items,
    )
    val_sample_rows = sample_candidates(
        users=list(val_rows.keys()),
        positive_map=val_rows,
        full_hist_map=full_hist_rows,
        total_items=total_items,
    )
    write_split(os.path.join(OUTPUT_DIR, "test_sample.txt"), test_sample_rows)
    write_split(os.path.join(OUTPUT_DIR, "val_sample.txt"), val_sample_rows)

    print("SUCCESS")
    print(f"  User ID range: 1 -> {total_users}")
    print(f"  Item ID range: 1 -> {total_items}")
    print(f"  Train users:   {len(train_rows)}")
    print(f"  Val users:     {len(val_rows)}")
    print(f"  Test users:    {len(test_rows)}")
    print(f"  Output dir:    {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
