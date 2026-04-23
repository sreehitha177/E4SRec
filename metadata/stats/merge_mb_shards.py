"""
Merge MusicBrainz shard SQLite DBs into enriched_genres.csv.

Run this after all array jobs finish:
    python merge_mb_shards.py

It consolidates musicbrainz_cache_shard0.db ... shard3.db into a single
combined DB, then applies genre + language to enriched_genres.csv.
"""

import sqlite3
import argparse
from pathlib import Path
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    track_index       INTEGER PRIMARY KEY,
    artist_name       TEXT,
    track_name        TEXT,
    mb_recording_id   TEXT,
    mb_score          INTEGER,
    first_release_year INTEGER,
    mb_genre          TEXT,
    mb_language       TEXT,
    mb_tags           TEXT,
    fetched_at        TEXT
);
CREATE TABLE IF NOT EXISTS errors (
    track_index INTEGER PRIMARY KEY,
    artist_name TEXT,
    track_name  TEXT,
    error       TEXT,
    fetched_at  TEXT
);
"""


def merge_dbs(shard_paths: list[Path], combined_path: Path):
    conn = sqlite3.connect(str(combined_path))
    conn.executescript(SCHEMA)
    conn.commit()

    for p in shard_paths:
        if not p.exists():
            print(f"  WARNING: {p} not found, skipping")
            continue
        shard = sqlite3.connect(str(p))
        rows = shard.execute("SELECT * FROM results").fetchall()
        conn.executemany(
            "INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?,?,?)", rows
        )
        errs = shard.execute("SELECT * FROM errors").fetchall()
        conn.executemany(
            "INSERT OR REPLACE INTO errors VALUES (?,?,?,?,?)", errs
        )
        shard.close()
        print(f"  {p.name}: {len(rows):,} results, {len(errs):,} errors")

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    errors = conn.execute("SELECT COUNT(*) FROM errors").fetchone()[0]
    print(f"\nCombined DB: {total:,} results, {errors:,} errors → {combined_path}")
    return conn


def apply_to_genres(conn: sqlite3.Connection, genres_path: Path, min_score: int):
    print(f"\nLoading {genres_path}...")
    gdf = pd.read_csv(genres_path)

    if "language" not in gdf.columns:
        gdf["language"] = None

    gdf["_key"] = (
        gdf["artist_name"].str.lower().str.strip()
        + " ||| "
        + gdf["track_name"].str.lower().str.strip()
    )

    rows = conn.execute(
        "SELECT artist_name, track_name, mb_genre, mb_language FROM results WHERE mb_score >= ?",
        (min_score,)
    ).fetchall()

    results_df = pd.DataFrame(rows, columns=["artist_name", "track_name", "mb_genre", "mb_language"])
    results_df["_key"] = (
        results_df["artist_name"].str.lower().str.strip()
        + " ||| "
        + results_df["track_name"].str.lower().str.strip()
    )

    key_to_genre = results_df.set_index("_key")["mb_genre"].to_dict()
    key_to_lang  = results_df.set_index("_key")["mb_language"].to_dict()

    before_genre = gdf["genre_enriched"].notna().sum()
    before_lang  = gdf["language"].notna().sum()

    mask_genre = gdf["genre_enriched"].isna() & gdf["_key"].isin(key_to_genre)
    gdf.loc[mask_genre, "genre_enriched"] = gdf.loc[mask_genre, "_key"].map(key_to_genre)

    mask_lang = gdf["language"].isna() & gdf["_key"].isin(key_to_lang)
    gdf.loc[mask_lang, "language"] = gdf.loc[mask_lang, "_key"].map(key_to_lang)

    gdf.drop(columns=["_key"], inplace=True)

    after_genre = gdf["genre_enriched"].notna().sum()
    after_lang  = gdf["language"].notna().sum()

    print(f"  genre_enriched: {before_genre:,} → {after_genre:,} (+{after_genre - before_genre:,})")
    print(f"  language:       {before_lang:,} → {after_lang:,} (+{after_lang - before_lang:,})")

    gdf.to_csv(genres_path, index=False)
    print(f"\nSaved {genres_path}  ({len(gdf):,} rows, columns: {list(gdf.columns)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-dir",
        default="/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata",
    )
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument(
        "--genres",
        default="/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/enriched_genres.csv",
    )
    parser.add_argument(
        "--combined-db",
        default="/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/musicbrainz_cache.db",
    )
    parser.add_argument("--min-score", type=int, default=70)
    args = parser.parse_args()

    meta = Path(args.metadata_dir)
    shard_paths = [meta / f"musicbrainz_cache_shard{i}.db" for i in range(args.num_shards)]

    print("Merging shard DBs...")
    conn = merge_dbs(shard_paths, Path(args.combined_db))
    apply_to_genres(conn, Path(args.genres), args.min_score)
    conn.close()


if __name__ == "__main__":
    main()
