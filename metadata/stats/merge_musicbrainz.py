"""
Merge enriched_genres.csv (genre + language) and MusicBrainz year fixes
into top_50k_full_augmented.csv, producing top_50k_full_augmented_v2.csv.

Sources:
  - genre   : enriched_genres.csv  (genre_enriched column)
  - language: enriched_genres.csv  (language column)
  - year    : SQLite checkpoint DB (first_release_year from MusicBrainz)

Usage:
    python merge_musicbrainz.py [--catalog PATH] [--genres PATH] [--db PATH] [--output PATH]
"""

import argparse
import sqlite3
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        default="/project/pi_dagarwal_umass_edu/project_7/hmagapu/top_50k_full_augmented.csv",
    )
    parser.add_argument(
        "--genres",
        default="/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/enriched_genres.csv",
    )
    parser.add_argument(
        "--db",
        default="/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/musicbrainz_cache.db",
        help="SQLite checkpoint DB (for year corrections)",
    )
    parser.add_argument(
        "--output",
        default="/project/pi_dagarwal_umass_edu/project_7/hmagapu/top_50k_full_augmented_v2.csv",
    )
    parser.add_argument("--min-score", type=int, default=70)
    args = parser.parse_args()

    print(f"Loading catalog: {args.catalog}")
    cat = pd.read_csv(args.catalog)
    print(f"  {len(cat):,} rows")

    # ── Genre + Language from enriched_genres.csv ───────────────────────────
    print(f"\nLoading genres/language: {args.genres}")
    gdf = pd.read_csv(args.genres)
    print(f"  genre_enriched filled: {gdf['genre_enriched'].notna().sum():,}")
    if "language" in gdf.columns:
        print(f"  language filled:       {gdf['language'].notna().sum():,}")

    merged = cat.merge(
        gdf[["artist_name", "track_name", "genre_enriched"] +
            (["language"] if "language" in gdf.columns else [])],
        on=["artist_name", "track_name"],
        how="left",
    )

    before_genre = merged["genre"].notna().sum()
    merged["genre"] = merged["genre"].combine_first(merged["genre_enriched"])
    after_genre = merged["genre"].notna().sum()
    merged.drop(columns=["genre_enriched"], inplace=True)
    print(f"  Genre coverage: {before_genre:,} → {after_genre:,} (+{after_genre - before_genre:,})")

    # ── Year from SQLite DB ─────────────────────────────────────────────────
    db_path = Path(args.db)
    if db_path.exists():
        print(f"\nLoading year corrections from {db_path}...")
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT artist_name, track_name, first_release_year, mb_score FROM results WHERE mb_score >= ?",
            (args.min_score,)
        ).fetchall()
        conn.close()

        year_df = pd.DataFrame(rows, columns=["artist_name", "track_name", "first_release_year", "mb_score"])
        merged = merged.merge(year_df[["artist_name", "track_name", "first_release_year"]],
                              on=["artist_name", "track_name"], how="left")

        before_wrong = (merged["year"] > 2009).sum()
        merged["year"] = merged["first_release_year"].combine_first(merged["year"])
        after_wrong = (merged["year"] > 2009).sum()
        merged.drop(columns=["first_release_year"], inplace=True)
        print(f"  Year: post-2009 songs {before_wrong:,} → {after_wrong:,}")
    else:
        print(f"\nNo SQLite DB found at {db_path} — skipping year correction")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\nFinal coverage:")
    print(f"  genre:    {merged['genre'].notna().sum():,} / {len(merged):,} ({merged['genre'].notna().mean()*100:.1f}%)")
    if "language" in merged.columns:
        print(f"  language: {merged['language'].notna().sum():,} / {len(merged):,} ({merged['language'].notna().mean()*100:.1f}%)")

    print(f"\nWriting to {args.output}...")
    merged.to_csv(args.output, index=False)
    print(f"Done. {len(merged):,} rows, {len(merged.columns)} columns.")


if __name__ == "__main__":
    main()
