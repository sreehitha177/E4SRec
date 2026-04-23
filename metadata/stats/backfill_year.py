"""
Backfill first_release_year for rows in musicbrainz_cache.db where year is NULL.

Uses mb_recording_id (already stored) for a direct get_recording_by_id lookup —
one API call per song, no search needed.

Supports sharding for parallel SLURM array jobs.

Usage:
    python backfill_year.py [--db PATH] [--shard-id N] [--num-shards N] [--limit N]
"""

import argparse
import re
import socket
import sqlite3
import time
from pathlib import Path

import musicbrainzngs

socket.setdefaulttimeout(30)


def parse_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    m = re.match(r"(\d{4})", date_str)
    return int(m.group(1)) if m else None


def earliest_year_from_releases(release_list: list) -> int | None:
    years = []
    for rel in release_list:
        y = parse_year(rel.get("date"))
        if y and y > 1900:
            years.append(y)
    return min(years) if years else None


def fetch_year_by_mbid(mbid: str) -> int | None:
    """Direct lookup using MBID — 1 call, no search needed."""
    result = musicbrainzngs.get_recording_by_id(mbid, includes=["releases"])
    releases = result["recording"].get("release-list", [])
    return earliest_year_from_releases(releases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        default="/project/pi_dagarwal_umass_edu/project_7/hmagapu/musicbrainz_cache.db",
    )
    parser.add_argument("--shard-id",   type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit",      type=int, default=None)
    args = parser.parse_args()

    musicbrainzngs.set_useragent(f"top50k-year-backfill-shard{args.shard_id}", "1.0", "research")

    conn = sqlite3.connect(args.db)

    all_rows = conn.execute(
        "SELECT track_index, mb_recording_id FROM results WHERE first_release_year IS NULL AND mb_recording_id IS NOT NULL"
    ).fetchall()

    # Shard
    rows = all_rows[args.shard_id::args.num_shards]
    if args.limit:
        rows = rows[:args.limit]

    total = len(rows)
    print(f"Shard {args.shard_id}/{args.num_shards}: {total:,} rows to backfill", flush=True)

    filled = 0
    for i, (track_index, mbid) in enumerate(rows):
        try:
            year = fetch_year_by_mbid(mbid)
            if year:
                conn.execute(
                    "UPDATE results SET first_release_year = ? WHERE track_index = ?",
                    (year, track_index),
                )
                conn.commit()
                filled += 1
        except Exception as e:
            print(f"  ERROR [track_index={track_index}]: {e}", flush=True)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}] {(i+1)/total*100:.1f}% — {filled} years filled", flush=True)

        time.sleep(1.1)

    conn.close()

    conn2 = sqlite3.connect(args.db)
    final = conn2.execute("SELECT COUNT(*) FROM results WHERE first_release_year IS NOT NULL").fetchone()[0]
    total_rows = conn2.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    conn2.close()

    print(f"\nShard {args.shard_id} done. Year filled globally: {final:,} / {total_rows:,} ({final/total_rows*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
