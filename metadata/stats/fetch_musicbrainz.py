"""
Fetch genre, language, and release year from MusicBrainz for the top-50k catalog.

Reads/writes enriched_genres.csv (artist_name, track_name, genre_enriched, language).
Skips songs that already have genre_enriched filled — only fetches for the ~39k
still-null songs. Language is captured for every song fetched.

One API call per song (search_recordings), extracting:
  - genre_enriched     : top tag by vote count (fills nulls in enriched_genres.csv)
  - language           : inferred from tag keywords (e.g. "french", "j-pop")
  - first_release_year : from first-release-date (written to SQLite only;
                         use merge_musicbrainz.py to apply to the main catalog)
  - mb_score           : search confidence (0–100); low-confidence results discarded
  - mb_recording_id    : MusicBrainz MBID

Checkpoints to SQLite so the script can be killed and safely resumed.

Usage:
    python fetch_musicbrainz.py [--genres PATH] [--db PATH] [--limit N]

Rate limit: 1 request/second (MusicBrainz ToS).
Expected runtime for ~39k songs: ~12 hours. Use --limit N for a quick test.
"""

import argparse
import sqlite3
import time
import re
import socket
from pathlib import Path

import musicbrainzngs
import pandas as pd

socket.setdefaulttimeout(30)  # 30s hard timeout — prevents hanging on unresponsive MB

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

musicbrainzngs.set_useragent("top50k-enrichment", "1.0", "research")

GENRE_BLOCKLIST = {
    # Tags that are moods/descriptors, not genres — skip them
    "seen live", "favorite", "beautiful", "mellow", "chill", "love",
    "sad", "happy", "relaxing", "workout", "driving", "sleepy",
    "good", "great", "awesome", "amazing", "classic",
}

LANG_TAG_MAP = {
    "french": "fr", "chanson": "fr",
    "german": "de", "deutsch": "de",
    "spanish": "es", "latin": "es", "flamenco": "es",
    "portuguese": "pt", "bossa nova": "pt", "mpb": "pt", "samba": "pt",
    "italian": "it", "opera": "it",
    "japanese": "ja", "j-pop": "ja", "jpop": "ja", "j-rock": "ja", "anime": "ja",
    "korean": "ko", "k-pop": "ko", "kpop": "ko",
    "chinese": "zh", "mandarin": "zh", "cantopop": "zh",
    "russian": "ru",
    "turkish": "tr",
    "arabic": "ar",
    "hindi": "hi", "bollywood": "hi",
    "swedish": "sv",
    "norwegian": "no",
    "danish": "da",
    "finnish": "fi",
    "polish": "pl",
    "dutch": "nl",
    "greek": "el",
    "hebrew": "he",
    "thai": "th",
    "indonesian": "id",
    "vietnamese": "vi",
}

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

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
    fetched_at        TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS errors (
    track_index INTEGER PRIMARY KEY,
    artist_name TEXT,
    track_name  TEXT,
    error       TEXT,
    fetched_at  TEXT DEFAULT (datetime('now'))
);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def get_done_indices(conn: sqlite3.Connection) -> set:
    rows = conn.execute("SELECT track_index FROM results").fetchall()
    errors = conn.execute("SELECT track_index FROM errors").fetchall()
    return {r[0] for r in rows} | {r[0] for r in errors}


def save_result(conn: sqlite3.Connection, row: dict):
    conn.execute(
        """INSERT OR REPLACE INTO results
           (track_index, artist_name, track_name, mb_recording_id, mb_score,
            first_release_year, mb_genre, mb_language, mb_tags)
           VALUES (:track_index, :artist_name, :track_name, :mb_recording_id,
                   :mb_score, :first_release_year, :mb_genre, :mb_language, :mb_tags)""",
        row,
    )
    conn.commit()


def save_error(conn: sqlite3.Connection, track_index: int, artist: str, track: str, error: str):
    conn.execute(
        "INSERT OR REPLACE INTO errors (track_index, artist_name, track_name, error) VALUES (?,?,?,?)",
        (track_index, artist, track, error),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# MusicBrainz query
# ---------------------------------------------------------------------------

def parse_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    m = re.match(r"(\d{4})", date_str)
    return int(m.group(1)) if m else None


def infer_language(tags: list[dict]) -> str | None:
    """Return ISO language code if any tag matches a known language keyword."""
    for tag in tags:
        name = tag.get("name", "").lower()
        for keyword, lang_code in LANG_TAG_MAP.items():
            if keyword in name:
                return lang_code
    return None


def pick_genre(tags: list[dict]) -> str | None:
    """Return the highest-voted tag that looks like a genre (not a mood/descriptor)."""
    genre_tags = [
        t for t in tags
        if t.get("name", "").lower() not in GENRE_BLOCKLIST
    ]
    if not genre_tags:
        return None
    best = max(genre_tags, key=lambda t: int(t.get("count", 0)))
    return best["name"]


def earliest_year_from_releases(release_list: list) -> int | None:
    """Extract the earliest release year from the release-list."""
    years = []
    for rel in release_list:
        y = parse_year(rel.get("date"))
        if y and y > 1900:
            years.append(y)
    return min(years) if years else None


# ISO 639-3 → ISO 639-1 for the most common codes returned by MusicBrainz
MB_LANG_MAP = {
    "eng": "en", "fra": "fr", "deu": "de", "spa": "es", "por": "pt",
    "ita": "it", "jpn": "ja", "kor": "ko", "zho": "zh", "rus": "ru",
    "tur": "tr", "ara": "ar", "hin": "hi", "swe": "sv", "nor": "no",
    "dan": "da", "fin": "fi", "pol": "pl", "nld": "nl", "ell": "el",
    "heb": "he", "tha": "th", "ind": "id", "vie": "vi",
    "mul": None,  # multilingual — not useful
    "zxx": None,  # no linguistic content (instrumentals)
}


def language_from_lookup(mbid: str) -> str | None:
    """
    Second MB API call: get_recording_by_id with includes=['releases'] to
    retrieve text-representation.language. Returns ISO 639-1 code or None.
    """
    try:
        time.sleep(1.1)  # respect rate limit between the two calls
        result = musicbrainzngs.get_recording_by_id(mbid, includes=["releases"])
        releases = result["recording"].get("release-list", [])
        for rel in releases:
            tr = rel.get("text-representation")
            if tr and tr.get("language"):
                lang3 = tr["language"].lower()
                return MB_LANG_MAP.get(lang3, lang3)
    except Exception:
        pass
    return None


def query_mb(artist: str, track: str) -> dict:
    """
    Call 1: search_recordings — gets score, tags, year, genre.
    Call 2: get_recording_by_id — gets proper language from text-representation.
    Call 2 is skipped if tags already gave a language signal.
    """
    result = musicbrainzngs.search_recordings(
        recording=track,
        artist=artist,
        limit=1,
    )
    recordings = result.get("recording-list", [])
    if not recordings:
        return {
            "mb_recording_id": None,
            "mb_score": 0,
            "first_release_year": None,
            "mb_genre": None,
            "mb_language": None,
            "mb_tags": None,
        }

    rec = recordings[0]
    score = int(rec.get("ext:score", 0))
    tags = rec.get("tag-list", [])
    releases = rec.get("release-list", [])
    mbid = rec.get("id")

    # Tag-based language first (free); only do the lookup call if tags gave nothing
    lang = infer_language(tags)
    if lang is None and mbid and score >= 70:
        lang = language_from_lookup(mbid)

    return {
        "mb_recording_id": mbid,
        "mb_score": score,
        "first_release_year": earliest_year_from_releases(releases),
        "mb_genre": pick_genre(tags),
        "mb_language": lang,
        "mb_tags": ", ".join(t["name"] for t in tags) if tags else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch MusicBrainz metadata for top-50k songs.")
    parser.add_argument(
        "--catalog",
        default="/project/pi_dagarwal_umass_edu/project_7/hmagapu/top_50k_full_augmented.csv",
        help="Path to top_50k_full_augmented.csv (source of track_index)",
    )
    parser.add_argument(
        "--genres",
        default="/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/enriched_genres.csv",
        help="Path to enriched_genres.csv (read + written in place)",
    )
    parser.add_argument(
        "--db", default="musicbrainz_cache.db",
        help="SQLite checkpoint database (default: musicbrainz_cache.db)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only fetch N songs (for testing)",
    )
    parser.add_argument(
        "--min-score", type=int, default=70,
        help="Minimum MB search score to trust result (default: 70)",
    )
    parser.add_argument(
        "--shard-id", type=int, default=0,
        help="Which shard this job handles (0-indexed, default: 0)",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Total number of shards / parallel jobs (default: 1)",
    )
    args = parser.parse_args()

    genres_path = Path(args.genres)
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = Path(__file__).parent / db_path

    # ── Load catalog (for real track_index) ────────────────────────────────
    print(f"Loading catalog: {args.catalog}...")
    cat = pd.read_csv(args.catalog, usecols=["track_index", "artist_name", "track_name"])
    cat["_key"] = cat["artist_name"].str.lower().str.strip() + " ||| " + cat["track_name"].str.lower().str.strip()
    key_to_idx = cat.set_index("_key")["track_index"].to_dict()
    print(f"  {len(cat):,} songs in catalog")

    # ── Load enriched_genres.csv ────────────────────────────────────────────
    print(f"Loading {genres_path}...")
    gdf = pd.read_csv(genres_path)
    print(f"  {len(gdf):,} rows")

    # Ensure language column exists
    if "language" not in gdf.columns:
        gdf["language"] = None

    # Build a key for quick lookup
    gdf["_key"] = gdf["artist_name"].str.lower().str.strip() + " ||| " + gdf["track_name"].str.lower().str.strip()

    # Songs still needing genre (language we want for all, but prioritise genre gaps)
    need_genre = gdf["genre_enriched"].isna()
    print(f"  {need_genre.sum():,} songs missing genre_enriched")
    print(f"  {gdf['language'].isna().sum():,} songs missing language")

    # Only fetch for songs missing genre (language is a free bonus from same call)
    todo = gdf[need_genre].copy().reset_index(drop=True)

    # Shard: each job takes every Nth row starting at shard_id
    if args.num_shards > 1:
        todo = todo.iloc[args.shard_id::args.num_shards].copy()
        print(f"  Shard {args.shard_id}/{args.num_shards}: {len(todo):,} songs")

    if args.limit:
        todo = todo.head(args.limit)
    print(f"  Will fetch: {len(todo):,} songs\n")

    # Use distinct user-agent per shard to be polite to MusicBrainz
    musicbrainzngs.set_useragent(
        f"top50k-enrichment-shard{args.shard_id}", "1.0", "research"
    )

    # ── Checkpoint DB ───────────────────────────────────────────────────────
    conn = init_db(str(db_path))
    done_keys = set(
        r[0].lower().strip() + " ||| " + r[1].lower().strip()
        for r in conn.execute("SELECT artist_name, track_name FROM results").fetchall()
    )
    error_keys = set(
        r[0].lower().strip() + " ||| " + r[1].lower().strip()
        for r in conn.execute("SELECT artist_name, track_name FROM errors").fetchall()
    )
    skip_keys = done_keys | error_keys
    todo = todo[~todo["_key"].isin(skip_keys)]
    print(f"  {len(skip_keys):,} already in checkpoint DB — {len(todo):,} remaining\n")

    # ── Fetch ───────────────────────────────────────────────────────────────
    for i, (_, song) in enumerate(todo.iterrows()):
        artist = song["artist_name"]
        track = song["track_name"]
        key = song["_key"]

        # Use real track_index from the catalog
        idx = int(key_to_idx.get(key, -1))

        try:
            data = query_mb(artist, track)

            if data["mb_score"] < args.min_score:
                data.update({
                    "mb_recording_id": None,
                    "first_release_year": None,
                    "mb_genre": None,
                    "mb_language": None,
                    "mb_tags": None,
                })

            save_result(conn, {
                "track_index": idx,
                "artist_name": artist,
                "track_name": track,
                **data,
            })

        except Exception as e:
            save_error(conn, idx, artist, track, str(e))

        if (i + 1) % 100 == 0:
            total_done = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            print(f"  [{i+1}/{len(todo)}] fetched — {total_done:,} total in DB")

        time.sleep(1.1)  # MusicBrainz ToS: max 1 req/sec

    total_done = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    total_err  = conn.execute("SELECT COUNT(*) FROM errors").fetchone()[0]
    print(f"\nShard {args.shard_id} done.")
    print(f"  Results: {total_done:,}  |  Errors: {total_err:,}")
    print(f"  DB: {db_path}")
    print(f"\nRun merge_mb_shards.py after all shards complete to apply to enriched_genres.csv.")
    conn.close()


if __name__ == "__main__":
    main()
