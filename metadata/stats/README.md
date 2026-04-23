# Catalog Enrichment: Release Year, Genre & Language

## Why

The top-50k song catalog (`top_50k_full_augmented.csv`) was built from LastFM scrobble data
spanning **2005–2009**. However, many tracks had release years from **2010–2022**, well outside
that window — because the year field reflected **re-releases and remasters** rather than original
release dates. Reporting those years would be inconsistent with the LastFM interaction data.

At the same time, the catalog was missing **genre** for ~82% of songs and **language** entirely.
Both are useful descriptive statistics for the benchmark.

This pipeline fixes all three by querying the
[MusicBrainz](https://musicbrainz.org/) open-source music database:

- **`first_release_year`** — earliest known release date for the recording, not a remaster
- **`genre`** — top community-voted genre tag for the recording
- **`language`** — language of the lyrics as stored in MusicBrainz (`text-representation.language`)

Language for songs MusicBrainz couldn't classify is filled in via **`langdetect`** on the track name
as a fallback.

The enriched output is written to **`top_50k_full_augmented_v2.csv`**.

---

## Scripts

| Script | What it does |
|--------|-------------|
| `fetch_musicbrainz.py` | Queries MusicBrainz for year, genre, and language for each song. Checkpoints results to a SQLite DB. Supports sharding (`--shard-id`, `--num-shards`) for parallel SLURM execution. Rate-limited to 1 req/s per MusicBrainz ToS. |
| `fetch_mb_array.sbatch` | SLURM array job (4 tasks) that runs `fetch_musicbrainz.py` in parallel, each handling ~12.5k songs. |
| `merge_mb_shards.py` | Merges the 4 shard SQLite DBs into one combined `musicbrainz_cache.db`, then writes `enriched_genres.csv` with genre and language per song. |
| `backfill_year.py` | Targeted backfill for songs where `first_release_year` is still NULL in the combined DB. Uses direct `get_recording_by_id` calls (faster, since MBIDs are already stored). Supports sharding. |
| `backfill_year_array.sbatch` | SLURM array job (4 tasks) for `backfill_year.py`. |
| `merge_musicbrainz.py` | Joins `musicbrainz_cache.db` + `enriched_genres.csv` back onto the original catalog CSV. Writes the final `top_50k_full_augmented_v2.csv`. |
| `fill_language_langdetect.py` | Fills any remaining NULL `language` values in v2 using `langdetect` on the track name. Run after `merge_musicbrainz.py`. |
| `benchmark_stats.ipynb` | Jupyter notebook computing summary statistics for both the LastFM 1K dataset and the enriched Top-50k catalog (session stats, audio features, genre distribution, language distribution). |

---

## Pipeline

```
top_50k_full_augmented.csv
        │
        ▼
fetch_musicbrainz.py  ×4 shards  (SLURM: fetch_mb_array.sbatch)
        │
        ▼  musicbrainz_cache_shard{0..3}.db
        │
merge_mb_shards.py
        │
        ▼  musicbrainz_cache.db  +  enriched_genres.csv
        │
        ├──► [optional] backfill_year.py  ×4 shards  (backfill_year_array.sbatch)
        │           └──► re-run merge_mb_shards.py to update combined DB
        │
merge_musicbrainz.py
        │
        ▼  top_50k_full_augmented_v2.csv  (year + genre + language merged in)
        │
fill_language_langdetect.py   (fallback: langdetect on track name for remaining NULLs)
        │
        ▼  top_50k_full_augmented_v2.csv  (final, ~99.8% language coverage)
```

---

## Usage

### 1. Fetch from MusicBrainz (SLURM)

```bash
# Expects top_50k_full_augmented.csv and output paths configured inside the sbatch file
sbatch fetch_mb_array.sbatch

# Monitor
squeue -u $USER
tail -f logs/mb_fetch_*_0.out
```

### 2. Merge shards

```bash
python merge_mb_shards.py \
  --shard-dbs /project/.../musicbrainz_cache_shard{0,1,2,3}.db \
  --combined-db /project/.../musicbrainz_cache.db \
  --genres-csv /project/.../enriched_genres.csv \
  --catalog /project/.../top_50k_full_augmented.csv
```

### 3. (Optional) Backfill missing years

```bash
sbatch backfill_year_array.sbatch
# then re-run merge_mb_shards.py
```

### 4. Build v2 catalog

```bash
python merge_musicbrainz.py
```

### 5. Fill remaining language gaps

```bash
python fill_language_langdetect.py
```

### 6. View stats

Open `benchmark_stats.ipynb` in Jupyter using the `dolby` conda environment.

---

## Output Coverage (final v2 file)

| Field | Filled | Missing |
|-------|--------|---------|
| `year` | ~65.8% | ~34.2% (MusicBrainz has no record) |
| `genre` | ~56.2% | ~43.8% |
| `language` | ~99.8% | ~0.2% (track name too short for langdetect) |

---

## Dependencies

```bash
conda activate dolby
pip install musicbrainzngs langdetect
```

MusicBrainz API: no key required, but rate-limited to **1 request/second**.
Scripts use `time.sleep(1.1)` between calls and `socket.setdefaulttimeout(30)` to avoid hangs.
