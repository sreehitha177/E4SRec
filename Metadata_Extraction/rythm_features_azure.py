import json
import os
import time
import pandas as pd
from typing import Optional
from litellm import completion

# ── Config ─────────────────────────────────────────────────────────────────────

# Azure
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_BASE = "https://azure-sm.services.ai.azure.com/api/projects/Azure-SM-project/openai/v1/"
MODEL_NAME = "gpt-5"   # use your Azure deployment name here if different

# Files
OUTPUT_PATH = "/work/pi_dagarwal_umass_edu/project_7/swetha/Metadata_Extraction/outputs/rhythm_annotations_remaining_optimized.csv"
LASTFM_PATH = "/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/shared/top_50k_songs.csv"
VALIDATION_PATH = "/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/shared/validation_df.csv"
GENES_PATH = "/work/pi_dagarwal_umass_edu/project_7/swetha/Metadata_Extraction/mgphot_genes.tsv"

# Runtime settings
BATCH_SIZE = 50
STOP_AFTER_FAILURES = 10
REQUEST_TIMEOUT = 60
RETRIES_PER_TRACK = 2
SLEEP_BETWEEN_CALLS = 0.4

# Budget guard
MAX_BUDGET_USD = 30.0
ESTIMATED_COST_PER_CALL_USD = 0.003  # conservative stop-early estimate
total_estimated_spend = 0.0

# ── Feature definitions ────────────────────────────────────────────────────────

rhythm_features = pd.read_csv(GENES_PATH, sep="\t").iloc[9:19]
RHYTHM_FEATURES = rhythm_features["name"].tolist()

FEATURE_DEFINITIONS = {
    "Tempo": "0 slow, 5 fast",
    "Cut Time Feel": "0 normal 4/4, 5 strong half-time/cut-time",
    "Triple Meter": "0 no triple feel, 5 waltz/triple throughout",
    "Compound Meter": "0 no 6/8 or 12/8 feel, 5 dominant compound meter",
    "Odd Meter": "0 standard meter, 5 strong odd meter",
    "Swing Feel": "0 straight 8ths, 5 strong swing",
    "Shuffle Feel": "0 none, 5 strong shuffle groove",
    "Syncopation Low to High": "0 on-beat, 5 highly syncopated",
    "Backbeat": "0 none, 5 very strong 2-and-4 accent",
    "Danceability": "0 not danceable, 5 highly danceable",
}

SYSTEM_PROMPT = (
    "Rate rhythm features of a song. "
    "Return only valid JSON with integer values from 0 to 5."
)

# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_user_prompt(artist: str, title: str) -> str:
    feature_block = "; ".join(
        f"{feat}: {FEATURE_DEFINITIONS[feat]}"
        for feat in RHYTHM_FEATURES
    )
    keys_str = ", ".join(RHYTHM_FEATURES)

    return (
        f'Track="{title}" Artist="{artist}". '
        f'Rate these rhythm features 0-5 using music knowledge only. '
        f'{feature_block}. '
        f'Return JSON only with keys: {keys_str}. '
        f'Use integers only.'
    )

# ── Budget helpers ─────────────────────────────────────────────────────────────

def check_budget_before_call():
    global total_estimated_spend
    projected = total_estimated_spend + ESTIMATED_COST_PER_CALL_USD

    if projected > MAX_BUDGET_USD:
        raise RuntimeError(
            f"Budget stop triggered. Current estimated spend=${total_estimated_spend:.2f}. "
            f"Next call may exceed limit ${MAX_BUDGET_USD:.2f}. Stopping to avoid charges."
        )

def record_estimated_spend():
    global total_estimated_spend
    total_estimated_spend += ESTIMATED_COST_PER_CALL_USD
    print(f"Estimated spend: ${total_estimated_spend:.2f} / ${MAX_BUDGET_USD:.2f}")

# ── Core annotation ────────────────────────────────────────────────────────────

def validate_config():
    if not API_KEY:
        raise RuntimeError(
            "Missing AZURE_OPENAI_API_KEY. Set it first:\n"
            'export AZURE_OPENAI_API_KEY="your_key_here"'
        )

def annotate_track(
    artist: str,
    title: str,
    retries: int = RETRIES_PER_TRACK,
    fail_fast: bool = False
) -> Optional[dict]:
    prompt = build_user_prompt(artist, title)
    last_error = None

    for attempt in range(retries):
        try:
            check_budget_before_call()

            response = completion(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                api_base=API_BASE,
                api_key=API_KEY,
                response_format={"type": "json_object"},
                timeout=REQUEST_TIMEOUT,
            )

            record_estimated_spend()

            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Empty response content")

            scores = json.loads(content)
            normalized = {}

            for feat in RHYTHM_FEATURES:
                val = scores.get(feat)

                if val is None:
                    raise RuntimeError(f"Missing key: {feat}")
                if not isinstance(val, (int, float)):
                    raise RuntimeError(f"Non-numeric value for {feat}: {val}")
                if val < 0 or val > 5:
                    raise RuntimeError(f"Out-of-range value for {feat}: {val}")

                normalized[feat] = float(val) / 5.0

            time.sleep(SLEEP_BETWEEN_CALLS)
            return normalized

        except Exception as e:
            last_error = e
            print(f"Attempt {attempt + 1}/{retries} failed for {artist} - {title}: {e}")

            if fail_fast:
                raise

            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    print(f"Failed for {artist} - {title}: {last_error}")
    return None

def preflight_check(df, artist_col="artist_name", title_col="track_name"):
    if df.empty:
        raise RuntimeError("No tracks left to annotate.")

    sample = df[[artist_col, title_col]].drop_duplicates().head(1)
    if sample.empty:
        raise RuntimeError("Could not prepare preflight sample.")

    artist = str(sample.iloc[0][artist_col])
    title = str(sample.iloc[0][title_col])

    print(f"Running preflight on: {artist} - {title}")

    result = annotate_track(
        artist=artist,
        title=title,
        retries=1,
        fail_fast=True
    )

    if result is None:
        raise RuntimeError("Preflight failed: no result returned.")

    missing = [feat for feat in RHYTHM_FEATURES if feat not in result or result[feat] is None]
    if missing:
        raise RuntimeError(f"Preflight failed: missing features {missing}")

    print("Preflight passed.")

# ── Batch annotation ───────────────────────────────────────────────────────────

def annotate_df_safe(
    df,
    artist_col="artist_name",
    title_col="track_name",
    stop_after_failures=1
):
    unique_tracks = df[[artist_col, title_col]].drop_duplicates().reset_index(drop=True)
    results = []
    consecutive_failures = 0

    for i, row in unique_tracks.iterrows():
        scores = annotate_track(
            artist=row[artist_col],
            title=row[title_col],
            retries=RETRIES_PER_TRACK,
            fail_fast=False
        )

        entry = {
            artist_col: row[artist_col],
            title_col: row[title_col]
        }
        entry.update(scores if scores else {feat: None for feat in RHYTHM_FEATURES})
        results.append(entry)

        if scores is not None:
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        print(f"{i + 1}/{len(unique_tracks)} done")

        if consecutive_failures >= stop_after_failures:
            raise RuntimeError(
                f"Stopping run: {consecutive_failures} consecutive failure(s)."
            )

    return pd.DataFrame(results)

# ── Data preparation ───────────────────────────────────────────────────────────

def normalize_name_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["artist_name"] = df["artist_name"].astype(str).str.strip().str.lower()
    df["track_name"] = df["track_name"].astype(str).str.strip().str.lower()
    return df

def load_remaining_tracks(output_path: str) -> pd.DataFrame:
    lastfm_df = pd.read_csv(LASTFM_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    lastfm_df = lastfm_df[["artist_name", "track_name"]]
    validation_df = validation_df[["artist_name", "track_name"]]

    lastfm_df = normalize_name_columns(lastfm_df)
    validation_df = normalize_name_columns(validation_df)

    all_tracks = lastfm_df.drop_duplicates()
    validation_tracks = validation_df.drop_duplicates()

    remaining_df = all_tracks.merge(
        validation_tracks,
        on=["artist_name", "track_name"],
        how="left",
        indicator=True
    )
    remaining_df = remaining_df[remaining_df["_merge"] == "left_only"].drop(columns="_merge")

    print(f"Total tracks: {len(all_tracks)}")
    print(f"Validation excluded: {len(validation_tracks)}")
    print(f"After validation exclusion: {len(remaining_df)}")

    # Resume from where previous output stopped
    if pd.io.common.file_exists(output_path):
        done_df = pd.read_csv(output_path)
        done_df = done_df[["artist_name", "track_name"]]
        done_df = normalize_name_columns(done_df)

        done_keys = set(zip(done_df["artist_name"], done_df["track_name"]))

        remaining_df = remaining_df[
            ~remaining_df.apply(
                lambda r: (r["artist_name"], r["track_name"]) in done_keys,
                axis=1
            )
        ]

        print(f"Skipping {len(done_keys)} already completed tracks")

    remaining_df = remaining_df.reset_index(drop=True)
    print(f"Tracks left to annotate: {len(remaining_df)}")

    return remaining_df

# ── Saving ─────────────────────────────────────────────────────────────────────

def append_and_save(batch_annotations: pd.DataFrame, output_path: str):
    if batch_annotations.empty:
        print("No new rows to save.")
        return

    if pd.io.common.file_exists(output_path):
        old_df = pd.read_csv(output_path)
        final_df = pd.concat([old_df, batch_annotations], ignore_index=True)
    else:
        final_df = batch_annotations

    final_df.to_csv(output_path, index=False)
    print(f"Saved {len(final_df)} total tracks to {output_path}")

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    validate_config()

    remaining_df = load_remaining_tracks(OUTPUT_PATH)

    if remaining_df.empty:
        print("No tracks left to annotate.")
        raise SystemExit(0)

    print(f"\nTracks queued this run: {len(remaining_df)}")

    # One-call safety check
    preflight_check(remaining_df)

    # Single-row safety batch
    first_batch_size = min(1, len(remaining_df))
    if first_batch_size > 0:
        print(f"\nRunning safety batch of {first_batch_size} track first...")
        safety_batch = remaining_df.iloc[:first_batch_size]

        safety_annotations = annotate_df_safe(
            safety_batch,
            stop_after_failures=1
        )
        append_and_save(safety_annotations, OUTPUT_PATH)

        remaining_df = remaining_df.iloc[first_batch_size:].reset_index(drop=True)

    # Main processing
    for start in range(0, len(remaining_df), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(remaining_df))
        batch_df = remaining_df.iloc[start:end]

        print(f"\nAnnotating batch rows {start} to {end - 1} ...")

        batch_annotations = annotate_df_safe(
            batch_df,
            stop_after_failures=STOP_AFTER_FAILURES
        )

        append_and_save(batch_annotations, OUTPUT_PATH)

    print("Annotation run completed.")
    print(f"Final estimated spend: ${total_estimated_spend:.2f}")