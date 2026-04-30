import json
import os
import time
from typing import Optional

import pandas as pd
from google import genai
from google.genai import types

# =============================================================================
# Config
# =============================================================================

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")

OUTPUT_PATH = "/work/pi_dagarwal_umass_edu/project_7/swetha/Metadata_Extraction/outputs/harmony_annotations_remaining_gemini.csv"
LASTFM_PATH = "/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/shared/top_50k_songs.csv"
VALIDATION_PATH = "/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/shared/validation_df.csv"

API_BATCH_SIZE = 5
REQUEST_TIMEOUT = 90
RETRIES_PER_BATCH = 3
STOP_AFTER_CONSECUTIVE_FAILURES = 3
INITIAL_RETRY_SLEEP = 2
MAX_RETRY_SLEEP = 20

SHORT_TO_FULL = {
    "majmin": "Minor / Major Key Tonality",
    "harm_soph": "Harmonic Sophistication",
}

SYSTEM_PROMPT = (
    "You rate harmony features for songs. "
    "Return only valid JSON. "
    "Values must be integers from 0 to 5."
)

RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "majmin": {"type": "integer"},
                    "harm_soph": {"type": "integer"},
                },
                "required": ["id", "majmin", "harm_soph"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


# =============================================================================
# Gemini client
# =============================================================================

def get_client():
    if not API_KEY:
        raise RuntimeError(
            "Missing GEMINI_API_KEY.\n"
            'Set it first: export GEMINI_API_KEY="your_key_here"'
        )
    return genai.Client(api_key=API_KEY)


# =============================================================================
# Helpers
# =============================================================================

def normalize_name_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["artist_name"] = df["artist_name"].astype(str).str.strip().str.lower()
    df["track_name"] = df["track_name"].astype(str).str.strip().str.lower()
    return df


def safe_backoff_sleep(attempt: int):
    sleep_seconds = min(INITIAL_RETRY_SLEEP * (2 ** attempt), MAX_RETRY_SLEEP)
    time.sleep(sleep_seconds)


def coerce_score(value, key_name: str) -> int:
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"Non-numeric value for {key_name}: {value}")
    ivalue = int(value)
    if ivalue < 0 or ivalue > 5:
        raise RuntimeError(f"Out-of-range value for {key_name}: {ivalue}")
    return ivalue


# =============================================================================
# Prompt
# =============================================================================

def build_batch_prompt(batch_df: pd.DataFrame) -> str:
    song_lines = []
    for i, row in batch_df.reset_index(drop=True).iterrows():
        artist = str(row["artist_name"])
        title = str(row["track_name"])
        song_lines.append(f'{i}. Track="{title}" Artist="{artist}"')

    songs_block = "\n".join(song_lines)

    return f"""
Rate these harmony features for each song.

Keys:
majmin: 0=minor, 5=major, 2-3=mixed/ambiguous
harm_soph: 0=simple diatonic/triadic, 5=highly chromatic/complex

Return only JSON with this shape:
{{
  "results": [
    {{"id": 0, "majmin": 0, "harm_soph": 0}}
  ]
}}

Songs:
{songs_block}
""".strip()


# =============================================================================
# Core annotation
# =============================================================================

def annotate_batch(
    client: genai.Client,
    batch_df: pd.DataFrame,
    retries: int = RETRIES_PER_BATCH,
    fail_fast: bool = False,
) -> Optional[pd.DataFrame]:
    prompt = build_batch_prompt(batch_df)
    last_error = None

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_json_schema=RESPONSE_JSON_SCHEMA,
                    temperature=0,
                ),
            )

            text = response.text
            if not text:
                raise RuntimeError("Empty response text")

            data = json.loads(text)
            if "results" not in data or not isinstance(data["results"], list):
                raise RuntimeError("Missing or invalid 'results' array")

            parsed_rows = []
            seen_ids = set()
            expected_batch_len = len(batch_df)

            for item in data["results"]:
                idx = item.get("id")
                if not isinstance(idx, int):
                    raise RuntimeError(f"Missing/invalid id in item: {item}")
                if idx < 0 or idx >= expected_batch_len:
                    raise RuntimeError(f"Returned id out of range: {idx}")
                if idx in seen_ids:
                    raise RuntimeError(f"Duplicate id in response: {idx}")

                seen_ids.add(idx)

                majmin = coerce_score(item.get("majmin"), "majmin")
                harm_soph = coerce_score(item.get("harm_soph"), "harm_soph")

                source_row = batch_df.iloc[idx]
                parsed_rows.append({
                    "artist_name": source_row["artist_name"],
                    "track_name": source_row["track_name"],
                    SHORT_TO_FULL["majmin"]: majmin / 5.0,
                    SHORT_TO_FULL["harm_soph"]: harm_soph / 5.0,
                })

            missing_ids = [i for i in range(expected_batch_len) if i not in seen_ids]
            if missing_ids:
                raise RuntimeError(f"Missing ids in response: {missing_ids}")

            result_df = pd.DataFrame(parsed_rows)

            # keep same order as input batch
            result_df["__input_order"] = range(len(result_df))
            result_df = result_df.drop(columns="__input_order").reset_index(drop=True)

            return result_df

        except Exception as e:
            last_error = e
            print(f"Batch attempt {attempt + 1}/{retries} failed: {e}")

            if fail_fast:
                raise

            if attempt < retries - 1:
                safe_backoff_sleep(attempt)

    print(f"Batch failed after {retries} attempt(s): {last_error}")
    return None


def preflight_check(client: genai.Client, df: pd.DataFrame):
    if df.empty:
        raise RuntimeError("No tracks left to annotate.")

    sample = df.head(min(API_BATCH_SIZE, len(df))).reset_index(drop=True)
    print(f"Running preflight on {len(sample)} track(s)...")

    result = annotate_batch(client, sample, retries=1, fail_fast=True)
    if result is None or result.empty:
        raise RuntimeError("Preflight failed: no result returned.")

    expected_cols = [
        "artist_name",
        "track_name",
        SHORT_TO_FULL["majmin"],
        SHORT_TO_FULL["harm_soph"],
    ]
    missing_cols = [c for c in expected_cols if c not in result.columns]
    if missing_cols:
        raise RuntimeError(f"Preflight failed: missing columns {missing_cols}")

    print("Preflight passed.")


# =============================================================================
# Data loading / resume
# =============================================================================

def load_remaining_tracks(output_path: str) -> pd.DataFrame:
    lastfm_df = pd.read_csv(LASTFM_PATH, usecols=["artist_name", "track_name"])
    validation_df = pd.read_csv(VALIDATION_PATH, usecols=["artist_name", "track_name"])

    lastfm_df = normalize_name_columns(lastfm_df).drop_duplicates()
    validation_df = normalize_name_columns(validation_df).drop_duplicates()

    remaining_df = lastfm_df.merge(
        validation_df,
        on=["artist_name", "track_name"],
        how="left",
        indicator=True,
    )
    remaining_df = remaining_df[remaining_df["_merge"] == "left_only"].drop(columns="_merge")

    print(f"Total tracks: {len(lastfm_df)}")
    print(f"Validation excluded: {len(validation_df)}")
    print(f"After validation exclusion: {len(remaining_df)}")

    if pd.io.common.file_exists(output_path):
        done_df = pd.read_csv(output_path, usecols=["artist_name", "track_name"])
        done_df = normalize_name_columns(done_df).drop_duplicates()

        remaining_df = remaining_df.merge(
            done_df,
            on=["artist_name", "track_name"],
            how="left",
            indicator=True,
        )
        remaining_df = remaining_df[remaining_df["_merge"] == "left_only"].drop(columns="_merge")

        print(f"Skipping {len(done_df)} already completed tracks")

    remaining_df = remaining_df.reset_index(drop=True)
    print(f"Tracks left to annotate: {len(remaining_df)}")
    return remaining_df


# =============================================================================
# Saving
# =============================================================================

def append_and_save(batch_annotations: pd.DataFrame, output_path: str):
    if batch_annotations is None or batch_annotations.empty:
        print("No new rows to save.")
        return

    file_exists = pd.io.common.file_exists(output_path)
    batch_annotations.to_csv(
        output_path,
        mode="a",
        header=not file_exists,
        index=False,
    )
    print(f"Appended {len(batch_annotations)} row(s) to {output_path}")


# =============================================================================
# Main batch loop
# =============================================================================

def process_batches(client: genai.Client, remaining_df: pd.DataFrame):
    consecutive_failures = 0

    for start in range(0, len(remaining_df), API_BATCH_SIZE):
        end = min(start + API_BATCH_SIZE, len(remaining_df))
        batch_df = remaining_df.iloc[start:end].reset_index(drop=True)

        print(f"\nAnnotating rows {start} to {end - 1} (batch size={len(batch_df)}) ...")

        batch_annotations = annotate_batch(client, batch_df, retries=RETRIES_PER_BATCH, fail_fast=False)

        # fallback to single-song calls if full batch fails
        if batch_annotations is None:
            print("Batch failed. Falling back to single-track processing for this batch...")
            fallback_rows = []

            for i in range(len(batch_df)):
                single_df = batch_df.iloc[[i]].reset_index(drop=True)
                single_result = annotate_batch(client, single_df, retries=RETRIES_PER_BATCH, fail_fast=False)

                if single_result is not None and not single_result.empty:
                    fallback_rows.append(single_result.iloc[0].to_dict())
                else:
                    row = single_df.iloc[0]
                    fallback_rows.append({
                        "artist_name": row["artist_name"],
                        "track_name": row["track_name"],
                        SHORT_TO_FULL["majmin"]: None,
                        SHORT_TO_FULL["harm_soph"]: None,
                    })

            batch_annotations = pd.DataFrame(fallback_rows)

        append_and_save(batch_annotations, OUTPUT_PATH)

        # reset or increment failure streak
        value_cols = [SHORT_TO_FULL["majmin"], SHORT_TO_FULL["harm_soph"]]
        if batch_annotations[value_cols].notna().any(axis=1).any():
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        print(f"Completed {end}/{len(remaining_df)} tracks")

        if consecutive_failures >= STOP_AFTER_CONSECUTIVE_FAILURES:
            raise RuntimeError(
                f"Stopping run: {consecutive_failures} consecutive failed batch(es)."
            )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    client = get_client()

    remaining_df = load_remaining_tracks(OUTPUT_PATH)

    if remaining_df.empty:
        print("No tracks left to annotate.")
        raise SystemExit(0)

    print(f"\nTracks queued this run: {len(remaining_df)}")
    print(f"Model: {MODEL_NAME}")
    print(f"API batch size: {API_BATCH_SIZE}")

    preflight_check(client, remaining_df)
    process_batches(client, remaining_df)

    print("\nAnnotation run completed.")