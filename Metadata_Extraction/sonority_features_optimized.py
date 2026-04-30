import json
import os
import time
from typing import Optional

import pandas as pd
from litellm import completion

# =============================================================================
# CONFIG
# =============================================================================

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_BASE = "https://azure-sm.services.ai.azure.com/api/projects/Azure-SM-project/openai/v1/"
MODEL_NAME = "gpt-5"

OUTPUT_PATH = "/work/pi_dagarwal_umass_edu/project_7/swetha/Metadata_Extraction/outputs/sonority_annotations_remaining_optimized.csv"
LASTFM_PATH = "/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/shared/top_50k_songs.csv"
VALIDATION_PATH = "/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/shared/validation_df.csv"

API_BATCH_SIZE = 5
RETRIES = 2
STOP_AFTER_FAILURES = 3

# =============================================================================
# FEATURES
# =============================================================================

SHORT_TO_FULL = {
    "live": "Live Recording",
    "prod": "Audio Production",
    "intense": "Aural Intensity",
    "acoustic": "Acoustic Sonority",
    "electric": "Electric Sonority",
    "synth": "Synthetic Sonority",
}

# Very short prompt to save tokens, but still grounded in the paper
SYSTEM_PROMPT = (
    "Rate MGPHot sonority features from Oramas et al. 2025. "
    "Use 0-5 integers only. Return JSON."
)

def build_prompt(batch_df: pd.DataFrame) -> str:
    songs = []
    for i, row in batch_df.reset_index(drop=True).iterrows():
        songs.append(f'{i}:{row["track_name"]}|{row["artist_name"]}')
    songs_block = "\n".join(songs)

    # Short schema + one example only
    return (
        "Fields: live,prod,intense,acoustic,electric,synth\n"
        '{"results":[{"id":0,"live":0,"prod":0,"intense":0,"acoustic":0,"electric":0,"synth":0}]}\n'
        f"{songs_block}"
    )

# =============================================================================
# HELPERS
# =============================================================================

def validate_config():
    if not API_KEY:
        raise RuntimeError("Set AZURE_OPENAI_API_KEY")

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["artist_name"] = df["artist_name"].str.lower().str.strip()
    df["track_name"] = df["track_name"].str.lower().str.strip()
    return df

# =============================================================================
# ANNOTATION
# =============================================================================

def annotate_batch(batch_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    prompt = build_prompt(batch_df)

    for attempt in range(RETRIES):
        try:
            response = completion(
                model=MODEL_NAME,
                api_base=API_BASE,
                api_key=API_KEY,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )

            data = json.loads(response.choices[0].message.content)

            rows = []
            for item in data["results"]:
                idx = item["id"]

                # Guard against bad ids
                if idx < 0 or idx >= len(batch_df):
                    continue

                row = batch_df.iloc[idx]

                parsed = {
                    "artist_name": row["artist_name"],
                    "track_name": row["track_name"],
                }

                for short, full in SHORT_TO_FULL.items():
                    val = item.get(short)
                    parsed[full] = None if val is None else float(val) / 5.0

                rows.append(parsed)

            if not rows:
                raise ValueError("No valid rows parsed from model output")

            return pd.DataFrame(rows)

        except Exception as e:
            print(f"Retry {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)

    return None

# =============================================================================
# DATA
# =============================================================================

def load_remaining(output_path: str) -> pd.DataFrame:
    lastfm = pd.read_csv(LASTFM_PATH)[["artist_name", "track_name"]]
    val = pd.read_csv(VALIDATION_PATH)[["artist_name", "track_name"]]

    lastfm = normalize(lastfm).drop_duplicates()
    val = normalize(val).drop_duplicates()

    # Exclude validation songs
    remaining = lastfm.merge(
        val,
        on=["artist_name", "track_name"],
        how="left",
        indicator=True
    )
    remaining = remaining[remaining["_merge"] == "left_only"].drop(columns="_merge")

    # Exclude already completed songs
    if os.path.exists(output_path):
        done = pd.read_csv(output_path)[["artist_name", "track_name"]]
        done = normalize(done).drop_duplicates()

        remaining = remaining.merge(
            done,
            on=["artist_name", "track_name"],
            how="left",
            indicator=True
        )
        remaining = remaining[remaining["_merge"] == "left_only"].drop(columns="_merge")

    return remaining.reset_index(drop=True)

# =============================================================================
# SAVE
# =============================================================================

def append(df: Optional[pd.DataFrame], path: str):
    if df is None or df.empty:
        return

    exists = os.path.exists(path)
    df.to_csv(path, mode="a", header=not exists, index=False)

# =============================================================================
# MAIN
# =============================================================================

def run():
    validate_config()

    df = load_remaining(OUTPUT_PATH)
    print(f"Tracks left: {len(df)}")

    failures = 0

    for i in range(0, len(df), API_BATCH_SIZE):
        batch = df.iloc[i:i + API_BATCH_SIZE].reset_index(drop=True)
        print(f"Processing {i} -> {i + len(batch)}")

        result = annotate_batch(batch)

        if result is None:
            failures += 1
        else:
            failures = 0
            append(result, OUTPUT_PATH)

        if failures >= STOP_AFTER_FAILURES:
            print("Too many failures — stopping")
            break

    print("Done")

# =============================================================================

if __name__ == "__main__":
    run()