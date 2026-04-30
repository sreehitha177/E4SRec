import json
import time
import pandas as pd
from typing import Optional
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "sk-O3lNsL5biATpqUmgUAKtHg"

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://azure-sm.openai.azure.com/",
    api_key=API_KEY
)

# ── Feature definitions ────────────────────────────────────────────────────────

harmony_features = pd.read_csv("mgphot_genes.tsv", sep="\t").iloc[7:9]
HARMONY_FEATURES = []
for idx, row in harmony_features.iterrows():
    print(row["name"])
    print(row["description"])
    print("**********************")
    HARMONY_FEATURES.append(row["name"])

# Shorter prompt to reduce API cost
SYSTEM_PROMPT = (
    "Rate harmonic features of a song. "
    "Return ONLY valid JSON with integer scores 0-5. "
    "No explanation, no extra text."
)

FEATURE_DEFINITIONS = {
    "Minor / Major Key Tonality": (
        "0=minor, 5=major, 2-3=mixed/ambiguous; based on tonic chord."
    ),
    "Harmonic Sophistication": (
        "0=I/IV/V, 1=diatonic, 2=secondary dominants, "
        "3=non-diatonic, 4=chromatic, 5=high chromatic."
    ),
}

def build_user_prompt(artist: str, title: str) -> str:
    feature_block = "\n".join(
        f'"{feat}": {FEATURE_DEFINITIONS[feat]}'
        for feat in HARMONY_FEATURES
    )
    return (
        f'{title} - {artist}\n'
        f'{feature_block}\n'
        f'Return JSON with keys: {HARMONY_FEATURES}'
    )

def annotate_track(artist: str, title: str, api_key: str, retries: int = 3) -> Optional[dict]:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5",  
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(artist, title)},
                ],
                response_format={"type": "json_object"},
            )

            scores = json.loads(response.choices[0].message.content)
            normalized = {}

            for feat in HARMONY_FEATURES:
                val = scores.get(feat)
                if val is None:
                    return None
                normalized[feat] = float(val) / 5.0

            return normalized

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Failed for {artist} – {title}: {e}")
                return None

def annotate_df_parallel(df, artist_col="artist_name", title_col="track_name", api_key: str = None, max_workers=10):
    unique_tracks = df[[artist_col, title_col]].drop_duplicates()
    results = []

    def annotate_row(row):
        scores = annotate_track(row[artist_col], row[title_col], api_key)
        entry = {artist_col: row[artist_col], title_col: row[title_col]}
        entry.update(scores if scores else {feat: None for feat in HARMONY_FEATURES})
        return entry

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(annotate_row, row): row for _, row in unique_tracks.iterrows()}
        for i, future in enumerate(as_completed(futures)):
            results.append(future.result())
            if i % 100 == 0:
                print(f"{i}/{len(unique_tracks)} done")

    return pd.DataFrame(results)

# ── Main execution: full remaining set only ───────────────────────────────────
if __name__ == "__main__":
    LASTFM_PATH = "swetha/lastfm_unique_tracks_formatted.csv"
    OUTPUT_PATH = "harmony_annotations.csv"
    VALIDATION_PATH = "hmagapu/metadata/shared/validation_df.csv"

    col_names = ["user_id", "timestamp", "artist_id", "artist_name", "track_id", "track_name", "session_id"]
    lastfm_df = pd.read_csv(LASTFM_PATH, names=col_names, header=None)

    # Load validation set (to exclude)
    validation_df = pd.read_csv(VALIDATION_PATH)

    # Normalize names for safe matching
    for df_ in [lastfm_df, validation_df]:
        df_["artist_name"] = df_["artist_name"].astype(str).str.strip().str.lower()
        df_["track_name"] = df_["track_name"].astype(str).str.strip().str.lower()

    # Unique tracks
    all_tracks = lastfm_df[["artist_name", "track_name"]].drop_duplicates()
    validation_tracks = validation_df[["artist_name", "track_name"]].drop_duplicates()

    # Remove validation tracks
    remaining_df = all_tracks.merge(
        validation_tracks,
        on=["artist_name", "track_name"],
        how="left",
        indicator=True
    )

    remaining_df = remaining_df[
        remaining_df["_merge"] == "left_only"
    ].drop(columns="_merge")

    print(f"Total tracks: {len(all_tracks)}")
    print(f"Validation excluded: {len(validation_tracks)}")
    print(f"After exclusion: {len(remaining_df)}")

    # Remove already processed tracks (your original logic)
    if pd.io.common.file_exists(OUTPUT_PATH):
        done_df = pd.read_csv(OUTPUT_PATH)

        # normalize for matching
        done_df["artist_name"] = done_df["artist_name"].astype(str).str.strip().str.lower()
        done_df["track_name"] = done_df["track_name"].astype(str).str.strip().str.lower()

        done_keys = set(zip(done_df["artist_name"], done_df["track_name"]))

        remaining_df = remaining_df[
            ~remaining_df.apply(
                lambda r: (r["artist_name"], r["track_name"]) in done_keys,
                axis=1
            )
        ]

        print(f"Skipping {len(done_keys)} already completed tracks")

    print(f"Annotating {len(remaining_df)} remaining tracks...")

    new_annotations = annotate_df_parallel(
        remaining_df,
        api_key=API_KEY,
        max_workers=10
    )

    # Append to existing file
    if pd.io.common.file_exists(OUTPUT_PATH):
        old_df = pd.read_csv(OUTPUT_PATH)
        final_df = pd.concat([old_df, new_annotations], ignore_index=True)
    else:
        final_df = new_annotations

    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(final_df)} total tracks to {OUTPUT_PATH}")