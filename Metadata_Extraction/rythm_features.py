import ast
import json
import time
import pandas as pd
from scipy.stats import spearmanr
from typing import Optional
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "YOUR_API_KEY"

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://dolby-feature-extraction.services.ai.azure.com/",
    api_key=API_KEY
)

# ── Load rhythm features ───────────────────────────────────────────────────────

rhythm_features = pd.read_csv("mgphot_genes.tsv", sep="\t").iloc[9:19]
RHYTHM_FEATURES = rhythm_features["name"].tolist()

# Much shorter definitions to reduce prompt tokens
FEATURE_SHORT = {
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

RHYTHM_FEATURE_INDICES = {
    "Tempo": 9,
    "Cut Time Feel": 10,
    "Triple Meter": 11,
    "Compound Meter": 12,
    "Odd Meter": 13,
    "Swing Feel": 14,
    "Shuffle Feel": 15,
    "Syncopation Low to High": 16,
    "Backbeat": 17,
    "Danceability": 18,
}

# ── Compact prompt builder ────────────────────────────────────────────────────

def build_user_prompt(artist: str, title: str) -> str:
    feature_block = "; ".join(
        f"{feat}: {FEATURE_SHORT[feat]}" for feat in RHYTHM_FEATURES
    )
    keys_str = ", ".join(RHYTHM_FEATURES)

    return (
        f'Track="{title}" Artist="{artist}". '
        f'Rate these rhythm features 0-5 using music knowledge only. '
        f'{feature_block}. '
        f'Return JSON only with keys: {keys_str}. '
        f'Use integers only.'
    )

# ── Annotation ────────────────────────────────────────────────────────────────

def annotate_track(artist: str, title: str, retries: int = 3) -> Optional[dict]:
    prompt = build_user_prompt(artist, title)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
            )

            scores = json.loads(response.choices[0].message.content)

            normalized = {}
            for feat in RHYTHM_FEATURES:
                val = scores.get(feat)
                if val is None:
                    return None
                normalized[feat] = float(val) / 5.0

            return normalized

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Failed for {artist} - {title}: {e}")
                return None

def annotate_df_parallel(
    df,
    artist_col="artist_name",
    title_col="track_name",
    max_workers=10
):
    unique_tracks = df[[artist_col, title_col]].drop_duplicates()
    results = []

    def annotate_row(row):
        scores = annotate_track(row[artist_col], row[title_col])
        entry = {
            artist_col: row[artist_col],
            title_col: row[title_col]
        }
        entry.update(scores if scores else {feat: None for feat in RHYTHM_FEATURES})
        return entry

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(annotate_row, row): row
            for _, row in unique_tracks.iterrows()
        }

        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if i % 100 == 0:
                print(f"{i}/{len(unique_tracks)} done")

    return pd.DataFrame(results)

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_annotations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for feat, idx in RHYTHM_FEATURE_INDICES.items():
        gt = df["gene_values"].apply(lambda x: x[idx])
        pred = df[feat]
        mask = pred.notna()
        gt, pred = gt[mask], pred[mask]

        if len(gt) < 10:
            continue

        mae = (gt - pred).abs().mean()
        rho, _ = spearmanr(gt, pred)

        rows.append({
            "feature": feat,
            "n": len(gt),
            "MAE": round(mae, 4),
            "Spearman_rho": round(rho, 4),
        })

    return pd.DataFrame(rows).sort_values("Spearman_rho", ascending=False)

def check_bias(df: pd.DataFrame):
    print(f"{'Feature':<30} {'GT mean':>8} {'Pred mean':>10} {'Bias':>8}")
    print("-" * 60)

    for feat, idx in RHYTHM_FEATURE_INDICES.items():
        gt = df["gene_values"].apply(lambda x: x[idx])
        pred = df[feat].dropna()

        if len(pred) > 0:
            bias = pred.mean() - gt.mean()
            print(f"{feat:<30} {gt.mean():>8.3f} {pred.mean():>10.3f} {bias:>8.3f}")

# ── Main execution example ────────────────────────────────────────────────────
if __name__ == "__main__":
    LASTFM_PATH = "hmagapu/metadata/shared/user_sessions_lastfm1k_minuser1000_minitem7_sessgap1200_minsesslen10_minhist50.csv"
    col_names = [
        "user_id", "timestamp", "artist_id", "artist_name",
        "track_id", "track_name", "session_id"
    ]
    lastfm_df = pd.read_csv(LASTFM_PATH, names=col_names, header=None)

    validation_df = pd.read_csv("hmagapu/validation_df.csv")

    # normalize names so merge matches better
    for df_ in [lastfm_df, validation_df]:
        df_["artist_name"] = df_["artist_name"].astype(str).str.strip().str.lower()
        df_["track_name"] = df_["track_name"].astype(str).str.strip().str.lower()

    # unique tracks in full data
    all_tracks = lastfm_df[["artist_name", "track_name"]].drop_duplicates()

    # unique tracks already present in validation
    validation_tracks = validation_df[["artist_name", "track_name"]].drop_duplicates()

    # keep only tracks not in validation
    tracks_to_annotate = all_tracks.merge(
        validation_tracks,
        on=["artist_name", "track_name"],
        how="left",
        indicator=True
    )

    tracks_to_annotate = tracks_to_annotate[
        tracks_to_annotate["_merge"] == "left_only"
    ].drop(columns="_merge")

    print(f"Total unique tracks in LastFM: {len(all_tracks)}")
    print(f"Validation tracks excluded: {len(validation_tracks)}")
    print(f"Tracks left to annotate: {len(tracks_to_annotate)}")

    print("\nAnnotating all tracks except validation set...")
    full_annotations = annotate_df_parallel(
        tracks_to_annotate,
        artist_col="artist_name",
        title_col="track_name",
        max_workers=10
    )

    full_annotations.to_csv("rhythm_annotations.csv", index=False)
    print(f"Saved {len(full_annotations)} tracks to rhythm_annotations.csv")