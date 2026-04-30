import ast
import json
import time
import pandas as pd
from scipy.stats import spearmanr
from typing import Optional
from litellm import completion
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Load sonority features ─────────────────────────────────────────────────────

sonority_features = pd.read_csv("mgphot_genes.tsv", sep="\t").iloc[43:49]
SONORITY_FEATURES = sonority_features["name"].tolist()

# Short definitions to reduce token usage
FEATURE_SHORT = {
    "Live Recording": "0 studio, 5 clearly live",
    "Audio Production": "0 poor/noisy, 5 polished/high quality",
    "Aural Intensity": "0 very soft, 5 very loud/intense",
    "Acoustic Sonority": "0 no acoustic sound, 5 mostly acoustic",
    "Electric Sonority": "0 no electric instruments, 5 mostly electric",
    "Synthetic Sonority": "0 no synth sound, 5 mostly synthetic",
}

SONORITY_FEATURE_INDICES = {
    "Live Recording": 43,
    "Audio Production": 44,
    "Aural Intensity": 45,
    "Acoustic Sonority": 46,
    "Electric Sonority": 47,
    "Synthetic Sonority": 48,
}

# ── Compact prompt builder ────────────────────────────────────────────────────

def build_user_prompt(artist: str, title: str) -> str:
    feature_block = "; ".join(
        f"{feat}: {FEATURE_SHORT[feat]}" for feat in SONORITY_FEATURES
    )
    keys_str = ", ".join(SONORITY_FEATURES)

    return (
        f'Track="{title}" Artist="{artist}". '
        f'Rate these sonority features 0-5 using music knowledge only. '
        f'{feature_block}. '
        f'Return JSON only with keys: {keys_str}. '
        f'Use integers only.'
    )

# ── Annotation ─────────────────────────────────────────────────────────────────

def annotate_track(artist: str, title: str, api_key: str, retries: int = 3) -> Optional[dict]:
    prompt = build_user_prompt(artist, title)

    for attempt in range(retries):
        try:
            response = completion(
                model="openai/gpt5",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                api_base="https://thekeymaker.umass.edu/",
                api_key=api_key,
                response_format={"type": "json_object"},
            )

            scores = json.loads(response.choices[0].message.content)

            normalized = {}
            for feat in SONORITY_FEATURES:
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

def annotate_df_parallel(df, artist_col="artist_name", title_col="track_name", api_key: str = None, max_workers=10):
    unique_tracks = df[[artist_col, title_col]].drop_duplicates()
    results = []

    def annotate_row(row):
        scores = annotate_track(row[artist_col], row[title_col], api_key)
        entry = {artist_col: row[artist_col], title_col: row[title_col]}
        entry.update(scores if scores else {feat: None for feat in SONORITY_FEATURES})
        return entry

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(annotate_row, row): row for _, row in unique_tracks.iterrows()}
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if i % 100 == 0:
                print(f"{i}/{len(unique_tracks)} done")

    return pd.DataFrame(results)

# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_annotations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat, idx in SONORITY_FEATURE_INDICES.items():
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

    for feat, idx in SONORITY_FEATURE_INDICES.items():
        gt = df["gene_values"].apply(lambda x: x[idx])
        pred = df[feat].dropna()

        if len(pred) > 0:
            bias = pred.mean() - gt.mean()
            print(f"{feat:<30} {gt.mean():>8.3f} {pred.mean():>10.3f} {bias:>8.3f}")

# ── Main execution example ─────────────────────────────────────────────────────
if __name__ == "__main__":
    API_KEY = "YOUR_API_KEY"
    LASTFM_PATH = "hmagapu/metadata/shared/user_sessions_lastfm1k_minuser1000_minitem7_sessgap1200_minsesslen10_minhist50.csv"

    # Load full LastFM data
    col_names = ["user_id", "timestamp", "artist_id", "artist_name", "track_id", "track_name", "session_id"]
    lastfm_df = pd.read_csv(LASTFM_PATH, names=col_names, header=None)

    # Load validation data (already annotated, so exclude it)
    validation_df = pd.read_csv("hmagapu/metadata/shared/validation_df.csv")

    # Keep only unique artist-track pairs from validation
    validation_tracks = validation_df[["artist_name", "track_name"]].drop_duplicates()

    # Keep only unique artist-track pairs from full dataset
    all_tracks = lastfm_df[["artist_name", "track_name"]].drop_duplicates()

    # Remove validation tracks from full set
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

    # Annotate only non-validation tracks
    print("\nAnnotating all tracks except validation set...")
    full_annotations = annotate_df_parallel(
        tracks_to_annotate,
        artist_col="artist_name",
        title_col="track_name",
        api_key=API_KEY,
        max_workers=10
    )

    # Save output
    full_annotations.to_csv("sonority_annotations.csv", index=False)
    print(f"Saved {len(full_annotations)} tracks to sonority_annotations.csv")