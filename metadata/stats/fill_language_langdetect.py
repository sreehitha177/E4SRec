"""
Fill missing `language` in top_50k_full_augmented_v2.csv using langdetect
on track names. Only processes rows where language is currently null.
"""
import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 42  # reproducibility

CSV_PATH = "/project/pi_dagarwal_umass_edu/project_7/hmagapu/top_50k_full_augmented_v2.csv"

def detect_lang(text) -> str | None:
    if pd.isna(text) or len(str(text).strip()) < 3:
        return None
    try:
        return detect(str(text))
    except LangDetectException:
        return None

df = pd.read_csv(CSV_PATH)
total = len(df)
before = df["language"].notna().sum()

mask = df["language"].isna()
n_to_fill = mask.sum()
print(f"Rows to process: {n_to_fill:,}  (already filled: {before:,} / {total:,})", flush=True)

results = []
for i, (idx, row) in enumerate(df[mask].iterrows()):
    results.append((idx, detect_lang(row["track_name"])))
    if (i + 1) % 2000 == 0:
        filled_so_far = sum(1 for _, v in results if v is not None)
        print(f"  {i+1:,}/{n_to_fill:,} processed — {filled_so_far:,} detected so far", flush=True)

for idx, lang in results:
    df.at[idx, "language"] = lang

after = df["language"].notna().sum()
print(f"\nDone.")
print(f"Language filled:  {after:,} / {total:,}  ({after/total*100:.1f}%)")
print(f"Added by langdetect: {after - before:,}")

df.to_csv(CSV_PATH, index=False)
print("Saved.")
