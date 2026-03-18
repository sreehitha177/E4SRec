import os
import pandas as pd
import random

# Paths
INPUT_FILE = "userid-timestamp-artid-artname-traid-traname.tsv"
# This is the NEW CSV your teammate gave you for audio embeddings
AUDIO_CSV = "/project/pi_dagarwal_umass_edu/project_7/srikar/output_sample/final/master_lyrics.csv" 
OUTPUT_DIR = "datasets/sequential/LastFM"

MIN_SEQ_LEN = 5
NUM_CANDIDATES = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Interaction Data
print(f"Loading {INPUT_FILE}...")
df = pd.read_csv(
    INPUT_FILE, sep="\t", header=None,
    usecols=[0, 1, 2, 3, 4, 5],
    names=["user", "timestamp", "artist_id", "artist_name", "track_id", "track_name"],
    engine="python", on_bad_lines="skip"
)

# Standardize text for matching
df = df.dropna(subset=["artist_name", "track_name"])
df["artist_name"] = df["artist_name"].astype(str).str.strip().str.lower()
df["track_name"] = df["track_name"].astype(str).str.strip().str.lower()
df = df.sort_values(["user", "timestamp"])

# 2. Load Audio Embeddings CSV
print(f"Loading Audio CSV: {AUDIO_CSV}...")
audio_df = pd.read_csv(AUDIO_CSV)
# Ensure these match the formatting of your interaction data
audio_df["artist_name"] = audio_df["artist_name"].astype(str).str.strip().str.lower()
audio_df["track_name"] = audio_df["track_name"].astype(str).str.strip().str.lower()

# 3. Inner Join: Only keep interactions for songs that exist in the Audio CSV
# This ensures that every item in your dataset has a track_index to look up the .pt file
merged_df = df.merge(audio_df, on=["artist_name", "track_name"], how="inner")
print(f"Matched rows with audio entries: {len(merged_df)}")

# 4. Generate Contiguous Item IDs
user_map, item_map, item_master_list = {}, {}, []
u_count, i_count = 1, 1
final_sequences = {}

for original_user, user_df in merged_df.groupby("user"):
    items = []
    for _, row in user_df.iterrows():
        # Using track_index from the Audio CSV as the primary key
        track_idx = row["track_index"] 
        
        if track_idx not in item_map:
            item_map[track_idx] = i_count
            item_master_list.append({
                "item_id": i_count,      # Contiguous ID for SASRec/LLM
                "track_index": track_idx, # Index to look up in the .pt file
                "artist_name": row["artist_name"],
                "track_name": row["track_name"]
            })
            i_count += 1
        items.append(item_map[track_idx])

    if len(items) >= MIN_SEQ_LEN:
        user_map[original_user] = u_count
        final_sequences[u_count] = items
        u_count += 1

# 5. Save the Master Map
# You MUST load this in your zeroshot_withAudioEmbeddings.py script
mapping_df = pd.DataFrame(item_master_list)
mapping_df.to_csv(f"{OUTPUT_DIR}/item_id_master_map.csv", index=False)

# 6. Save standard sequential files
total_items = i_count - 1
with open(f"{OUTPUT_DIR}/LastFM.txt", "w") as f:
    for u_idx, items in final_sequences.items():
        f.write(f"{u_idx} " + " ".join(map(str, items)) + "\n")

with open(f"{OUTPUT_DIR}/LastFM_sample.txt", "w") as f:
    for u_idx, items in final_sequences.items():
        test_item = items[-1]
        user_items_set = set(items)
        candidates = [test_item]
        while len(candidates) < NUM_CANDIDATES:
            neg = random.randint(1, total_items)
            if neg not in user_items_set and neg not in candidates:
                candidates.append(neg)
        f.write(f"{u_idx} " + " ".join(map(str, candidates)) + "\n")

print(f"SUCCESS: {len(user_map)} users, {total_items} items.")