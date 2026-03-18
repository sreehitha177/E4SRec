import pandas as pd
from collections import defaultdict
import random
import os


INPUT_FILE = "userid-timestamp-artid-artname-traid-traname.tsv"
METADATA_FILE = "/work/pi_dagarwal_umass_edu/project_7/swetha/lastfm_all_mgphot_labels.csv"   # your provided CSV
OUTPUT_DIR = "datasets/sequential/LastFM"

MIN_SEQ_LEN = 5
NUM_CANDIDATES = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)


print(f"Loading {INPUT_FILE}...")

df = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    header=None,
    usecols=[0, 1, 2, 3, 4, 5],
    names=["user", "timestamp", "artist_id", "artist_name", "track_id", "track_name"],
    engine="python",
    on_bad_lines="skip"
)

# Clean
df = df.dropna(subset=["artist_name", "track_name"])
df = df.sort_values(["user", "timestamp"])

df["artist_name"] = df["artist_name"].astype(str).str.strip().str.lower()
df["track_name"] = df["track_name"].astype(str).str.strip().str.lower()

print("LastFM loaded.")

# Load Metadata file
print(f"Loading {METADATA_FILE}...")

meta_df = pd.read_csv(METADATA_FILE)

meta_df["artist_name"] = meta_df["artist_name"].astype(str).str.strip().str.lower()
meta_df["track_name"] = meta_df["track_name"].astype(str).str.strip().str.lower()

print("Metadata loaded.")


print("Matching tracks with metadata...")
# Merge and keep rows where both artist_name and track_name match between df and meta_df
merged_df = df.merge(
    meta_df,
    on=["artist_name", "track_name"],
    how="inner"   
)

print(f"Matched rows: {len(merged_df)}")


print("Filtering users and reindexing IDs...")

user_map = {}
item_map = {}
item_master_list = []

u_count = 1
i_count = 1

final_sequences = {}

# Group by user
for original_user, user_df in merged_df.groupby("user"):

    items = []

    for _, row in user_df.iterrows():
        track_index = row["track_index"]  # from metadata
        key = track_index

        # Map item
        if key not in item_map:
            item_map[key] = i_count

            item_master_list.append({
                "item_id": i_count,
                "track_index": track_index,
                "artist_name": row["artist_name"],
                "track_name": row["track_name"]
            })

            i_count += 1

        items.append(item_map[key])

    # Keep only long sequences
    if len(items) >= MIN_SEQ_LEN:
        user_map[original_user] = u_count
        final_sequences[u_count] = items
        u_count += 1

total_users = u_count - 1
total_items = i_count - 1

print(f"Final Users: {total_users}")
print(f"Final Items: {total_items}")


pd.DataFrame(item_master_list).to_csv(
    f"{OUTPUT_DIR}/item_id_master_map.csv",
    index=False
)


print("Saving LastFM.txt...")

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

print("SUCCESS")
print(f"User ID range: 1 → {total_users}")
print(f"Item ID range: 1 → {total_items}")