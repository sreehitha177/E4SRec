import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS']      = '1'
os.environ['MKL_NUM_THREADS']      = '1'

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
AUDIO_ROOT = Path("/scratch3/workspace/skandagatla_umass_edu-dolby/raw_audio_files/batch_1")
STATUS_CSV = Path("/home/snarayana_umass_edu/E4SRec-1/datasets/sequential/LastFM/all_download_status.csv")
TOP_50K    = Path("/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/shared/top_50k_songs.csv")
OUT_DIR    = Path("/scratch3/workspace/skandagatla_umass_edu-dolby/librosa_features")
CKPT_DIR   = OUT_DIR / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ── Step 1: Build track_index → audio_path mapping ────────────────────────
def build_mapping():
    top50k = pd.read_csv(TOP_50K)  # columns: track_index, artist_name, track_name, play_count

    status = pd.read_csv(STATUS_CSV, header=0)
    status = status[status['track_index'] != 'track_index']  # drop duplicate headers
    status['download_success'] = status['download_success'].astype(str)
    status = status[status['download_success'] == 'True'].copy()
    status['track_index']    = status['track_index'].astype(int)
    status['node_processed'] = status['node_processed'].astype(int)

    # Direct join on track_index — clean, no fuzzy name matching needed
    merged = top50k.merge(status, on='track_index', how='inner')

    merged['audio_path'] = merged.apply(
        lambda r: str(AUDIO_ROOT / f"node_{r['node_processed']}" / f"{r['track_index']}.wav"),
        axis=1
    )

    print(f"Top-50k tracks found with audio: {len(merged)}")
    return merged[['track_index', 'audio_path']].drop_duplicates('track_index')


# ── Step 2: Feature extraction for one track ─────────────────────────────
def extract_features(args):
    track_index, audio_path = args
    ckpt = CKPT_DIR / f"{track_index}.npy"

    if ckpt.exists():
        return track_index, np.load(ckpt, allow_pickle=True).item()

    try:
        p = Path(audio_path)
        if not p.exists() or p.stat().st_size < 1000:
            print(f"  SKIP track_index={track_index}: file missing or too small")
            return track_index, None

        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=60.0)

        if len(y) < sr * 5:
            print(f"  SKIP track_index={track_index}: audio too short")
            return track_index, None

        f = {}

        # Timbral (20+20 = 40 dims)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        f['mfcc_mean'] = mfcc.mean(axis=1)
        f['mfcc_std']  = mfcc.std(axis=1)

        # Spectral (1+1+7+7+1+1 = 18 dims)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        f['spectral_centroid_mean'] = np.array([cent.mean()])
        f['spectral_centroid_std']  = np.array([cent.std()])

        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        f['spectral_contrast_mean'] = contrast.mean(axis=1)
        f['spectral_contrast_std']  = contrast.std(axis=1)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        f['spectral_rolloff_mean'] = np.array([rolloff.mean()])
        f['spectral_rolloff_std']  = np.array([rolloff.std()])

        # Energy (1+1 = 2 dims)
        rms = librosa.feature.rms(y=y)
        f['rms_mean'] = np.array([rms.mean()])
        f['rms_std']  = np.array([rms.std()])

        # Rhythm (1+1 = 2 dims)
        zcr = librosa.feature.zero_crossing_rate(y)
        f['zcr_mean'] = np.array([zcr.mean()])
        f['zcr_std']  = np.array([zcr.std()])

        # Tempo (1 dim) — tempo() is stable, beat_track() causes segfault
        tempo = librosa.feature.tempo(y=y, sr=sr)
        f['tempo'] = np.array([float(tempo[0])])

        # Total: 40 + 18 + 2 + 2 + 1 = 63 dims

        np.save(ckpt, f)
        return track_index, f

    except Exception as e:
        print(f"  FAILED track_index={track_index}: {e}")
        return track_index, None


# ── Step 3: Flatten feature dict → single fixed-size vector ──────────────
def flatten(feats):
    keys = [
        'mfcc_mean',              # 20
        'mfcc_std',               # 20
        'spectral_centroid_mean', #  1
        'spectral_centroid_std',  #  1
        'spectral_contrast_mean', #  7
        'spectral_contrast_std',  #  7
        'spectral_rolloff_mean',  #  1
        'spectral_rolloff_std',   #  1
        'rms_mean',               #  1
        'rms_std',                #  1
        'zcr_mean',               #  1
        'zcr_std',                #  1
        'tempo',                  #  1
    ]
    return np.concatenate([feats[k] for k in keys])  # 63 dims total


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Building track_index → audio_path mapping...")
    mapping_df = build_mapping()

    # Skip already checkpointed tracks
    items = [
        (int(idx), path)
        for idx, path in zip(mapping_df['track_index'], mapping_df['audio_path'])
        if not (CKPT_DIR / f"{idx}.npy").exists()
    ]
    already_done = len(mapping_df) - len(items)
    print(f"Already checkpointed: {already_done}  |  Remaining: {len(items)}")

    if len(items) > 0:
        N_WORKERS  = 16
        SAVE_EVERY = 1000
        print(f"\nStep 2: Extracting features with {N_WORKERS} threads...")
        print("=" * 60)

        results = []
        failed  = []
        batch   = []

        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {
                executor.submit(extract_features, item): item[0]
                for item in items
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                track_index = futures[future]
                try:
                    track_index, feats = future.result(timeout=120)
                    if feats is not None:
                        batch.append({
                            'track_index':    track_index,
                            'feature_vector': flatten(feats)
                        })
                    else:
                        failed.append(track_index)
                except Exception as e:
                    print(f"  Worker exception track_index={track_index}: {e}")
                    failed.append(track_index)

                # Periodic save every 1000 so progress isn't lost on interruption
                if len(batch) >= SAVE_EVERY:
                    results.extend(batch)
                    batch = []
                    pd.DataFrame(results).to_parquet(
                        OUT_DIR / "librosa_features_partial.parquet", index=False
                    )

        results.extend(batch)
        print(f"\nSuccessful: {len(results)}  |  Failed: {len(failed)}")
        np.save(OUT_DIR / "failed_track_indices.npy", np.array(failed))

    # ── Collect all checkpoints into final parquet ────────────────────────
    print("\nStep 3: Collecting all checkpointed features...")
    all_results = []
    for ckpt_file in tqdm(list(CKPT_DIR.glob("*.npy"))):
        track_index = int(ckpt_file.stem)
        feats       = np.load(ckpt_file, allow_pickle=True).item()
        all_results.append({
            'track_index':    track_index,
            'feature_vector': flatten(feats)
        })

    out_df = pd.DataFrame(all_results)
    out_df.to_parquet(OUT_DIR / "librosa_features.parquet", index=False)

    print("=" * 60)
    print(f"Total features saved: {len(out_df)}")
    print(f"Feature vector dim:   {all_results[0]['feature_vector'].shape[0]}")  # 63
    print(f"Output → {OUT_DIR / 'librosa_features.parquet'}")
    print(f"Failed → {OUT_DIR / 'failed_track_indices.npy'}")