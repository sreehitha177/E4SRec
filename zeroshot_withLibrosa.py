import os
import torch
import numpy as np
import pandas as pd
from typing import List
import fire
import pickle
from model import LLM4Rec
from utils.data_utils import SequentialDataset
from utils.eval_utils import RecallPrecision_atK, MRR_atK, MAP_atK, NDCG_atK, getLabel
from utils.prompter import Prompter


def zero_shot_evaluate(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    data_path: str = "datasets/sequential/LastFM/",
    # Path to the librosa features parquet produced by librosa_features.py
    librosa_parquet: str = "/scratch3/workspace/skandagatla_umass_edu-dolby/librosa_features/librosa_features.parquet",
    # top_50k CSV to resolve track_index → artist/track name
    top50k_csv: str = "/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/shared/top_50k_songs.csv",
    # master map to resolve artist/track name → item_id
    mapping_path: str = "datasets/sequential/LastFM/item_id_master_map.csv",
    model_name: str = "LIBROSA",
    cache_dir: str = "",
    output_dir: str = "/home/snarayana_umass_edu/E4SRec-1/results",
    task_type: str = "sequential",
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"],
    max_test_users: int = 0,
    prompt_template_name: str = "alpaca",
    device_map: str = "auto",
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nConfiguration:")
    print(f"  Model name:      {model_name}")
    print(f"  Librosa parquet: {librosa_parquet}")
    print(f"  Base model:      {base_model}")

    # ------------------------------------------------------------------ #
    # 1. Dataset
    # ------------------------------------------------------------------ #
    print(f"\n1. Loading dataset...")
    dataset = SequentialDataset(data_path, 50)

    # ------------------------------------------------------------------ #
    # 2. SASRec base embeddings
    # ------------------------------------------------------------------ #
    print(f"\n2. Loading SASRec embeddings...")
    sasrec_file = os.path.join(data_path, 'SASRec_item_embed.pkl')
    with open(sasrec_file, 'rb') as f:
        raw_sasrec = pickle.load(f)
    if isinstance(raw_sasrec, torch.Tensor):
        sasrec_embed = raw_sasrec.float()
    elif isinstance(raw_sasrec, np.ndarray):
        sasrec_embed = torch.from_numpy(raw_sasrec).float()
    else:
        sasrec_embed = torch.tensor(raw_sasrec).float()
    sasrec_embed = sasrec_embed.cpu()
    print(f"   SASRec shape: {sasrec_embed.shape}")

    # ------------------------------------------------------------------ #
    # 3. Load and align librosa features
    # ------------------------------------------------------------------ #
    print(f"\n3. Loading librosa features from {librosa_parquet}...")
    librosa_df = pd.read_parquet(librosa_parquet)
    # librosa_df columns: track_index (int), feature_vector (np.ndarray, 63-dim)

    top50k = pd.read_csv(top50k_csv)[['track_index', 'artist_name', 'track_name']]
    master = pd.read_csv(mapping_path)[['item_id', 'artist_name', 'track_name']]

    # Merge librosa → top50k → master on normalised name keys
    top50k['_key'] = (top50k['artist_name'].str.lower().str.strip() + '||' +
                      top50k['track_name'].str.lower().str.strip())
    master['_key'] = (master['artist_name'].str.lower().str.strip() + '||' +
                      master['track_name'].str.lower().str.strip())

    merged = (
        librosa_df
        .merge(top50k[['track_index', '_key']], on='track_index', how='inner')
        .merge(master[['item_id', '_key']], on='_key', how='inner')
        .drop_duplicates(subset=['track_index'])
    )
    print(f"   Librosa tracks: {len(librosa_df)} | matched to master: {len(merged)}")

    # Determine feature dimension from data
    emb_dim = merged['feature_vector'].iloc[0].shape[0]
    print(f"   Librosa feature dim: {emb_dim}")

    n_items = sasrec_embed.shape[0]
    aligned = torch.zeros(n_items, emb_dim, dtype=torch.float32)

    loaded = 0
    for _, row in merged.iterrows():
        item_idx = int(row['item_id']) - 1   # item_id is 1-indexed
        if 0 <= item_idx < n_items:
            vec = row['feature_vector']
            aligned[item_idx] = torch.tensor(vec, dtype=torch.float32)
            loaded += 1

    print(f"   Aligned {loaded} / {len(merged)} tracks into item matrix.")

    item_embed = torch.cat([sasrec_embed, aligned], dim=-1)
    print(f"   Final item embedding shape: {item_embed.shape}")

    # ------------------------------------------------------------------ #
    # 4. Build model
    # ------------------------------------------------------------------ #
    print(f"\n4. Loading base model: {base_model}")
    prompter = Prompter(prompt_template_name)
    model = LLM4Rec(
        base_model=base_model,
        task_type=task_type,
        cache_dir=cache_dir,
        input_dim=item_embed.shape[1],
        output_dim=dataset.m_item,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        device_map=device_map,
        instruction_text=prompter.generate_prompt(task_type),
        user_embeds=None,
        input_embeds=item_embed,
    )
    model.eval()

    # ------------------------------------------------------------------ #
    # 5. Evaluation loop
    # ------------------------------------------------------------------ #
    print(f"\n5. Running zero-shot evaluation...")
    topk = [1, 5, 10, 20, 100]
    metrics = ['Precision', 'Recall', 'MRR', 'MAP', 'NDCG']
    results = {m: np.zeros(len(topk)) for m in metrics}

    test_keys = list(dataset.testData.keys())
    if max_test_users and max_test_users > 0:
        test_keys = test_keys[:max_test_users]

    with torch.no_grad():
        device = next(model.llama_model.parameters()).device
        for u in test_keys:
            full_history = dataset.testData[u][0]
            seq = full_history[-256:] if len(full_history) > 256 else full_history
            selected_items = [dataset.allPos[u]]
            groundTruth = [[0]]

            inputs      = torch.LongTensor(seq).to(device).unsqueeze(0)
            inputs_mask = torch.ones(inputs.size()).to(device)

            _, ratings = model.predict(inputs, inputs_mask)
            idx_row    = torch.arange(ratings.size(0)).unsqueeze(1)
            ratings    = ratings[idx_row, selected_items]

            _, ratings_K = torch.topk(ratings, k=topk[-1])
            r = getLabel(groundTruth, ratings_K.cpu().numpy())

            for j, k in enumerate(topk):
                pre, rec = RecallPrecision_atK(groundTruth, r, k)
                results['Precision'][j] += pre
                results['Recall'][j]    += rec
                results['MRR'][j]       += MRR_atK(groundTruth, r, k)
                results['MAP'][j]       += MAP_atK(groundTruth, r, k)
                results['NDCG'][j]      += NDCG_atK(groundTruth, r, k)

    num_eval = len(test_keys)
    for k in results:
        results[k] /= float(num_eval)

    df_results = pd.DataFrame(
        {k: np.round(results[k], 3) for k in results},
        index=[f"Top-{k}" for k in topk]
    )
    np.set_printoptions(precision=3, suppress=True)
    print("\n" + df_results.to_string(float_format=lambda x: f"{x:.3f}"))

    output_file = os.path.join(output_dir, f"zeroshot_withLibrosa.txt")
    with open(output_file, "w") as f:
        f.write(f"Zero-Shot Evaluation — SASRec + {model_name}\n")
        f.write(f"Base model: {base_model}\n")
        f.write(f"Librosa parquet: {librosa_parquet}\n\n")
        f.write(df_results.to_string(float_format=lambda x: f"{x:.3f}"))
        f.write("\n\nDetailed arrays:\n")
        for key in results:
            f.write(f"{key}: {np.round(results[key], 3)}\n")
    print(f"\nResults saved to {output_file}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    fire.Fire(zero_shot_evaluate)
