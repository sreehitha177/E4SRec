import os
import torch
import numpy as np
import pandas as pd
from typing import List
import fire
import pickle
from concurrent.futures import ThreadPoolExecutor
from model import LLM4Rec
from utils.data_utils import SequentialDataset
from utils.eval_utils import RecallPrecision_atK, MRR_atK, MAP_atK, NDCG_atK, getLabel
from utils.prompter import Prompter

def zero_shot_evaluate(
    base_model: str = "/datasets/ai/qwen2/hub/models--Qwen--Qwen2.5-32B/snapshots/1818d35814b8319459f4bd55ed1ac8709630f003",
    data_path: str = "datasets/sequential/LastFM/",
    # Path to the node directory (e.g. .../audio_embeddings/node_0 or .../lyrics_embeddings/node_5)
    node_path: str = "",
    model_name: str = "embedding",   # used for output filename only
    mapping_path: str = "datasets/sequential/LastFM/item_id_master_map.csv",
    cache_dir: str = "",
    output_dir: str = "results",
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
    print(f"  Model name: {model_name}")
    print(f"  Node path:  {node_path}")
    print(f"  Base model: {base_model}")

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
    sasrec_embed = sasrec_embed.cpu()   # ensure CPU before cat with aligned zeros
    print(f"   SASRec shape: {sasrec_embed.shape}")

    # ------------------------------------------------------------------ #
    # 3. Load and align external embeddings (audio or lyrics)
    # ------------------------------------------------------------------ #
    if node_path and os.path.isdir(node_path):
        print(f"\n3. Loading {model_name} embeddings from {node_path}...")
        emb_csv = pd.read_csv(os.path.join(node_path, 'embeddings.csv'))
        master  = pd.read_csv(mapping_path)

        # Join on normalised artist + track name
        emb_csv['_key'] = emb_csv['artist_name'].str.lower().str.strip() + '||' + \
                           emb_csv['track_name'].str.lower().str.strip()
        master['_key']  = master['artist_name'].str.lower().str.strip() + '||' + \
                           master['track_name'].str.lower().str.strip()

        merged = (
            emb_csv.merge(master[['item_id', '_key']], on='_key', how='inner')
                   .drop_duplicates(subset=['track_index'])   # 1 item_id per file
        )
        print(f"   {model_name} files: {len(emb_csv)} | master items: {len(master)} | matched: {len(merged)}")

        # Detect embedding dimension from the first file
        sample_filename = os.path.basename(merged['embedding_path'].iloc[0])
        sample_tensor   = torch.load(os.path.join(node_path, sample_filename),
                                     map_location='cpu', weights_only=False)
        emb_dim = sample_tensor.shape[0]
        print(f"   Embedding dim: {emb_dim}")

        n_items = sasrec_embed.shape[0]
        aligned = torch.zeros(n_items, emb_dim)

        # -- parallel load with ThreadPoolExecutor --
        def _load_one(row):
            pt_path = os.path.join(node_path, os.path.basename(row['embedding_path']))
            item_idx = int(row['item_id']) - 1   # item_id is 1-indexed
            if os.path.exists(pt_path) and 0 <= item_idx < n_items:
                t = torch.load(pt_path, map_location='cpu', weights_only=False)
                return item_idx, t.float()
            return None

        rows = [row for _, row in merged.iterrows()]
        found = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            for result in pool.map(_load_one, rows):
                if result is not None:
                    idx, emb = result
                    aligned[idx] = emb
                    found += 1

        print(f"   Loaded {found} / {len(merged)} matched embeddings.")
        item_embed = torch.cat([sasrec_embed, aligned], dim=-1)
    else:
        print(f"\n3. node_path not provided or invalid — using SASRec only.")
        item_embed = sasrec_embed

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

    output_file = os.path.join(output_dir, f"zeroshot_{model_name}.txt")
    with open(output_file, "w") as f:
        f.write(f"Zero-Shot Evaluation — SASRec + {model_name}\n")
        f.write(f"Base model: {base_model}\n")
        f.write(f"Node path:  {node_path}\n\n")
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
