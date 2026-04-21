import os
import torch
import numpy as np
import pandas as pd
from typing import List, Optional
import fire
import pickle
from concurrent.futures import ThreadPoolExecutor
from model import LLM4Rec
from utils.data_utils import SequentialDataset
from utils.eval_utils import RecallPrecision_atK, MRR_atK, MAP_atK, NDCG_atK, getLabel
from utils.prompter import Prompter


def load_sasrec_embeddings(data_path: str) -> torch.Tensor:
    sasrec_file = os.path.join(data_path, 'SASRec_item_embed.pkl')
    with open(sasrec_file, 'rb') as f:
        raw_sasrec = pickle.load(f)
    if isinstance(raw_sasrec, torch.Tensor):
        sasrec_embed = raw_sasrec.float()
    elif isinstance(raw_sasrec, np.ndarray):
        sasrec_embed = torch.from_numpy(raw_sasrec).float()
    else:
        sasrec_embed = torch.tensor(raw_sasrec).float()
    return sasrec_embed.cpu()


def load_node_embeddings(
    node_path: str,
    mapping_path: str,
    sasrec_embed: torch.Tensor,
    model_name: str,
) -> Optional[torch.Tensor]:
    if not node_path or not os.path.isdir(node_path):
        print(f"Skipping {model_name}: node_path not provided or invalid.")
        return None

    print(f"\nLoading {model_name} embeddings from {node_path}...")
    emb_csv = pd.read_csv(os.path.join(node_path, 'embeddings.csv'))
    master = pd.read_csv(mapping_path)

    emb_csv['_key'] = emb_csv['artist_name'].str.lower().str.strip() + '||' + emb_csv['track_name'].str.lower().str.strip()
    master['_key'] = master['artist_name'].str.lower().str.strip() + '||' + master['track_name'].str.lower().str.strip()

    merged = (
        emb_csv.merge(master[['item_id', '_key']], on='_key', how='inner')
               .drop_duplicates(subset=['track_index'])
    )
    print(f"   {model_name} files: {len(emb_csv)} | master items: {len(master)} | matched: {len(merged)}")

    if len(merged) == 0:
        print(f"   WARNING: no matched {model_name} embeddings found. Returning None.")
        return None

    sample_filename = os.path.basename(merged['embedding_path'].iloc[0])
    sample_tensor = torch.load(os.path.join(node_path, sample_filename), map_location='cpu', weights_only=False)
    emb_dim = sample_tensor.shape[0]
    print(f"   {model_name} embedding dim: {emb_dim}")

    n_items = sasrec_embed.shape[0]
    aligned = torch.zeros(n_items, emb_dim, dtype=torch.float32)

    def _load_one(row):
        embedding_file = os.path.join(node_path, os.path.basename(row['embedding_path']))
        item_idx = int(row['item_id']) - 1
        if not os.path.exists(embedding_file) or item_idx < 0 or item_idx >= n_items:
            return None
        tensor = torch.load(embedding_file, map_location='cpu', weights_only=False)
        return item_idx, tensor.float()

    rows = [row for _, row in merged.iterrows()]
    loaded = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(_load_one, rows):
            if result is not None:
                idx, emb = result
                aligned[idx] = emb
                loaded += 1

    print(f"   Loaded {loaded} / {len(merged)} matched {model_name} embeddings.")
    return aligned


def fuse_item_embeddings(
    sasrec_embed: torch.Tensor,
    audio_embed: Optional[torch.Tensor] = None,
    lyric_embed: Optional[torch.Tensor] = None,
    strategy: str = 'concat',
) -> torch.Tensor:
    """Fuse SASRec, audio, and lyric embeddings.

    Current default: concatenation, matching existing separate zeroshot scripts.
    Later strategies can be added here without changing the evaluation logic.
    """
    embeds = [sasrec_embed]
    if audio_embed is not None:
        embeds.append(audio_embed)
    if lyric_embed is not None:
        embeds.append(lyric_embed)

    if strategy == 'concat':
        fused = torch.cat(embeds, dim=-1)
    else:
        raise ValueError(f"Unsupported fusion strategy: {strategy}")

    print(f"   Fused item embedding shape: {fused.shape} (strategy={strategy})")
    return fused


def build_metadata_lookup(data_path: str, metadata_path: str):
    master_map = pd.read_csv(os.path.join(data_path, 'item_id_master_map.csv'))
    full_meta = pd.read_csv(metadata_path)

    full_meta['_key'] = full_meta['artist_name'].str.lower().str.strip() + '||' + full_meta['track_name'].str.lower().str.strip()
    master_map['_key'] = master_map['artist_name'].str.lower().str.strip() + '||' + master_map['track_name'].str.lower().str.strip()

    merged_meta = full_meta.merge(master_map[['item_id', '_key']], on='_key', how='inner').drop_duplicates(subset=['_key'])
    print(f"   Metadata file: {len(full_meta)} rows | master items: {len(master_map)} | matched: {len(merged_meta)}")

    name_lookup = {
        row['item_id']: f"'{row['track_name']}' by {row['artist_name']}"
        for _, row in master_map.iterrows()
        if pd.notna(row.get('track_name')) and pd.notna(row.get('artist_name'))
    }

    def _fmt(val, suffix=''):
        return f"{val}{suffix}" if pd.notna(val) and str(val).strip() not in ('', 'nan') else None

    meta_lookup = {}
    for _, row in merged_meta.iterrows():
        parts = []
        if _fmt(row.get('genre')): parts.append(f"genre: {row['genre']}")
        if _fmt(row.get('year')):  parts.append(f"year: {int(row['year'])}")
        if _fmt(row.get('tags')):  parts.append(f"tags: {row['tags']}")
        if _fmt(row.get('tempo')): parts.append(f"tempo: {row['tempo']:.0f} bpm")
        if _fmt(row.get('valence')): parts.append(f"valence: {row['valence']:.2f}")
        if _fmt(row.get('energy')): parts.append(f"energy: {row['energy']:.2f}")
        meta_str = ' | '.join(parts)
        desc = f"'{row['track_name']}' by {row['artist_name']}"
        if meta_str:
            desc += f" [{meta_str}]"
        meta_lookup[row['item_id']] = desc

    return meta_lookup, name_lookup


def zero_shot_evaluate(
    base_model: str = "/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    data_path: str = "datasets/sequential/LastFM/",
    audio_node_path: str = "/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/audio_embeddings/node_3",
    lyric_node_path: str = "/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/lyrics_embeddings/node_7",
    metadata_path: str = "/project/pi_dagarwal_umass_edu/project_7/hmagapu/top_50k_full_augmented.csv",
    mapping_path: str = "datasets/sequential/LastFM/item_id_master_map.csv",
    cache_dir: str = "",
    output_dir: str = "results",
    task_type: str = "sequential",
    fusion_strategy: str = "concat",
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"],
    max_test_users: int = 0,
    prompt_template_name: str = "alpaca",
    device_map: str = "auto",
):
    os.makedirs(output_dir, exist_ok=True)
    print("\nConfiguration:")
    print(f"  Base model:       {base_model}")
    print(f"  Dataset:          {data_path}")
    print(f"  Audio node path:  {audio_node_path}")
    print(f"  Lyric node path:  {lyric_node_path}")
    print(f"  Metadata path:    {metadata_path}")
    print(f"  Fusion strategy:  {fusion_strategy}")

    print("\n1. Loading dataset...")
    dataset = SequentialDataset(data_path, 50)

    print("\n2. Loading SASRec embeddings...")
    sasrec_embed = load_sasrec_embeddings(data_path)
    print(f"   SASRec shape: {sasrec_embed.shape}")

    audio_embed = load_node_embeddings(audio_node_path, mapping_path, sasrec_embed, "audio")
    lyric_embed = load_node_embeddings(lyric_node_path, mapping_path, sasrec_embed, "lyric")

    item_embed = fuse_item_embeddings(sasrec_embed, audio_embed=audio_embed, lyric_embed=lyric_embed, strategy=fusion_strategy)

    print("\n3. Building metadata lookup table...")
    meta_lookup, name_lookup = build_metadata_lookup(data_path, metadata_path)

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

    print("\n5. Running zero-shot evaluation...")
    topk = [1, 5, 10, 20, 100]
    results = {m: np.zeros(len(topk)) for m in ['Precision', 'Recall', 'MRR', 'MAP', 'NDCG']}

    test_keys = list(dataset.testData.keys())
    if max_test_users and max_test_users > 0:
        test_keys = test_keys[:max_test_users]

    testData = {k: dataset.testData[k] for k in test_keys}
    num_evaluated_users = 0

    with torch.no_grad():
        for u in test_keys:
            if u not in testData or len(testData[u]) == 0:
                continue

            full_history = testData[u][0]
            seq = full_history[-256:] if len(full_history) > 256 else full_history
            selected_items = [[testData[u][1]] + dataset.allPos[u]]
            groundTruth = [[0]]

            recent_history = full_history[-10:] if len(full_history) > 10 else full_history
            history_lines = []
            for i, item_id in enumerate(recent_history):
                desc = meta_lookup.get(item_id, name_lookup.get(item_id, "Unknown Track"))
                history_lines.append(f"{i+1}. {desc}")
            prompt_texts = prompter.generate_prompt(task_type, "\n".join(history_lines))

            device = next(model.llama_model.parameters()).device
            inputs = torch.LongTensor(seq).to(device).unsqueeze(0)
            inputs_mask = torch.ones(inputs.size()).to(device)

            _, ratings = model.predict(inputs, inputs_mask, history_metadata=[prompt_texts[0]])
            idx_row = torch.arange(ratings.size(0)).unsqueeze(1)
            ratings = ratings[idx_row, selected_items]

            _, ratings_K = torch.topk(ratings, k=topk[-1])
            ratings_K = ratings_K.cpu().numpy()
            r = getLabel(groundTruth, ratings_K)

            for j, k in enumerate(topk):
                results['Precision'][j] += RecallPrecision_atK(groundTruth, r, k)[0]
                results['Recall'][j] += RecallPrecision_atK(groundTruth, r, k)[1]
                results['MRR'][j] += MRR_atK(groundTruth, r, k)
                results['MAP'][j] += MAP_atK(groundTruth, r, k)
                results['NDCG'][j] += NDCG_atK(groundTruth, r, k)

            num_evaluated_users += 1

    if num_evaluated_users == 0:
        raise RuntimeError("No eligible users found for zero-shot evaluation.")

    for key in results:
        results[key] /= float(num_evaluated_users)

    df_results = pd.DataFrame(
        {k: np.round(results[k], 3) for k in results},
        index=[f"Top-{k}" for k in topk]
    )
    np.set_printoptions(precision=3, suppress=True)
    print("\n" + df_results.to_string(float_format=lambda x: f"{x:.3f}"))

    safe_model_name = f"SASRec_audio_lyric_metadata_{fusion_strategy}"
    output_file = os.path.join(output_dir, f"zeroshot_{safe_model_name}.txt")
    with open(output_file, 'w') as f:
        f.write(f"Zero-Shot Evaluation — {safe_model_name}\n")
        f.write(f"Base model: {base_model}\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Audio node path: {audio_node_path}\n")
        f.write(f"Lyric node path: {lyric_node_path}\n")
        f.write(f"Metadata path: {metadata_path}\n")
        f.write(f"Fusion strategy: {fusion_strategy}\n\n")
        f.write(df_results.to_string(float_format=lambda x: f"{x:.3f}"))
        f.write("\n\nDetailed arrays:\n")
        for key in results:
            f.write(f"{key}: {np.round(results[key], 3)}\n")

    print(f"\nResults saved to {output_file}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


if __name__ == '__main__':
    fire.Fire(zero_shot_evaluate)
