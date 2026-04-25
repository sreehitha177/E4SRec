import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Optional
import fire
import pickle
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoConfig
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


def _project_to_dim(embed: torch.Tensor, target_dim: int, seed_offset: int = 0) -> torch.Tensor:
    """Project an embedding to target_dim using a deterministic random linear layer.
 
    The projection is seeded so results are reproducible across runs. In a
    fine-tuning setup you would replace this with a trained nn.Linear that is
    jointly optimised with the rest of the model.
    """
    src_dim = embed.shape[1]
    if src_dim == target_dim:
        return embed
 
    # Deterministic seed per modality so each gets a distinct projection matrix.
    gen = torch.Generator()
    gen.manual_seed(42 + seed_offset)
 
    proj = nn.Linear(src_dim, target_dim, bias=False)
    nn.init.kaiming_uniform_(proj.weight, generator=gen)  # He init, common for projections
    proj.eval()
 
    with torch.no_grad():
        projected = proj(embed)
 
    print(f"   Projected {src_dim} → {target_dim} (seed_offset={seed_offset})")
    return projected


def fuse_item_embeddings(
    sasrec_embed: torch.Tensor,
    audio_embed: Optional[torch.Tensor] = None,
    lyric_embed: Optional[torch.Tensor] = None,
    strategy: str = 'concat',
    target_dim: Optional[int] = None,
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """Fuse SASRec, audio, and lyric embeddings.

    Current default: concatenation, matching existing separate zeroshot scripts.

    weighted_sum — project every modality to `target_dim` (defaults to the
                 SASRec embedding dim), L2-normalise each one, then compute a
                 weighted sum. `weights` is an optional list of non-negative
                 scalars in the order [sasrec, audio, lyric] (missing modalities
                 are skipped). Weights are automatically normalised to sum to 1,
                 so you can pass raw importance values like [2, 1, 1].
    """
    embeds_raw = [sasrec_embed]
    if audio_embed is not None:
        embeds_raw.append(audio_embed)
    if lyric_embed is not None:
        embeds_raw.append(lyric_embed)

    modality_names = ['sasrec']
    if audio_embed is not None:
        modality_names.append('audio')
    if lyric_embed is not None:
        modality_names.append('lyric')
        
    if strategy == 'concat':
        fused = torch.cat(embeds_raw, dim=-1)
    elif strategy == 'weighted_sum':
        out_dim = target_dim or sasrec_embed.shape[1]
        print(f"\n   [weighted_sum] target_dim={out_dim}")
 
        # 1. Project every modality to target_dim (no-op if already correct).
        projected = [
            _project_to_dim(emb, out_dim, seed_offset=i)
            for i, emb in enumerate(embeds_raw)
        ]
 
        # 2. L2-normalise each modality so weights control contribution cleanly.
        normalised = [F.normalize(e, p=2, dim=-1) for e in projected]
 
        # 3. Resolve and validate weights.
        n = len(normalised)
        if weights is None:
            w = [1.0 / n] * n
            print(f"   No weights supplied — using equal weights: {w}")
        else:
            if len(weights) != n:
                raise ValueError(
                    f"Expected {n} weights (one per active modality: "
                    f"{modality_names}), got {len(weights)}: {weights}"
                )
            total = sum(weights)
            if total <= 0:
                raise ValueError(f"Weights must be positive, got {weights}")
            w = [wi / total for wi in weights]
 
        print(f"   Modalities : {modality_names}")
        print(f"   Raw weights: {weights}")
        print(f"   Normalised : {[round(wi, 4) for wi in w]}")
 
        # 4. Weighted sum.
        fused = sum(wi * e for wi, e in zip(w, normalised))
    else:
        raise ValueError(f"Unsupported fusion strategy: {strategy}")

    print(f"   Fused item embedding shape: {fused.shape} (strategy={strategy})")
    return fused


def build_name_lookup(mapping_path: str):
    """Creates a simple ID -> 'Track by Artist' mapping."""
    master_map = pd.read_csv(mapping_path)
    name_lookup = {}
    for _, row in master_map.iterrows():
        item_id = row['item_id']
        track = row.get('track_name', 'Unknown Track')
        artist = row.get('artist_name', 'Unknown Artist')
        name_lookup[item_id] = f"'{track}' by {artist}"
    return name_lookup


def zero_shot_evaluate(
    base_model: str = "/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    data_path: str = "datasets/sequential/LastFM/",
    audio_node_path: str = "/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/audio_embeddings/node_3",
    lyric_node_path: str = "/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/lyrics_embeddings/node_7",
    mapping_path: str = "datasets/sequential/LastFM/item_id_master_map.csv",
    cache_dir: str = "",
    output_dir: str = "results",
    task_type: str = "sequential",
    use_completion_ratio: bool = False,
    completion_path: str = "data_preproc/user_sessions_with_completion.csv",
    source_path: str = "data_preproc/user_sessions_lastfm1k_minuser1000_minitem7_sessgap1200_minsesslen10_minhist50.csv",
    fusion_strategy: str = "concat",
    fusion_weights: str = "",
    fusion_target_dim: int = 0,     # 0 → defaults to SASRec dim
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
    print(f"  Fusion strategy:  {fusion_strategy}")
    print(f"  Fusion weights:   {fusion_weights or 'equal (default)'}")
    print(f"  Fusion target dim:{fusion_target_dim or 'SASRec dim (default)'}")

    parsed_weights: Optional[List[float]] = None
    
    if isinstance(fusion_weights, (list, tuple)):
        if len(fusion_weights) > 0:
            parsed_weights = [float(x) for x in fusion_weights]
    elif isinstance(fusion_weights, str) and fusion_weights.strip():
        try:
            parsed_weights = [float(x) for x in fusion_weights.split(',')]
        except ValueError:
            raise ValueError(
                f"--fusion_weights must be a comma-separated list of floats, "
                f"e.g. '0.5,0.3,0.2'. Got: '{fusion_weights}'"
            )

    print("\n1. Loading dataset...")
    dataset = SequentialDataset(
        data_path,
        50,
        use_completion_ratio=use_completion_ratio,
        completion_path=completion_path,
        source_path=source_path,
    )

    print("\n2. Loading SASRec embeddings...")
    sasrec_embed = load_sasrec_embeddings(data_path)
    print(f"   SASRec shape: {sasrec_embed.shape}")

    audio_embed = load_node_embeddings(audio_node_path, mapping_path, sasrec_embed, "audio")
    lyric_embed = load_node_embeddings(lyric_node_path, mapping_path, sasrec_embed, "lyric")

    # item_embed = fuse_item_embeddings(sasrec_embed, audio_embed=audio_embed, lyric_embed=lyric_embed, strategy=fusion_strategy)

    print("\n3. Fusing embeddings...")
    fusion_dim = fusion_target_dim if fusion_target_dim > 0 else None
    if fusion_strategy == 'weighted_sum' and fusion_dim is None:
        cfg = AutoConfig.from_pretrained(base_model, cache_dir=cache_dir or None)
        fusion_dim = cfg.hidden_size
        print(f"   Auto-detected LLM hidden_size as projection target: {fusion_dim}")
    item_embed = fuse_item_embeddings(
        sasrec_embed,
        audio_embed=audio_embed,
        lyric_embed=lyric_embed,
        strategy=fusion_strategy,
        target_dim=fusion_dim,
        weights=parsed_weights,
    )

    name_lookup = build_name_lookup(mapping_path)

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
        use_completion_ratio=use_completion_ratio,
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

            full_history, full_history_ratio, target = dataset.get_eval_record(u, subset='test')
            seq = full_history[-256:] if len(full_history) > 256 else full_history
            seq_ratio = None
            if full_history_ratio is not None:
                seq_ratio = full_history_ratio[-256:] if len(full_history_ratio) > 256 else full_history_ratio
            selected_items = [dataset.allPos[u]]
            groundTruth = [[0]]

            recent_history = full_history[-10:] if len(full_history) > 10 else full_history
            history_lines = []
            for i, item_id in enumerate(recent_history):
                # desc = meta_lookup.get(item_id, name_lookup.get(item_id, "Unknown Track"))
                # history_lines.append(f"{i+1}. {desc}")
                name_desc = name_lookup.get(item_id, f"Item {item_id}")
                history_lines.append(f"{i+1}. {name_desc}")

            prompt_texts = prompter.generate_prompt(task_type, "\n".join(history_lines))

            device = next(model.llama_model.parameters()).device
            inputs = torch.LongTensor(seq).to(device).unsqueeze(0)
            inputs_mask = torch.ones(inputs.size()).to(device)
            completion_tensor = None
            if seq_ratio is not None:
                completion_tensor = torch.FloatTensor(seq_ratio).to(device).unsqueeze(0)

            _, ratings = model.predict(
                inputs,
                inputs_mask,
                history_metadata=[prompt_texts[0]],
                completion_ratio=completion_tensor,
            )
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

    if parsed_weights:
        weight_tag = '-'.join(str(w) for w in parsed_weights)
    else:
        weight_tag = "equal"
    safe_model_name = f"SASRec_audio_lyric_{fusion_strategy}_{weight_tag}"
    output_file = os.path.join(output_dir, f"zeroshot_{safe_model_name}.txt")
    with open(output_file, 'w') as f:
        f.write(f"Zero-Shot Evaluation — {safe_model_name}\n")
        f.write(f"Base model: {base_model}\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Audio node path: {audio_node_path}\n")
        f.write(f"Lyric node path: {lyric_node_path}\n")
        f.write(f"Fusion strategy: {fusion_strategy}\n\n")
        f.write(f"Fusion weights:   {fusion_weights or 'equal'}\n")
        f.write(f"Fusion target dim:{fusion_target_dim or 'SASRec dim'}\n\n")
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
