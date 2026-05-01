import os
import pickle
import numpy as np
import pandas as pd
import torch
from typing import List, Optional
import fire

from model import LLM4Rec
from utils.data_utils import SequentialDataset
from utils.eval_utils import RecallPrecision_atK, MRR_atK, MAP_atK, NDCG_atK, getLabel
from utils.fusion import FusionModule, load_node_embeddings
from utils.prompter import Prompter


def _load_sasrec(data_path: str) -> torch.Tensor:
    with open(os.path.join(data_path, 'SASRec_item_embed.pkl'), 'rb') as f:
        raw = pickle.load(f)
    if isinstance(raw, torch.Tensor):
        return raw.float().cpu()
    if isinstance(raw, np.ndarray):
        return torch.from_numpy(raw).float()
    return torch.tensor(raw).float()


def build_name_lookup(mapping_path: str, item_map: dict) -> dict:
    master = pd.read_csv(mapping_path)
    lookup = {}
    for _, row in master.iterrows():
        new_idx = item_map.get(int(row['item_id']), None)
        if new_idx is not None:
            lookup[new_idx] = f"'{row.get('track_name', '?')}' by {row.get('artist_name', '?')}"
    return lookup


def zero_shot_evaluate(
    base_model: str = "/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    data_path: str = "datasets/sequential/LastFM/",
    audio_node_path: str = "/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/audio_embeddings/node_3",
    lyric_node_path: str = "/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/lyrics_embeddings/node_7",
    mapping_path: str = "datasets/sequential/LastFM/item_id_master_map.csv",
    cache_dir: str = "",
    output_dir: str = "results",
    task_type: str = "sequential",
    # fusion_strategy: concat | weighted_sum | cross_attention | film
    fusion_strategy: str = "concat",
    # comma-separated weights for weighted_sum, e.g. "0.5,0.3,0.2"; default = equal
    fusion_weights: str = "",
    # target dim for non-concat strategies; 0 = use SASRec dim
    fusion_target_dim: int = 0,
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
    print(f"  Base model:        {base_model}")
    print(f"  Dataset:           {data_path}")
    print(f"  Audio node path:   {audio_node_path}")
    print(f"  Lyric node path:   {lyric_node_path}")
    print(f"  Fusion strategy:   {fusion_strategy}")
    print(f"  Fusion weights:    {fusion_weights or 'equal (default)'}")
    print(f"  Fusion target dim: {fusion_target_dim or 'SASRec dim (default)'}")

    # Parse optional fixed weights for weighted_sum.
    parsed_weights: Optional[List[float]] = None
    if isinstance(fusion_weights, (list, tuple)) and len(fusion_weights) > 0:
        parsed_weights = [float(x) for x in fusion_weights]
    elif isinstance(fusion_weights, str) and fusion_weights.strip():
        try:
            parsed_weights = [float(x) for x in fusion_weights.split(',')]
        except ValueError:
            raise ValueError(
                f"--fusion_weights must be comma-separated floats, e.g. '0.5,0.3,0.2'. "
                f"Got: '{fusion_weights}'"
            )

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    print("\n1. Loading dataset...")
    dataset  = SequentialDataset(data_path, 50)
    item_map = dataset.item_map
    n_items  = len(item_map) + 1

    # ── 2. Load & remap embeddings ────────────────────────────────────────────
    print("\n2. Loading SASRec embeddings...")
    raw_sasrec   = _load_sasrec(data_path)
    sasrec_embed = torch.zeros(n_items, raw_sasrec.shape[1])
    for raw_id, new_idx in item_map.items():
        if raw_id < raw_sasrec.shape[0]:
            sasrec_embed[new_idx] = raw_sasrec[raw_id]
    print(f"   SASRec: {raw_sasrec.shape} → remapped {sasrec_embed.shape}")
    del raw_sasrec

    audio_embed = load_node_embeddings(audio_node_path, mapping_path, n_items, item_map, "audio")
    lyric_embed = load_node_embeddings(lyric_node_path, mapping_path, n_items, item_map, "lyric")

    # ── 3. Fuse ───────────────────────────────────────────────────────────────
    print(f"\n3. Fusing embeddings (strategy={fusion_strategy})...")
    embeds = [sasrec_embed]
    dims   = [sasrec_embed.shape[1]]
    if audio_embed is not None:
        embeds.append(audio_embed); dims.append(audio_embed.shape[1])
    if lyric_embed is not None:
        embeds.append(lyric_embed); dims.append(lyric_embed.shape[1])

    if len(embeds) == 1:
        # Only SASRec available — skip fusion entirely.
        item_embed = sasrec_embed
        print("   Only SASRec available; no fusion applied.")
    else:
        target_dim = fusion_target_dim if fusion_target_dim > 0 else dims[0]
        # For weighted_sum, use fixed weights (deterministic zero-shot results).
        # For cross_attention / film, weights are random-init but frozen — results
        # are baseline numbers only.
        fw = parsed_weights if fusion_strategy == "weighted_sum" else None
        fusion = FusionModule(
            modality_dims=dims,
            strategy=fusion_strategy,
            output_dim=target_dim,
            fixed_weights=fw,
        )
        fusion.eval()
        with torch.no_grad():
            item_embed = fusion(embeds)
        print(f"   Fused shape: {item_embed.shape}")

    # ── 4. Model ──────────────────────────────────────────────────────────────
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

    name_lookup = build_name_lookup(mapping_path, item_map)

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    print("\n5. Running zero-shot evaluation...")
    topk    = [1, 5, 10, 20, 100]
    results = {m: np.zeros(len(topk)) for m in ['Precision', 'Recall', 'MRR', 'MAP', 'NDCG']}

    test_keys = list(dataset.testData.keys())
    if max_test_users and max_test_users > 0:
        test_keys = test_keys[:max_test_users]
    testData = {k: dataset.testData[k] for k in test_keys}
    n_eval   = 0

    with torch.no_grad():
        for u in test_keys:
            if u not in testData or len(testData[u]) == 0:
                continue

            full_history   = testData[u][0]
            seq            = full_history[-256:] if len(full_history) > 256 else full_history
            selected_items = [dataset.allPos[u]]
            groundTruth    = [[0]]

            history_lines = [
                f"{i+1}. {name_lookup.get(iid, f'Item {iid}')}"
                for i, iid in enumerate(full_history[-10:])
            ]
            prompt_texts = prompter.generate_prompt(task_type, "\n".join(history_lines))

            device      = next(model.llama_model.parameters()).device
            inputs      = torch.LongTensor(seq).to(device).unsqueeze(0)
            inputs_mask = torch.ones(inputs.size()).to(device)

            _, ratings = model.predict(inputs, inputs_mask, history_metadata=[prompt_texts[0]])
            idx_row    = torch.arange(ratings.size(0)).unsqueeze(1)
            ratings    = ratings[idx_row, selected_items]

            _, ratings_K = torch.topk(ratings, k=topk[-1])
            ratings_K    = ratings_K.cpu().numpy()
            r = getLabel(groundTruth, ratings_K)

            for j, k in enumerate(topk):
                results['Precision'][j] += RecallPrecision_atK(groundTruth, r, k)[0]
                results['Recall'][j]    += RecallPrecision_atK(groundTruth, r, k)[1]
                results['MRR'][j]       += MRR_atK(groundTruth, r, k)
                results['MAP'][j]       += MAP_atK(groundTruth, r, k)
                results['NDCG'][j]      += NDCG_atK(groundTruth, r, k)
            n_eval += 1

    if n_eval == 0:
        raise RuntimeError("No eligible users found for zero-shot evaluation.")

    for key in results:
        results[key] /= float(n_eval)

    df = pd.DataFrame(
        {k: np.round(results[k], 3) for k in results},
        index=[f"Top-{k}" for k in topk],
    )
    np.set_printoptions(precision=3, suppress=True)
    print("\n" + df.to_string(float_format=lambda x: f"{x:.3f}"))

    # Build output filename from active modalities + strategy.
    modality_tag = '_'.join(
        ['SASRec']
        + (['audio'] if audio_embed is not None else [])
        + (['lyric'] if lyric_embed is not None else [])
    )
    if audio_embed is None and lyric_embed is None:
        # No fusion was applied — omit strategy from filename.
        tag = modality_tag
    elif fusion_strategy == 'weighted_sum' and parsed_weights:
        weight_tag = '-'.join(f"{w:.2f}" for w in parsed_weights)
        tag = f"{modality_tag}_weighted_sum_{weight_tag}"
    else:
        tag = f"{modality_tag}_{fusion_strategy}"
    output_file = os.path.join(output_dir, f"zeroshot_{tag}.txt")
    with open(output_file, 'w') as f:
        f.write(f"Zero-Shot Evaluation — {tag}\n")
        f.write(f"Base model: {base_model}\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Audio node path: {audio_node_path}\n")
        f.write(f"Lyric node path: {lyric_node_path}\n")
        f.write(f"Fusion strategy: {fusion_strategy}\n")
        f.write(f"Fusion weights:  {fusion_weights or 'equal'}\n")
        f.write(f"Fusion target dim: {fusion_target_dim or 'SASRec dim'}\n\n")
        f.write(df.to_string(float_format=lambda x: f"{x:.3f}"))
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
