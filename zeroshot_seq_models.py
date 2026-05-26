import math
import os
import pickle
from typing import List, Optional

import fire
import numpy as np
import pandas as pd
import torch

from model import LLM4Rec
from utils.data_utils import SequentialDataset
from utils.eval_utils import RecallPrecision_atK, MRR_atK, MAP_atK, NDCG_atK, getLabel
from utils.fusion import FusionModule, load_node_embeddings
from utils.prompter import Prompter

# Features from the mgphot annotation CSV that are human-readable in the prompt
_META_FEATURES = [
    "Tempo", "Danceability", "Minor / Major Key Tonality", "Vocal Register",
    "Aural Intensity", "Syncopation Low to High",
    "Sad Lyrics", "Happy/Joyful Lyrics", "Love/Romance Lyrics",
    "Focus on Lead Vocal", "Focus on Melody", "Focus on Rhythmic Groove",
]


def _load_seq_embed(seq_model: str, seq_embed_path: str, data_path: str) -> torch.Tensor:
    if seq_embed_path:
        pkl_file = seq_embed_path
    else:
        pkl_file = os.path.join(data_path, f"{seq_model}_item_embed.pkl")
    with open(pkl_file, "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, torch.Tensor):
        return raw.float().cpu()
    if isinstance(raw, np.ndarray):
        return torch.from_numpy(raw).float()
    return torch.tensor(raw).float()


def _build_meta_lookup(metadata_path: str, mapping_path: str, item_map: dict) -> dict:
    """Build a lookup {new_idx: description_string} using the mgphot annotations."""
    master = pd.read_csv(mapping_path)
    meta   = pd.read_csv(metadata_path)

    meta["_key"]   = meta["artist_name"].str.lower().str.strip() + "||" + \
                     meta["track_name"].str.lower().str.strip()
    master["_key"] = master["artist_name"].str.lower().str.strip() + "||" + \
                     master["track_name"].str.lower().str.strip()

    # Merge only item_id from master; track_name/artist_name come from meta (suffix _x)
    merged = meta.merge(master[["item_id", "_key"]],
                        on="_key", how="inner").drop_duplicates(subset=["_key"])
    print(f"   Metadata: {len(meta)} rows | master: {len(master)} | matched: {len(merged)}")

    lookup = {}
    for _, row in merged.iterrows():
        new_idx = item_map.get(int(row["item_id"]), None)
        if new_idx is None:
            continue
        desc = f"'{row['track_name']}' by {row['artist_name']}"
        parts = []
        for feat in _META_FEATURES:
            val = row.get(feat, None)
            if pd.notna(val):
                fval = float(val)
                if fval > 0.4:
                    parts.append(f"{feat}: {fval:.1f}")
        if parts:
            desc += f" [{' | '.join(parts)}]"
        lookup[new_idx] = desc
    return lookup


def _build_name_lookup(mapping_path: str, item_map: dict) -> dict:
    master = pd.read_csv(mapping_path)
    lookup = {}
    for _, row in master.iterrows():
        new_idx = item_map.get(int(row["item_id"]), None)
        if new_idx is not None:
            lookup[new_idx] = f"'{row.get('track_name', '?')}' by {row.get('artist_name', '?')}"
    return lookup


def zero_shot_evaluate(
    base_model: str = "/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    data_path: str = "datasets/sequential/LastFM/",
    # Sequential model: SASRec | BERT4Rec | GRU4Rec
    seq_model: str = "SASRec",
    # Full path to the .pkl file; if empty, defaults to <data_path>/<seq_model>_item_embed.pkl
    seq_embed_path: str = "",
    # Audio / lyric nodes — empty string to disable
    audio_node_path: str = "/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/audio_embeddings/node_3",
    lyric_node_path: str = "/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/lyrics_embeddings/node_7",
    # Metadata CSV (mgphot annotations_no_meta_few_shot.csv); empty to disable
    metadata_path: str = "",
    mapping_path: str = "datasets/sequential/LastFM/item_id_master_map.csv",
    cache_dir: str = "",
    output_dir: str = "results",
    task_type: str = "sequential",
    fusion_strategy: str = "concat",
    fusion_weights: str = "",
    fusion_target_dim: int = 0,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"],
    max_test_users: int = 0,
    prompt_template_name: str = "alpaca",
    device_map: str = "auto",
    load_in_4bit: bool = False,
    # Completion ratios: "none" | "prompt" | "embed"
    completion_ratios_path: str = "",
    completion_ratios_mode: str = "none",
):
    os.makedirs(output_dir, exist_ok=True)

    print("\nConfiguration:")
    print(f"  Base model:        {base_model}")
    print(f"  Seq model:         {seq_model}")
    print(f"  Seq embed path:    {seq_embed_path or '(default)'}")
    print(f"  Dataset:           {data_path}")
    print(f"  Audio node path:   {audio_node_path or '(disabled)'}")
    print(f"  Lyric node path:   {lyric_node_path or '(disabled)'}")
    print(f"  Metadata path:     {metadata_path or '(disabled)'}")
    print(f"  Fusion strategy:   {fusion_strategy}")
    print(f"  Completion mode:   {completion_ratios_mode}")

    parsed_weights: Optional[List[float]] = None
    if isinstance(fusion_weights, (list, tuple)) and len(fusion_weights) > 0:
        parsed_weights = [float(x) for x in fusion_weights]
    elif isinstance(fusion_weights, str) and fusion_weights.strip():
        try:
            parsed_weights = [float(x) for x in fusion_weights.split(",")]
        except ValueError:
            raise ValueError(f"--fusion_weights must be comma-separated floats. Got: '{fusion_weights}'")

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    print("\n1. Loading dataset...")
    dataset  = SequentialDataset(data_path, 50)
    item_map = dataset.item_map
    n_items  = len(item_map) + 1

    # ── 2. Completion ratios ──────────────────────────────────────────────────
    comp_data = None
    if completion_ratios_path and completion_ratios_mode != "none":
        with open(completion_ratios_path, "rb") as _f:
            comp_data = pickle.load(_f)
        print(f"   Loaded completion ratios for {len(comp_data)} users.")

    # ── 3. Sequential model embeddings ───────────────────────────────────────
    print(f"\n2. Loading {seq_model} embeddings...")
    raw_seq   = _load_seq_embed(seq_model, seq_embed_path, data_path)
    seq_embed = torch.zeros(n_items, raw_seq.shape[1])
    for raw_id, new_idx in item_map.items():
        if raw_id < raw_seq.shape[0]:
            seq_embed[new_idx] = raw_seq[raw_id]
    print(f"   {seq_model}: {raw_seq.shape} → remapped {seq_embed.shape}")
    del raw_seq

    # ── 4. Audio / lyric embeddings ───────────────────────────────────────────
    audio_embed = load_node_embeddings(audio_node_path, mapping_path, n_items, item_map, "audio")
    lyric_embed = load_node_embeddings(lyric_node_path, mapping_path, n_items, item_map, "lyric")

    # ── 5. Fuse ───────────────────────────────────────────────────────────────
    embeds = [seq_embed]
    dims   = [seq_embed.shape[1]]
    if audio_embed is not None:
        embeds.append(audio_embed); dims.append(audio_embed.shape[1])
    if lyric_embed is not None:
        embeds.append(lyric_embed); dims.append(lyric_embed.shape[1])

    if len(embeds) == 1:
        item_embed = seq_embed
        print(f"\n3. No audio/lyric embeddings — using {seq_model} only.")
    else:
        print(f"\n3. Fusing embeddings (strategy={fusion_strategy})...")
        target_dim = fusion_target_dim if fusion_target_dim > 0 else dims[0]
        fw = parsed_weights if fusion_strategy == "weighted_sum" else None
        fusion = FusionModule(modality_dims=dims, strategy=fusion_strategy,
                              output_dim=target_dim, fixed_weights=fw)
        fusion.eval()
        with torch.no_grad():
            item_embed = fusion(embeds)
        print(f"   Fused shape: {item_embed.shape}")

    # ── 6. Name / metadata lookups ────────────────────────────────────────────
    if metadata_path:
        print("\n4. Building metadata lookup...")
        desc_lookup = _build_meta_lookup(metadata_path, mapping_path, item_map)
        name_lookup = _build_name_lookup(mapping_path, item_map)
    else:
        desc_lookup = {}
        name_lookup = _build_name_lookup(mapping_path, item_map)

    # ── 7. Model ──────────────────────────────────────────────────────────────
    print(f"\n5. Loading base model: {base_model}")
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
        load_in_4bit=load_in_4bit,
        instruction_text=prompter.generate_prompt(task_type),
        user_embeds=None,
        input_embeds=item_embed,
        use_completion_embed=(completion_ratios_mode == "embed"),
    )
    model.eval()

    # ── 8. Evaluation ─────────────────────────────────────────────────────────
    print("\n6. Running zero-shot evaluation...")
    topk    = [1, 5, 10, 20, 100]
    results = {m: np.zeros(len(topk)) for m in ["Precision", "Recall", "MRR", "MAP", "NDCG"]}

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
                f"{i+1}. {desc_lookup.get(iid, name_lookup.get(iid, f'Item {iid}'))}"
                for i, iid in enumerate(full_history[-10:])
            ]
            prompt_texts = prompter.generate_prompt(task_type, "\n".join(history_lines))

            device      = next(model.llama_model.parameters()).device
            inputs      = torch.LongTensor(seq).to(device).unsqueeze(0)
            inputs_mask = torch.ones(inputs.size()).to(device)

            eval_hist_meta   = [prompt_texts[0]]
            eval_comp_ratios = None

            if comp_data is not None:
                user_ratios  = comp_data.get(u + 1, [])
                # Only include ratios for the 10 songs shown in the prompt
                prompt_window = full_history[-10:]
                ratios_slice  = user_ratios[-len(seq):] if user_ratios else []
                # Align: take the last min(10, len(prompt_window)) ratios
                prompt_ratios = ratios_slice[-len(prompt_window):] if ratios_slice else []

                if completion_ratios_mode == "prompt":
                    ratio_strs = [
                        "?" if (v is None or math.isnan(v)) else f"{v:.2f}"
                        for v in prompt_ratios
                    ]
                    paired = "\n".join(
                        f"{line}  [completion: {r}]"
                        for line, r in zip(history_lines, ratio_strs)
                    )
                    base_prompt = prompter.generate_prompt(task_type, paired)[0]
                    eval_hist_meta = [base_prompt]
                elif completion_ratios_mode == "embed":
                    vals = [0.0 if (v is None or math.isnan(v)) else float(v) for v in ratios_slice]
                    eval_comp_ratios = torch.FloatTensor([vals]).to(device)

            if n_eval == 0:
                print("\n--- Prompt sent to LLM (first user) ---")
                print(eval_hist_meta[0])
                print("--- End of prompt ---\n")

            _, ratings = model.predict(
                inputs, inputs_mask,
                history_metadata=eval_hist_meta,
                completion_ratios=eval_comp_ratios,
            )
            idx_row  = torch.arange(ratings.size(0)).unsqueeze(1)
            ratings  = ratings[idx_row, selected_items]

            _, ratings_K = torch.topk(ratings, k=topk[-1])
            ratings_K    = ratings_K.cpu().numpy()
            r = getLabel(groundTruth, ratings_K)

            for j, k in enumerate(topk):
                results["Precision"][j] += RecallPrecision_atK(groundTruth, r, k)[0]
                results["Recall"][j]    += RecallPrecision_atK(groundTruth, r, k)[1]
                results["MRR"][j]       += MRR_atK(groundTruth, r, k)
                results["MAP"][j]       += MAP_atK(groundTruth, r, k)
                results["NDCG"][j]      += NDCG_atK(groundTruth, r, k)
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

    # ── 9. Build output filename ──────────────────────────────────────────────
    modalities = [seq_model]
    if audio_embed is not None:
        modalities.append("audio")
    if lyric_embed is not None:
        modalities.append("lyric")
    if metadata_path:
        modalities.append("metadata")

    if len(modalities) == 1:
        tag = f"{seq_model}_only"
    elif fusion_strategy == "weighted_sum" and parsed_weights:
        weight_tag = "-".join(f"{w:.2f}" for w in parsed_weights)
        tag = "_".join(modalities) + f"_weighted_sum_{weight_tag}"
    else:
        tag = "_".join(modalities) + f"_{fusion_strategy}"

    if completion_ratios_mode != "none":
        tag = f"{tag}_completion_{completion_ratios_mode}"

    output_file = os.path.join(output_dir, f"zeroshot_{tag}.txt")
    with open(output_file, "w") as f:
        f.write(f"Zero-Shot Evaluation — {tag}\n")
        f.write(f"Base model: {base_model}\n")
        f.write(f"Seq model:  {seq_model}\n")
        f.write(f"Dataset:    {data_path}\n")
        f.write(f"Audio node path: {audio_node_path}\n")
        f.write(f"Lyric node path: {lyric_node_path}\n")
        f.write(f"Metadata path:   {metadata_path}\n")
        f.write(f"Fusion strategy: {fusion_strategy}\n")
        f.write(f"Completion mode: {completion_ratios_mode}\n\n")
        f.write(df.to_string(float_format=lambda x: f"{x:.3f}"))
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
