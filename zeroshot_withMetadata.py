import os
import torch
import numpy as np
import pandas as pd
from typing import List
import fire
from model import LLM4Rec
from utils.data_utils import SequentialDataset
from utils.eval_utils import RecallPrecision_atK, MRR_atK, MAP_atK, NDCG_atK, getLabel
from utils.prompter import Prompter

def zero_shot_evaluate(
    base_model: str = "/datasets/ai/qwen2/hub/models--Qwen--Qwen2.5-32B/snapshots/1818d35814b8319459f4bd55ed1ac8709630f003",
    data_path: str = "datasets/sequential/LastFM/",
    metadata_path: str = "/project/pi_dagarwal_umass_edu/project_7/hmagapu/top_50k_full_augmented.csv",
    cache_dir: str = "/datasets/ai/qwen2/hub",
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
    print(f"\nConfiguration:")
    print(f"  Base model: {base_model}")
    print(f"  Dataset: {data_path}")
    print(f"  Embedding type: SASRec ")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nLoading dataset...")
    dataset = SequentialDataset(data_path, 50)
    
    print(f"\nLoading SASRec embeddings...")
    user_embed = None
    import pickle
    item_embed = pickle.load(open(data_path + 'SASRec_item_embed.pkl', 'rb'))
    print(f"   Loaded SASRec embeddings with shape: {item_embed.shape}")

    
    print("Building Metadata Lookup Table...")
    master_map = pd.read_csv(os.path.join(data_path, "item_id_master_map.csv"))
    full_meta = pd.read_csv(metadata_path)

    # Join on normalised artist + track name 
    full_meta['_key'] = full_meta['artist_name'].str.lower().str.strip() + '||' + \
                        full_meta['track_name'].str.lower().str.strip()
    master_map['_key'] = master_map['artist_name'].str.lower().str.strip() + '||' + \
                         master_map['track_name'].str.lower().str.strip()

    merged_meta = full_meta.merge(master_map[['item_id', '_key']], on='_key', how='inner') \
                            .drop_duplicates(subset=['_key'])
    print(f"   Metadata file: {len(full_meta)} rows | master items: {len(master_map)} | matched: {len(merged_meta)}")

    # Fallback lookup: { item_id -> "'track' by artist" } for items without full metadata
    name_lookup = {
        row['item_id']: f"'{row['track_name']}' by {row['artist_name']}"
        for _, row in master_map.iterrows()
        if pd.notna(row.get('track_name')) and pd.notna(row.get('artist_name'))
    }

    # Build lookup: { item_id -> description string }
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
        if _fmt(row.get('energy')):  parts.append(f"energy: {row['energy']:.2f}")
        meta_str = ' | '.join(parts)
        desc = f"'{row['track_name']}' by {row['artist_name']}"
        if meta_str:
            desc += f" [{meta_str}]"
        meta_lookup[row['item_id']] = desc

    print(f"\nLoading base model: {base_model}")
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
        user_embeds=user_embed,
        input_embeds=item_embed,
    )
    model.eval()
    
    print(f"\nRunning zero-shot evaluation...")
    topk = [1, 5, 10, 20, 100]
    results = {
        'Precision': np.zeros(len(topk)),
        'Recall': np.zeros(len(topk)),
        'MRR': np.zeros(len(topk)),
        'MAP': np.zeros(len(topk)),
        'NDCG': np.zeros(len(topk))
    }
    
    test_keys = list(dataset.testData.keys())
    if max_test_users and max_test_users > 0:
        test_keys = test_keys[:max_test_users]
    
    testData = {k: dataset.testData[k] for k in test_keys}
    users = np.array(test_keys) # Only evaluate selected test keys

    num_evaluated_users = 0
    with torch.no_grad():
        for u in users:
            if u not in testData or len(testData[u]) == 0:
                continue
            
            full_history = testData[u][0]
            seq = full_history[-256:] if len(full_history) > 256 else full_history
            selected_items = [[testData[u][1]] + dataset.allPos[u]]
            groundTruth = [[0]]
            
            
            recent_history = full_history[-10:] if len(full_history) > 10 else full_history
            history_lines = []
            for i, item_id in enumerate(recent_history):
                # Fetch the text description from our lookup
                desc = meta_lookup.get(item_id, name_lookup.get(item_id, "Unknown Track"))
                history_lines.append(f"{i+1}. {desc}")
            
            # history 
            history_text = "\n".join(history_lines)
            
            # Get the dynamic prompt (Instruction + Metadata)
            prompt_texts = prompter.generate_prompt(task_type, history_text)


            if u == 0: 
            #     print(f"DEBUG: VERIFYING USER ID: {u}")
            #     print(f"DEBUG: RAW SEQUENCE FROM DATASET: {full_history}")
                print("DEBUG: PROMPT SENT TO MODEL:")
                print(prompt_texts[0])
            
           

            device = next(model.llama_model.parameters()).device
            inputs = torch.LongTensor(seq).to(device).unsqueeze(0)
            inputs_mask = torch.ones(inputs.size()).to(device)
            
            # _, ratings = model.predict(inputs, inputs_mask)
            _, ratings = model.predict(
                inputs, 
                inputs_mask, 
                history_metadata=[prompt_texts[0]]
            )
            
            idx_row = torch.arange(ratings.size(0)).unsqueeze(1)
            ratings = ratings[idx_row, selected_items]
            
            _, ratings_K = torch.topk(ratings, k=topk[-1])
            ratings_K = ratings_K.cpu().numpy()
            
            r = getLabel(groundTruth, ratings_K)
            for j, k in enumerate(topk):
                pre, rec = RecallPrecision_atK(groundTruth, r, k)
                mrr = MRR_atK(groundTruth, r, k)
                map_val = MAP_atK(groundTruth, r, k)
                ndcg = NDCG_atK(groundTruth, r, k)
                
                results['Precision'][j] += pre
                results['Recall'][j] += rec
                results['MRR'][j] += mrr
                results['MAP'][j] += map_val
                results['NDCG'][j] += ndcg
            num_evaluated_users += 1

    if num_evaluated_users == 0:
        raise RuntimeError("No eligible users found for zero-shot evaluation.")
    for key in results.keys():
        results[key] /= float(num_evaluated_users)

    # Output Table
    df_results = pd.DataFrame(
        {k: np.round(results[k], 3) for k in results},
        index=[f"Top-{k}" for k in topk]
    )
    np.set_printoptions(precision=3, suppress=True)
    print("\n" + df_results.to_string(float_format=lambda x: f"{x:.3f}"))

    # Save results
    output_file = os.path.join(output_dir, f"zeroshot_METADATA_{base_model.replace('/', '_')}.txt")
    with open(output_file, "w") as f:
        f.write("Zero-Shot Evaluation Results\n")
        f.write(f"Base model: {base_model}\n")
        f.write(f"Dataset: {data_path}\n\n")
        f.write(df_results.to_string(float_format=lambda x: f"{x:.3f}"))
        f.write("\n\nDetailed Arrays:\n")
        for key in results:
            f.write(f"{key}: {np.round(results[key], 3)}\n")

    print(f"\nResults saved to {output_file}")


    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return results

if __name__ == "__main__":
    fire.Fire(zero_shot_evaluate)