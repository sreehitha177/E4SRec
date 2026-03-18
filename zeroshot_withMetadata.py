import os
import torch
import numpy as np
import pandas as pd
from typing import List
import fire
from model_new import LLM4Rec
from utils.data_utils import SequentialDataset
from utils.eval_utils import RecallPrecision_atK, MRR_atK, MAP_atK, NDCG_atK, getLabel
from utils.prompter import Prompter

def zero_shot_evaluate(
    base_model: str = "huggyllama/llama-7b", 
    data_path: str = "datasets/sequential/LastFM/",
    cache_dir: str = "",
    output_dir: str = "results",
    task_type: str = "sequential",
    cutoff_len: int = 4096,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"],
    max_test_users: int = 100,  
    batch_size: int = 1,  
    prompt_template_name: str = "alpaca",
    use_sasrec_embedding: bool = True, 
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
    full_meta = pd.read_csv("/work/pi_dagarwal_umass_edu/project_7/swetha/lastfm_all_mgphot_labels.csv")
    
    # Merge them to link YOUR item_id to metadata
    merged_meta = master_map.merge(full_meta, on="track_index")
    
    # Create the lookup: { item_id: "Description String" }
    meta_lookup = {}
    for _, row in merged_meta.iterrows():
        # Using track_name_x/artist_name_x because they come from the master_map columns
        desc = f"'{row['track_name_x']}' by {row['artist_name_x']} [{row['genre']}, {row['tempo']}]"
        meta_lookup[row['item_id']] = desc
    
    # print(f"Meta lookup size: {len(meta_lookup)}", flush=True)
    # # Check the first few entries of your lookup
    # print("--- LOOKUP SAMPLE ---")
    # for k in list(meta_lookup.keys())[:5]:
    #     print(f"ID {k}: {meta_lookup[k]}")

    # # Check what ID 1 actually is in your lookup
    # print(f"--- VERIFYING ID 1 ---")
    # print(f"Metadata for ID 1: {meta_lookup.get(1, 'NOT FOUND')}")
    # # --- VERIFICATION BLOCK ---
    # test_id = 1
    # # 1. What does your current lookup say ID 1 is?
    # print(f"DEBUG: My lookup says ID {test_id} is: {meta_lookup.get(test_id)}")

    # # 2. What does the Master Map say ID 1's track_name is?
    # master_map = pd.read_csv(os.path.join(data_path, "item_id_master_map.csv"))
    # sample_row = master_map[master_map['item_id'] == test_id]
    # print(f"DEBUG: Master Map says ID {test_id} is: {sample_row['track_name'].values[0]}")
    # # --------------------------

    print(f"\nLoading base model: {base_model}")
    prompter = Prompter(prompt_template_name)
    
    model = LLM4Rec(
        base_model=base_model,
        task_type=task_type,
        cache_dir=cache_dir,
        input_dim=item_embed.shape[1], # Dynamic based on SASRec weights
        output_dim=dataset.m_item,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        device_map="auto", # Recommended for 7B models
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
    if max_test_users:
        test_keys = test_keys[:max_test_users]
    
    testData = {k: dataset.testData[k] for k in test_keys}
    users = np.array(test_keys) # Only evaluate selected test keys
    
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
                desc = meta_lookup.get(item_id, "Unknown Track")
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
    
    num_evaluated_users = len(users)
    for key in results.keys():
        results[key] /= float(num_evaluated_users)
    
    # Output Table
    df_results = pd.DataFrame({
        "Metric": results.keys(),
        "Top-1": [results[k][0] for k in results],
        "Top-5": [results[k][1] for k in results],
        "Top-10": [results[k][2] for k in results],
        "Top-100": [results[k][4] for k in results]
    })
    print("\n" + df_results.to_string(index=False))

    # Save results
    output_file = os.path.join(output_dir, "zero_shot_with_metadata_results.txt")
    with open(output_file, "w") as f:
        f.write("Zero-Shot Evaluation Results\n")
        f.write(f"Base model: {base_model}\n")
        f.write(f"Dataset: {data_path}\n\n")
        f.write(df_results.to_string(index=False))
        f.write("\n\nDetailed Arrays:\n")
        for key in results:
            f.write(f"{key}: {results[key]}\n")

    print(f"\nResults saved to {output_file}")


    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return results

if __name__ == "__main__":
    fire.Fire(zero_shot_evaluate)