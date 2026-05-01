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
    # base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    base_model: str = "/datasets/ai/qwen2/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
    data_path: str = "datasets/sequential/LastFM/",
    cache_dir: str = "",
    output_dir: str = "results",
    task_type: str = "sequential",
    cutoff_len: int = 4096,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # Llama-specific target modules
    lora_target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"],

    max_test_users: int = 0,  
    batch_size: int = 1,  
    prompt_template_name: str = "alpaca",
    
    use_sasrec_embedding: bool = True,
    device_map: str = "auto",
):
    print(f"\nConfiguration:")
    print(f"  Base model: {base_model}")
    print(f"  Dataset: {data_path}")
    print(f"  Embedding type: SASRec ")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\n1. Loading dataset...")
    dataset = SequentialDataset(data_path, 50)

    print(f"\n2. Loading SASRec embeddings...")
    user_embed = None
    import pickle
    raw_embed = pickle.load(open(data_path + 'SASRec_item_embed.pkl', 'rb'))
    if not isinstance(raw_embed, torch.Tensor):
        raw_embed = torch.tensor(raw_embed)
    raw_embed = raw_embed.float().cpu()
    print(f"   Raw SASRec shape: {raw_embed.shape}")
    input_dim = raw_embed.shape[1]
    n_items = len(dataset.item_map) + 1
    item_embed = torch.zeros(n_items, input_dim)
    for raw_id, new_idx in dataset.item_map.items():
        if raw_id < raw_embed.shape[0]:
            item_embed[new_idx] = raw_embed[raw_id]
    print(f"   Remapped embeddings: {item_embed.shape}")
    
    print(f"\n3. Loading base model: {base_model}")
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
        device_map=device_map,        
        instruction_text=prompter.generate_prompt(task_type),
        user_embeds=user_embed,
        input_embeds=item_embed,
    )
    model.eval()
    
    print(f"\n4. Running zero-shot evaluation...")
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

            selected_items = [dataset.allPos[u]]
            groundTruth = [[0]]

            device = next(model.llama_model.parameters()).device
            inputs = torch.LongTensor(seq).to(device).unsqueeze(0)
            inputs_mask = torch.ones(inputs.size()).to(device)

            _, ratings = model.predict(inputs, inputs_mask)
            
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
    output_file = os.path.join(output_dir, f"zeroshot_results_{base_model.replace('/', '_')}.txt")
    with open(output_file, "w") as f:
        f.write("Zero-Shot Evaluation Results (Global Temporal Split)\n")
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
