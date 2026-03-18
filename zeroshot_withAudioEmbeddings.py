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
    base_model: str = "huggyllama/llama-7b", 
    data_path: str = "datasets/sequential/LastFM/",
    # Paths for your audio ablation
    audio_emb_path: str = "/project/pi_dagarwal_umass_edu/project_7/srikar/output_sample/final/audio_encodec.pt",
    mapping_path: str = "datasets/sequential/LastFM/item_id_master_map.csv",  
    cache_dir: str = "",
    output_dir: str = "results",
    task_type: str = "sequential",
    cutoff_len: int = 4096,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"],
    max_test_users: int = 0,  
    batch_size: int = 1,  
    prompt_template_name: str = "alpaca",
    use_sasrec_embedding: bool = True, 
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n1. Loading dataset...")
    dataset = SequentialDataset(data_path, 50)
    
    print(f"\n2. Loading and mapping Embeddings...")
    # Load SASRec Baseline 
    sasrec_file = os.path.join(data_path, 'SASRec_item_embed.pkl')
    print(f"   Loading SASRec from {sasrec_file}...")
    
    with open(sasrec_file, 'rb') as f:
        raw_sasrec = pickle.load(f)
    
    if isinstance(raw_sasrec, torch.Tensor):
        sasrec_embed = raw_sasrec.float()
    elif isinstance(raw_sasrec, np.ndarray):
        sasrec_embed = torch.from_numpy(raw_sasrec).float()
    else:
        sasrec_embed = torch.tensor(raw_sasrec).float()
    

    
    if os.path.exists(audio_emb_path) and os.path.exists(mapping_path):
        print(f"   Loading audio from {audio_emb_path}...")
        
        audio_data = torch.load(audio_emb_path, map_location="cpu")

        track_ids = audio_data["track_ids"]
        embeddings = audio_data["embeddings"]

        audio_dict = {
            str(track_ids[i]): torch.tensor(embeddings[i])
            for i in range(len(track_ids))
        }
        id_map = pd.read_csv(mapping_path)
        
        # Debug prints
        print(f"Audio dict keys type: {type(list(audio_dict.keys())[0])}")
        print(f"Audio dict keys sample: {list(audio_dict.keys())[:5]}")
        print(f"Track index type: {type(id_map['track_index'].iloc[0])}")
        print(f"Track index sample: {id_map['track_index'].head().tolist()}")
        
        # Detect audio dimension
        sample_val = next(iter(audio_dict.values()))
        audio_dim = torch.tensor(sample_val).shape[0]
        
        aligned_audio = torch.zeros((sasrec_embed.shape[0], audio_dim))
        
        # Create mapping lookup 
        # id_lookup = {str(v): k for v, k in zip(id_map['item_id'], id_map['track_index'])}
        id_lookup = {str(k): v for v, k in zip(id_map['item_id'], id_map['track_index'])}
        
        # print(f"Number of audio embeddings: {len(audio_dict)}")
        # print(f"Number of mappings: {len(id_lookup)}")
        print("audio_dict len:", len(audio_dict))
        audio_ids = set(audio_dict.keys())
        csv_ids = set(id_map["track_index"].astype(str))
        print("in audio only:", len(audio_ids - csv_ids))
        print("in csv only:", len(csv_ids - audio_ids))
        print("matched:", len(audio_ids & csv_ids))

        
        # Check first 10 audio keys
        missed = []
        for raw_track_id in list(audio_dict.keys())[:10]:
            str_id = str(raw_track_id)
            if str_id not in id_lookup:
                missed.append(str_id)
        print(f"Missed keys sample (first 10): {missed}")
        
        device = sasrec_embed.device
        aligned_audio = torch.zeros((sasrec_embed.shape[0], audio_dim), device=device)
        
        found_count = 0
        for raw_track_id, emb in audio_dict.items():
            str_id = str(raw_track_id)
            if str_id in id_lookup:
                item_id = id_lookup[str_id]
                if item_id - 1 < sasrec_embed.shape[0]:
                    emb_tensor = torch.tensor(emb) if not isinstance(emb, torch.Tensor) else emb
                    aligned_audio[item_id - 1] = emb_tensor.float()
                    found_count += 1
        
        print(f"   Matched {found_count} audio embeddings.")
        item_embed = torch.cat([sasrec_embed, aligned_audio], dim=-1)

    else:
        print("   WARNING: Audio or mapping not found. Using SASRec only.")
        item_embed = sasrec_embed

    # Final move to GPU is handled by LLM4Rec's device_map="auto"
    print(f"   Final item embedding shape: {item_embed.shape}")

    print(f"\n3. Loading base model: {base_model}")
    prompter = Prompter(prompt_template_name)
    model = LLM4Rec(
        base_model=base_model,
        task_type=task_type,
        cache_dir=cache_dir,
        input_dim=item_embed.shape[1], # Now accounts for concatenated audio
        output_dim=dataset.m_item,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        device_map="auto",
        instruction_text=prompter.generate_prompt(task_type),
        user_embeds=None,
        input_embeds=item_embed,
    )
    model.eval()
    
    print(f"\n4. Running evaluation...")
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
            selected_items = [[dataset.testData[u][1]] + dataset.allPos[u]]
            groundTruth = [[0]]

            inputs = torch.LongTensor(seq).to(device).unsqueeze(0)
            inputs_mask = torch.ones(inputs.size()).to(device)
            
            _, ratings = model.predict(inputs, inputs_mask)
            
            idx_row = torch.arange(ratings.size(0)).unsqueeze(1)
            ratings = ratings[idx_row, selected_items]
            
            _, ratings_K = torch.topk(ratings, k=topk[-1])
            r = getLabel(groundTruth, ratings_K.cpu().numpy())
            
            for j, k in enumerate(topk):
                pre, rec = RecallPrecision_atK(groundTruth, r, k)
                results['Precision'][j] += pre
                results['Recall'][j] += rec
                results['MRR'][j] += MRR_atK(groundTruth, r, k)
                results['MAP'][j] += MAP_atK(groundTruth, r, k)
                results['NDCG'][j] += NDCG_atK(groundTruth, r, k)

    # Average Results
    num_eval = len(test_keys)
    for k in results: results[k] /= float(num_eval)
    
    rounded_results = {k: np.round(results[k], 4) for k in results}
    
    df_results = pd.DataFrame(rounded_results, index=[f"Top-{k}" for k in topk])
    print("\n" + df_results.to_string())

    # Save Results
    output_file = os.path.join(output_dir, "with_AudioEmbeddings.txt")
    with open(output_file, "w") as f:
        f.write(f"Base model: {base_model}\n")
        f.write(df_results.to_string())

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return results

if __name__ == "__main__":
    fire.Fire(zero_shot_evaluate)
