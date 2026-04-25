import os
from typing import List

import fire
import pickle
import numpy as np
import torch
import transformers

from model import LLM4Rec
from utils.data_utils import (
    BipartiteGraphCollator,
    BipartiteGraphDataset,
    SequentialCollator,
    SequentialDataset,
)
from utils.eval_utils import MAP_atK, MRR_atK, NDCG_atK, RecallPrecision_atK, getLabel
from utils.prompter import Prompter


def train(
    # model/data params
    base_model: str = "/home/snarayana_umass_edu/E4SRec-1/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    data_path: str = "datasets/sequential/LastFM/",
    cache_dir: str = "",
    output_dir: str = "",
    task_type: str = "sequential",
    use_completion_ratio: bool = False,
    completion_path: str = "data_preproc/user_sessions_with_completion.csv",
    source_path: str = "data_preproc/user_sessions_lastfm1k_minuser1000_minitem7_sessgap1200_minsesslen10_minhist50.csv",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4096,
    val_set_size: int = 0,
    lr_scheduler: str = "cosine",
    warmup_steps: int = 100, 
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # Qwen 2.5 attention projections; override from CLI for other backbones.
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca"
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Params using prompt template {prompt_template_name}:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"cache_dir: {cache_dir}\n"
            f"output_dir: {output_dir}\n"
            f"task_type: {task_type}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"warmup_steps: {warmup_steps}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )
    assert base_model, "Please specify a --base_model, e.g. a local Qwen snapshot path"
    if batch_size % micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print(f"DDP enabled: LOCAL_RANK={os.environ.get('LOCAL_RANK')}, WORLD_SIZE={world_size}")  
        print("gradient_accumulation_steps: ", gradient_accumulation_steps)

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if task_type == 'general':
        dataset = BipartiteGraphDataset(data_path)
        user_embed, item_embed = (
            pickle.load(open(data_path + 'VanillaMF_user_embed.pkl', 'rb')),
            pickle.load(open(data_path + 'VanillaMF_item_embed.pkl', 'rb')),
        )
        item_embed = torch.cat([item_embed.mean(dim=0).unsqueeze(0), item_embed], dim=0)
        data_collator = BipartiteGraphCollator()
        input_dim = 64
    elif task_type == 'sequential':
        sasrec_embed_path = os.path.join(data_path, "SASRec_item_embed.pkl")
        if not os.path.exists(sasrec_embed_path):
            raise FileNotFoundError(
                f"Missing sequential item embeddings: {sasrec_embed_path}. "
                "Generate SASRec_item_embed.pkl before finetuning."
            )
        dataset = SequentialDataset(
            data_path,
            50,
            use_completion_ratio=use_completion_ratio,
            completion_path=completion_path,
            source_path=source_path,
        )
        user_embed, item_embed = None, pickle.load(open(sasrec_embed_path, 'rb'))
        data_collator = SequentialCollator()
        input_dim = item_embed.shape[1]
    else:
        raise ValueError("task_type must be either 'general' or 'sequential'")

    model = LLM4Rec(
        base_model=base_model,
        task_type=task_type,
        cache_dir=cache_dir,
        input_dim=input_dim,
        output_dim=dataset.m_item,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        device_map=device_map,
        instruction_text=prompter.generate_prompt(task_type),
        user_embeds=user_embed,
        input_embeds=item_embed,
        use_completion_ratio=use_completion_ratio,
    )

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=None,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            # fp16=True,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            gradient_checkpointing=True,         # trades compute for memory
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=0,
            # evaluation_strategy="steps" if val_set_size > 0 else "no",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=500,
            lr_scheduler_type=lr_scheduler,
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="none",
            run_name=None,
        ),
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.eval()
    topk = [1, 5, 10, 20, 100]
    results = {'Precision': np.zeros(len(topk)),
               'Recall': np.zeros(len(topk)),
               'MRR': np.zeros(len(topk)),
               'MAP': np.zeros(len(topk)),
               'NDCG': np.zeros(len(topk))}

    testData = dataset.testData
    users = np.arange(dataset.n_user)
    for u in users:
        if task_type == 'general':
            all_pos = [dataset.allPos[u]]
            groundTruth = [testData[u]]
            inputs = torch.LongTensor([u] + all_pos[0]).cuda().unsqueeze(0)
            inputs_mask = torch.ones(inputs.shape).cuda()
            _, ratings = model.predict(inputs, inputs_mask)
            exclude_index = []
            exclude_items = []
            for range_i, its in enumerate(all_pos):
                exclude_index.extend([range_i] * len(its))
                exclude_items.extend(its)
            ratings[exclude_index, exclude_items] = -(1 << 10)

        elif task_type == 'sequential':
            seq, completion_ratio, target = dataset.get_eval_record(u, subset='test')
            if len(seq) == 0:
                continue
            seq = seq[-dataset.maxlen:]
            if completion_ratio is not None:
                completion_ratio = completion_ratio[-dataset.maxlen:]
            selected_items = [[target] + dataset.allPos[u]]
            groundTruth = [[0]]
            device = next(model.llama_model.parameters()).device
            inputs = torch.LongTensor(seq).to(device).unsqueeze(0)
            inputs_mask = torch.ones(inputs.shape, device=device)
            completion_tensor = None
            if completion_ratio is not None:
                completion_tensor = torch.FloatTensor(completion_ratio).to(device).unsqueeze(0)
            _, ratings = model.predict(inputs, inputs_mask, completion_ratio=completion_tensor)
            ratings = ratings[[[[k] * len(selected_items[0]) for k in range(len(ratings))], selected_items]]

        _, ratings_K = torch.topk(ratings, k=topk[-1])
        ratings_K = ratings_K.cpu().numpy()

        r = getLabel(groundTruth, ratings_K)
        for j, k in enumerate(topk):
            pre, rec = RecallPrecision_atK(groundTruth, r, k)
            mrr = MRR_atK(groundTruth, r, k)
            map = MAP_atK(groundTruth, r, k)
            ndcg = NDCG_atK(groundTruth, r, k)
            results['Precision'][j] += pre
            results['Recall'][j] += rec
            results['MRR'][j] += mrr
            results['MAP'][j] += map
            results['NDCG'][j] += ndcg

    for key in results.keys():
        results[key] /= float(len(users))
    print(f'Evaluation for User: \n')
    for j, k in enumerate(topk):
        print(f'Precision@{k}: {results["Precision"][j]} \n '
              f'Recall@{k}: {results["Recall"][j]} \n '
              f'MRR@{k}: {results["MRR"][j]} \n '
              f'MAP@{k}: {results["MAP"][j]} \n '
              f'NDCG@{k}: {results["NDCG"][j]} \n')

    os.makedirs(output_dir, exist_ok=True)
    model.llama_model.save_pretrained(output_dir)
    model_path = os.path.join(output_dir, "adapter.pth")
    if task_type == 'general':
        user_proj, input_proj, score = model.user_proj.state_dict(), model.input_proj.state_dict(), model.score.state_dict()
        torch.save({'user_proj': user_proj, 'input_proj': input_proj, 'score': score}, model_path)
    elif task_type == 'sequential':
        input_proj, score = model.input_proj.state_dict(), model.score.state_dict()
        torch.save({'input_proj': input_proj, 'score': score}, model_path)
    print(f"Saved PEFT adapter to {output_dir}")
    print(f"Saved projection heads to {model_path}")


if __name__ == "__main__":
    torch.cuda.empty_cache() 
    fire.Fire(train)
