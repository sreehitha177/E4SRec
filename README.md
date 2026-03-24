### Environment

At least 2 * A800 is required. Preferable 8 * A800.

    pip install requirements.txt
  

### Training

    torchrun --nproc_per_node=8 --master_port=1234 finetune.py \
        --base_model garage-bAInd/Platypus2-70B-instruct \
        --data_path Beauty \
        --task_type sequential \
        --cache_dir cache_dir/ \
        --output_dir output_dir/ \
        --batch_size 16 \
        --micro_batch_size 1 \
        --num_epochs 3 \
        --learning_rate 0.0003 \
        --cutoff_len 4096 \
        --val_set_size 0 \
        --lora_r 16 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_target_modules '[gate_proj, down_proj, up_proj]' \
        --train_on_inputs False \
        --add_eos_token False \
        --group_by_length False \
        --prompt_template_name alpaca \
        --lr_scheduler 'cosine' \
        --warmup_steps 100

### Inference

    torchrun --nproc_per_node=8 --master_port=1234 inference.py \
        --base_model garage-bAInd/Platypus2-70B-instruct \
        --data_path Beauty \
        --task_type sequential \
        --checkpoint_dir checkpoint_dir \
        --cache_dir cache_dir/ \
        --output_dir output_dir/ \
        --batch_size 16 \
        --micro_batch_size 1

---

## LastFM-1K Data Configuration

### Dataset

Source: `user_sessions_lastfm1k_minuser1000_minitem7_sessgap1200_minsesslen10_minhist50.csv`
(4.2M interactions, 814 users with ≥5 interactions, 383,670 unique tracks)

### Data Split — Global Temporal Split

The original E4SRec leave-last-out split was replaced with a **global temporal split** that respects
chronological order across all users.

**Test cutoff**: 90th-percentile timestamp of all interactions → `2009-01-26 20:24:25`  
**Val cutoff**: 90th-percentile timestamp of pre-test interactions → `2008-10-17 00:44:19`

| Split | Rule | Users |
|-------|------|-------|
| Train | interactions ≤ val cutoff (val users) or ≤ test cutoff (others) | 638 |
| Val   | interactions in (val cutoff, test cutoff] — 10% of users only | 60 |
| Test  | interactions > test cutoff | 638 |

~10% of users (81 selected, 60 retained after filtering) are designated as **validation users**;
their train window is cut off at the val cutoff instead of the test cutoff, so their val
interactions are never seen during training. All users contribute to the test set.

Preprocessing outputs written to `datasets/sequential/LastFM/`:
- `train.txt`, `val.txt`, `test.txt` — space-separated `uid item1 item2 ...`
- `test_sample.txt`, `val_sample.txt` — 1 positive + 199 random negatives per user
- `item_id_master_map.csv` — mapping from contiguous item IDs to original track IDs

### Running

**Step 1 — Preprocess** (generates split files):

    conda activate dolby
    cd /work/pi_dagarwal_umass_edu/project_7/E4SRec
    python preprocess_lastfm.py

**Step 2 — Train SASRec baseline** (produces item embeddings):

    sbatch train_sasrec.sh

SASRec is trained on `train.txt`, evaluated on `test.txt` using 200-candidate ranking.
Best checkpoint is saved to `datasets/sequential/LastFM/SASRec_item_embed.pkl`.

**Step 3 — Zero-shot LLaMA evaluation**:

    sbatch zeroshot.sh

Or directly:

    python zeroshot.py \
        --base_model "huggyllama/llama-7b" \
        --data_path "datasets/sequential/LastFM/" \
        --output_dir "./results" \
        --task_type "sequential"

Results are written to `results/zeroshot_results.txt`.

### Zero-Shot Results

#### Global Temporal Split (current)

| Metric    | @1     | @5     | @10    | @100   |
|-----------|--------|--------|--------|--------|
| Precision | 0.0063 | 0.0047 | 0.0056 | 0.0053 |
| Recall    | 0.0063 | 0.0235 | 0.0564 | 0.5282 |
| MRR       | 0.0063 | 0.0128 | 0.0172 | 0.0286 |
| MAP       | 0.0063 | 0.0128 | 0.0172 | 0.0286 |
| NDCG      | 0.0063 | 0.0155 | 0.0261 | 0.1114 |

#### Leave-Last-Out / Session Split (previous)

| Metric    | @1     | @5     | @10    | @100   |
|-----------|--------|--------|--------|--------|
| Precision | 0.0061 | 0.0047 | 0.0060 | 0.0050 |
| Recall    | 0.0061 | 0.0233 | 0.0602 | 0.4988 |
| MRR       | 0.0061 | 0.0123 | 0.0171 | 0.0283 |
| MAP       | 0.0061 | 0.0123 | 0.0171 | 0.0283 |
| NDCG      | 0.0061 | 0.0150 | 0.0268 | 0.1068 |

### Modified Files

| File | Change |
|------|--------|
| `preprocess_lastfm.py` | Rewritten — global temporal split, outputs `train/val/test/*.txt` |
| `preprocess_lastfm_loo.py` | Original leave-last-out preprocessor (kept for reference) |
| `utils/data_utils.py` | `SequentialDataset` reads from split files instead of `LastFM.txt` |
| `baseline/data_utils.py` | Same; adds `valCandidates`/`testCandidates`, `get_eval_users()` |
| `baseline/main_sequential.py` | `evaluate(subset)` replaces `test()`; val evaluated separately |
| `zeroshot.py` | Reads candidate list from `test_sample.txt`; counts only eligible users |

