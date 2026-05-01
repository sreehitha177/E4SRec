#!/bin/bash
#SBATCH --job-name=ft_concat
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/finetune_concat_%j.log
#SBATCH --error=logs/finetune_concat_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate /work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec
cd /home/snarayana_umass_edu/E4SRec-1
mkdir -p logs results

export HF_HOME=/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

python -c "import bitsandbytes" >/dev/null 2>&1 || { echo "bitsandbytes not installed"; exit 1; }

echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "GPU(s)  : $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_MODEL="/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DATA_PATH="datasets/sequential/LastFM/"
MAPPING_PATH="datasets/sequential/LastFM/item_id_master_map.csv"
AUDIO_NODE_PATH="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/audio_embeddings/node_3"
LYRIC_NODE_PATH="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/lyrics_embeddings/node_7"
OUTPUT_DIR="/project/pi_dagarwal_umass_edu/project_7/snarayana/checkpoints/SASRec_audio_lyric_concat"
MAX_STEPS="${MAX_STEPS:--1}"

[ ! -d "$BASE_MODEL" ] && { echo "Base model not found: $BASE_MODEL"; exit 1; }
mkdir -p "$OUTPUT_DIR"

# concat input_dim = SASRec_dim + audio_dim + lyric_dim — no target dim needed
CHECKPOINT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -n "$CHECKPOINT" ] && echo "Resuming from: $CHECKPOINT" || echo "Starting fresh"

python -u finetune.py \
    --base_model        "$BASE_MODEL" \
    --task_type         "sequential" \
    --data_path         "$DATA_PATH" \
    --output_dir        "$OUTPUT_DIR" \
    --fusion_strategy   "concat" \
    --audio_node_path   "$AUDIO_NODE_PATH" \
    --lyric_node_path   "$LYRIC_NODE_PATH" \
    --mapping_path      "$MAPPING_PATH" \
    --batch_size        64 \
    --micro_batch_size  2 \
    --num_epochs        5 \
    --max_steps         "$MAX_STEPS" \
    --learning_rate     1e-4 \
    --cutoff_len        128 \
    --val_set_size      500 \
    --warmup_steps      0 \
    --warmup_ratio      0.1 \
    --lora_r            32 \
    --lora_alpha        64 \
    --lora_dropout      0.05 \
    --lora_target_modules '["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
    --load_in_4bit      True \
    ${CHECKPOINT:+--resume_from_checkpoint "$CHECKPOINT"}

echo "Training finished with exit code $?"
