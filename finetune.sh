#!/bin/bash
#SBATCH --job-name=e4srec_finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram80     
#SBATCH --mem=32G               
#SBATCH --time=24:00:00         
#SBATCH --output=logs/finetune_%j.log
#SBATCH --error=logs/finetune_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec             
cd /home/snarayana_umass_edu/E4SRec-1
mkdir -p logs results

export HF_HOME=/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

# ── Sanity check ──────────────────────────────────────────────────────────────
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "GPU(s)     : $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_MODEL="/home/snarayana_umass_edu/E4SRec-1/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DATA_PATH="datasets/sequential/LastFM/"
OUTPUT_DIR="./trainer_output_qwen_7b"

# ── Resume from latest checkpoint if one exists ───────────────────────────────
CHECKPOINT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -n "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $CHECKPOINT"
else
    echo "No checkpoint found, starting fresh"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
# On A100-80GB:
#   micro_batch_size=4, batch_size=64 → gradient_accumulation = 16 steps
#   micro_batch_size=2, batch_size=64 → gradient_accumulation = 32 steps
#   micro_batch_size=4 is safe and faster; drop to 2 if you hit OOM

python finetune.py \
    --base_model "$BASE_MODEL" \
    --task_type "sequential" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 32 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --cutoff_len 256 \
    --val_set_size 0 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '["q_proj","k_proj","v_proj","o_proj"]' \
    ${CHECKPOINT:+--resume_from_checkpoint "$CHECKPOINT"}

echo "Training finished with exit code $?"