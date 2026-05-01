#!/bin/bash
#SBATCH --job-name=e4srec_finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/finetune_%j.log
#SBATCH --error=logs/finetune_%j.err


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

python -c "import bitsandbytes" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "bitsandbytes is not installed in the active conda env"
    exit 1
fi

# ── Sanity check ──────────────────────────────────────────────────────────────
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "GPU(s)     : $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_MODEL="/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DATA_PATH="datasets/sequential/LastFM/"
MAPPING_PATH="datasets/sequential/LastFM/item_id_master_map.csv"

# ── Fusion config ─────────────────────────────────────────────────────────────
# FUSION_STRATEGY: concat | weighted_sum | cross_attention | film
FUSION_STRATEGY="${FUSION_STRATEGY:-concat}"
FUSION_TARGET_DIM="${FUSION_TARGET_DIM:-64}"   # ignored for concat
# Leave empty to skip that modality.
AUDIO_NODE_PATH="${AUDIO_NODE_PATH:-}"
LYRIC_NODE_PATH="${LYRIC_NODE_PATH:-}"

# Derive a modality tag and output dir so every experiment gets its own folder.
MODALITY_TAG="SASRec"
[ -n "$AUDIO_NODE_PATH" ] && MODALITY_TAG="${MODALITY_TAG}_audio"
[ -n "$LYRIC_NODE_PATH" ] && MODALITY_TAG="${MODALITY_TAG}_lyric"

CHECKPOINT_BASE="/project/pi_dagarwal_umass_edu/project_7/snarayana/checkpoints"

if [ -z "$AUDIO_NODE_PATH" ] && [ -z "$LYRIC_NODE_PATH" ]; then
    # SASRec-only: fusion strategy is irrelevant, omit from dir name.
    OUTPUT_DIR="${CHECKPOINT_BASE}/SASRec"
else
    OUTPUT_DIR="${CHECKPOINT_BASE}/${MODALITY_TAG}_${FUSION_STRATEGY}"
fi

DEBUG_RUN="${DEBUG_RUN:-0}"
MAX_STEPS="${MAX_STEPS:--1}"

if [ ! -d "$BASE_MODEL" ]; then
    echo "Base model path does not exist: $BASE_MODEL"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

DEBUG_VAL_SIZE="${DEBUG_VAL_SIZE:-10}"
if [ "$DEBUG_RUN" = "1" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_debug"
    MAX_STEPS="${MAX_STEPS:-2}"
    echo "DEBUG_RUN enabled: output_dir=$OUTPUT_DIR max_steps=$MAX_STEPS val_size=$DEBUG_VAL_SIZE"
fi

mkdir -p "$OUTPUT_DIR"

# ── Resume from latest checkpoint if one exists ───────────────────────────────
CHECKPOINT=$(
    find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
    | sort -t- -k2 -n \
    | tail -1
)
if [ -n "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $CHECKPOINT"
else
    echo "No checkpoint found, starting fresh"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
# L4-24GB budget with 4-bit NF4 + paged_adamw_8bit:
#   ~6GB weights + ~4GB activations + ~2GB optimizer states ≈ 12GB peak
#   micro_batch_size=2, batch_size=64 → gradient_accumulation = 32 steps

python -u finetune.py \
    --base_model "$BASE_MODEL" \
    --task_type "sequential" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --fusion_strategy "$FUSION_STRATEGY" \
    --fusion_target_dim "$FUSION_TARGET_DIM" \
    ${AUDIO_NODE_PATH:+--audio_node_path "$AUDIO_NODE_PATH"} \
    ${LYRIC_NODE_PATH:+--lyric_node_path "$LYRIC_NODE_PATH"} \
    --mapping_path "$MAPPING_PATH" \
    --batch_size 64 \
    --micro_batch_size 2 \
    --num_epochs 5 \
    --max_steps "$MAX_STEPS" \
    --learning_rate 1e-4 \
    --cutoff_len 128 \
    --val_set_size 500 \
    --warmup_steps 0 \
    --warmup_ratio 0.1 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_target_modules '["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
    --load_in_4bit True \
    ${CHECKPOINT:+--resume_from_checkpoint "$CHECKPOINT"}

echo "Training finished with exit code $?"
