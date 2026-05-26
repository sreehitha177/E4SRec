#!/bin/bash
#SBATCH --job-name=e4srec_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_%j.err

module load conda/latest
eval "$(conda shell.bash hook)"
conda activate /work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec
cd /home/snarayana_umass_edu/E4SRec-1
mkdir -p logs

export HF_HOME="${HF_CACHE:-/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

BASE_MODEL="${BASE_MODEL:-/datasets/ai/llama2/hub/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1}"
DATA_PATH="datasets/sequential/LastFM/"
MAPPING_PATH="datasets/sequential/LastFM/item_id_master_map.csv"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/scratch3/workspace/snarayana_umass_edu-checkpoints/Llama-2-13B}"
FUSION_STRATEGY="${FUSION_STRATEGY:-concat}"
FUSION_TARGET_DIM="${FUSION_TARGET_DIM:-64}"
AUDIO_NODE_PATH="${AUDIO_NODE_PATH:-}"
LYRIC_NODE_PATH="${LYRIC_NODE_PATH:-}"
COMPLETION_RATIOS_PATH="${COMPLETION_RATIOS_PATH:-datasets/sequential/LastFM/interaction_completion_ratios.pkl}"
COMPLETION_RATIOS_MODE="${COMPLETION_RATIOS_MODE:-none}"

MODALITY_TAG="SASRec"
[ -n "$AUDIO_NODE_PATH" ] && MODALITY_TAG="${MODALITY_TAG}_audio"
[ -n "$LYRIC_NODE_PATH" ] && MODALITY_TAG="${MODALITY_TAG}_lyric"

if [ -z "$AUDIO_NODE_PATH" ] && [ -z "$LYRIC_NODE_PATH" ]; then
    OUTPUT_DIR="${CHECKPOINT_BASE}/SASRec"
else
    OUTPUT_DIR="${CHECKPOINT_BASE}/${MODALITY_TAG}_${FUSION_STRATEGY}"
fi
[ "$COMPLETION_RATIOS_MODE" != "none" ] && OUTPUT_DIR="${OUTPUT_DIR}_completion_${COMPLETION_RATIOS_MODE}"

echo "Evaluating checkpoint: $OUTPUT_DIR"

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
    --micro_batch_size "${MICRO_BATCH:-2}" \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_target_modules '["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
    --load_in_4bit True \
    --completion_ratios_mode "$COMPLETION_RATIOS_MODE" \
    ${COMPLETION_RATIOS_MODE:+--completion_ratios_path "$COMPLETION_RATIOS_PATH"} \
    --eval_only True

echo "Eval finished with exit code $?"
