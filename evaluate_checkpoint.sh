#!/bin/bash
#SBATCH --job-name=e4srec_eval_qwen7b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram80
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_%j.err

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec

cd /home/snarayana_umass_edu/E4SRec-1
mkdir -p logs results/finetuned_qwen_7b

export HF_HOME=/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "GPU(s)     : $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

BASE_MODEL="/work/pi_dagarwal_umass_edu/project_7/srikar/models/qwen2.5-7b-instruct"
DATA_PATH="datasets/sequential/LastFM/"
CHECKPOINT_DIR="/work/pi_dagarwal_umass_edu/project_7/srikar/E4SRec/trainer_output_qwen_7b/checkpoint-100"
OUTPUT_DIR="./results/finetuned_qwen_7b"

python inference.py \
    --base_model "$BASE_MODEL" \
    --task_type "sequential" \
    --data_path "$DATA_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 32 \
    --micro_batch_size 4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules '["q_proj","k_proj","v_proj","o_proj"]'

echo "Evaluation finished with exit code $?"
