#!/bin/bash
#SBATCH --job-name=finetune 
#SBATCH --partition=gpu    
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1              
#SBATCH --constraint=vram48
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/finetune_%j.log
#SBATCH --error=logs/finetune_%j.err

module load conda/latest
conda activate e4srec

cd /home/snarayana_umass_edu/E4SRec-1
mkdir -p logs results

export HF_HOME=/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false


# Set paths
DATA_PATH="datasets/sequential/LastFM/"
OUTPUT_DIR="./trainer_output"

# Run the script
CHECKPOINT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

python finetune.py \
  --base_model "huggyllama/llama-7b" \
  --task_type "sequential" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 128 \
  --micro_batch_size 2 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --lora_r 16 \
  --cutoff_len 256 \
  ${CHECKPOINT:+--resume_from_checkpoint "$CHECKPOINT"}