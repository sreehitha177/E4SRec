#!/bin/bash
#SBATCH --job-name=E4SRec_baseline
#SBATCH --partition=gpu-preempt    # Use preempt if 'gpu' is full
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --constraint=a100|l40s|a40          # Requesting A100 for higher VRAM (highly recommended)
#SBATCH --mem=64G                  # 7B models need decent system RAM
#SBATCH --time=08:00:00            # Give it plenty of time
#SBATCH --output=logs/finetune_%j.log

module load conda/latest
conda activate e4srec

# Set paths
DATA_PATH="/home/snarayana_umass_edu/E4SRec-1/datasets/sequential/LastFM/"
OUTPUT_DIR="./results"

# Run the script
/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec/bin/python finetune.py \
  --base_model "huggyllama/llama-7b" \
  --task_type "sequential" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --results_file "$OUTPUT_DIR/finetune_results.txt"
  --batch_size 128 \
  --micro_batch_size 4 \
  --num_epochs 1 \
  --learning_rate 3e-4 \
  --lora_r 16 \
  --cutoff_len 512