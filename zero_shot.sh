#!/bin/bash
#SBATCH --job-name=E4SRec_L4_ZeroShot
#SBATCH --output=logs/L4-Zero-Shot.log
#SBATCH --error=logs/L4-Zero-Shot.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1            # Requesting A100 for sufficient VRAM
#SBATCH --time=04:00:00              # MoE models can be slower to load
#SBATCH --partition=gpu

# Define the local Llama 4 path
MODEL_PATH="/datasets/ai/llama4/hub/models--meta-llama--Llama-4-Maverick-17B-128E-Instruct/snapshots/c33770d34695b047805ab5b952a20d4d76b5b3dc"

# Execute the python script
/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec/bin/python zeroshot.py \
    --base_model "$MODEL_PATH" \
    --data_path "datasets/sequential/LastFM/" \
    --output_dir "./results" \
    --task_type "sequential"