#!/bin/bash
#SBATCH --job-name=zero_shot
#SBATCH --output=logs/Zero-Shot.log
#SBATCH --error=logs/Zero-Shot.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|a40          # Requesting A100 for higher VRAM (highly recommended)
#SBATCH --time=02:00:00    # 2 hours should be plenty for 100 users
#SBATCH --partition=gpu    # Change this to your cluster's GPU partition name



export HF_HOME=/work/pi_dagarwal_umass_edu/snarayana_umass_edu/hf_cache

eval "$(conda shell.bash hook)"
conda activate dolby

mkdir -p ./results ./logs

cd /work/pi_dagarwal_umass_edu/project_7/E4SRec

echo "Starting Zero-Shot Evaluation for Llama-7B + SASRec..."

python zeroshot.py \
    --base_model "huggyllama/llama-7b" \
    --data_path "datasets/sequential/LastFM/" \
    --output_dir "./results" \
    --task_type "sequential"

echo "Evaluation complete. Results saved in ./results/"
