#!/bin/bash
#SBATCH --job-name=zero_shot
#SBATCH --output=logs/WithMetadata.log
#SBATCH --error=logs/WithMetadata.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|a40          # Requesting A100 for higher VRAM (highly recommended)
#SBATCH --time=01:00:00    # 2 hours should be plenty for 100 users
#SBATCH --partition=gpu    # Change this to your cluster's GPU partition name



export HF_HOME=/work/pi_dagarwal_umass_edu/snarayana_umass_edu/hf_cache

# Create results directory if it doesn't exist
mkdir -p ./results

echo "Starting Zero-Shot Evaluation with Metadata..."

# Execute the python script
# We pass parameters directly via Fire
/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec/bin/python zeroshot_withMetadata.py \
    --base_model "huggyllama/llama-7b" \
    --data_path "datasets/sequential/LastFM/" \
    --output_dir "./results" \
    --max_test_users 100 \
    --task_type "sequential"

echo "Evaluation complete. Results saved in ./results/"