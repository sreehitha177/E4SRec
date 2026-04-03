#!/bin/bash
#SBATCH --job-name=zero_shot
#SBATCH --output=logs/WithMetadata.log
#SBATCH --error=logs/WithMetadata.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00    
#SBATCH --partition=gpu-preempt    



export HF_HOME=/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache

mkdir -p ./results

echo "Starting Zero-Shot Evaluation with Metadata..."


/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec/bin/python zeroshot_withMetadata.py \
    --base_model "huggyllama/llama-7b" \
    --data_path "datasets/sequential/LastFM/" \
    --output_dir "./results" \
    --max_test_users 100 \
    --task_type "sequential"

echo "Evaluation complete. Results saved in ./results/"