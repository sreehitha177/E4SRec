#!/bin/bash
#SBATCH --job-name=zero_shot
#SBATCH --output=logs/WithAudioEmbeddings.log
#SBATCH --error=logs/WithAudioEmbeddings.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|a40         
#SBATCH --time=01:00:00    
#SBATCH --partition=gpu 



export HF_HOME=/work/pi_dagarwal_umass_edu/snarayana_umass_edu/hf_cache

# Create results directory if it doesn't exist
mkdir -p ./results

echo "Starting Zero-Shot Evaluation with Audio Embeddings..."

# Execute the python script
# We pass parameters directly via Fire
/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec/bin/python zeroshot_withAudioEmbeddings.py \
    --base_model "huggyllama/llama-7b" \
    --data_path "datasets/sequential/LastFM/" \
    --output_dir "./results" \
    --task_type "sequential"

echo "Evaluation complete. Results saved in ./results/"
