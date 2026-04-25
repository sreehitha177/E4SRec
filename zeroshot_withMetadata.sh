#!/bin/bash
#SBATCH --job-name=zs_metadata
#SBATCH --output=logs/WithMetadata.log
#SBATCH --error=logs/WithMetadata.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --constraint=a40
#SBATCH --partition=gpu-preempt

export HF_HOME=/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache
export PYTHONPATH=/home/snarayana_umass_edu/E4SRec-1

cd /home/snarayana_umass_edu/E4SRec-1
mkdir -p ./results ./logs

echo "Starting Zero-Shot Evaluation with Spotify Metadata..."

/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec/bin/python zeroshot_withMetadata.py \
    --base_model "Qwen/Qwen2.5-7B-Instruct" \
    --data_path "datasets/sequential/LastFM/" \
    --metadata_path "/project/pi_dagarwal_umass_edu/project_7/hmagapu/top_50k_full_augmented.csv" \
    --output_dir "./results" \
    --task_type "sequential"

echo "Evaluation complete. Results saved in ./results/"