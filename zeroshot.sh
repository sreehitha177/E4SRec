#!/bin/bash
#SBATCH --job-name=zero_shot
#SBATCH --output=logs/Zero-Shot.log
#SBATCH --error=logs/Zero-Shot.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00    
#SBATCH --partition=gpu-preempt    



export HF_HOME=/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache

eval "$(conda shell.bash hook)"
conda activate dolby

mkdir -p ./results ./logs

cd /work/pi_dagarwal_umass_edu/project_7/E4SRec

echo "Starting Zero-Shot Evaluation for Llama-7B + SASRec..."

# Execute the python script
# We pass parameters directly via Fire
/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec/bin/python zeroshot.py \
    --base_model "Qwen/Qwen2.5-7B-Instruct" \
    --data_path "datasets/sequential/LastFM/" \
    --output_dir "./results" \
    --task_type "sequential"

echo "Evaluation complete. Results saved in ./results/"
