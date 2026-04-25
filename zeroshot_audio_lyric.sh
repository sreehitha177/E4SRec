#!/bin/bash
#SBATCH --job-name=Multimodal_zero_shot
#SBATCH --output=logs/Multimodal_Zero-Shot_%j.log
#SBATCH --error=logs/Multimodal_Zero-Shot_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00    
#SBATCH --partition=gpu
#SBATCH --constraint=vram48



export HF_HOME=/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec

mkdir -p ./results ./logs

cd /home/snarayana_umass_edu/E4SRec-1

echo "Starting Zero-Shot Evaluation for Qwen-7B + SASRec + Encodec + Multilingual"

# Execute the python script
# We pass parameters directly via Fire
/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec/bin/python zeroshot_audio_lyric.py \
    --fusion_strategy=weighted_sum \
    --fusion_weights="2,1,1"

echo "Evaluation complete. Results saved in ./results/"
