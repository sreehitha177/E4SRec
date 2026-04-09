#!/bin/bash
#SBATCH --job-name=zero_shot
#SBATCH --output=logs/Zero-Shot.log
#SBATCH --error=logs/Zero-Shot.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00    
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=a40



export HF_HOME=/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec

mkdir -p ./results ./logs

cd /home/snarayana_umass_edu/E4SRec-1

echo "Starting Zero-Shot Evaluation for Llama-7B + SASRec..."

# Execute the python script
# We pass parameters directly via Fire
/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec/bin/python zeroshot.py \
    --base_model "Qwen/Qwen2.5-7B-Instruct" \
    --data_path "datasets/sequential/LastFM/" \
    --output_dir "./results" \
    --device_map "cpu"

echo "Evaluation complete. Results saved in ./results/"
