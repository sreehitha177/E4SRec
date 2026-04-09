#!/bin/bash
#SBATCH --job-name=zs_librosa
#SBATCH --output=logs/zeroshot_librosa_%j.log
#SBATCH --error=logs/zeroshot_librosa_%j.err
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

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec

cd /home/snarayana_umass_edu/E4SRec-1
mkdir -p ./results ./logs

echo "Starting Zero-Shot Evaluation: SASRec + LIBROSA"

/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec/bin/python zeroshot_withLibrosa.py \
    --base_model "Qwen/Qwen2.5-7B-Instruct" \
    --data_path "datasets/sequential/LastFM/" \
    --librosa_parquet "/scratch3/workspace/skandagatla_umass_edu-dolby/librosa_features/librosa_features.parquet" \
    --top50k_csv "/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/shared/top_50k_songs.csv" \
    --mapping_path "datasets/sequential/LastFM/item_id_master_map.csv" \
    --model_name "LIBROSA" \
    --output_dir "./results" \
    --task_type "sequential"

echo "Done. Results saved to ./results/zeroshot_withLibrosa.txt"
