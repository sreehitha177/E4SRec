#!/bin/bash
#SBATCH --job-name=librosa_feat
#SBATCH --output=logs/librosa_features.log
#SBATCH --error=logs/librosa_features.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu

export HF_HOME=/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec

cd /home/snarayana_umass_edu/E4SRec-1
mkdir -p ./logs

echo "Starting librosa feature extraction..."
/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec/bin/python librosa_features.py

echo "Done."
