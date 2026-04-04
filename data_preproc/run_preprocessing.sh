#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=preprocess_%j.out
#SBATCH --error=preprocess_%j.err

eval "$(conda shell.bash hook)"
conda activate dolby

cd /work/pi_dagarwal_umass_edu/project_7/hmagapu
python data_preprocessing.py
