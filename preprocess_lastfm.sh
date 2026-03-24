#!/bin/bash
#SBATCH --job-name=lastfm_prep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prep_%j.log
#SBATCH --error=logs/prep_%j.err

mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate dolby

cd /work/pi_dagarwal_umass_edu/project_7/E4SRec
python preprocess_lastfm.py
