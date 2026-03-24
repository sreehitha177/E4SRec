#!/bin/bash
#SBATCH --job-name=sasrec_train
#SBATCH --partition=gpu
#SBATCH --constraint=sm_70
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/sasrec_%j.log
#SBATCH --error=logs/sasrec_%j.err

mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate dolby

cd /work/pi_dagarwal_umass_edu/project_7/E4SRec/baseline

python main_sequential.py \
    --model SASRec \
    --dataset LastFM \
    --save 1 \
    --epochs 300 \
    --eval_freq 10 \
    --dim 64 \
    --maxlen 50 \
    --batch_size 128 \
    --lr 0.001 \
    --cuda 0
