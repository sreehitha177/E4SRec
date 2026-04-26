#!/bin/bash
#SBATCH --job-name=ZeroShot_WeightedSum
#SBATCH --output=logs/ZeroShot_WeightedSum_%j.log
#SBATCH --error=logs/ZeroShot_WeightedSum_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=vram48

export HF_HOME=/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec

mkdir -p ./results ./logs

cd /home/snarayana_umass_edu/E4SRec-1

PYTHON=/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec/bin/python

echo "=============================================="
echo " Zero-Shot Weighted Sum Experiments"
echo " Job ID : $SLURM_JOB_ID"
echo " Node   : $SLURMD_NODENAME"
echo " Start  : $(date)"
echo "=============================================="

# --- 1. SASRec + Audio (equal weights) ---
echo ""
echo "[1/3] SASRec + Audio — weighted_sum (equal weights)"
$PYTHON zeroshot_audio_lyric.py \
    --fusion_strategy=weighted_sum \
    --lyric_node_path=""
echo "Done: $(date)"

# --- 2. SASRec + Lyric (equal weights) ---
echo ""
echo "[2/3] SASRec + Lyric — weighted_sum (equal weights)"
$PYTHON zeroshot_audio_lyric.py \
    --fusion_strategy=weighted_sum \
    --audio_node_path=""
echo "Done: $(date)"

# --- 3. SASRec + Audio + Lyric (equal weights) ---
echo ""
echo "[3/3] SASRec + Audio + Lyric — weighted_sum (equal weights)"
$PYTHON zeroshot_audio_lyric.py \
    --fusion_strategy=weighted_sum
echo "Done: $(date)"

echo ""
echo "=============================================="
echo " All weighted_sum experiments complete: $(date)"
echo " Results saved to ./results/"
echo "=============================================="
