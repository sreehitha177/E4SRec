#!/bin/bash
#SBATCH --job-name=Multimodal_Weight_Sweep
#SBATCH --output=logs/Weight_Sweep_%j.log
#SBATCH --error=logs/Weight_Sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=vram48

export HF_HOME=/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache
module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec

mkdir -p ./results ./logs

cd /home/snarayana_umass_edu/E4SRec-1

PYTHON=/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec/bin/python

echo "=============================================="
echo " Multimodal Fusion Weight Sweep"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $SLURMD_NODENAME"
echo " Start    : $(date)"
echo "=============================================="

$PYTHON sweep_weights.py \
    --python_bin=$PYTHON \
    --eval_script=zeroshot_audio_lyric.py \
    --results_dir=results

echo ""
echo "=============================================="
echo " Sweep complete : $(date)"
echo " Summary saved to results/weight_sweep_summary.csv"
echo "=============================================="