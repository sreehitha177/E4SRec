#!/bin/bash
#SBATCH --job-name=vocal_annot
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=05:00:00
#SBATCH --array=0-15
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/vocal/logs/vocal_annot_%A_%a.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/vocal/logs/vocal_annot_%A_%a.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail

BASE_DIR=/work/pi_dagarwal_umass_edu/project_7/hmagapu
METADATA_DIR="${BASE_DIR}/metadata"
VOCAL_DIR="${METADATA_DIR}/vocal"

mkdir -p "${VOCAL_DIR}/logs"

eval "$(conda shell.bash hook)"
conda activate dolby

set -a
source "${METADATA_DIR}/.env"
set +a

echo "========================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Array Task   : ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_COUNT}"
echo "Node         : ${SLURMD_NODENAME:-unknown}"
echo "Start        : $(date)"
echo "Python       : $(which python)"
echo "========================================"

SEED_ANNOTATIONS_CSV="${VOCAL_DIR}/output/validation_vocal_annotations.csv"

python "${VOCAL_DIR}/annotate_vocal_features.py" annotate \
  --song-list-path "${METADATA_DIR}/shared/top_50k_songs.csv" \
  --output-csv  "${VOCAL_DIR}/output/vocal_annotations.csv" \
  --seed-annotations-csv "${SEED_ANNOTATIONS_CSV}" \
  --max-workers 10

echo "========================================"
echo "End          : $(date)"
echo "========================================"
