#!/bin/bash
#SBATCH --job-name=mb_fetch_v2
#SBATCH --array=0-3
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/project/pi_dagarwal_umass_edu/project_7/hmagapu/logs/mb_fetch_%A_%a.out
#SBATCH --error=/project/pi_dagarwal_umass_edu/project_7/hmagapu/logs/mb_fetch_%A_%a.err

SHARD_ID=${SLURM_ARRAY_TASK_ID}
NUM_SHARDS=4
PYTHON=/home/hmagapu_umass_edu/.conda/envs/dolby/bin/python
SCRIPT=/project/pi_dagarwal_umass_edu/project_7/hmagapu/fetch_musicbrainz.py
PROJECT=/project/pi_dagarwal_umass_edu/project_7/hmagapu

mkdir -p "${PROJECT}/logs"

echo "=========================================="
echo "Job array task: ${SHARD_ID} / ${NUM_SHARDS}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

DB="${PROJECT}/musicbrainz_cache_shard${SHARD_ID}.db"

${PYTHON} -u "${SCRIPT}" \
    --catalog /project/pi_dagarwal_umass_edu/project_7/hmagapu/top_50k_full_augmented.csv \
    --genres  "${PROJECT}/enriched_genres.csv" \
    --db      "${DB}" \
    --shard-id   ${SHARD_ID} \
    --num-shards ${NUM_SHARDS}

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
