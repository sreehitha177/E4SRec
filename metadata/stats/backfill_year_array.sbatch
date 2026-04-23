#!/bin/bash
#SBATCH --job-name=mb_year_backfill
#SBATCH --array=0-3
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/project/pi_dagarwal_umass_edu/project_7/hmagapu/logs/mb_year_%A_%a.out
#SBATCH --error=/project/pi_dagarwal_umass_edu/project_7/hmagapu/logs/mb_year_%A_%a.err

PYTHON=/home/hmagapu_umass_edu/.conda/envs/dolby/bin/python
PROJECT=/project/pi_dagarwal_umass_edu/project_7/hmagapu

echo "=========================================="
echo "Backfill year — shard ${SLURM_ARRAY_TASK_ID} / 4"
echo "Node: $(hostname)  Start: $(date)"
echo "=========================================="

${PYTHON} -u "${PROJECT}/backfill_year.py" \
    --db        "${PROJECT}/musicbrainz_cache.db" \
    --shard-id   ${SLURM_ARRAY_TASK_ID} \
    --num-shards 4

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
