#!/bin/bash
# submit_zeroshot_completion.sh
#
# Submits three SLURM jobs — one per completion-ratio mode:
#   1. none   — no completion ratio (baseline fusion experiments)
#   2. prompt — completion ratios injected as text into the LLM prompt
#   3. embed  — completion ratio concatenated to item embeddings
#
# Results land in results/ with filenames like:
#   zeroshot_SASRec_concat.txt
#   zeroshot_SASRec_audio_lyric_concat_completion_prompt.txt
#   zeroshot_SASRec_audio_lyric_concat_completion_embed.txt
#
# Usage:
#   bash submit_zeroshot_completion.sh            # submit all three
#   bash submit_zeroshot_completion.sh --dry-run  # print without submitting

DRY_RUN=0
[ "${1}" = "--dry-run" ] && DRY_RUN=1

COMPLETION_RATIOS_PATH="datasets/sequential/LastFM/interaction_completion_ratios.pkl"

if [ ! -f "$COMPLETION_RATIOS_PATH" ] && [ "$DRY_RUN" -eq 0 ]; then
    echo "ERROR: $COMPLETION_RATIOS_PATH not found."
    echo "Run first: python compute_completion_ratio.py"
    exit 1
fi

submit() {
    local mode=$1
    local cmd="sbatch \
        --job-name=zs_cr_${mode} \
        --output=logs/zs_completion_${mode}_%j.log \
        --error=logs/zs_completion_${mode}_%j.err \
        --export=ALL,COMPLETION_MODE=${mode} \
        zeroshot_completion.sh"

    echo "  Submitting: completion_mode=${mode}"
    if [ "$DRY_RUN" -eq 0 ]; then
        eval "$cmd"
    else
        echo "  [dry-run] $cmd"
    fi
}

mkdir -p logs

echo "=========================================="
echo " Submitting zeroshot completion experiments"
echo "=========================================="
submit none
submit prompt
submit embed

echo ""
echo "Done. Monitor with: squeue -u \$USER"
