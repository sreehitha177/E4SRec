#!/bin/bash
# submit_finetune_all.sh
#
# Submits all 13 finetuning experiments as independent SLURM jobs so they run
# in parallel.  Each job writes to its own output directory derived from the
# modality combo + fusion strategy.
#
# Usage:
#   bash submit_finetune_all.sh          # submit all 13
#   bash submit_finetune_all.sh --dry-run # print commands without submitting

DRY_RUN=0
[ "${1}" = "--dry-run" ] && DRY_RUN=1

AUDIO_NODE_PATH="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/audio_embeddings/node_3"
LYRIC_NODE_PATH="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/lyrics_embeddings/node_7"

submit() {
    local fusion=$1
    local audio=$2
    local lyric=$3

    # Build a short tag used for job name and log filenames.
    local tag="SASRec"
    [ -n "$audio" ] && tag="${tag}_audio"
    [ -n "$lyric" ] && tag="${tag}_lyric"
    [ -z "$audio" ] && [ -z "$lyric" ] && tag="SASRec"
    # Omit strategy suffix for SASRec-only (no fusion applied).
    if [ -n "$audio" ] || [ -n "$lyric" ]; then
        tag="${tag}_${fusion}"
    fi

    local micro=2
    local gres_override=""
    if [ -n "$lyric" ]; then
        # Lyric embeddings are 768-dim; the backward pass needs more than 24GB L4.
        # Request a 48GB GPU instead (overrides #SBATCH --gres in finetune.sh).
        micro=1
        gres_override="--gres=gpu:1 --constraint=vram48"
    fi
    local export_str="ALL,FUSION_STRATEGY=${fusion},AUDIO_NODE_PATH=${audio},LYRIC_NODE_PATH=${lyric},MICRO_BATCH=${micro}"
    local cmd="sbatch \
        --job-name=ft_${tag} \
        --output=logs/ft_${tag}_%j.log \
        --error=logs/ft_${tag}_%j.err \
        $gres_override \
        --export=${export_str} \
        finetune.sh"

    echo "  Submitting: ${tag}"
    if [ "$DRY_RUN" -eq 0 ]; then
        eval "$cmd"
    else
        echo "  [dry-run] $cmd"
    fi
}

echo "=========================================="
echo " Submitting 13 finetuning experiments"
echo "=========================================="

# ── 0. SASRec only (baseline, 1 job) ──────────────────────────────────────────
echo ""
echo "--- SASRec only (baseline) ---"
submit "concat" "" ""

# ── 1. SASRec + Audio (4 jobs) ────────────────────────────────────────────────
echo ""
echo "--- SASRec + Audio ---"
for strategy in concat weighted_sum cross_attention film; do
    submit "$strategy" "$AUDIO_NODE_PATH" ""
done

# ── 2. SASRec + Lyric (4 jobs) ────────────────────────────────────────────────
echo ""
echo "--- SASRec + Lyric ---"
for strategy in concat weighted_sum cross_attention film; do
    submit "$strategy" "" "$LYRIC_NODE_PATH"
done

# ── 3. SASRec + Audio + Lyric (4 jobs) ───────────────────────────────────────
echo ""
echo "--- SASRec + Audio + Lyric ---"
for strategy in concat weighted_sum cross_attention film; do
    submit "$strategy" "$AUDIO_NODE_PATH" "$LYRIC_NODE_PATH"
done

echo ""
echo "Done. Monitor with: squeue -u \$USER"
