#!/bin/bash
# submit_finetune_qwen7b.sh
#
# Submits all 39 finetuning experiments for Qwen2.5-7B-Instruct as independent
# SLURM jobs: 13 fusion combos × 3 completion modes (none, prompt, embed).
# Baseline/audio runs use L4 (24GB); lyric runs request vram48.
#
# Usage:
#   bash submit_finetune_qwen7b.sh           # submit all 39
#   bash submit_finetune_qwen7b.sh --dry-run  # print commands without submitting

DRY_RUN=0
[ "${1}" = "--dry-run" ] && DRY_RUN=1

BASE_MODEL="/datasets/ai/qwen2/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
HF_CACHE="/datasets/ai/qwen2/hub"
CHECKPOINT_BASE="/scratch3/workspace/snarayana_umass_edu-checkpoints/Qwen2.5-7B"
AUDIO_NODE_PATH="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/audio_embeddings/node_3"
LYRIC_NODE_PATH="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/lyrics_embeddings/node_7"
COMPLETION_RATIOS_PATH="datasets/sequential/LastFM/interaction_completion_ratios.pkl"

submit() {
    local fusion=$1
    local audio=$2
    local lyric=$3
    local cr_mode=$4   # none | prompt | embed

    local tag="SASRec"
    [ -n "$audio" ] && tag="${tag}_audio"
    [ -n "$lyric" ] && tag="${tag}_lyric"
    [ -z "$audio" ] && [ -z "$lyric" ] && tag="SASRec"
    [ -n "$audio" ] || [ -n "$lyric" ] && tag="${tag}_${fusion}"
    [ "$cr_mode" != "none" ] && tag="${tag}_completion_${cr_mode}"

    local micro=2
    local gres_override=""
    if [ -n "$lyric" ]; then
        micro=1
        gres_override="--gres=gpu:1 --constraint=vram48"
    fi

    local export_str="ALL,BASE_MODEL=${BASE_MODEL},HF_CACHE=${HF_CACHE},CHECKPOINT_BASE=${CHECKPOINT_BASE},FUSION_STRATEGY=${fusion},AUDIO_NODE_PATH=${audio},LYRIC_NODE_PATH=${lyric},MICRO_BATCH=${micro},COMPLETION_RATIOS_MODE=${cr_mode},COMPLETION_RATIOS_PATH=${COMPLETION_RATIOS_PATH}"
    local cmd="sbatch \
        --job-name=ft7b_${tag} \
        --output=logs/ft7b_${tag}_%j.log \
        --error=logs/ft7b_${tag}_%j.err \
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

run_all_combos() {
    local cr_mode=$1
    echo ""
    echo "── Completion mode: ${cr_mode} ──────────────────────────────────────────"

    echo "  --- SASRec only ---"
    submit "concat" "" "" "$cr_mode"

    echo "  --- SASRec + Audio ---"
    for strategy in concat weighted_sum cross_attention film; do
        submit "$strategy" "$AUDIO_NODE_PATH" "" "$cr_mode"
    done

    echo "  --- SASRec + Lyric ---"
    for strategy in concat weighted_sum cross_attention film; do
        submit "$strategy" "" "$LYRIC_NODE_PATH" "$cr_mode"
    done

    echo "  --- SASRec + Audio + Lyric ---"
    for strategy in concat weighted_sum cross_attention film; do
        submit "$strategy" "$AUDIO_NODE_PATH" "$LYRIC_NODE_PATH" "$cr_mode"
    done
}

echo "=========================================="
echo " Submitting 39 finetuning experiments (Qwen2.5-7B)"
echo "=========================================="

for mode in none prompt; do
    run_all_combos "$mode"
done

echo ""
echo "Done. Monitor with: squeue -u \$USER"
