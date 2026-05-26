#!/bin/bash
# submit_finetune_llama70b.sh
#
# Submits all 39 finetuning experiments for Llama-3-70B as independent
# SLURM jobs: 13 fusion combos × 3 completion modes (none, prompt, embed).
# All jobs request 2x vram48 GPUs (70B in 4-bit needs ~35GB+).
#
# Usage:
#   bash submit_finetune_llama70b.sh           # submit all 39
#   bash submit_finetune_llama70b.sh --dry-run  # print commands without submitting

DRY_RUN=0
[ "${1}" = "--dry-run" ] && DRY_RUN=1

BASE_MODEL="/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3-70B/snapshots/c82494877ce7f6d7d317c56ec081328e382c72fe"
HF_CACHE="/datasets/ai/llama3/hub"
CHECKPOINT_BASE="/scratch3/workspace/snarayana_umass_edu-checkpoints/Llama-3-70B"
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

    # 70B in 4-bit needs ~35GB; use 2x vram48. Lyric runs use micro_batch=1.
    local micro=2
    [ -n "$lyric" ] && micro=1

    local export_str="ALL,BASE_MODEL=${BASE_MODEL},HF_CACHE=${HF_CACHE},CHECKPOINT_BASE=${CHECKPOINT_BASE},FUSION_STRATEGY=${fusion},AUDIO_NODE_PATH=${audio},LYRIC_NODE_PATH=${lyric},MICRO_BATCH=${micro},COMPLETION_RATIOS_MODE=${cr_mode},COMPLETION_RATIOS_PATH=${COMPLETION_RATIOS_PATH}"
    local cmd="sbatch \
        --job-name=ft70b_${tag} \
        --output=logs/ft70b_${tag}_%j.log \
        --error=logs/ft70b_${tag}_%j.err \
        --gres=gpu:2 --constraint=vram48 --mem=128G \
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
echo " Submitting 39 finetuning experiments (Llama-3-70B)"
echo "=========================================="

for mode in none prompt embed; do
    run_all_combos "$mode"
done

echo ""
echo "Done. Monitor with: squeue -u \$USER"
