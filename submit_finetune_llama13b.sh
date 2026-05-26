#!/bin/bash
# submit_finetune_llama13b.sh
#
# Submits all 39 finetuning experiments for Llama-2-13b-hf as independent
# SLURM jobs: 13 fusion combos × 3 completion modes (none, prompt, embed).
# All jobs request a 48GB GPU (13B needs more VRAM than L4).
#
# Usage:
#   bash submit_finetune_llama13b.sh           # submit all 39
#   bash submit_finetune_llama13b.sh --dry-run  # print commands without submitting

DRY_RUN=0
[ "${1}" = "--dry-run" ] && DRY_RUN=1

BASE_MODEL="/datasets/ai/llama2/hub/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
HF_CACHE="/datasets/ai/llama2/hub"
CHECKPOINT_BASE="/scratch3/workspace/snarayana_umass_edu-checkpoints/Llama-2-13B"
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

    # 13B needs vram48 for all runs; lyric needs smaller micro_batch.
    local micro=2
    [ -n "$lyric" ] && micro=1

    local export_str="ALL,BASE_MODEL=${BASE_MODEL},HF_CACHE=${HF_CACHE},CHECKPOINT_BASE=${CHECKPOINT_BASE},FUSION_STRATEGY=${fusion},AUDIO_NODE_PATH=${audio},LYRIC_NODE_PATH=${lyric},MICRO_BATCH=${micro},COMPLETION_RATIOS_MODE=${cr_mode},COMPLETION_RATIOS_PATH=${COMPLETION_RATIOS_PATH}"
    local cmd="sbatch \
        --job-name=ft13b_${tag} \
        --output=logs/ft13b_${tag}_%j.log \
        --error=logs/ft13b_${tag}_%j.err \
        --gres=gpu:1 --constraint=vram48 --mem=64G \
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
echo " Submitting 39 finetuning experiments (Llama-2-13B)"
echo "=========================================="

for mode in none prompt; do
    run_all_combos "$mode"
done

echo ""
echo "Done. Monitor with: squeue -u \$USER"
