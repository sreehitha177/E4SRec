#!/bin/bash
# submit_eval_llama13b.sh
#
# Submits eval-only jobs for Llama-2-13B experiments that have a saved
# adapter.pth but no metrics.json (or need metrics re-verified).
#
# Usage:
#   bash submit_eval_llama13b.sh           # submit all
#   bash submit_eval_llama13b.sh --dry-run

DRY_RUN=0
[ "${1}" = "--dry-run" ] && DRY_RUN=1

BASE_MODEL="/datasets/ai/llama2/hub/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
HF_CACHE="/datasets/ai/llama2/hub"
CHECKPOINT_BASE="/scratch3/workspace/snarayana_umass_edu-checkpoints/Llama-2-13B"
AUDIO_NODE_PATH="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/audio_embeddings/node_3"
LYRIC_NODE_PATH="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/lyrics_embeddings/node_7"
COMPLETION_RATIOS_PATH="datasets/sequential/LastFM/interaction_completion_ratios.pkl"

submit_eval() {
    local fusion=$1
    local audio=$2
    local lyric=$3
    local cr_mode=$4

    local tag="SASRec"
    [ -n "$audio" ] && tag="${tag}_audio"
    [ -n "$lyric" ] && tag="${tag}_lyric"
    [ -z "$audio" ] && [ -z "$lyric" ] && tag="SASRec"
    [ -n "$audio" ] || [ -n "$lyric" ] && tag="${tag}_${fusion}"
    [ "$cr_mode" != "none" ] && tag="${tag}_completion_${cr_mode}"

    local output_dir="${CHECKPOINT_BASE}/${tag}"
    local adapter_pth="${output_dir}/adapter.pth"

    if [ ! -f "$adapter_pth" ]; then
        echo "  SKIP (no adapter.pth): ${tag}"
        return
    fi

    local micro=2
    [ -n "$lyric" ] && micro=1

    local export_str="ALL,BASE_MODEL=${BASE_MODEL},HF_CACHE=${HF_CACHE},CHECKPOINT_BASE=${CHECKPOINT_BASE},FUSION_STRATEGY=${fusion},AUDIO_NODE_PATH=${audio},LYRIC_NODE_PATH=${lyric},MICRO_BATCH=${micro},COMPLETION_RATIOS_MODE=${cr_mode},COMPLETION_RATIOS_PATH=${COMPLETION_RATIOS_PATH},EVAL_ONLY=1"
    local cmd="sbatch \
        --job-name=eval13b_${tag} \
        --output=logs/eval13b_${tag}_%j.log \
        --error=logs/eval13b_${tag}_%j.err \
        --gres=gpu:1 --constraint=vram48 --mem=64G \
        --time=4:00:00 \
        --export=${export_str} \
        eval.sh"

    echo "  Submitting eval: ${tag}"
    if [ "$DRY_RUN" -eq 0 ]; then
        eval "$cmd"
    else
        echo "  [dry-run] $cmd"
    fi
}

echo "── Eval-only jobs for Llama-2-13B ──────────────────────────────────────────"

# SASRec only
submit_eval "concat" "" "" "none"
submit_eval "concat" "" "" "prompt"

# Audio only
for strategy in concat weighted_sum cross_attention film; do
    submit_eval "$strategy" "$AUDIO_NODE_PATH" "" "none"
    submit_eval "$strategy" "$AUDIO_NODE_PATH" "" "prompt"
done

# Lyric only
for strategy in concat weighted_sum cross_attention film; do
    submit_eval "$strategy" "" "$LYRIC_NODE_PATH" "none"
    submit_eval "$strategy" "" "$LYRIC_NODE_PATH" "prompt"
done

# Audio + Lyric
for strategy in concat weighted_sum cross_attention film; do
    submit_eval "$strategy" "$AUDIO_NODE_PATH" "$LYRIC_NODE_PATH" "none"
    submit_eval "$strategy" "$AUDIO_NODE_PATH" "$LYRIC_NODE_PATH" "prompt"
done
