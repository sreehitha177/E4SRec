#!/bin/bash
#SBATCH --job-name=zs_completion
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram48
#SBATCH --time=24:00:00
#SBATCH --output=logs/zs_completion_%j.log
#SBATCH --error=logs/zs_completion_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate /work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec
cd /home/snarayana_umass_edu/E4SRec-1
mkdir -p logs results

export HF_HOME=/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "GPU(s)  : $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_MODEL="/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DATA_PATH="datasets/sequential/LastFM/"
MAPPING_PATH="datasets/sequential/LastFM/item_id_master_map.csv"
AUDIO_NODE_PATH="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/audio_embeddings/node_3"
LYRIC_NODE_PATH="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/lyrics_embeddings/node_7"
COMPLETION_RATIOS_PATH="datasets/sequential/LastFM/interaction_completion_ratios.pkl"
OUTPUT_DIR="results"
FUSION_TARGET_DIM=64

# COMPLETION_MODE is set by the submission script via --export.
# Values: "prompt" | "embed"
COMPLETION_MODE="${COMPLETION_MODE:-prompt}"

echo "Completion mode: $COMPLETION_MODE"

if [ ! -f "$COMPLETION_RATIOS_PATH" ]; then
    echo "ERROR: $COMPLETION_RATIOS_PATH not found."
    echo "Run: python compute_completion_ratio.py"
    exit 1
fi

# ── Helper ────────────────────────────────────────────────────────────────────
run() {
    local combo=$1
    local strategy=$2
    local audio=${3:-""}
    local lyric=${4:-""}

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ${combo}  |  strategy=${strategy}  |  completion=${COMPLETION_MODE}"
    echo "════════════════════════════════════════════════════════════════"

    python -u zeroshot_multimodal.py \
        --base_model                "$BASE_MODEL" \
        --data_path                 "$DATA_PATH" \
        --mapping_path              "$MAPPING_PATH" \
        --output_dir                "$OUTPUT_DIR" \
        --fusion_strategy           "$strategy" \
        --fusion_target_dim         "$FUSION_TARGET_DIM" \
        --audio_node_path           "$audio" \
        --lyric_node_path           "$lyric" \
        --completion_ratios_path    "$COMPLETION_RATIOS_PATH" \
        --completion_ratios_mode    "$COMPLETION_MODE"

    echo "  → exit code $?"
}

# ── Experiments (mirrors zeroshot_fusion_all.sh, 13 combinations) ─────────────

echo "################################################################"
echo "  COMBO 0 — SASRec only"
echo "################################################################"
run "SASRec" concat "" ""

echo "################################################################"
echo "  COMBO 1 — SASRec + Audio"
echo "################################################################"
run "SASRec+audio" concat          "$AUDIO_NODE_PATH" ""
run "SASRec+audio" weighted_sum    "$AUDIO_NODE_PATH" ""
run "SASRec+audio" cross_attention "$AUDIO_NODE_PATH" ""
run "SASRec+audio" film            "$AUDIO_NODE_PATH" ""

echo "################################################################"
echo "  COMBO 2 — SASRec + Lyric"
echo "################################################################"
run "SASRec+lyric" concat          "" "$LYRIC_NODE_PATH"
run "SASRec+lyric" weighted_sum    "" "$LYRIC_NODE_PATH"
run "SASRec+lyric" cross_attention "" "$LYRIC_NODE_PATH"
run "SASRec+lyric" film            "" "$LYRIC_NODE_PATH"

echo "################################################################"
echo "  COMBO 3 — SASRec + Audio + Lyric"
echo "################################################################"
run "SASRec+audio+lyric" concat          "$AUDIO_NODE_PATH" "$LYRIC_NODE_PATH"
run "SASRec+audio+lyric" weighted_sum    "$AUDIO_NODE_PATH" "$LYRIC_NODE_PATH"
run "SASRec+audio+lyric" cross_attention "$AUDIO_NODE_PATH" "$LYRIC_NODE_PATH"
run "SASRec+audio+lyric" film            "$AUDIO_NODE_PATH" "$LYRIC_NODE_PATH"

echo ""
echo "All 13 zero-shot experiments complete (completion_mode=${COMPLETION_MODE})."
echo "Results in ./${OUTPUT_DIR}/"
