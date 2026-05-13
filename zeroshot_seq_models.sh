#!/bin/bash
#SBATCH --job-name=zs_seq_models
#SBATCH --output=logs/zs_seq_models_%j.log
#SBATCH --error=logs/zs_seq_models_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=vram48

export HF_HOME=/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec

mkdir -p ./results ./logs

cd /home/snarayana_umass_edu/E4SRec-1

PYTHON=/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec/bin/python

BASE_MODEL="/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

# Comma-separated: batch_1 (tracks 0-49k) + batch_2 (tracks 50k+) for full coverage
AUDIO_NODE="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/audio_embeddings/node_3,/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_2/audio_embeddings/node_3"
LYRIC_NODE="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/lyrics_embeddings/node_7,/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_2/lyrics_embeddings/node_8"

METADATA="/project/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/mgphot/output/run-4-top50k/annotations_no_meta_few_shot.csv"

COMPLETION_PKL="datasets/sequential/LastFM/interaction_completion_ratios.pkl"

# SASRec embed lives in the default data_path; BERT4Rec and GRU4Rec use srikar's directory
BERTGRU_EMBED_DIR="/work/pi_dagarwal_umass_edu/project_7/srikar/E4SRec/datasets/sequential/LastFM"

RUN=0
run() {
    RUN=$((RUN + 1))
    local label="$1"; shift
    echo ""
    echo "======================================"
    echo " [${RUN}/6] ${label}"
    echo " Start: $(date)"
    echo "======================================"
    "$@"
    echo " Done:  $(date)"
}

# ── 1. SASRec only ─────────────────────────────────────────────────────────
run "SASRec — seq only" \
    $PYTHON zeroshot_seq_models.py \
        --base_model="$BASE_MODEL" \
        --seq_model=SASRec \
        --audio_node_path="" \
        --lyric_node_path="" \
        --metadata_path="" \
        --completion_ratios_mode=none

# ── 2. SASRec + audio + lyric + metadata + completion (prompt) ─────────────
run "SASRec + audio + lyric + metadata + completion" \
    $PYTHON zeroshot_seq_models.py \
        --base_model="$BASE_MODEL" \
        --seq_model=SASRec \
        --audio_node_path="$AUDIO_NODE" \
        --lyric_node_path="$LYRIC_NODE" \
        --metadata_path="$METADATA" \
        --completion_ratios_path="$COMPLETION_PKL" \
        --completion_ratios_mode=prompt \
        --fusion_strategy=concat

# ── 3. BERT4Rec only ────────────────────────────────────────────────────────
run "BERT4Rec — seq only" \
    $PYTHON zeroshot_seq_models.py \
        --base_model="$BASE_MODEL" \
        --seq_model=BERT4Rec \
        --seq_embed_path="${BERTGRU_EMBED_DIR}/BERT4Rec_item_embed.pkl" \
        --audio_node_path="" \
        --lyric_node_path="" \
        --metadata_path="" \
        --completion_ratios_mode=none

# ── 4. BERT4Rec + audio + lyric + metadata + completion (prompt) ───────────
run "BERT4Rec + audio + lyric + metadata + completion" \
    $PYTHON zeroshot_seq_models.py \
        --base_model="$BASE_MODEL" \
        --seq_model=BERT4Rec \
        --seq_embed_path="${BERTGRU_EMBED_DIR}/BERT4Rec_item_embed.pkl" \
        --audio_node_path="$AUDIO_NODE" \
        --lyric_node_path="$LYRIC_NODE" \
        --metadata_path="$METADATA" \
        --completion_ratios_path="$COMPLETION_PKL" \
        --completion_ratios_mode=prompt \
        --fusion_strategy=concat

# ── 5. GRU4Rec only ─────────────────────────────────────────────────────────
run "GRU4Rec — seq only" \
    $PYTHON zeroshot_seq_models.py \
        --base_model="$BASE_MODEL" \
        --seq_model=GRU4Rec \
        --seq_embed_path="${BERTGRU_EMBED_DIR}/GRU4Rec_item_embed.pkl" \
        --audio_node_path="" \
        --lyric_node_path="" \
        --metadata_path="" \
        --completion_ratios_mode=none

# ── 6. GRU4Rec + audio + lyric + metadata + completion (prompt) ────────────
run "GRU4Rec + audio + lyric + metadata + completion" \
    $PYTHON zeroshot_seq_models.py \
        --base_model="$BASE_MODEL" \
        --seq_model=GRU4Rec \
        --seq_embed_path="${BERTGRU_EMBED_DIR}/GRU4Rec_item_embed.pkl" \
        --audio_node_path="$AUDIO_NODE" \
        --lyric_node_path="$LYRIC_NODE" \
        --metadata_path="$METADATA" \
        --completion_ratios_path="$COMPLETION_PKL" \
        --completion_ratios_mode=prompt \
        --fusion_strategy=concat

echo ""
echo "======================================"
echo " All 6 runs complete: $(date)"
echo " Results saved to ./results/"
echo "======================================"
