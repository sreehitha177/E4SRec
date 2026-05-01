#!/bin/bash
#SBATCH --job-name=ZeroShot_AllFusion_Sweep
#SBATCH --output=logs/ZeroShot_AllFusion_Sweep_%j.log
#SBATCH --error=logs/ZeroShot_AllFusion_Sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=vram48

export HF_HOME=/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec

mkdir -p ./results ./logs

cd /home/snarayana_umass_edu/E4SRec-1

PYTHON=/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec/bin/python

# BERT4Rec and GRU4Rec pkl files live in srikar's dataset directory;
# SASRec pkl uses the default data_path so no --seq_embed_dir needed.
BERTGRU_EMBED_DIR=/work/pi_dagarwal_umass_edu/project_7/srikar/E4SRec/datasets/sequential/LastFM

echo "=============================================="
echo " Zero-Shot All-Fusion Sweep"
echo " Models    : SASRec, BERT4Rec, GRU4Rec"
echo " Strategies: concat, weighted_sum, cross_attention, film"
echo " Modalities: seq-only | audio-only | lyric-only | audio+lyric"
echo " Total runs: 39"
echo " Job ID    : $SLURM_JOB_ID"
echo " Node      : $SLURMD_NODENAME"
echo " Start     : $(date)"
echo "=============================================="

RUN=0

run() {
    RUN=$((RUN + 1))
    echo ""
    echo "[${RUN}/39] $*"
    shift
    "$@"
    echo "Done: $(date)"
}

# ──────────────────────────────────────────────
# SASRec
# ──────────────────────────────────────────────

run "SASRec — seq only" \
    $PYTHON zeroshot_audio_lyric.py \
        --seq_model=SASRec \
        --audio_node_path="" \
        --lyric_node_path=""

for STRATEGY in concat weighted_sum cross_attention film; do
# for STRATEGY in concat weighted_sum; do


    run "SASRec + Audio only — ${STRATEGY}" \
        $PYTHON zeroshot_audio_lyric.py \
            --seq_model=SASRec \
            --fusion_strategy=${STRATEGY} \
            --lyric_node_path=""

    run "SASRec + Lyric only — ${STRATEGY}" \
        $PYTHON zeroshot_audio_lyric.py \
            --seq_model=SASRec \
            --fusion_strategy=${STRATEGY} \
            --audio_node_path=""

    run "SASRec + Audio + Lyric — ${STRATEGY}" \
        $PYTHON zeroshot_audio_lyric.py \
            --seq_model=SASRec \
            --fusion_strategy=${STRATEGY}

done

# ──────────────────────────────────────────────
# BERT4Rec
# ──────────────────────────────────────────────

run "BERT4Rec — seq only" \
    $PYTHON zeroshot_audio_lyric.py \
        --seq_model=BERT4Rec \
        --seq_embed_dir=${BERTGRU_EMBED_DIR} \
        --audio_node_path="" \
        --lyric_node_path=""

for STRATEGY in concat weighted_sum cross_attention film; do

    run "BERT4Rec + Audio only — ${STRATEGY}" \
        $PYTHON zeroshot_audio_lyric.py \
            --seq_model=BERT4Rec \
            --seq_embed_dir=${BERTGRU_EMBED_DIR} \
            --fusion_strategy=${STRATEGY} \
            --lyric_node_path=""

    run "BERT4Rec + Lyric only — ${STRATEGY}" \
        $PYTHON zeroshot_audio_lyric.py \
            --seq_model=BERT4Rec \
            --seq_embed_dir=${BERTGRU_EMBED_DIR} \
            --fusion_strategy=${STRATEGY} \
            --audio_node_path=""

    run "BERT4Rec + Audio + Lyric — ${STRATEGY}" \
        $PYTHON zeroshot_audio_lyric.py \
            --seq_model=BERT4Rec \
            --seq_embed_dir=${BERTGRU_EMBED_DIR} \
            --fusion_strategy=${STRATEGY}

done

# ──────────────────────────────────────────────
# GRU4Rec
# ──────────────────────────────────────────────

run "GRU4Rec — seq only" \
    $PYTHON zeroshot_audio_lyric.py \
        --seq_model=GRU4Rec \
        --seq_embed_dir=${BERTGRU_EMBED_DIR} \
        --audio_node_path="" \
        --lyric_node_path=""

for STRATEGY in concat weighted_sum cross_attention film; do

    run "GRU4Rec + Audio only — ${STRATEGY}" \
        $PYTHON zeroshot_audio_lyric.py \
            --seq_model=GRU4Rec \
            --seq_embed_dir=${BERTGRU_EMBED_DIR} \
            --fusion_strategy=${STRATEGY} \
            --lyric_node_path=""

    run "GRU4Rec + Lyric only — ${STRATEGY}" \
        $PYTHON zeroshot_audio_lyric.py \
            --seq_model=GRU4Rec \
            --seq_embed_dir=${BERTGRU_EMBED_DIR} \
            --fusion_strategy=${STRATEGY} \
            --audio_node_path=""

    run "GRU4Rec + Audio + Lyric — ${STRATEGY}" \
        $PYTHON zeroshot_audio_lyric.py \
            --seq_model=GRU4Rec \
            --seq_embed_dir=${BERTGRU_EMBED_DIR} \
            --fusion_strategy=${STRATEGY}

done

echo ""
echo "=============================================="
echo " All 39 runs complete: $(date)"
echo " Results saved to ./results/"
echo "=============================================="
