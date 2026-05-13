#!/bin/bash
#SBATCH --job-name=zs_allllms
#SBATCH --output=logs/zs_allllms_%j.log
#SBATCH --error=logs/zs_allllms_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=vram48

export HF_HOME=/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec

mkdir -p ./results/Qwen2.5-7B-Instruct \
         ./results/Llama-2-13B \
         ./results/Llama-3-70B \
         ./logs

cd /home/snarayana_umass_edu/E4SRec-1

PYTHON=/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec/bin/python

QWEN7B="/datasets/ai/qwen2/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
QWEN7B_CACHE="/datasets/ai/qwen2/hub"

LLAMA13B="/datasets/ai/llama2/hub/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
LLAMA13B_CACHE="/datasets/ai/llama2/hub"

LLAMA70B="/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3-70B/snapshots/c82494877ce7f6d7d317c56ec081328e382c72fe"
LLAMA70B_CACHE="/datasets/ai/llama3/hub"

AUDIO_NODE="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/audio_embeddings/node_3,/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_2/audio_embeddings/node_3"
LYRIC_NODE="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/lyrics_embeddings/node_7,/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_2/lyrics_embeddings/node_8"

METADATA="/project/pi_dagarwal_umass_edu/project_7/hmagapu/metadata/mgphot/output/run-4-top50k/annotations_no_meta_few_shot.csv"
COMPLETION_PKL="datasets/sequential/LastFM/interaction_completion_ratios.pkl"

BERTGRU_EMBED_DIR="/work/pi_dagarwal_umass_edu/project_7/srikar/E4SRec/datasets/sequential/LastFM"

RUN=0
TOTAL=18

run() {
    RUN=$((RUN + 1))
    local label="$1"; shift
    echo ""
    echo "======================================"
    echo " [${RUN}/${TOTAL}] ${label}"
    echo " Start: $(date)"
    echo "======================================"
    "$@"
    echo " Done:  $(date)"
}

# Runs all 6 configs (3 seq-models × only|full) for one LLM
run_block() {
    local llm_label="$1"
    local base_model="$2"
    local cache_dir="$3"
    local output_dir="$4"

    run "${llm_label} | SASRec only" \
        $PYTHON zeroshot_seq_models.py \
            --base_model="$base_model" --cache_dir="$cache_dir" \
            --seq_model=SASRec \
            --audio_node_path="" --lyric_node_path="" --metadata_path="" \
            --completion_ratios_mode=none \
            --output_dir="$output_dir"

    run "${llm_label} | SASRec + audio + lyric + metadata + completion" \
        $PYTHON zeroshot_seq_models.py \
            --base_model="$base_model" --cache_dir="$cache_dir" \
            --seq_model=SASRec \
            --audio_node_path="$AUDIO_NODE" --lyric_node_path="$LYRIC_NODE" \
            --metadata_path="$METADATA" \
            --completion_ratios_path="$COMPLETION_PKL" --completion_ratios_mode=prompt \
            --fusion_strategy=concat \
            --output_dir="$output_dir"

    run "${llm_label} | BERT4Rec only" \
        $PYTHON zeroshot_seq_models.py \
            --base_model="$base_model" --cache_dir="$cache_dir" \
            --seq_model=BERT4Rec \
            --seq_embed_path="${BERTGRU_EMBED_DIR}/BERT4Rec_item_embed.pkl" \
            --audio_node_path="" --lyric_node_path="" --metadata_path="" \
            --completion_ratios_mode=none \
            --output_dir="$output_dir"

    run "${llm_label} | BERT4Rec + audio + lyric + metadata + completion" \
        $PYTHON zeroshot_seq_models.py \
            --base_model="$base_model" --cache_dir="$cache_dir" \
            --seq_model=BERT4Rec \
            --seq_embed_path="${BERTGRU_EMBED_DIR}/BERT4Rec_item_embed.pkl" \
            --audio_node_path="$AUDIO_NODE" --lyric_node_path="$LYRIC_NODE" \
            --metadata_path="$METADATA" \
            --completion_ratios_path="$COMPLETION_PKL" --completion_ratios_mode=prompt \
            --fusion_strategy=concat \
            --output_dir="$output_dir"

    run "${llm_label} | GRU4Rec only" \
        $PYTHON zeroshot_seq_models.py \
            --base_model="$base_model" --cache_dir="$cache_dir" \
            --seq_model=GRU4Rec \
            --seq_embed_path="${BERTGRU_EMBED_DIR}/GRU4Rec_item_embed.pkl" \
            --audio_node_path="" --lyric_node_path="" --metadata_path="" \
            --completion_ratios_mode=none \
            --output_dir="$output_dir"

    run "${llm_label} | GRU4Rec + audio + lyric + metadata + completion" \
        $PYTHON zeroshot_seq_models.py \
            --base_model="$base_model" --cache_dir="$cache_dir" \
            --seq_model=GRU4Rec \
            --seq_embed_path="${BERTGRU_EMBED_DIR}/GRU4Rec_item_embed.pkl" \
            --audio_node_path="$AUDIO_NODE" --lyric_node_path="$LYRIC_NODE" \
            --metadata_path="$METADATA" \
            --completion_ratios_path="$COMPLETION_PKL" --completion_ratios_mode=prompt \
            --fusion_strategy=concat \
            --output_dir="$output_dir"
}

echo "=============================================="
echo " Zero-Shot All-LLM Sweep"
echo " LLMs : Qwen2.5-7B-Instruct, Llama-2-13B, Llama-3-70B"
echo " Seq  : SASRec, BERT4Rec, GRU4Rec"
echo " Total: ${TOTAL} runs"
echo " Job  : $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME"
echo " Start: $(date)"
echo "=============================================="

run_block "Qwen2.5-7B-Instruct" "$QWEN7B"   "$QWEN7B_CACHE"   "results/Qwen2.5-7B-Instruct"
run_block "Llama-2-13B"         "$LLAMA13B" "$LLAMA13B_CACHE" "results/Llama-2-13B"
run_block "Llama-3-70B"         "$LLAMA70B" "$LLAMA70B_CACHE" "results/Llama-3-70B"

echo ""
echo "=============================================="
echo " All ${TOTAL} runs complete: $(date)"
echo " Results in:"
echo "   results/Qwen2.5-7B-Instruct/"
echo "   results/Llama-2-13B/"
echo "   results/Llama-3-70B/"
echo "=============================================="
