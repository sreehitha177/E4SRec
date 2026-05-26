#!/bin/bash
# submit_finetune_bert4rec_llama70b.sh
#
# Submits all 26 BERT4Rec finetuning experiments for Llama-3-70B:
#   13 fusion combos × 2 completion modes (none, prompt)
# Each job requests 1x A100 80GB.
#
# Usage:
#   bash submit_finetune_bert4rec_llama70b.sh           # submit all 26
#   bash submit_finetune_bert4rec_llama70b.sh --dry-run

DRY_RUN=0
[ "${1}" = "--dry-run" ] && DRY_RUN=1

BASE_MODEL="/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3-70B/snapshots/c82494877ce7f6d7d317c56ec081328e382c72fe"
HF_CACHE="/datasets/ai/llama3/hub"
CHECKPOINT_BASE="/scratch3/workspace/snarayana_umass_edu-checkpoints/BERT4Rec_Llama-3-70B"
BERT4REC_EMBED="/work/pi_dagarwal_umass_edu/project_7/srikar/E4SRec/datasets/sequential/LastFM/BERT4Rec_item_embed.pkl"
AUDIO_NODE_PATH="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/audio_embeddings/node_3"
LYRIC_NODE_PATH="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/lyrics_embeddings/node_7"
COMPLETION_RATIOS_PATH="datasets/sequential/LastFM/interaction_completion_ratios.pkl"

submit() {
    local fusion=$1
    local audio=$2
    local lyric=$3
    local cr_mode=$4

    local tag="BERT4Rec"
    [ -n "$audio" ] && tag="${tag}_audio"
    [ -n "$lyric" ] && tag="${tag}_lyric"
    [ -n "$audio" ] || [ -n "$lyric" ] && tag="${tag}_${fusion}"
    [ "$cr_mode" != "none" ] && tag="${tag}_completion_${cr_mode}"

    local output_dir="${CHECKPOINT_BASE}/BERT4Rec"
    [ -n "$audio" ] || [ -n "$lyric" ] && output_dir="${CHECKPOINT_BASE}/BERT4Rec_$([ -n "$audio" ] && echo -n "audio")$([ -n "$audio" ] && [ -n "$lyric" ] && echo -n "_")$([ -n "$lyric" ] && echo -n "lyric")_${fusion}"
    [ "$cr_mode" != "none" ] && output_dir="${output_dir}_completion_${cr_mode}"

    local job_script=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=ftB4R70b_${tag}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120G
#SBATCH --time=120:00:00
#SBATCH --qos=long
#SBATCH --output=logs/ftB4R70b_${tag}_%j.log
#SBATCH --error=logs/ftB4R70b_${tag}_%j.err

module load conda/latest
eval "\$(conda shell.bash hook)"
conda activate /work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec
cd /home/snarayana_umass_edu/E4SRec-1
mkdir -p logs results

export HF_HOME=${HF_CACHE}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

python -c "import bitsandbytes" >/dev/null 2>&1 || { echo "bitsandbytes not installed"; exit 1; }

echo "Job ID : \$SLURM_JOB_ID  |  Node: \$(hostname)  |  GPU: \$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CHECKPOINT=\$(find "${output_dir}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -n "\$CHECKPOINT" ] && rm -f "\${CHECKPOINT}/optimizer.pt" "\${CHECKPOINT}/scaler.pt" "\${CHECKPOINT}/training_args.bin"

python -u finetune.py \\
    --base_model        "${BASE_MODEL}" \\
    --task_type         "sequential" \\
    --seq_model         "BERT4Rec" \\
    --seq_embed_path    "${BERT4REC_EMBED}" \\
    --data_path         "datasets/sequential/LastFM/" \\
    --output_dir        "${output_dir}" \\
    --mapping_path      "datasets/sequential/LastFM/item_id_master_map.csv" \\
    --fusion_strategy   "${fusion}" \\
    --fusion_target_dim 64 \\
    --audio_node_path   "${audio}" \\
    --lyric_node_path   "${lyric}" \\
    --completion_ratios_mode "${cr_mode}" \\
    --completion_ratios_path "${COMPLETION_RATIOS_PATH}" \\
    --batch_size        64 \\
    --micro_batch_size  1 \\
    --num_epochs        5 \\
    --learning_rate     1e-4 \\
    --cutoff_len        128 \\
    --val_set_size      500 \\
    --warmup_ratio      0.1 \\
    --lora_r            32 \\
    --lora_alpha        64 \\
    --lora_dropout      0.05 \\
    --lora_target_modules '["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \\
    --load_in_4bit      True \\
    \${CHECKPOINT:+--resume_from_checkpoint "\$CHECKPOINT"}

echo "Training finished with exit code \$?"
EOF
)

    echo "  Submitting: ftB4R70b_${tag}"
    if [ "$DRY_RUN" -eq 0 ]; then
        echo "$job_script" | sbatch
    else
        echo "  [dry-run] would submit: ftB4R70b_${tag}"
    fi
}

run_all_combos() {
    local cr_mode=$1
    echo ""
    echo "── Completion mode: ${cr_mode} ──────────────────────────────────────────"

    submit "concat" "" "" "$cr_mode"
    for strategy in concat weighted_sum cross_attention film; do
        submit "$strategy" "$AUDIO_NODE_PATH" "" "$cr_mode"
        submit "$strategy" "" "$LYRIC_NODE_PATH" "$cr_mode"
        submit "$strategy" "$AUDIO_NODE_PATH" "$LYRIC_NODE_PATH" "$cr_mode"
    done
}

echo "=========================================="
echo " Submitting 26 BERT4Rec finetune experiments (Llama-3-70B, A100)"
echo "=========================================="
mkdir -p logs

for mode in none prompt; do
    run_all_combos "$mode"
done

echo ""
echo "Done. Monitor with: squeue -u \$USER"
