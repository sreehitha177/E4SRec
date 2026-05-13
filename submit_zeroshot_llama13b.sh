#!/bin/bash
# submit_zeroshot_llama13b.sh
#
# Submits zero-shot experiments for Llama-2-13b-hf covering:
#   3 seq models (SASRec, BERT4Rec, GRU4Rec)
#   × 13 fusion combos (baseline + audio×4 + lyric×4 + audio+lyric×4)
#   × 3 completion modes (none, prompt, embed)
#   = 117 independent SLURM jobs
#
# Usage:
#   bash submit_zeroshot_llama13b.sh           # submit all 117
#   bash submit_zeroshot_llama13b.sh --dry-run  # print without submitting

DRY_RUN=0
[ "${1}" = "--dry-run" ] && DRY_RUN=1

BASE_MODEL="/datasets/ai/llama2/hub/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
CACHE_DIR="/datasets/ai/llama2/hub"
AUDIO_NODE="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/audio_embeddings/node_3"
LYRIC_NODE="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1/lyrics_embeddings/node_7"
COMPLETION_PKL="datasets/sequential/LastFM/interaction_completion_ratios.pkl"
PYTHON=/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec/bin/python

submit() {
    local seq_model=$1
    local fusion=$2
    local audio=$3
    local lyric=$4
    local cr_mode=$5

    local tag="${seq_model}"
    [ -n "$audio" ] && tag="${tag}_audio"
    [ -n "$lyric" ] && tag="${tag}_lyric"
    [ -n "$audio" ] || [ -n "$lyric" ] && tag="${tag}_${fusion}"
    [ "$cr_mode" != "none" ] && tag="${tag}_completion_${cr_mode}"

    local job_script=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=zs13b_${tag}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram48
#SBATCH --time=12:00:00
#SBATCH --output=logs/zs13b_${tag}_%j.log
#SBATCH --error=logs/zs13b_${tag}_%j.err

module load conda/latest
eval "\$(conda shell.bash hook)"
conda activate /work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec
cd /home/snarayana_umass_edu/E4SRec-1
mkdir -p logs results

export HF_HOME=${CACHE_DIR}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "Job: zs13b_${tag}  |  Node: \$(hostname)  |  GPU: \$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PYTHON zeroshot_seq_models.py \\
    --base_model="${BASE_MODEL}" \\
    --cache_dir="${CACHE_DIR}" \\
    --seq_model="${seq_model}" \\
    --audio_node_path="${audio}" \\
    --lyric_node_path="${lyric}" \\
    --metadata_path="" \\
    --fusion_strategy="${fusion}" \\
    --completion_ratios_path="${COMPLETION_PKL}" \\
    --completion_ratios_mode="${cr_mode}"
EOF
)

    echo "  Submitting: zs13b_${tag}"
    if [ "$DRY_RUN" -eq 0 ]; then
        echo "$job_script" | sbatch
    else
        echo "  [dry-run] would submit: zs13b_${tag}"
    fi
}

run_all_combos() {
    local seq_model=$1
    local cr_mode=$2
    echo ""
    echo "  ── ${seq_model} | completion=${cr_mode} ────────────────────────────"

    submit "$seq_model" "concat"          ""           ""            "$cr_mode"
    for strategy in concat weighted_sum cross_attention film; do
        submit "$seq_model" "$strategy"   "$AUDIO_NODE" ""           "$cr_mode"
        submit "$seq_model" "$strategy"   ""            "$LYRIC_NODE" "$cr_mode"
        submit "$seq_model" "$strategy"   "$AUDIO_NODE" "$LYRIC_NODE" "$cr_mode"
    done
}

echo "============================================================"
echo " Submitting 117 zero-shot experiments (Llama-2-13B)"
echo "============================================================"

mkdir -p logs

for seq_model in SASRec BERT4Rec GRU4Rec; do
    echo ""
    echo "══ Seq model: ${seq_model} ══════════════════════════════════════════"
    for cr_mode in none prompt embed; do
        run_all_combos "$seq_model" "$cr_mode"
    done
done

echo ""
echo "Done. Monitor with: squeue -u \$USER"
