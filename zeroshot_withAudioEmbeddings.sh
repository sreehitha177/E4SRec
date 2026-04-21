#!/bin/bash
#SBATCH --job-name=zs_emb
#SBATCH --output=logs/zeroshot_emb_%A_%a.log
#SBATCH --error=logs/zeroshot_emb_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --constraint=a40
#SBATCH --partition=gpu-preempt
#SBATCH --array=0-9

BASE_EMB="/scratch3/workspace/skandagatla_umass_edu-dolby/embeddings/batch_1"

# Map array index → (model_name, node_path)
MODEL_NAMES=( CLAP MERT MUSIC2VEC ENCODEC MFCC MINILM BGEM3 MPNET MULTILINGUAL BERT )
NODE_PATHS=(
    "$BASE_EMB/audio_embeddings/node_0"
    "$BASE_EMB/audio_embeddings/node_1"
    "$BASE_EMB/audio_embeddings/node_2"
    "$BASE_EMB/audio_embeddings/node_3"
    "$BASE_EMB/audio_embeddings/node_4"
    "$BASE_EMB/lyrics_embeddings/node_5"
    "$BASE_EMB/lyrics_embeddings/node_6"
    "$BASE_EMB/lyrics_embeddings/node_7"
    "$BASE_EMB/lyrics_embeddings/node_8"
    "$BASE_EMB/lyrics_embeddings/node_9"
)

MODEL_NAME="${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}"
NODE_PATH="${NODE_PATHS[$SLURM_ARRAY_TASK_ID]}"

export HF_HOME=/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache
export PYTHONPATH=/home/snarayana_umass_edu/E4SRec-1

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec

cd /home/snarayana_umass_edu/E4SRec-1
mkdir -p ./results ./logs

echo "Starting Zero-Shot Evaluation: SASRec + ${MODEL_NAME}"
echo "Node path: ${NODE_PATH}"


/work/pi_dagarwal_umass_edu/project_7/snarayana_umass_edu/.conda/envs/e4srec/bin/python zeroshot_withAudioEmbeddings.py \
    --base_model "/project/pi_dagarwal_umass_edu/project_7/snarayana/hf_cache/hub/models--prithivMLmods--Llama-Song-Stream-3B-Instruct/snapshots/56536079fccb29711c0ab5aff0de9372317cba3b" \
    --data_path "datasets/sequential/LastFM/" \
    --node_path "$NODE_PATH" \
    --model_name "$MODEL_NAME" \
    --output_dir "./results" \
    --task_type "sequential"

echo "Done. Results saved to ./results/zeroshot_${MODEL_NAME}.txt"
