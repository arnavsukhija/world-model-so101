#!/usr/bin/env bash
# Precompute T5-XXL language embeddings for the SO-101 action zarr dataset.
# Adds a language_embedding key to every zarr episode. CPU-only, no GPU needed.
# Run this BEFORE euler_scripts/verify/ or euler_scripts/train/w2a_train.sbatch.
#
# Usage (from world-model-so101/):
#   DATA_DIR=/cluster/scratch/$USER/so101_push_zarr sbatch euler_scripts/embeddings/t5_action.sbatch
#SBATCH --job-name=t5-action
#SBATCH --output=logs/t5_action_%j.out
#SBATCH --error=logs/t5_action_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --gpus=rtx_4090:1

DATA_DIR="${DATA_DIR:-/cluster/scratch/$USER/so101_push_zarr}"

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
MODEL_DIR="$REPO_DIR/model"
mkdir -p "$REPO_DIR/logs"

echo "=== T5-XXL Embedding Precompute (action zarr) ==="
echo "Job     : $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "DATA_DIR: $DATA_DIR"
echo ""

source "$MODEL_DIR/.venv/bin/activate"

cd "$MODEL_DIR"

python ../data_preprocessing/action/precompute_t5.py \
    --dataset-path "$DATA_DIR"

echo ""
echo "=== Done. Each zarr now contains language_embedding. ==="
echo "Next: DATA_DIR=$DATA_DIR sbatch $REPO_DIR/euler_scripts/verify/w2a_verify.sbatch"
