#!/usr/bin/env bash
# Quick smoke test: 100 training iterations to confirm data loads and loss is finite.
# Use this BEFORE submitting euler_train.sbatch to catch config/data errors cheaply.
#SBATCH --job-name=w2a-so101-verify
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

DATA_DIR="${DATA_DIR:-/cluster/scratch/$USER/so101_push_zarr}"
EXPERIMENT="w2a_so101_push_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$REPO_DIR/model"
mkdir -p "$REPO_DIR/logs"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="/cluster/scratch/$USER/.cache/uv"
export UV_PYTHON_INSTALL_DIR="/cluster/scratch/$USER/.cache/uv-python"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0

cd "$MODEL_DIR"

echo "=== Smoke test: config dryrun ==="
uv run torchrun --nproc_per_node=1 --master_port=12341 \
    -m scripts.train --config=cosmos_predict2/configs/config.py --dryrun \
    -- experiment="$EXPERIMENT" \
    +data_config.dataset.dataset.data_dir="$DATA_DIR" \
    +data_config.dataset.dataset.num_val_episodes=0 \
    trainer.run_validation=false

echo ""
echo "=== Smoke test: 100 training iterations ==="
uv run torchrun --nproc_per_node=1 --master_port=12342 \
    -m scripts.train --config=cosmos_predict2/configs/config.py \
    -- experiment="$EXPERIMENT" \
    +data_config.dataset.dataset.data_dir="$DATA_DIR" \
    +data_config.dataset.dataset.num_val_episodes=0 \
    trainer.max_iter=100 trainer.logging_iter=10 \
    trainer.run_validation=false checkpoint.save_iter=100

echo ""
echo "Check logs/verify_${SLURM_JOB_ID}.out — loss should be finite and roughly decreasing."
