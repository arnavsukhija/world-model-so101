#!/usr/bin/env bash
#SBATCH --job-name=w2a-so101-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
# Uncomment to request a specific GPU type (A100 has 40 GB VRAM, recommended):
# #SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# ── Config ────────────────────────────────────────────────────────────────────
# zarr dataset directory — must already contain language_embedding in every zarr
# (run euler_setup.sh --data-dir <PATH> first).
DATA_DIR="${DATA_DIR:-/cluster/scratch/$USER/so101_push_zarr}"

# SO-101 experiment: pretrained Cosmos backbone, lr=1e-4, layer-20 x-attn, bsz=1 globally.
# Gradient accumulation of 4 simulates effective batch size 4 — better signal for tiny dataset.
EXPERIMENT="w2a_so101_push_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1"

# Training length. 5000 iterations at ~0.5–1 s/iter ≈ 1–1.5 h on an A100.
# Increase to 20000 for a longer run (use --time=12:00:00).
MAX_ITER=5000
LOG_ITER=50
SAVE_ITER=500
VAL_ITER=500
GRAD_ACCUM=4      # effective batch size = 4

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$REPO_DIR/model"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

# ── Environment ───────────────────────────────────────────────────────────────
echo "=== World2Action SO-101 Training ==="
echo "Job      : $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "DATA_DIR : $DATA_DIR"
echo "Iter     : $MAX_ITER (grad_accum=$GRAD_ACCUM → eff_bsz=$GRAD_ACCUM)"
echo ""

module purge
module load gcc/12.2.0
module load cuda/12.6.0
module load python/3.10.14

export PATH="$HOME/.local/bin:$PATH"

# Required by megatron / NCCL
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0

cd "$MODEL_DIR"

# ── Dryrun: verify config resolves before wasting GPU time ────────────────────
echo "--- Config dryrun ---"
uv run torchrun \
    --nproc_per_node=1 \
    --master_port=12341 \
    -m scripts.train \
    --config=cosmos_predict2/configs/config.py \
    --dryrun \
    -- \
    experiment="$EXPERIMENT" \
    +data_config.dataset.dataset.data_dir="$DATA_DIR"

CONFIG_YAML=$(ls -t ../logs/train_${SLURM_JOB_ID}* 2>/dev/null | head -1 || echo "")
echo "Config OK. Starting training..."
echo ""

# ── Training ──────────────────────────────────────────────────────────────────
# The frozen Cosmos video backbone (2B DiT, bfloat16) extracts video features.
# Only the LoRA adapters on the backbone + the SO-101 action decoder are trained.
# Action decoder: 12-block, 512-ch DiT, 5-joint I/O, ~30M params.
uv run torchrun \
    --nproc_per_node=1 \
    --master_port=12342 \
    -m scripts.train \
    --config=cosmos_predict2/configs/config.py \
    -- \
    experiment="$EXPERIMENT" \
    +data_config.dataset.dataset.data_dir="$DATA_DIR" \
    trainer.max_iter=$MAX_ITER \
    trainer.logging_iter=$LOG_ITER \
    trainer.validation_iter=$VAL_ITER \
    trainer.grad_accum_iter=$GRAD_ACCUM \
    checkpoint.save_iter=$SAVE_ITER

echo ""
echo "=== Training complete ==="
echo ""
echo "Checkpoints are in: $MODEL_DIR/training_runs/<experiment_name>/"
echo ""
echo "To run longer, resubmit with MAX_ITER=20000 and --time=12:00:00."
echo "Training resumes automatically from the latest checkpoint."
