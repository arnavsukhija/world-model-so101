#!/usr/bin/env bash
# Train the SO-101 World2Action decoder on pushing data.
# Run from world-model-so101/model/ after:
#   1. Processing data with data_preprocessing/action/process_so101.py
#   2. Running precompute_t5.py to add language_embedding
#   3. Setting DATA_DIR below

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/data/so101_push_zarr}"    # path to processed zarr directory
EXPERIMENT="${EXPERIMENT:-w2a_so101_push_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1}"
NPROC="${NPROC:-1}"                              # GPUs (set to 4 on A100 quad node)
MASTER_PORT="${MASTER_PORT:-12341}"
LOG_DIR="${LOG_DIR:-./logs}"

mkdir -p "$LOG_DIR"

# ─── Patch data_dir into the config ──────────────────────────────────────────
# The so101_push.yaml has data_dir=??? which must be overridden at launch.
# We pass it via +dataset.dataset.data_dir= override.

cd "$(dirname "$0")/model"

echo "Starting World2Action training for SO-101 pushing …"
echo "  data_dir  : $DATA_DIR"
echo "  experiment: $EXPERIMENT"
echo "  GPUs      : $NPROC"

TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
NVTE_FUSED_ATTN=0 \
torchrun \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  -m scripts.train \
  --config=cosmos_predict2/configs/config.py \
  -- \
  experiment="$EXPERIMENT" \
  +data_config.dataset.dataset.data_dir="$DATA_DIR" \
  trainer.logging_iter=50 \
  trainer.max_iter=20000 \
  checkpoint.save_iter=1000 \
  2>&1 | tee "$LOG_DIR/train_so101.log"
