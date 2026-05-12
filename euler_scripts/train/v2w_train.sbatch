#!/usr/bin/env bash
#SBATCH --job-name=v2w-so101-train
#SBATCH --output=logs/v2w_%j.out
#SBATCH --error=logs/v2w_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=6G
# V2W needs more VRAM than W2A: 2B frozen backbone + LoRA activations over 61 frames.
# A100 40GB is sufficient for bsz=1 with grad_accum=32.
#SBATCH --gpus=a100-pcie-40gb:1

# ── Config ────────────────────────────────────────────────────────────────────
# Path to the preprocessed video dataset (video/ metas/ t5_xxl/ layout).
# Build it first with:
#   python data_preprocessing/video/process_so101_video.py \
#       --dataset-root <lerobot_v3_root> --output-dir $VIDEO_DATA_DIR \
#       --use-lerobot-tasks   # or --prompts-json /path/to/prompts.json
#   sbatch euler_scripts/embeddings/t5_video.sbatch
VIDEO_DATA_DIR="${VIDEO_DATA_DIR:-/cluster/scratch/$USER/so101_push_video}"

EXPERIMENT="v2w_so101_push_lora_rank256_lr3.162e-04_bsz32_gradacc32"

MAX_ITER=10000
LOG_ITER=50
SAVE_ITER=1000

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
MODEL_DIR="$REPO_DIR/model"
mkdir -p "$REPO_DIR/logs"

# ── Environment ───────────────────────────────────────────────────────────────
echo "=== Video2World SO-101 Finetuning ==="
echo "Job       : $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "GPUs      : $CUDA_VISIBLE_DEVICES"
echo "VIDEO_DATA: $VIDEO_DATA_DIR"
echo "Experiment: $EXPERIMENT"
echo "Iter      : $MAX_ITER (grad_accum=32 → eff_bsz=32)"
echo ""

source "$MODEL_DIR/.venv/bin/activate"

SITE_PKGS="$MODEL_DIR/.venv/lib/python3.10/site-packages"
NVRTC_LIB="$SITE_PKGS/nvidia/cuda_nvrtc/lib"
export LD_LIBRARY_PATH="${NVRTC_LIB}:${SITE_PKGS}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

REAL_LDCONFIG="$(command -v ldconfig 2>/dev/null || echo /usr/sbin/ldconfig)"
JOB_TMP="/tmp/euler_$SLURM_JOB_ID"
mkdir -p "$JOB_TMP"
cat > "$JOB_TMP/ldconfig" << LDWRAP
#!/bin/bash
${REAL_LDCONFIG} "\$@" 2>/dev/null || true
if [[ "\${1:-}" == "-p" ]]; then
    for lib in ${NVRTC_LIB}/libnvrtc*.so*; do
        [ -e "\$lib" ] && echo "  \$(basename "\$lib") (libc6,x86-64) => \$lib"
    done
fi
LDWRAP
chmod +x "$JOB_TMP/ldconfig"
export PATH="$JOB_TMP:$PATH"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0

cd "$MODEL_DIR"

# ── Dryrun ────────────────────────────────────────────────────────────────────
echo "--- Config dryrun ---"
torchrun --nproc_per_node=1 --master_port=12343 \
    -m scripts.train --config=cosmos_predict2/configs/config.py --dryrun \
    -- experiment="$EXPERIMENT" \
    video_dataset_train.dataset_dir="$VIDEO_DATA_DIR" \
    video_dataset_val.dataset_dir="$VIDEO_DATA_DIR" \
    trainer.run_validation=false

echo "Config OK. Starting V2W training..."
echo ""

# ── Training ──────────────────────────────────────────────────────────────────
torchrun --nproc_per_node=1 --master_port=12344 \
    -m scripts.train --config=cosmos_predict2/configs/config.py \
    -- experiment="$EXPERIMENT" \
    video_dataset_train.dataset_dir="$VIDEO_DATA_DIR" \
    video_dataset_val.dataset_dir="$VIDEO_DATA_DIR" \
    trainer.max_iter=$MAX_ITER \
    trainer.logging_iter=$LOG_ITER \
    trainer.run_validation=false \
    checkpoint.save_iter=$SAVE_ITER

echo ""
echo "=== V2W training complete ==="
echo ""

# ── Extract slim video backbone ────────────────────────────────────────────────
CKPT_DIR="checkpoints/posttraining/so101/$EXPERIMENT/checkpoints/model"
LATEST_FUSED=$(ls -t "$CKPT_DIR"/iter_*_fused.pt 2>/dev/null | head -1)

if [[ -z "$LATEST_FUSED" ]]; then
    # VideoEvalCallback may not have run if save_iter wasn't hit; fuse manually.
    LATEST_RAW=$(ls -t "$CKPT_DIR"/iter_*.pt 2>/dev/null | grep -v fused | head -1)
    if [[ -n "$LATEST_RAW" ]]; then
        echo "--- Fusing LoRA weights from $LATEST_RAW ---"
        python scripts/fuse_lora_ckpt.py "$LATEST_RAW"
        LATEST_FUSED="${LATEST_RAW%.pt}_fused.pt"
    fi
fi

if [[ -n "$LATEST_FUSED" ]]; then
    ITER=$(basename "$LATEST_FUSED" | sed 's/iter_\([0-9]*\)_fused.pt/\1/')
    BACKBONE_NAME="v2w_so101_push_lora_rank256_lr3.162e-04_bsz32_iter_${ITER}_fused"
    echo "--- Extracting video backbone: $BACKBONE_NAME ---"
    python scripts/extract_v2w_backbone.py \
        "$LATEST_FUSED" \
        --output-name "$BACKBONE_NAME"
    echo ""
    echo "Backbone saved to: checkpoints/video_backbone/${BACKBONE_NAME}.pt"
    echo ""
    echo "Next steps:"
    echo "  1. Add '$BACKBONE_NAME' to VIDEO_MODEL_CKPT_NAMES in"
    echo "     cosmos_predict2/configs/defaults/world2action_model.py"
    echo "  2. Retrain W2A: DATA_DIR=<zarr> sbatch euler_scripts/train/w2a_train.sbatch"
    echo "  3. Or to train V2W longer: MAX_ITER=20000 sbatch euler_scripts/train/v2w_train.sbatch"
else
    echo "WARNING: No checkpoint found in $CKPT_DIR — extraction skipped."
fi
