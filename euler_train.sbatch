#!/usr/bin/env bash
#SBATCH --job-name=w2a-so101-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=6G
# A100 required: 2B Cosmos backbone at bf16 needs ~15 GB VRAM; V100 16 GB is too small.
#SBATCH --gpus=a100-pcie-40gb:1

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/cluster/scratch/$USER/so101_push_zarr}"
EXPERIMENT="w2a_so101_push_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1"

MAX_ITER=5000
LOG_ITER=50
SAVE_ITER=500
GRAD_ACCUM=4

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
MODEL_DIR="$REPO_DIR/model"
mkdir -p "$REPO_DIR/logs"

# ── Environment ───────────────────────────────────────────────────────────────
echo "=== World2Action SO-101 Training ==="
echo "Job      : $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "DATA_DIR : $DATA_DIR"
echo "Iter     : $MAX_ITER (grad_accum=$GRAD_ACCUM → eff_bsz=$GRAD_ACCUM)"
echo ""

# Activate the pre-built venv (cu126 wheels bundle their own CUDA runtime)
source "$MODEL_DIR/.venv/bin/activate"

# transformer_engine calls `ldconfig -p | grep libnvrtc` to find the bundled lib,
# but ldconfig only searches system paths. Inject a wrapper that also reports the
# venv's libnvrtc so the check passes without any system CUDA module.
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
torchrun --nproc_per_node=1 --master_port=12341 \
    -m scripts.train --config=cosmos_predict2/configs/config.py --dryrun \
    -- experiment="$EXPERIMENT" \
    trainer.run_validation=false

echo "Config OK. Starting training..."
echo ""

# ── Training ──────────────────────────────────────────────────────────────────
torchrun --nproc_per_node=1 --master_port=12342 \
    -m scripts.train --config=cosmos_predict2/configs/config.py \
    -- experiment="$EXPERIMENT" \
    trainer.max_iter=$MAX_ITER \
    trainer.logging_iter=$LOG_ITER \
    trainer.run_validation=false \
    trainer.grad_accum_iter=$GRAD_ACCUM \
    checkpoint.save_iter=$SAVE_ITER

echo ""
echo "=== Training complete ==="
echo ""

# ── Extract slim inference artifacts ──────────────────────────────────────────
CKPT_DIR="checkpoints/vam/so101/$EXPERIMENT/checkpoints"
LATEST_CKPT=$(ls -t "$CKPT_DIR"/iter_*.pt 2>/dev/null | head -1)

if [[ -n "$LATEST_CKPT" ]]; then
    echo "--- Extracting inference artifacts from $LATEST_CKPT ---"
    python scripts/extract_checkpoint.py \
        "$LATEST_CKPT" \
        --stats-dir "$DATA_DIR" \
        --output-dir checkpoints/so101_inference
    echo ""
    echo "Download inference files with:"
    echo "  scp -r $SLURMD_NODENAME:$(pwd)/checkpoints/so101_inference/ ."
else
    echo "WARNING: No checkpoint found in $CKPT_DIR — extraction skipped."
fi

echo ""
echo "To run longer, resubmit with MAX_ITER=20000 and --time=12:00:00."
echo "Training resumes automatically from the latest checkpoint."
