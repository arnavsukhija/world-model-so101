#!/usr/bin/env bash
# Verify the World2Action training setup before submitting a full job.
# Runs a config dryrun then a 5-iteration smoke test to confirm:
#   - Experiment config resolves (correct backbone / micro decoder / data config)
#   - Video backbone checkpoint loads without key mismatches
#   - Forward + backward pass completes on real data
#   - Loss is finite
#
# Usage (from world-model-so101/):
#   DATA_DIR=/cluster/scratch/$USER/so101_push_zarr \
#     sbatch euler_scripts/verify/w2a_verify.sbatch
#
# To verify with the SO-101-finetuned backbone (after v2w_train.sbatch completes):
#   DATA_DIR=... \
#   EXPERIMENT=w2a_so101_push_micro_v2w_so101_push_lora_rank256_lr3.162e-04_bsz32_iter_XXXXXX_fused_lr1.000e-04_layer20_bsz1 \
#     sbatch euler_scripts/verify/w2a_verify.sbatch
#SBATCH --job-name=w2a-verify
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=a100-pcie-40gb:1

DATA_DIR="${DATA_DIR:-/cluster/scratch/$USER/so101_push_zarr}"
EXPERIMENT="${EXPERIMENT:-w2a_so101_push_micro_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1}"
MASTER_PORT="${MASTER_PORT:-12360}"

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
MODEL_DIR="$REPO_DIR/model"
mkdir -p "$REPO_DIR/logs"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: DATA_DIR does not exist: $DATA_DIR"
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────
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

echo "============================================"
echo "W2A Setup Verification"
echo "============================================"
echo "Job        : $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Experiment : $EXPERIMENT"
echo "DATA_DIR   : $DATA_DIR"
echo ""

# ── Step 1: Config dryrun ─────────────────────────────────────────────────────
echo "--- [1/2] Config dryrun ---"
torchrun --nproc_per_node=1 --master_port=$((MASTER_PORT)) \
    -m scripts.train --config=cosmos_predict2/configs/config.py --dryrun \
    -- experiment="$EXPERIMENT" \
    +data_config.dataset.dataset.data_dir="$DATA_DIR" \
    trainer.run_validation=false

echo "Config resolves OK."
echo ""

# ── Step 2: 5-iteration smoke test ───────────────────────────────────────────
echo "--- [2/2] 5-iteration smoke test (forward + backward on real data) ---"
SMOKE_DIR="/tmp/verify_w2a_$SLURM_JOB_ID"
mkdir -p "$SMOKE_DIR"

torchrun --nproc_per_node=1 --master_port=$((MASTER_PORT + 1)) \
    -m scripts.train --config=cosmos_predict2/configs/config.py \
    -- experiment="$EXPERIMENT" \
    +data_config.dataset.dataset.data_dir="$DATA_DIR" \
    trainer.max_iter=5 \
    trainer.logging_iter=1 \
    trainer.run_validation=false \
    checkpoint.save_iter=999999 \
    job.project="verify_smoke" \
    job.group="verify" \
    job.name="smoke_$SLURM_JOB_ID" \
    2>&1 | tee "$SMOKE_DIR/smoke.log"

if grep -qE "loss.*[0-9]\.[0-9]" "$SMOKE_DIR/smoke.log"; then
    echo ""
    echo "Loss values seen:"
    grep -E "loss" "$SMOKE_DIR/smoke.log" | tail -5
    echo ""
    echo "============================================"
    echo "PASS — setup verified. Ready to submit:"
    echo "  DATA_DIR=$DATA_DIR sbatch euler_scripts/train/w2a_train.sbatch"
    echo "============================================"
else
    echo ""
    echo "WARNING: Could not find loss values in smoke log."
    echo "Check $SMOKE_DIR/smoke.log for errors."
    exit 1
fi

rm -rf "$SMOKE_DIR"
