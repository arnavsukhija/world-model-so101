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
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=a100-pcie-40gb:1

DATA_DIR="${DATA_DIR:-/cluster/scratch/$USER/so101_push_zarr}"
EXPERIMENT="w2a_so101_push_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1"

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
MODEL_DIR="$REPO_DIR/model"
mkdir -p "$REPO_DIR/logs"

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

echo "=== Smoke test: config dryrun ==="
torchrun --nproc_per_node=1 --master_port=12341 \
    -m scripts.train --config=cosmos_predict2/configs/config.py --dryrun \
    -- experiment="$EXPERIMENT" \
    trainer.run_validation=false

echo ""
echo "=== Smoke test: 100 training iterations ==="
torchrun --nproc_per_node=1 --master_port=12342 \
    -m scripts.train --config=cosmos_predict2/configs/config.py \
    -- experiment="$EXPERIMENT" \
    trainer.max_iter=100 trainer.logging_iter=10 \
    trainer.run_validation=false checkpoint.save_iter=100

echo ""
echo "Check logs/verify_${SLURM_JOB_ID}.out — loss should be finite and roughly decreasing."
