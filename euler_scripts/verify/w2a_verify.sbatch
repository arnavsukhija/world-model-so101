#!/usr/bin/env bash
# SLURM job wrapper for verify_w2a_setup.sh.
# Runs the 3-step W2A smoke test (config dryrun, param counts, 5-iter forward+backward)
# on an A100 node. Use this when submitting via sbatch from a login node.
# For interactive verification, run verify_w2a_setup.sh directly.
#
# Usage (from world-model-so101/):
#   DATA_DIR=/cluster/scratch/$USER/so101_push_zarr \
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

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
MODEL_DIR="$REPO_DIR/model"
mkdir -p "$REPO_DIR/logs"

# Activate the pre-built venv (cu126 wheels bundle their own CUDA runtime)
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

DATA_DIR="$DATA_DIR" EXPERIMENT="$EXPERIMENT" \
    bash "$REPO_DIR/euler_scripts/verify/verify_w2a_setup.sh"
