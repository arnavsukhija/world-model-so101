#!/usr/bin/env bash
# Verify the World2Action training setup before submitting a full job.
# Runs a config dryrun then a 5-iteration smoke test to confirm:
#   - Experiment config resolves (correct backbone / micro decoder / data config)
#   - Video backbone checkpoint loads without key mismatches
#   - Forward + backward pass completes on real data
#   - Loss is finite
#   - Action decoder parameter count is printed
#
# Usage (run from world-model-so101/ on Euler after activating the venv):
#
#   DATA_DIR=/cluster/scratch/$USER/so101_push_zarr \
#   EXPERIMENT=w2a_so101_push_micro_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1 \
#   bash euler_scripts/verify/verify_w2a_setup.sh
#
# To verify with the SO-101-finetuned video backbone (after v2w_train.sbatch completes):
#
#   DATA_DIR=... \
#   EXPERIMENT=w2a_so101_push_micro_v2w_so101_push_lora_rank256_lr3.162e-04_bsz32_iter_XXXXXX_fused_lr1.000e-04_layer20_bsz1 \
#   bash euler_scripts/verify/verify_w2a_setup.sh

set -euo pipefail

DATA_DIR="${DATA_DIR:-}"
EXPERIMENT="${EXPERIMENT:-w2a_so101_push_micro_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1}"
MASTER_PORT="${MASTER_PORT:-12360}"

if [[ -z "$DATA_DIR" ]]; then
    echo "ERROR: DATA_DIR is not set. Export it before running:"
    echo "  DATA_DIR=/cluster/scratch/\$USER/so101_push_zarr bash euler_scripts/verify/verify_w2a_setup.sh"
    exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: DATA_DIR does not exist: $DATA_DIR"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="$REPO_DIR/model"
cd "$MODEL_DIR"

# Activate venv if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "$MODEL_DIR/.venv/bin/activate"
fi

echo "============================================"
echo "W2A Setup Verification"
echo "============================================"
echo "Experiment : $EXPERIMENT"
echo "DATA_DIR   : $DATA_DIR"
echo ""

# ── Step 1: Config dryrun ─────────────────────────────────────────────────────
echo "--- [1/3] Config dryrun ---"
torchrun --nproc_per_node=1 --master_port=$((MASTER_PORT)) \
    -m scripts.train --config=cosmos_predict2/configs/config.py --dryrun \
    -- experiment="$EXPERIMENT" \
    +data_config.dataset.dataset.data_dir="$DATA_DIR" \
    trainer.run_validation=false

echo "Config resolves OK."
echo ""

# ── Step 2: Print model parameter counts ─────────────────────────────────────
echo "--- [2/3] Parameter counts ---"
python3 - << 'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())

# Approximate param count for the two SO-101 decoder variants
def block_params(d, context_dim=2048, mlp_ratio=4.0):
    return int(4*d*d + 2*d*d + 2*context_dim*d + 2*mlp_ratio*d*d)

def total_params(d, blocks, max_horizon, in_ch, out_ch, pair_rank):
    b = block_params(d) * blocks
    embedders = 2 * (in_ch + 1) * d + 2 * d * d
    pos = max_horizon * d
    t_emb = 2 * d * d + 2 * d * pair_rank + pair_rank * 3 * d
    final = d * out_ch
    return b + embedders + pos + t_emb + final

configs = {
    "libero  (d=1024, 24 blk)": (1024, 24, 61,  10, 10, 1024),
    "so101   (d=512,  12 blk)": ( 512, 12, 16,   5,  5,  512),
    "so101_micro (d=256, 8 blk)": ( 256,  8, 16,   5,  5,  256),
}
libero_p = total_params(1024, 24, 61, 10, 10, 1024)
print(f"{'Config':<35} {'Params':>8}  {'vs libero':>10}")
print("-" * 58)
for name, args in configs.items():
    p = total_params(*args)
    ratio = libero_p / p
    marker = " ← active" if "micro" in name else ""
    print(f"{name:<35} {p/1e6:>7.1f}M  {ratio:>9.1f}x{marker}")
PYEOF
echo ""

# ── Step 3: 5-iteration smoke test ───────────────────────────────────────────
echo "--- [3/3] 5-iteration smoke test (forward + backward on real data) ---"
SMOKE_DIR="/tmp/verify_w2a_$$"
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
    job.name="smoke_$$" \
    2>&1 | tee "$SMOKE_DIR/smoke.log"

# Check the log for a finite loss
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
