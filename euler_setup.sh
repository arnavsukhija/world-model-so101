#!/usr/bin/env bash
# Run this ONCE on an Euler LOGIN node before submitting training jobs.
# Sets up the Python env, downloads checkpoints, and precomputes T5 embeddings.
#
# Usage:
#   bash euler_setup.sh --data-dir /cluster/scratch/$USER/so101_push_zarr
#
# Options:
#   --data-dir PATH      Path to zarr dataset directory (required for T5 precompute)
#   --skip-checkpoints   Skip HuggingFace download (if already done)
#   --skip-t5            Skip T5 embedding precomputation

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$REPO_DIR/model"
DATA_DIR=""
SKIP_CKPTS=0
SKIP_T5=0

for arg in "$@"; do
  case $arg in
    --data-dir=*) DATA_DIR="${arg#*=}" ;;
    --data-dir)   shift; DATA_DIR="$1" ;;
    --skip-checkpoints) SKIP_CKPTS=1 ;;
    --skip-t5) SKIP_T5=1 ;;
  esac
  shift 2>/dev/null || true
done

echo "=== Euler setup for world-model-so101 ==="
echo "REPO_DIR : $REPO_DIR"
echo "MODEL_DIR: $MODEL_DIR"
echo "DATA_DIR : ${DATA_DIR:-'(not provided — T5 precompute will be skipped)'}"
echo ""

# ── 1. Modules ─────────────────────────────────────────────────────────────
# Verify available versions with: module avail gcc python cuda
# CUDA MUST be 12.6 — the prebuilt wheels are cu126-specific.
echo "[1/4] Loading modules..."
module purge
module load gcc/12.2.0
module load cuda/12.6.0
module load python/3.10.14

python --version
nvcc --version | head -1
echo ""

# ── 2. Install uv ──────────────────────────────────────────────────────────
echo "[2/4] Installing uv (user-local, no sudo needed)..."
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
echo "uv: $(uv --version)"
echo ""

# ── 3. Python environment ──────────────────────────────────────────────────
echo "[3/4] Installing Python dependencies..."
echo "      (apex, flash-attn, natten, transformer-engine come as prebuilt cu126 wheels"
echo "       from the NVIDIA cosmos index — no compilation needed)"
cd "$MODEL_DIR"
uv sync --extra cu126
echo ""
echo "Verifying critical packages..."
uv run python scripts/test_environment.py --training
echo ""

# ── 4a. Download checkpoints ───────────────────────────────────────────────
if [[ $SKIP_CKPTS -eq 0 ]]; then
    echo "[4/4a] Downloading checkpoints from jonpai/mimic-video..."
    echo "       Downloads: text_encoder (~5 GB), video_backbone/tokenizer (~1 GB),"
    echo "       video_backbone/v2w_pretrained_cosmos.pt (~7 GB)"
    echo "       Total: ~13 GB. Target: $MODEL_DIR/checkpoints/"
    echo ""

    if [[ -n "${HF_TOKEN:-}" ]]; then
        uv run python -c "import huggingface_hub; huggingface_hub.login(token='$HF_TOKEN')"
    fi

    # Only the pretrained video backbone — no need to download LIBERO/Bridge decoders.
    uv run python scripts/download_checkpoints.py \
        --models pretrained_cosmos_bridge \
        --checkpoint-dir checkpoints
else
    echo "[4/4a] Skipping checkpoint download (--skip-checkpoints)."
fi

# ── 4b. Precompute T5 language embeddings ─────────────────────────────────
# The T5-XXL encoder (~40 GB on disk) is loaded into CPU memory (~20 GB).
# All 15 SO-101 episodes share the same prompt, so this runs quickly.
# Result: adds language_embedding [1, 512, 1024] float16 to every zarr episode.
if [[ $SKIP_T5 -eq 0 ]]; then
    if [[ -z "$DATA_DIR" ]]; then
        echo "[4/4b] WARNING: --data-dir not provided, skipping T5 precompute."
        echo "       Run manually: cd model && uv run python ../data_preprocessing/action/precompute_t5.py \\"
        echo "         --dataset-path /path/to/so101_push_zarr \\"
        echo "         --prompt 'push the cube from left to right'"
    else
        echo "[4/4b] Precomputing T5-XXL language embeddings for all zarr episodes..."
        echo "       Dataset: $DATA_DIR"
        echo "       Prompt:  'push the cube from left to right'"
        uv run python ../data_preprocessing/action/precompute_t5.py \
            --dataset-path "$DATA_DIR" \
            --prompt "push the cube from left to right"
        echo "       Done. Each zarr now contains language_embedding."
    fi
else
    echo "[4/4b] Skipping T5 precompute (--skip-t5)."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Submit training:"
echo "  DATA_DIR=$DATA_DIR sbatch $REPO_DIR/euler_train.sbatch"
