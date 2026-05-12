#!/usr/bin/env bash
# Precompute T5-XXL language embeddings for the SO-101 video dataset.
# Reads metas/*.txt captions and writes t5_xxl/*.pickle alongside the videos.
# Required BEFORE euler_scripts/train/v2w_train.sbatch.
# Note: only processes text captions, not video frames — same resource profile as t5_action.
#
# The video dataset must already exist at VIDEO_DATA_DIR with the layout:
#   video/ep_000000.mp4  metas/ep_000000.txt  (produced by process_so101_video.py)
#
# Usage (from world-model-so101/):
#   VIDEO_DATA_DIR=/cluster/scratch/$USER/so101_push_video \
#     sbatch euler_scripts/embeddings/t5_video.sbatch
#SBATCH --job-name=t5-video
#SBATCH --output=logs/t5_video_%j.out
#SBATCH --error=logs/t5_video_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --gpus=rtx_4090:1

VIDEO_DATA_DIR="${VIDEO_DATA_DIR:-/cluster/scratch/$USER/so101_push_video}"

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
MODEL_DIR="$REPO_DIR/model"
mkdir -p "$REPO_DIR/logs"

echo "=== T5-XXL Embedding Precompute (video dataset) ==="
echo "Job           : $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "VIDEO_DATA_DIR: $VIDEO_DATA_DIR"
echo ""

if [[ ! -d "$VIDEO_DATA_DIR/metas" ]]; then
    echo "ERROR: $VIDEO_DATA_DIR/metas not found."
    echo "Run data_preprocessing/video/process_so101_video.py first."
    exit 1
fi

source "$MODEL_DIR/.venv/bin/activate"

cd "$MODEL_DIR"

# T5_MODEL_DIR resolves to checkpoints/text_encoder/t5-11b (set in imaginaire/constants.py)
python ../data_preprocessing/video/get_t5_embeddings.py \
    --dataset_path "$VIDEO_DATA_DIR"

echo ""
echo "=== Done. t5_xxl/*.pickle written to $VIDEO_DATA_DIR/t5_xxl/ ==="
echo "Next: VIDEO_DATA_DIR=$VIDEO_DATA_DIR sbatch $REPO_DIR/euler_scripts/train/v2w_train.sbatch"
