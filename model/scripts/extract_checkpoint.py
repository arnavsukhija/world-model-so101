"""Extract slim inference artifacts from a World2Action training checkpoint.

Training checkpoints are fat (~7 GB) because they contain the full frozen Cosmos
video backbone alongside the trainable action decoder and optimizer state. This
script extracts only what inference needs: the action decoder DiT weights (~60 MB)
and the dataset statistics JSON for normalizer calibration.

Usage (run from world-model-so101/model/ on Euler after training):

    uv run python scripts/extract_checkpoint.py \\
        checkpoints/vam/so101/<exp>/checkpoints/iter_000005000.pt \\
        --stats-dir /cluster/scratch/$USER/so101_push_zarr \\
        --output-dir checkpoints/so101_inference

Then download to local machine:

    scp -r euler:/path/to/world-model-so101/model/checkpoints/so101_inference/ .

And pass to inference:

    python eval/so101/run.py \\
        --action-model-path checkpoints/so101_inference/action_decoder.pt \\
        --dataset-stats checkpoints/so101_inference/dataset_stats.json \\
        --video-model-path checkpoints/video_backbone/v2w_pretrained_cosmos.pt \\
        ...
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="Path to training checkpoint (.pt)")
    parser.add_argument(
        "--stats-dir",
        required=True,
        help="Zarr dataset root directory (the one passed as DATA_DIR at training time). Contains .statistics_cache/.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/so101_inference",
        help="Directory to write inference-ready files into (default: checkpoints/so101_inference).",
    )
    args = parser.parse_args()

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Extract action decoder DiT weights ─────────────────────────────────
    ckpt_path = pathlib.Path(args.checkpoint)
    print(f"Loading {ckpt_path} …")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model" not in ckpt:
        raise ValueError(f"Checkpoint does not contain a 'model' key. Keys found: {list(ckpt.keys())}")

    model_state = ckpt["model"]
    prefix = "pipe.dit."
    dit_state = {k[len(prefix):]: v for k, v in model_state.items() if k.startswith(prefix)}

    if not dit_state:
        top_prefixes = sorted({k.split(".")[0] for k in model_state})
        raise ValueError(
            f"No keys found with prefix '{prefix}'.\n"
            f"Top-level prefixes in state dict: {top_prefixes}"
        )

    action_ckpt_path = out / "action_decoder.pt"
    torch.save(dit_state, action_ckpt_path)
    size_mb = action_ckpt_path.stat().st_size / 1024 / 1024
    print(f"Saved {len(dit_state)} tensors → {action_ckpt_path} ({size_mb:.1f} MB)")

    # ── 2. Find and copy dataset statistics ───────────────────────────────────
    stats_cache = pathlib.Path(args.stats_dir) / ".statistics_cache"
    stats_files = sorted(stats_cache.glob("*")) if stats_cache.exists() else []

    if not stats_files:
        print(
            f"WARNING: No stats cache found under {stats_cache}.\n"
            f"Stats are written on the first training run. Re-run this script after training starts."
        )
    else:
        stats_src = max(stats_files, key=lambda p: p.stat().st_mtime)
        stats_dst = out / "dataset_stats.json"
        shutil.copy(stats_src, stats_dst)
        size_kb = stats_dst.stat().st_size / 1024
        print(f"Copied stats ({size_kb:.1f} KB) → {stats_dst}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_mb = sum(f.stat().st_size for f in out.iterdir()) / 1024 / 1024
    print(f"\nInference artifacts ({total_mb:.1f} MB total) written to {out}/")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
