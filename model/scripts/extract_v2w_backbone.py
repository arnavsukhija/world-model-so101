"""Extract a slim video backbone checkpoint from a Video2World training checkpoint.

After V2W LoRA training, the model checkpoint contains all keys of the full
Video2WorldModel (video backbone + LoRA layers + tokenizer states). This script:

  1. Loads the (already LoRA-fused) training checkpoint.
  2. Extracts keys under the ``pipe.dit.`` prefix (the DiT network).
  3. Renames them to ``net.`` prefix (the format expected by video2world.py's
     ``from_config`` loader, which strips ``net.`` on load).
  4. Saves the result to ``checkpoints/video_backbone/<output_name>.pt``.

The output checkpoint can then be referenced in world2action_model.py as a new
entry in VIDEO_MODEL_CKPT_NAMES, enabling W2A training on top of the SO-101
finetuned video backbone.

Usage (run from world-model-so101/model/ on Euler after V2W training):

    python scripts/extract_v2w_backbone.py \\
        checkpoints/posttraining/so101/<exp>/checkpoints/model/iter_010000_fused.pt \\
        --output-name v2w_so101_push_lora_rank256_lr3.162e-04_bsz32_iter_010000_fused

This saves to checkpoints/video_backbone/<output_name>.pt (~7 GB).
Add <output_name> to VIDEO_MODEL_CKPT_NAMES in
cosmos_predict2/configs/defaults/world2action_model.py to use it in W2A training.
"""

from __future__ import annotations

import argparse
import pathlib

import torch


BACKBONE_DIR = pathlib.Path(__file__).parents[1] / "checkpoints" / "video_backbone"
SRC_PREFIX = "pipe.dit."
DST_PREFIX = "net."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "checkpoint",
        help="Path to LoRA-fused V2W training checkpoint (model/iter_XXXXXX_fused.pt)",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Output filename stem (without .pt). Saved to checkpoints/video_backbone/<name>.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=BACKBONE_DIR,
        help=f"Output directory (default: {BACKBONE_DIR})",
    )
    args = parser.parse_args()

    ckpt_path = pathlib.Path(args.checkpoint)
    print(f"Loading {ckpt_path} …")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Handle case where checkpoint is wrapped (has a 'model' key)
    if isinstance(state_dict, dict) and "model" in state_dict and not any(
        k.startswith(SRC_PREFIX) for k in state_dict
    ):
        print("Detected wrapped checkpoint — unwrapping 'model' key")
        state_dict = state_dict["model"]

    # Extract DiT keys and rename prefix
    backbone_state = {}
    for k, v in state_dict.items():
        if k.startswith(SRC_PREFIX):
            new_key = DST_PREFIX + k[len(SRC_PREFIX):]
            backbone_state[new_key] = v

    if not backbone_state:
        top_prefixes = sorted({k.split(".")[0] for k in state_dict})
        raise ValueError(
            f"No keys found with prefix '{SRC_PREFIX}'.\n"
            f"Top-level key prefixes in checkpoint: {top_prefixes}\n"
            f"Is this a LoRA-fused V2W training checkpoint? Run fuse_lora_ckpt.py first if not."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.output_name}.pt"
    torch.save(backbone_state, out_path)

    size_gb = out_path.stat().st_size / 1024 / 1024 / 1024
    print(f"Saved {len(backbone_state)} tensors → {out_path} ({size_gb:.2f} GB)")
    print()
    print("Next steps:")
    print(f"  1. Add '{args.output_name}' to VIDEO_MODEL_CKPT_NAMES in")
    print("     cosmos_predict2/configs/defaults/world2action_model.py")
    print("  2. Retrain W2A using the new video backbone experiment, e.g.:")
    exp = f"w2a_so101_push_{args.output_name}_lr1.000e-04_layer20_bsz1"
    print(f"     experiment={exp}")


if __name__ == "__main__":
    main()
