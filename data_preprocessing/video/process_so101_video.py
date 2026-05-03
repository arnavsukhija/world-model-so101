"""Extract mp4 + language text files from a LeRobot dataset for Video2World finetuning.

Output layout expected by mimic-video video finetuning:
  <output_dir>/
    video/episode_000000.mp4
    metas/episode_000000.txt     # language instruction
    t5_xxl/episode_000000.pickle # precomputed T5 embeddings (run get_t5_embeddings.py)

Usage:
  python process_so101_video.py \\
      --dataset-root ~/.cache/huggingface/lerobot/my_push_dataset \\
      --output-dir /data/so101_push_video \\
      --camera-key observation.images.cam_wrist \\
      --prompt "push the cube from left to right"
"""

import argparse
import pathlib
import shutil

import tqdm


def find_all_videos(dataset_root: pathlib.Path, camera_key: str) -> list[tuple[int, pathlib.Path]]:
    """Return sorted list of (episode_index, video_path)."""
    video_root = dataset_root / "videos" / camera_key
    result = []
    for mp4 in sorted(video_root.glob("**/*.mp4")):
        stem = mp4.stem  # e.g. "episode_000042"
        ep_idx = int(stem.split("_")[-1])
        result.append((ep_idx, mp4))
    return sorted(result)


def process_dataset(
    dataset_root: pathlib.Path,
    output_dir: pathlib.Path,
    camera_key: str,
    prompt: str,
    skip_existing: bool = True,
) -> None:
    video_out = output_dir / "video"
    metas_out = output_dir / "metas"
    video_out.mkdir(parents=True, exist_ok=True)
    metas_out.mkdir(parents=True, exist_ok=True)

    episodes = find_all_videos(dataset_root, camera_key)
    print(f"Found {len(episodes)} videos in {dataset_root}/videos/{camera_key}")

    for ep_idx, src_mp4 in tqdm.tqdm(episodes, desc="Copying videos"):
        dst_mp4 = video_out / f"episode_{ep_idx:06d}.mp4"
        dst_txt = metas_out / f"episode_{ep_idx:06d}.txt"

        if skip_existing and dst_mp4.exists():
            continue

        shutil.copy2(src_mp4, dst_mp4)
        dst_txt.write_text(prompt)

    print(f"Done. Output: {output_dir}")
    print("Next: run get_t5_embeddings.py to precompute T5 embeddings in t5_xxl/")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--output-dir", type=pathlib.Path, required=True)
    p.add_argument("--camera-key", default="observation.images.cam_wrist")
    p.add_argument("--prompt", default="push the cube from left to right")
    p.add_argument("--no-skip-existing", action="store_true")
    args = p.parse_args()

    process_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        camera_key=args.camera_key,
        prompt=args.prompt,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
