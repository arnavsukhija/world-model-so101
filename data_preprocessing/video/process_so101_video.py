"""Extract mp4 + language text files from a LeRobot dataset for Video2World finetuning.

Output layout expected by mimic-video video finetuning:
  <output_dir>/
    video/episode_000000.mp4
    metas/episode_000000.txt     # language instruction
    t5_xxl/episode_000000.pickle # precomputed T5 embeddings (run get_t5_embeddings.py)

Usage (single prompt for all episodes):
  python process_so101_video.py \\
      --dataset-root ~/.cache/huggingface/lerobot/my_push_dataset \\
      --output-dir /data/so101_push_video \\
      --camera-key observation.images.cam_wrist \\
      --prompt "push the cube from left to right"

Usage (per-episode prompts from JSON):
  python process_so101_video.py \\
      --dataset-root ~/.cache/huggingface/lerobot/my_push_dataset \\
      --output-dir /data/so101_push_video \\
      --prompts-json /path/to/prompts.json

  prompts.json format: {"0": "push the red cube", "1": "push the blue sphere", ...}
  Episodes not in the JSON fall back to --prompt (default: "push the object").
"""

import argparse
import json
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


def load_prompts_from_lerobot(dataset_root: pathlib.Path) -> dict[int, str]:
    """Read per-episode task strings from a LeRobot v3 meta/episodes.jsonl."""
    episodes_jsonl = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_jsonl.exists():
        return {}
    prompts: dict[int, str] = {}
    with episodes_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ep_idx = record.get("episode_index")
            task = record.get("tasks", [None])[0] if record.get("tasks") else record.get("task")
            if ep_idx is not None and task:
                prompts[int(ep_idx)] = task
    return prompts


def process_dataset(
    dataset_root: pathlib.Path,
    output_dir: pathlib.Path,
    camera_key: str,
    default_prompt: str,
    per_episode_prompts: dict[int, str],
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

        prompt = per_episode_prompts.get(ep_idx, default_prompt)
        shutil.copy2(src_mp4, dst_mp4)
        dst_txt.write_text(prompt)

    print(f"Done. Output: {output_dir}")
    print("Next: run get_t5_embeddings.py to precompute T5 embeddings in t5_xxl/")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--output-dir", type=pathlib.Path, required=True)
    p.add_argument("--camera-key", default="observation.images.cam_wrist")
    p.add_argument("--prompt", default="push the object", help="Fallback prompt when no per-episode prompt is found")
    p.add_argument(
        "--prompts-json",
        type=pathlib.Path,
        default=None,
        help='JSON file mapping episode index (str) to prompt text: {"0": "...", "1": "..."}',
    )
    p.add_argument(
        "--use-lerobot-tasks",
        action="store_true",
        help="Auto-read per-episode prompts from meta/episodes.jsonl in the LeRobot dataset",
    )
    p.add_argument("--no-skip-existing", action="store_true")
    args = p.parse_args()

    per_episode_prompts: dict[int, str] = {}

    if args.use_lerobot_tasks:
        per_episode_prompts = load_prompts_from_lerobot(args.dataset_root)
        print(f"Loaded {len(per_episode_prompts)} per-episode prompts from LeRobot meta/episodes.jsonl")

    if args.prompts_json is not None:
        with args.prompts_json.open() as f:
            raw = json.load(f)
        per_episode_prompts.update({int(k): v for k, v in raw.items()})
        print(f"Loaded {len(per_episode_prompts)} per-episode prompts from {args.prompts_json}")

    if not per_episode_prompts:
        print(f"Using single prompt for all episodes: '{args.prompt}'")

    process_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        camera_key=args.camera_key,
        default_prompt=args.prompt,
        per_episode_prompts=per_episode_prompts,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
