"""Convert a LeRobot HuggingFace dataset of SO-101 teleoperation to mimic-video zarr format.

LeRobot dataset layout (after `lerobot record`):
  <dataset_root>/
    meta/info.json                                   # keys, fps, shapes
    data/chunk-000/episode_000000.parquet            # one parquet per episode
    videos/<camera_key>/chunk-000/episode_000000.mp4

Parquet columns of interest:
  observation.state  [n_joints]  joint positions in degrees
  action             [n_joints]  commanded joint positions in degrees
  timestamp          float       seconds since episode start
  frame_index        int
  episode_index      int
  next.done          bool

Output zarr layout (one .zarr directory per episode):
  workspace_rgb              [N, H, W, 3]  uint8
  workspace_rgb_timestamps   [N]           uint64  nanoseconds
  joint_pos_lowdim           [N, 5]        float32  (5 arm joints, gripper excluded)
  joint_pos_lowdim_timestamps[N]           uint64  nanoseconds
  language_instruction       [1]           bytes

Run precompute_t5.py afterwards to add language_embedding.

Usage:
  python process_so101.py \\
      --dataset-root ~/.cache/huggingface/lerobot/my_push_dataset \\
      --output-dir /data/so101_push \\
      --camera-key observation.images.cam_wrist \\
      --prompt "push the cube from left to right" \\
      [--joints shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll]
"""

import argparse
import json
import pathlib
import subprocess

import cv2
import numpy as np
import pandas as pd
import tqdm
import zarr
from numcodecs import Blosc

S_TO_NS = 1_000_000_000

# SO-101 joint order as recorded by LeRobot (degrees, excluding gripper for pushing tasks)
DEFAULT_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def load_info(dataset_root: pathlib.Path) -> dict:
    with open(dataset_root / "meta" / "info.json") as f:
        return json.load(f)


def iter_episodes(dataset_root: pathlib.Path):
    """Yield (episode_index, parquet_path) sorted by episode_index."""
    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    for p in parquet_files:
        df = pd.read_parquet(p)
        for ep_idx, group in df.groupby("episode_index"):
            yield ep_idx, group.reset_index(drop=True)


def extract_video_frames(
    video_path: pathlib.Path,
    *,
    target_height: int = 480,
    target_width: int = 640,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (frames [N, H, W, 3] uint8, timestamps [N] float64 seconds)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames, timestamps = [], []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (target_height, target_width):
            frame = cv2.resize(frame, (target_width, target_height))
        frames.append(frame)
        timestamps.append(frame_idx / fps)
        frame_idx += 1
    cap.release()

    return np.stack(frames, axis=0), np.array(timestamps, dtype=np.float64)


def find_video_for_episode(
    dataset_root: pathlib.Path,
    camera_key: str,
    episode_index: int,
) -> pathlib.Path | None:
    """Locate the mp4 for a given episode across all chunks."""
    video_root = dataset_root / "videos" / camera_key
    pattern = f"episode_{episode_index:06d}.mp4"
    matches = list(video_root.glob(f"**/{pattern}"))
    return matches[0] if matches else None


def parse_state_column(df: pd.DataFrame, joints: list[str]) -> np.ndarray:
    """Extract joint positions. LeRobot stores observation.state as a list column."""
    raw = df["observation.state"].tolist()
    arr = np.array(raw, dtype=np.float32)  # [N, n_joints_total]

    # Try to match joints by name via info.json motor order; fall back to positional slice.
    # LeRobot SO-101 default order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    n_wanted = len(joints)
    return arr[:, :n_wanted]


def write_episode_zarr(
    output_path: pathlib.Path,
    *,
    frames: np.ndarray,        # [N, H, W, 3] uint8
    video_timestamps: np.ndarray,  # [N] float64 seconds
    joint_positions: np.ndarray,   # [N, J] float32
    joint_timestamps: np.ndarray,  # [N] float64 seconds
    language_instruction: str,
) -> None:
    """Write one episode to zarr format expected by mimic-video."""
    compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)

    with zarr.open(str(output_path), mode="w") as root:
        N, H, W, C = frames.shape
        root.create_dataset(
            "workspace_rgb",
            data=frames,
            shape=(N, H, W, C),
            dtype="uint8",
            chunks=(1, H, W, C),
            compressor=compressor,
        )
        root.create_dataset(
            "workspace_rgb_timestamps",
            data=(video_timestamps * S_TO_NS).astype(np.uint64),
            shape=(N,),
            dtype="uint64",
            chunks=(N,),
            compressor=compressor,
        )

        J = joint_positions.shape[1]
        root.create_dataset(
            "joint_pos_lowdim",
            data=joint_positions,
            shape=(N, J),
            dtype="float32",
            chunks=(N, J),
            compressor=compressor,
        )
        root.create_dataset(
            "joint_pos_lowdim_timestamps",
            data=(joint_timestamps * S_TO_NS).astype(np.uint64),
            shape=(N,),
            dtype="uint64",
            chunks=(N,),
            compressor=compressor,
        )

        encoded = language_instruction.encode("utf-8")
        root.create_dataset(
            "language_instruction",
            data=np.frombuffer(encoded, dtype="uint8").reshape(1, -1),
            shape=(1, len(encoded)),
            dtype="uint8",
            compressor=compressor,
        )


def process_dataset(
    dataset_root: pathlib.Path,
    output_dir: pathlib.Path,
    camera_key: str,
    prompt: str,
    joints: list[str],
    skip_existing: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = list(iter_episodes(dataset_root))
    print(f"Found {len(episodes)} episodes in {dataset_root}")

    for ep_idx, df in tqdm.tqdm(episodes, desc="Processing episodes"):
        out_path = output_dir / f"episode_{ep_idx:06d}.zarr"
        if skip_existing and out_path.exists():
            continue

        video_path = find_video_for_episode(dataset_root, camera_key, ep_idx)
        if video_path is None:
            print(f"  [warn] No video found for episode {ep_idx}, skipping.")
            continue

        frames, video_ts = extract_video_frames(video_path)
        joint_positions = parse_state_column(df, joints)
        joint_timestamps = df["timestamp"].to_numpy(dtype=np.float64)

        # Align lengths: video and joints may differ by ±1 frame at boundaries
        min_n = min(len(frames), len(joint_positions))
        frames = frames[:min_n]
        video_ts = video_ts[:min_n]
        joint_positions = joint_positions[:min_n]
        joint_timestamps = joint_timestamps[:min_n]

        write_episode_zarr(
            out_path,
            frames=frames,
            video_timestamps=video_ts,
            joint_positions=joint_positions,
            joint_timestamps=joint_timestamps,
            language_instruction=prompt,
        )

    print(f"Done. Written to {output_dir}")
    print("Next: run precompute_t5.py to add language_embedding to each zarr.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=pathlib.Path, required=True,
                   help="Root of the LeRobot HuggingFace dataset directory")
    p.add_argument("--output-dir", type=pathlib.Path, required=True,
                   help="Directory to write per-episode .zarr files")
    p.add_argument("--camera-key", default="observation.images.cam_wrist",
                   help="Camera key used during recording (matches videos/ subdirectory)")
    p.add_argument("--prompt", default="push the cube from left to right",
                   help="Language instruction stored in every episode")
    p.add_argument("--joints", nargs="+", default=DEFAULT_JOINTS,
                   help="Joint names to extract in order (must match LeRobot motor order)")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Reprocess episodes even if output zarr already exists")
    args = p.parse_args()

    process_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        camera_key=args.camera_key,
        prompt=args.prompt,
        joints=args.joints,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
