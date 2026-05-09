"""Convert a LeRobot v3 HuggingFace dataset of SO-101 teleoperation to mimic-video zarr format.

LeRobot v3 dataset layout (after `lerobot record`):
  <dataset_root>/
    meta/info.json
    meta/episodes/chunk-000/file-000.parquet   # per-episode index → frame range
    data/chunk-000/file-000.parquet            # all frames, grouped by episode_index
    videos/<camera_key>/chunk-000/file-000.mp4 # ALL episodes concatenated in one mp4

Parquet columns of interest (data/):
  observation.state  [n_joints]  joint positions in degrees
  action             [n_joints]  commanded joint positions in degrees
  timestamp          float       seconds since episode start
  frame_index        int
  episode_index      int

Episodes meta columns of interest:
  episode_index, length, dataset_from_index, dataset_to_index
  videos/<camera_key>/chunk_index, videos/<camera_key>/file_index

Output zarr layout (one .zarr directory per episode):
  workspace_rgb              [N, H, W, 3]  uint8
  workspace_rgb_timestamps   [N]           uint64  nanoseconds
  joint_pos_lowdim           [N, 5]        float32  (5 arm joints, gripper excluded)
  joint_pos_lowdim_timestamps[N]           uint64  nanoseconds
  language_instruction       [1]           bytes

Run precompute_t5.py afterwards to add language_embedding.

Usage:
  python process_so101.py \\
      --dataset-root ~/workspace/RobotLearning/record-test_20260506_184055 \\
      --output-dir /data/so101_push_zarr \\
      --camera-key observation.images.front \\
      --prompt "push the cube from left to right"
"""

import argparse
import json
import pathlib

import cv2
import numpy as np
import pandas as pd
import tqdm
import zarr
from numcodecs import Blosc

S_TO_NS = 1_000_000_000

DEFAULT_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
DEFAULT_CAMERA_KEY = "observation.images.front"


def load_info(dataset_root: pathlib.Path) -> dict:
    with open(dataset_root / "meta" / "info.json") as f:
        return json.load(f)


def load_episodes_meta(dataset_root: pathlib.Path, camera_key: str) -> dict[int, dict]:
    """Return {episode_index: {from_frame, length, chunk_index, file_index}} from meta/episodes/."""
    ep_files = sorted((dataset_root / "meta" / "episodes").glob("**/*.parquet"))
    frames = [pd.read_parquet(f) for f in ep_files]
    meta = pd.concat(frames, ignore_index=True)

    vid_chunk_col = f"videos/{camera_key}/chunk_index"
    vid_file_col = f"videos/{camera_key}/file_index"

    result = {}
    for _, row in meta.iterrows():
        ep_idx = int(row["episode_index"])
        result[ep_idx] = {
            "from_frame": int(row["dataset_from_index"]),
            "length": int(row["length"]),
            "chunk_index": int(row[vid_chunk_col]),
            "file_index": int(row[vid_file_col]),
        }
    return result


def iter_episodes(dataset_root: pathlib.Path):
    """Yield (episode_index, dataframe) sorted by episode_index."""
    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    all_dfs = [pd.read_parquet(p) for p in parquet_files]
    combined = pd.concat(all_dfs, ignore_index=True)
    for ep_idx, group in combined.groupby("episode_index"):
        yield ep_idx, group.reset_index(drop=True)


def extract_episode_frames(
    video_path: pathlib.Path,
    from_frame: int,
    n_frames: int,
    *,
    target_height: int = 480,
    target_width: int = 640,
) -> tuple[np.ndarray, np.ndarray]:
    """Seek to from_frame in the combined mp4 and read n_frames. Returns (frames [N,H,W,3], timestamps [N])."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, from_frame)

    frames, timestamps = [], []
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (target_height, target_width):
            frame = cv2.resize(frame, (target_width, target_height))
        frames.append(frame)
        timestamps.append(i / fps)  # relative to episode start
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from {video_path} at position {from_frame}")

    return np.stack(frames, axis=0), np.array(timestamps, dtype=np.float64)


def parse_state_column(df: pd.DataFrame, joints: list[str]) -> np.ndarray:
    """Extract first len(joints) columns from observation.state.
    LeRobot SO-101 order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper."""
    raw = df["observation.state"].tolist()
    arr = np.array(raw, dtype=np.float32)  # [N, n_joints_total]
    return arr[:, : len(joints)]


def write_episode_zarr(
    output_path: pathlib.Path,
    *,
    frames: np.ndarray,
    video_timestamps: np.ndarray,
    joint_positions: np.ndarray,
    joint_timestamps: np.ndarray,
    language_instruction: str,
) -> None:
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
    episode_offset: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_meta = load_episodes_meta(dataset_root, camera_key)

    episodes = list(iter_episodes(dataset_root))
    print(f"Found {len(episodes)} episodes in {dataset_root}")

    # Cache video paths per (chunk, file) — avoids re-resolving the same path every episode
    video_path_cache: dict[tuple[int, int], pathlib.Path] = {}

    for ep_idx, df in tqdm.tqdm(episodes, desc="Processing episodes"):
        out_path = output_dir / f"episode_{ep_idx + episode_offset:06d}.zarr"
        if skip_existing and out_path.exists():
            continue

        if ep_idx not in episodes_meta:
            print(f"  [warn] No episode meta for episode {ep_idx}, skipping.")
            continue

        meta = episodes_meta[ep_idx]
        chunk_idx = meta["chunk_index"]
        file_idx = meta["file_index"]
        from_frame = meta["from_frame"]

        cache_key = (chunk_idx, file_idx)
        if cache_key not in video_path_cache:
            vid_path = (
                dataset_root
                / "videos"
                / camera_key
                / f"chunk-{chunk_idx:03d}"
                / f"file-{file_idx:03d}.mp4"
            )
            if not vid_path.exists():
                print(f"  [warn] Video not found: {vid_path}, skipping episode {ep_idx}.")
                continue
            video_path_cache[cache_key] = vid_path

        video_path = video_path_cache[cache_key]
        joint_positions = parse_state_column(df, joints)
        joint_timestamps = df["timestamp"].to_numpy(dtype=np.float64)
        n_frames = len(joint_positions)

        frames, video_ts = extract_episode_frames(video_path, from_frame, n_frames)

        # Align if video returned fewer frames than expected (edge case near end of file)
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

    print(f"\nDone. Written to {output_dir}")
    print("Next: run precompute_t5.py to add language_embedding to each zarr.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--output-dir", type=pathlib.Path, required=True)
    p.add_argument("--camera-key", default=DEFAULT_CAMERA_KEY)
    p.add_argument("--prompt", default="push the cube from left to right")
    p.add_argument("--joints", nargs="+", default=DEFAULT_JOINTS)
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--episode-offset", type=int, default=0,
                   help="Add this value to all episode indices in the output filenames. "
                        "Use when merging multiple datasets that each start from episode 0.")
    args = p.parse_args()

    process_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        camera_key=args.camera_key,
        prompt=args.prompt,
        joints=args.joints,
        skip_existing=not args.no_skip_existing,
        episode_offset=args.episode_offset,
    )


if __name__ == "__main__":
    main()
