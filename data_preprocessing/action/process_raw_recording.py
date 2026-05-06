"""Convert mission-impushible raw continuous recordings directly to mimic-video zarr format.

Raw recording layout:
  <recording_dir>/
    metadata.json          # fps, resolution, camera_key, task, format
    frames.jsonl           # one JSON line per control loop frame
    video_000000.mp4       # 60s segments of continuous recording
    video_000001.mp4
    ...

frames.jsonl keys per row:
  frame_index, monotonic_s, video_file, segment_frame_index
  observation          {joint.pos: ...}   robot state
  sent_action          {joint.pos: ...}   commanded position (used as action)
  teleop_action        {joint.pos: ...}   fallback if no sent_action

Episode segmentation: either --episode-length-s (auto-chunk) or an --episodes JSON file with
  [[start_frame, end_frame], ...] or [{"start_frame": N, "end_frame": M}, ...]

Output zarr layout (one .zarr per episode, mimic-video compatible):
  workspace_rgb              [N, H, W, 3]  uint8
  workspace_rgb_timestamps   [N]           uint64  nanoseconds
  joint_pos_lowdim           [N, 5]        float32  (5 arm joints, gripper excluded)
  joint_pos_lowdim_timestamps[N]           uint64  nanoseconds
  language_instruction       [1]           bytes

Usage:
  # Auto-chunk eval-1 into 15s episodes, resize to 640x480:
  python process_raw_recording.py \\
      --input ~/workspace/RobotLearning/data/eval-1 \\
      --output-dir /data/so101_push_zarr \\
      --episode-length-s 15 \\
      --resize 640x480 \\
      --prompt "push the cube from left to right"

  # Use explicit episode JSON:
  python process_raw_recording.py \\
      --input ~/workspace/RobotLearning/data/eval-1 \\
      --output-dir /data/so101_push_zarr \\
      --episodes episodes_eval1.json \\
      --prompt "push the cube from left to right"

  # Dry-run to preview episode count without writing:
  python process_raw_recording.py --input data/eval-1 --output-dir /tmp/out \\
      --episode-length-s 15 --dry-run
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import tqdm
import zarr
from numcodecs import Blosc

S_TO_NS = 1_000_000_000

MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
MOTOR_POS_NAMES = tuple(f"{name}.pos" for name in MOTOR_NAMES)
DEFAULT_JOINTS = list(MOTOR_NAMES[:5])  # exclude gripper


# ---------------------------------------------------------------------------
# Episode range
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpisodeRange:
    start: int   # inclusive frame_index
    end: int     # exclusive frame_index

    @property
    def length(self) -> int:
        return self.end - self.start


def default_episode_ranges(
    rows: list[dict[str, Any]],
    episode_length_s: float | None,
    fps: int,
    min_frames: int = 4,
) -> list[EpisodeRange]:
    if not rows:
        return []
    first = int(rows[0]["frame_index"])
    last_exclusive = int(rows[-1]["frame_index"]) + 1
    if episode_length_s is None:
        return [EpisodeRange(first, last_exclusive)]
    chunk = max(1, int(round(episode_length_s * fps)))
    ranges: list[EpisodeRange] = []
    start = first
    while start < last_exclusive:
        end = min(start + chunk, last_exclusive)
        if end - start >= min_frames:
            ranges.append(EpisodeRange(start, end))
        start = end
    return ranges


def load_episode_ranges(path: pathlib.Path, min_frames: int = 4) -> list[EpisodeRange]:
    raw = json.loads(path.read_text())
    ranges: list[EpisodeRange] = []
    for item in raw:
        if isinstance(item, list):
            start, end = int(item[0]), int(item[1])
        elif isinstance(item, dict):
            start, end = int(item["start_frame"]), int(item["end_frame"])
        else:
            raise ValueError(f"Unknown episode entry: {item}")
        if end - start < min_frames:
            raise ValueError(f"Episode {start}:{end} is shorter than {min_frames} frames")
        ranges.append(EpisodeRange(start, end))
    return ranges


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def load_frame_rows(recording_dir: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with open(recording_dir / "frames.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["frame_index"]))
    return rows


def _normalize_motor(values: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in MOTOR_NAMES:
        for key in (f"{name}.pos", name):
            if key in values:
                out[f"{name}.pos"] = float(values[key])
                break
    return out


def get_state_vector(row: dict[str, Any], n_joints: int = 5) -> np.ndarray:
    src = row.get("observation") or row.get("follower_joints")
    if src is None:
        raise KeyError("Row has no observation or follower_joints")
    values = _normalize_motor(src)
    return np.asarray([values[k] for k in MOTOR_POS_NAMES[:n_joints]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Video reading
# ---------------------------------------------------------------------------

class VideoFrameReader:
    """Sequential reader across multiple video segment files."""

    def __init__(self, recording_dir: pathlib.Path) -> None:
        self.recording_dir = recording_dir
        self._video_file: str | None = None
        self._cap: cv2.VideoCapture | None = None
        self._next_idx = 0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._video_file = None
        self._next_idx = 0

    def read_rgb(self, row: dict[str, Any]) -> np.ndarray:
        video_file = row["video_file"]
        seg_idx = int(row["segment_frame_index"])
        if video_file != self._video_file:
            self.close()
            path = self.recording_dir / video_file
            self._cap = cv2.VideoCapture(str(path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open: {path}")
            self._video_file = video_file
            self._next_idx = 0
        assert self._cap is not None
        if seg_idx != self._next_idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, seg_idx)
            self._next_idx = seg_idx
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"Failed to read frame {seg_idx} from {video_file}")
        self._next_idx += 1
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __enter__(self) -> "VideoFrameReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def transform_image(
    rgb: np.ndarray,
    crop: tuple[int, int, int, int] | None,
    resize: tuple[int, int] | None,
) -> np.ndarray:
    if crop is not None:
        left, top, right, bottom = crop
        rgb = rgb[top:bottom, left:right]
    if resize is not None:
        w, h = resize
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)


# ---------------------------------------------------------------------------
# Zarr writing
# ---------------------------------------------------------------------------

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
            "workspace_rgb", data=frames,
            shape=(N, H, W, C), dtype="uint8",
            chunks=(1, H, W, C), compressor=compressor,
        )
        root.create_dataset(
            "workspace_rgb_timestamps",
            data=(video_timestamps * S_TO_NS).astype(np.uint64),
            shape=(N,), dtype="uint64", chunks=(N,), compressor=compressor,
        )
        J = joint_positions.shape[1]
        root.create_dataset(
            "joint_pos_lowdim", data=joint_positions,
            shape=(N, J), dtype="float32",
            chunks=(N, J), compressor=compressor,
        )
        root.create_dataset(
            "joint_pos_lowdim_timestamps",
            data=(joint_timestamps * S_TO_NS).astype(np.uint64),
            shape=(N,), dtype="uint64", chunks=(N,), compressor=compressor,
        )
        encoded = language_instruction.encode("utf-8")
        root.create_dataset(
            "language_instruction",
            data=np.frombuffer(encoded, dtype="uint8").reshape(1, -1),
            shape=(1, len(encoded)), dtype="uint8", compressor=compressor,
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_crop(s: str | None) -> tuple[int, int, int, int] | None:
    if not s:
        return None
    l, t, r, b = [int(x) for x in s.split(",")]
    return l, t, r, b


def parse_resize(s: str | None) -> tuple[int, int] | None:
    if not s:
        return None
    w, h = s.lower().replace(" ", "").split("x")
    return int(w), int(h)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_recording(
    recording_dir: pathlib.Path,
    output_dir: pathlib.Path,
    episode_ranges: list[EpisodeRange],
    prompt: str,
    crop: tuple[int, int, int, int] | None,
    resize: tuple[int, int] | None,
    skip_existing: bool = True,
    output_offset: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_frame_rows(recording_dir)
    row_map = {int(r["frame_index"]): r for r in rows}

    # Compute metadata for timestamps
    fps = json.loads((recording_dir / "metadata.json").read_text())["fps"]

    with VideoFrameReader(recording_dir) as reader:
        for ep_idx, ep in enumerate(tqdm.tqdm(episode_ranges, desc=f"Processing {recording_dir.name}")):
            out_idx = ep_idx + output_offset
            out_path = output_dir / f"episode_{out_idx:06d}.zarr"
            if skip_existing and out_path.exists():
                continue

            frames_list, joints_list, timestamps_list = [], [], []
            ep_start_mono: float | None = None

            for fi in range(ep.start, ep.end):
                row = row_map.get(fi)
                if row is None:
                    print(f"  [warn] Missing frame_index={fi} in episode {ep_idx}, stopping episode early.")
                    break
                rgb = transform_image(reader.read_rgb(row), crop=crop, resize=resize)
                frames_list.append(rgb)
                joints_list.append(get_state_vector(row))
                mono = float(row["monotonic_s"])
                if ep_start_mono is None:
                    ep_start_mono = mono
                timestamps_list.append(mono - ep_start_mono)

            if len(frames_list) < 4:
                print(f"  [warn] Episode {ep_idx} has only {len(frames_list)} frames, skipping.")
                continue

            write_episode_zarr(
                out_path,
                frames=np.stack(frames_list, axis=0),
                video_timestamps=np.array(timestamps_list, dtype=np.float64),
                joint_positions=np.stack(joints_list, axis=0),
                joint_timestamps=np.array(timestamps_list, dtype=np.float64),
                language_instruction=prompt,
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=pathlib.Path, required=True,
                   help="Raw recording directory (contains metadata.json + frames.jsonl)")
    p.add_argument("--output-dir", type=pathlib.Path, required=True)
    p.add_argument("--prompt", default="push the cube from left to right")
    p.add_argument("--episode-length-s", type=float, default=None,
                   help="Auto-chunk into fixed-length episodes (e.g. 15). Required if --episodes not given.")
    p.add_argument("--episodes", type=pathlib.Path, default=None,
                   help="JSON file with explicit [[start, end], ...] episode frame ranges.")
    p.add_argument("--crop", default=None, help="left,top,right,bottom pixels to crop before resize.")
    p.add_argument("--resize", default="640x480", help="WIDTHxHEIGHT for output frames (default: 640x480).")
    p.add_argument("--min-episode-frames", type=int, default=4)
    p.add_argument("--episode-index-offset", type=int, default=0,
                   help="Start numbering output zarr files at this index (for combining multiple recordings).")
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print episode plan without writing zarr.")
    args = p.parse_args()

    if args.episodes is None and args.episode_length_s is None:
        p.error("Provide either --episode-length-s or --episodes.")

    meta = json.loads((args.input / "metadata.json").read_text())
    fps = int(meta["fps"])
    rows = load_frame_rows(args.input)

    if args.episodes is not None:
        episodes = load_episode_ranges(args.episodes, args.min_episode_frames)
    else:
        episodes = default_episode_ranges(rows, args.episode_length_s, fps, args.min_episode_frames)

    total_frames = sum(e.length for e in episodes)
    crop = parse_crop(args.crop)
    resize = parse_resize(args.resize)

    print(f"Recording  : {args.input}")
    print(f"FPS        : {fps}")
    print(f"Episodes   : {len(episodes)}")
    print(f"Total frames: {total_frames}  ({total_frames/fps:.1f}s)")
    print(f"Resize     : {args.resize}")
    if crop:
        print(f"Crop       : {args.crop}")
    print(f"Output     : {args.output_dir}")

    if args.dry_run:
        print("\nDry run — episode frame ranges:")
        for i, ep in enumerate(episodes[:20]):
            print(f"  ep{i+args.episode_index_offset:03d}: frames {ep.start}–{ep.end-1}  ({ep.length} frames, {ep.length/fps:.1f}s)")
        if len(episodes) > 20:
            print(f"  ... and {len(episodes)-20} more")
        return

    process_recording(
        recording_dir=args.input,
        output_dir=args.output_dir,
        episode_ranges=episodes,
        prompt=args.prompt,
        crop=crop,
        resize=resize,
        skip_existing=not args.no_skip_existing,
        output_offset=args.episode_index_offset,
    )
    print(f"\nDone. Next: run precompute_t5.py --dataset-path {args.output_dir}")


if __name__ == "__main__":
    main()
