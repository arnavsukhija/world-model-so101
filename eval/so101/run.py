"""SO-101 real-robot inference with the Video-Action World Model (VAM).

Two modes:
  --mode open_loop   Plan once from the initial observation, execute all actions.
                     Fastest; best for deterministic tasks like straight pushing.
  --mode receding    Query the model every `--replan-every` steps using a sliding
                     window of recent observations. More reactive; needed for
                     obstacle avoidance or when the object deviates.

Usage (open-loop, typical for Eval 1):
  python run.py \\
      --experiment w2a_so101_push_v2w_pretrained_cosmos_lr3.162e-05_layer20_bsz1 \\
      --video-model-path ../../checkpoints/video_backbone/cosmos-predict2_v2w_480p_10fps.pt \\
      --action-model-path ../../checkpoints/action_decoder/so101_push.pt \\
      --dataset-stats ../../checkpoints/action_decoder/so101_push_stats.json \\
      --robot-port /dev/ttyUSB0 \\
      --camera-index 0 \\
      --mode open_loop \\
      --num-execute-actions 15

For inference on Brev (SSH tunnel to local camera/robot not practical):
  Run this script directly on the Brev machine, capturing frames from a local
  USB camera or pre-captured images. For real-time local execution, use a local
  GPU machine with the smaller action decoder (no video generation needed if
  features are pre-extracted, but the full pipeline still requires ~12GB VRAM).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from collections import deque

import cv2
import numpy as np
import torch
from einops import rearrange

from cosmos_predict2.configs.config import make_config
from cosmos_predict2.data.action.utils import extract_normalization_types
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import override

# SO-101 joint order (must match process_so101.py DEFAULT_JOINTS)
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

PROMPT = "push the cube from left to right"

IMG_H, IMG_W = 480, 640


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def load_pipeline(
    experiment_name: str,
    video_model_path: str,
    action_model_path: str,
    dataset_stats_path: pathlib.Path,
    dtype: torch.dtype = torch.bfloat16,
) -> Video2World2ActionPipeline:
    config = make_config()
    config = override(config, ["--", f"experiment={experiment_name}"])
    config.model.config.video_pipe_config.guardrail_config.enabled = False

    video2world_pipe = Video2WorldPipeline.from_config(
        config=config.model.config.video_pipe_config,
        dit_path=video_model_path,
        device="cuda",
        torch_dtype=dtype,
        load_ema_to_reg=False,
    )

    world2action_pipe = World2ActionPipeline.from_config(
        config.model.config.pipe_config,
        dit_path=action_model_path,
        device="cuda",
        dtype=dtype,
    )

    data_config = instantiate(config.data_config)
    with dataset_stats_path.open() as f:
        stats = json.load(f)
    world2action_pipe.normalizer.build_from_stats(
        stats,
        normalization_types=extract_normalization_types(data_config.policy_io.policy_io),
        concat_groups=data_config.policy_io.concat_groups,
        device="cuda",
        dtype=dtype,
    )

    return Video2World2ActionPipeline(video2world_pipe, world2action_pipe).cuda()


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def open_camera(camera_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
    return cap


def capture_frame(cap: cv2.VideoCapture) -> np.ndarray:
    """Return [H, W, 3] uint8 RGB frame."""
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Camera read failed")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame.shape[:2] != (IMG_H, IMG_W):
        frame = cv2.resize(frame, (IMG_W, IMG_H))
    return frame


def process_image(frame: np.ndarray) -> np.ndarray:
    """Normalize to [-1, 1] and reshape to [C, 1, H, W]."""
    tensor = rearrange(frame, "h w c -> c h w")[:, None, :, :]
    return 2.0 * (tensor.astype(np.float32) / 255.0 - 0.5)


# ---------------------------------------------------------------------------
# Robot helpers (LeRobot)
# ---------------------------------------------------------------------------

def try_import_lerobot():
    try:
        from lerobot.robots import make_robot_from_config
        from lerobot.configs import parser
        return make_robot_from_config, parser
    except ImportError:
        return None, None


def read_joint_positions(robot) -> np.ndarray:
    """Read 5 arm joint positions [deg] from the SO-101 follower."""
    obs = robot.get_observation()
    # obs keys: "shoulder_pan.pos", "shoulder_lift.pos", ...
    return np.array(
        [float(obs[f"{j}.pos"]) for j in JOINT_NAMES],
        dtype=np.float32,
    )


def send_joint_positions(robot, joint_pos: np.ndarray, joint_names: list[str]) -> None:
    """Send 5 joint positions to the SO-101 follower."""
    action = {f"{j}.pos": float(joint_pos[i]) for i, j in enumerate(joint_names)}
    # Keep gripper at current position (fixed per task rules)
    obs = robot.get_observation()
    action["gripper.pos"] = float(obs.get("gripper.pos", 0.0))
    robot.send_action(action)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def build_input_tensors(
    image_history: deque,
    joint_history: deque,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack histories into model input tensors."""
    # image_history contains [C, 1, H, W] arrays at ~5Hz cadence (stride-4 from 20Hz)
    images = np.concatenate(list(image_history), axis=1)  # [C, T, H, W]
    input_vid = torch.from_numpy(images[None]).cuda().to(dtype=dtype)  # [1, C, T, H, W]

    joints = np.stack(list(joint_history), axis=0)           # [HO, 5]
    state = torch.from_numpy(joints[None]).cuda().to(dtype=dtype)  # [1, HO, 5]

    return input_vid, state


@torch.no_grad()
def query_model(
    pipeline: Video2World2ActionPipeline,
    input_vid: torch.Tensor,
    state: torch.Tensor,
    num_sampling_steps: int = 35,
    stop_video_denoising_step: int = 17,
) -> np.ndarray:
    """Returns [T, 5] array of predicted joint positions."""
    pred_actions = pipeline(
        input_vid=input_vid,
        state_B_HO_O=state,
        prompt=PROMPT,
        num_sampling_step=num_sampling_steps,
        stop_after_step=stop_video_denoising_step,
        use_cuda_graphs=False,  # safer for one-off inference
    )
    return pred_actions[0].float().cpu().numpy()  # [T, 5]


# ---------------------------------------------------------------------------
# Open-loop execution
# ---------------------------------------------------------------------------

def run_open_loop(
    pipeline: Video2World2ActionPipeline,
    cap: cv2.VideoCapture,
    robot,
    *,
    img_horizon: int = 5,
    num_execute_actions: int = 15,
    execution_hz: float = 5.0,
    save_rollout: pathlib.Path | None = None,
) -> None:
    """Capture one observation, plan once, execute the full action sequence."""
    print("Capturing initial observation …")
    frame = capture_frame(cap)
    joints = read_joint_positions(robot) if robot else np.zeros(5, dtype=np.float32)

    # Fill histories with the single initial frame/joints (pad)
    processed = process_image(frame)
    image_history: deque = deque([processed] * img_horizon, maxlen=img_horizon)
    joint_history: deque = deque([joints], maxlen=1)

    input_vid, state = build_input_tensors(image_history, joint_history)

    print("Querying world model …")
    t0 = time.time()
    actions = query_model(pipeline, input_vid, state)  # [T, 5]
    print(f"Planning took {time.time()-t0:.1f}s. Executing {num_execute_actions} steps at {execution_hz}Hz …")

    dt = 1.0 / execution_hz
    rollout_frames = []
    for step_i in range(min(num_execute_actions, len(actions))):
        t_start = time.time()
        action = actions[step_i]

        if robot:
            send_joint_positions(robot, action, JOINT_NAMES)

        frame = capture_frame(cap)
        if save_rollout:
            rollout_frames.append(frame)

        elapsed = time.time() - t_start
        time.sleep(max(0.0, dt - elapsed))

    print("Execution complete.")

    if save_rollout and rollout_frames:
        save_rollout.parent.mkdir(parents=True, exist_ok=True)
        import imageio
        imageio.mimsave(str(save_rollout), rollout_frames, fps=int(execution_hz))
        print(f"Saved rollout to {save_rollout}")


# ---------------------------------------------------------------------------
# Receding-horizon execution
# ---------------------------------------------------------------------------

def run_receding_horizon(
    pipeline: Video2World2ActionPipeline,
    cap: cv2.VideoCapture,
    robot,
    *,
    img_horizon: int = 5,
    num_execute_actions: int = 4,
    replan_every: int = 4,
    total_steps: int = 60,
    execution_hz: float = 5.0,
    save_rollout: pathlib.Path | None = None,
) -> None:
    """Sliding-window receding horizon control."""
    processed = process_image(capture_frame(cap))
    joints = read_joint_positions(robot) if robot else np.zeros(5, dtype=np.float32)

    image_history: deque = deque([processed] * img_horizon, maxlen=img_horizon)
    joint_history: deque = deque([joints], maxlen=1)

    dt = 1.0 / execution_hz
    action_buffer: np.ndarray | None = None
    buf_idx = 0
    rollout_frames = []

    for step_i in range(total_steps):
        t_start = time.time()

        frame = capture_frame(cap)
        joints = read_joint_positions(robot) if robot else np.zeros(5, dtype=np.float32)
        image_history.append(process_image(frame))
        joint_history.append(joints)

        if save_rollout:
            rollout_frames.append(frame)

        if action_buffer is None or buf_idx >= replan_every:
            print(f"Step {step_i}: replanning …")
            input_vid, state = build_input_tensors(image_history, joint_history)
            action_buffer = query_model(pipeline, input_vid, state)
            buf_idx = 0

        action = action_buffer[buf_idx]
        buf_idx += 1

        if robot:
            send_joint_positions(robot, action, JOINT_NAMES)

        elapsed = time.time() - t_start
        time.sleep(max(0.0, dt - elapsed))

    print("Execution complete.")

    if save_rollout and rollout_frames:
        save_rollout.parent.mkdir(parents=True, exist_ok=True)
        import imageio
        imageio.mimsave(str(save_rollout), rollout_frames, fps=int(execution_hz))
        print(f"Saved rollout to {save_rollout}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, help="Hydra experiment name registered in world2action.py")
    p.add_argument("--video-model-path", required=True)
    p.add_argument("--action-model-path", required=True)
    p.add_argument("--dataset-stats", type=pathlib.Path, required=True,
                   help="JSON file with dataset statistics (produced during training)")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--robot-port", default=None,
                   help="Serial port for SO-101 follower. Omit to run without a robot (dry-run).")
    p.add_argument("--robot-id", default="follower")
    p.add_argument("--mode", choices=["open_loop", "receding"], default="open_loop")
    p.add_argument("--num-execute-actions", type=int, default=15)
    p.add_argument("--replan-every", type=int, default=4,
                   help="[receding mode] steps between model queries")
    p.add_argument("--total-steps", type=int, default=60,
                   help="[receding mode] total number of control steps")
    p.add_argument("--execution-hz", type=float, default=5.0)
    p.add_argument("--save-rollout", type=pathlib.Path, default=None,
                   help="If set, save rollout video to this path")
    args = p.parse_args()

    print("Loading pipeline …")
    pipeline = load_pipeline(
        experiment_name=args.experiment,
        video_model_path=args.video_model_path,
        action_model_path=args.action_model_path,
        dataset_stats_path=args.dataset_stats,
    )
    print("Pipeline loaded.")

    robot = None
    if args.robot_port:
        make_robot_from_config, _ = try_import_lerobot()
        if make_robot_from_config is None:
            raise ImportError("lerobot not found. Install it in mission-impushible/ with `uv sync`.")
        from lerobot.robots.so_follower.configuration_so_follower import SoFollowerConfig
        robot_cfg = SoFollowerConfig(port=args.robot_port, id=args.robot_id)
        robot = make_robot_from_config(robot_cfg)
        robot.connect(calibrate=False)
        print("Robot connected.")

    cap = open_camera(args.camera_index)
    print(f"Camera {args.camera_index} open at {IMG_W}×{IMG_H}.")

    try:
        if args.mode == "open_loop":
            run_open_loop(
                pipeline, cap, robot,
                num_execute_actions=args.num_execute_actions,
                execution_hz=args.execution_hz,
                save_rollout=args.save_rollout,
            )
        else:
            run_receding_horizon(
                pipeline, cap, robot,
                num_execute_actions=args.num_execute_actions,
                replan_every=args.replan_every,
                total_steps=args.total_steps,
                execution_hz=args.execution_hz,
                save_rollout=args.save_rollout,
            )
    finally:
        cap.release()
        if robot:
            robot.disconnect()


if __name__ == "__main__":
    main()
