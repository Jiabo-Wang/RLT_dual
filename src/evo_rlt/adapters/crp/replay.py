#!/usr/bin/env python

"""
Replay a recorded CRP episode on the real arms.

    evo-rlt-crp-replay --root data/crp_dual/0814_teleop_full/record_teleop_full_191728 \
        --repo-id local/record_teleop_full_191728 --episode 0

Not ``lerobot-replay`` because the CRP cell needs two things it does not have:

* a window to press green START. ``connect()`` switches work mode and powers the
  servos, which stops a teach program that was already running, so the program has
  to be (re)started after connect and before the first action -- otherwise nothing
  consumes the GP registers and the replay silently moves nothing.
* a first-frame jump guard. The dataset stores absolute cartesian GP poses, so
  replaying frame 0 commands the arm straight to wherever the recording started.
  From an arbitrary current pose that is a full-speed move across the workspace.

The action vector is used exactly as recorded: ``send_action`` consumes the same
14 keys the dataset stores, so there is no re-mapping step to get wrong.
"""

from __future__ import annotations

import argparse
import logging
import math
import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from .arm_dual import CRPArmDual
from .config_arm_dual import CRPArmDualConfig
from .teleop_config import TeleoperateDualCRPConfig
from .teleop_loop import wait_before_gp_align

logger = logging.getLogger(__name__)

_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def _action_to_dict(names: list[str], values) -> dict[str, float]:
    return {name: float(v) for name, v in zip(names, values, strict=True)}


def _first_frame_jump_mm(robot: CRPArmDual, action: dict[str, float]) -> dict[str, float]:
    """Distance from each arm's current TCP to the pose frame 0 would command."""
    out: dict[str, float] = {}
    for label, read in (("left", robot.read_end_pose_first), ("right", robot.read_end_pose_second)):
        try:
            tcp = read()
        except Exception as exc:
            logger.warning("[%s] TCP read failed, cannot check jump: %s", label, exc)
            continue
        target = [action[f"{label}_{a}.pos"] for a in _AXES]
        out[label] = math.dist(tcp[:3], target[:3])
    return out


def replay(args: argparse.Namespace) -> None:
    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    fps = args.fps or ds.fps
    names = list(ds.meta.features["action"]["names"])

    from_idx = ds.meta.episodes["dataset_from_index"][args.episode]
    to_idx = ds.meta.episodes["dataset_to_index"][args.episode]
    actions = ds.hf_dataset.select_columns("action")[from_idx:to_idx]["action"]
    logger.info(
        "Episode %d: %d frames @ %g fps (%.1fs), action keys: %s",
        args.episode, len(actions), fps, len(actions) / fps, names[:3] + ["..."],
    )

    robot = CRPArmDual(CRPArmDualConfig(ip1=args.ip1, ip2=args.ip2, cameras={}))
    robot.connect()
    try:
        wait_before_gp_align(delay_s=args.start_delay_s)

        first = _action_to_dict(names, actions[0])
        jumps = _first_frame_jump_mm(robot, first)
        for label, dist in jumps.items():
            logger.info("[%s] frame 0 is %.0f mm from the current TCP", label, dist)
        too_far = {k: v for k, v in jumps.items() if v > args.max_jump_mm}
        if too_far and not args.allow_jump:
            raise SystemExit(
                f"Refusing to replay: {too_far} exceed --max-jump-mm={args.max_jump_mm}. "
                "Jog the arms near the episode's start pose, or pass --allow-jump if the "
                "path is known to be clear."
            )

        for i, values in enumerate(actions):
            t0 = time.perf_counter()
            robot.send_action(_action_to_dict(names, values))
            if i % int(fps) == 0:
                logger.info("frame %d/%d", i, len(actions))
            precise_sleep(max(1 / fps - (time.perf_counter() - t0), 0.0))
        logger.info("Replay finished: %d frames", len(actions))
    finally:
        robot.disconnect()


def main() -> None:
    init_logging()
    p = argparse.ArgumentParser(description="Replay a recorded episode on the CRP arms.")
    p.add_argument("--root", required=True, help="dataset root directory")
    p.add_argument("--repo-id", required=True, help="e.g. local/record_teleop_full_191728")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--fps", type=float, default=None, help="default: the dataset's own fps")
    cfg = TeleoperateDualCRPConfig()
    p.add_argument("--ip1", default=cfg.robot.ip1)
    p.add_argument("--ip2", default=cfg.robot.ip2)
    p.add_argument(
        "--start-delay-s", type=float, default=cfg.gp_align_delay_s,
        help="countdown before the first action, to press green START",
    )
    p.add_argument(
        "--max-jump-mm", type=float, default=50.0,
        help="refuse to start if frame 0 is further than this from the current TCP",
    )
    p.add_argument("--allow-jump", action="store_true", help="skip the frame-0 distance check")
    replay(p.parse_args())


if __name__ == "__main__":
    main()
