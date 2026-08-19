#!/usr/bin/env python

"""
Point the CRP GP registers at where the arms currently are.

    python -m evo_rlt.adapters.crp.lock_gp

The GP registers persist across runs and across power cycles of the PC: whatever
pose was written last is still in there. Start a teach program without overwriting
them and its first ``MoveL`` drives the arm back to wherever the previous session
left off, with no leader input at all.

Teleop and recording both do this on their own at startup. This exists for the times
in between -- after an alarm, after jogging the arm by hand, or just to make sure
before pressing green START.

There is no "clear" to be had: an empty GP is a pose of all zeros, which is the base
origin, and moving there is exactly what must not happen. Writing the arm's own
current pose is the only safe way to neutralise a stale target -- by construction it
cannot move anything.
"""

from __future__ import annotations

import argparse
import logging

from lerobot.utils.utils import init_logging

from .arm_dual import CRPArmDual
from .config_arm_dual import CRPArmDualConfig
from .teleop_config import TeleoperateDualCRPConfig

logger = logging.getLogger(__name__)


def main() -> None:
    init_logging()
    cfg = TeleoperateDualCRPConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--ip1", default=cfg.robot.ip1)
    p.add_argument("--ip2", default=cfg.robot.ip2)
    args = p.parse_args()

    robot = CRPArmDual(CRPArmDualConfig(ip1=args.ip1, ip2=args.ip2, cameras={}))
    robot.connect()
    try:
        for label, read in (
            ("left", robot.read_end_pose_first),
            ("right", robot.read_end_pose_second),
        ):
            try:
                tcp = read()
                logger.info(
                    "[%s] current TCP (%.0f, %.0f, %.0f)", label, tcp[0], tcp[1], tcp[2]
                )
            except Exception as exc:
                logger.warning("[%s] TCP read failed: %s", label, exc)

        ok_left, ok_right = robot.seed_gp_registers_from_current_tcp()
        if ok_left and ok_right:
            logger.info(
                "GP registers now hold the current pose. Safe to press green START -- "
                "the teach program's first move will be a no-op."
            )
        else:
            logger.error(
                "GP write rejected (left ok=%s, right ok=%s). The registers may still "
                "hold an old target; do NOT press START until this succeeds.",
                ok_left,
                ok_right,
            )
            raise SystemExit(1)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
