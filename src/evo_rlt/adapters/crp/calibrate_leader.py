"""Calibrate one SO101 leader arm into CRP's own calibration slot.

One arm at a time, because the two are calibrated on different days for different
reasons and ``lerobot-calibrate --teleop.type=bi_so_leader`` insists on redoing both.

The port, the id and the directory all come from ``teleop_config`` -- the same module
``evo-rlt-crp-teleop`` reads. That shared source is the point: previously the calibrate
step and the teleop step resolved the calibration path independently, they disagreed
about the directory, and a recalibration silently wrote somewhere teleop never looked.
Deriving both from one place makes "the file I calibrated" and "the file teleop uses"
the same string by construction rather than by convention.

Usage:
    python -m evo_rlt.adapters.crp.calibrate_leader --side left
    python -m evo_rlt.adapters.crp.calibrate_leader --side right
"""

from __future__ import annotations

import argparse
import logging

from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

from .teleop_config import (
    LEADER_ID,
    LEADER_PORT_LEFT,
    LEADER_PORT_RIGHT,
    default_leader_calibration_dir,
)

logger = logging.getLogger(__name__)

PORTS = {"left": LEADER_PORT_LEFT, "right": LEADER_PORT_RIGHT}


def build_leader(side: str) -> SOLeader:
    return SOLeader(
        SOLeaderTeleopConfig(
            port=PORTS[side],
            use_degrees=True,
            id=f"{LEADER_ID}_{side}",
            calibration_dir=default_leader_calibration_dir(),
        )
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side", choices=["left", "right"], required=True)
    p.add_argument(
        "--show-path",
        action="store_true",
        help="print the target file and exit without touching the arm",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)

    leader = build_leader(args.side)
    print(f"arm    : {args.side}")
    print(f"port   : {PORTS[args.side]}")
    print(f"writes : {leader.calibration_fpath}")
    if args.show_path:
        return 0

    leader.connect(calibrate=False)
    try:
        # Writes the servos' EEPROM and the file together, so the two cannot drift --
        # a mismatch is what makes teleop prompt for recalibration on every connect.
        leader.calibrate()
    finally:
        leader.disconnect()

    print(f"\ndone -> {leader.calibration_fpath}")
    print("evo-rlt-crp-teleop reads this exact file; it should no longer prompt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
