"""Record the arms' current TCP as the go-home pose, into the manifest.

    # put the arms where you want them to return to, then:
    evo-rlt-crp-set-home --setup-json configs/crp_dual_manifest.json

Reads ``read_end_pose_user()`` / ``read_end_pose_user_second()`` and writes the
result to each CRP follower's ``home_tcp`` in the manifest. Nothing is commanded,
so this cannot move the arms.

Cameras are deliberately left out of the config: ``CRPArmDual.connect()`` skips
its camera loop when ``cameras`` is empty, and opening three v4l2 nodes to read
two poses would only add ways to fail.

Connecting still runs ``switch_work_mode`` + ``servo_power_on``, which stops a
teach program that is already running -- restart it from the pendant afterwards.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

TCP_LABELS = ("x", "y", "z", "roll", "pitch", "yaw")


def read_current_tcps(manifest: dict) -> dict[str, list[float]]:
    """Connect to the arms, read both TCPs, disconnect. Commands nothing."""
    from evo_rlt.adapters.crp.arm_dual import CRPArmDual
    from evo_rlt.adapters.crp.config_arm_dual import CRPArmDualConfig

    followers = [a for a in manifest["arms"] if a.get("type") == "follower"]
    if len(followers) != 2:
        raise ValueError(f"expected 2 CRP followers in the manifest, found {len(followers)}")
    left, right = followers

    robot = CRPArmDual(
        CRPArmDualConfig(
            id=manifest.get("robot_id", "crp_dual"),
            ip1=left["ip"],
            ip2=right["ip"],
            gp_register_left=int(left.get("gp_index", 10)),
            gp_register_right=int(right.get("gp_index", 20)),
            cameras={},
            # Seeding writes the pose the arm is already at, so it stays still; leaving
            # it on keeps this identical to how every other session connects.
            init_gp_on_connect=True,
        )
    )
    robot.connect()
    try:
        crp = robot.crp_arm_robot
        poses = {
            left["alias"]: [float(v) for v in crp.read_end_pose_user()],
            right["alias"]: [float(v) for v in crp.read_end_pose_user_second()],
        }
    finally:
        robot.disconnect()
    return poses


def write_home_poses(manifest_path: Path, poses: dict[str, list[float]]) -> None:
    """Set ``home_tcp`` on the named followers, leaving the rest of the file alone."""
    manifest = json.loads(manifest_path.read_text())
    for arm in manifest["arms"]:
        if arm.get("alias") in poses:
            arm["home_tcp"] = [round(v, 3) for v in poses[arm["alias"]]]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--setup-json", type=Path,
                    default=Path("configs/crp_dual_manifest.json"))
    ap.add_argument("--dry-run", action="store_true",
                    help="read and print the poses without touching the manifest")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    manifest_path = args.setup_json.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())

    print("连接机械臂读取当前 TCP（不会下发任何动作）...")
    poses = read_current_tcps(manifest)

    print("\n当前位姿:")
    for alias, tcp in poses.items():
        cells = "  ".join(f"{lbl}={v:9.2f}" for lbl, v in zip(TCP_LABELS, tcp))
        print(f"  {alias:16s} {cells}")

    if args.dry_run:
        print("\n--dry-run: manifest 未修改。")
        return 0

    write_home_poses(manifest_path, poses)
    print(f"\n已写入 {manifest_path} 的 home_tcp。")
    print("在线 RL / 录制时加 --go-home-after-episode 即可在每个 episode 结束后回到这里。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
