"""Load robot + camera configuration from a JSON file.

Supports the roboclaw `setup.json` format and converts it into
lerobot `BiSOFollowerConfig` / `SOFollowerRobotConfig` dataclasses.

Camera aliases in the JSON use **final** feature names (e.g. `left_wrist`).
The loader strips the `left_` / `right_` prefix automatically so that
`BiSOFollower`'s auto-prefixing produces the correct observation key.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig, SOFollowerRobotConfig

from .camera_resolve import resolve_camera_port

if TYPE_CHECKING:  # avoid importing the CRP stack (and its native SDK) at module load
    from evo_rlt.adapters.crp.config_arm_dual import CRPArmDualConfig

logger = logging.getLogger(__name__)

# BiSOFollower prepends these prefixes to each arm's features.
_ARM_PREFIXES = {"left": "left_", "right": "right_"}


def _strip_arm_prefix(alias: str, prefix: str) -> str:
    """Strip the arm prefix so BiSOFollower's auto-prefix restores it."""
    if alias.startswith(prefix):
        return alias[len(prefix):]
    return alias


def _find_arm_port(arms: list[dict], side: str) -> str:
    """Find the follower port for the given side from the arms list."""
    side_lower = side.lower()
    for arm in arms:
        alias = arm.get("alias", "").lower()
        arm_type = arm.get("type", "").lower()
        if side_lower in alias and ("follower" in alias or "follower" in arm_type):
            return arm["port"]
    raise ValueError(
        f"Cannot find {side} follower arm in setup.json. "
        f"Available arms: {[a.get('alias') for a in arms]}"
    )


def _build_camera_config(cam: dict) -> OpenCVCameraConfig:
    """Build one camera config, resolving ``port`` to a stable device path.

    ``fourcc`` is forwarded rather than dropped: a depth camera's colour node often
    offers exactly one format (the D405s here are YUYV-only), and letting OpenCV
    negotiate can land on a different one and silently change frame timing.
    """
    return OpenCVCameraConfig(
        index_or_path=resolve_camera_port(cam["port"]),
        fps=cam.get("fps", 30),
        width=cam.get("width", 640),
        height=cam.get("height", 480),
        fourcc=cam.get("fourcc"),
    )


def _assign_camera_to_arm(
    alias: str,
    cam_cfg: OpenCVCameraConfig,
    left_cams: dict[str, OpenCVCameraConfig],
    right_cams: dict[str, OpenCVCameraConfig],
) -> None:
    """Assign a camera to left or right arm based on its alias prefix."""
    for side, prefix in _ARM_PREFIXES.items():
        if alias.startswith(prefix):
            short_name = _strip_arm_prefix(alias, prefix)
            if side == "left":
                left_cams[short_name] = cam_cfg
            else:
                right_cams[short_name] = cam_cfg
            return
    # No prefix match - default to right arm (convention: shared cameras go to right)
    right_cams[alias] = cam_cfg


def _load_crp_dual_config(
    followers: list[dict], cameras: list[dict], robot_id: str | None,
) -> "CRPArmDualConfig":
    """Build a ``CRPArmDualConfig`` from manifest followers marked ``"kind": "crp"``.

    Unlike ``BiSOFollower``, ``CRPArmDual`` holds one flat camera dict and does not
    prefix features per arm, so aliases are used verbatim -- ``top`` stays ``top``
    rather than being forced onto one arm. That also means ``top`` needs no
    ``left_``/``right_`` prefix just to be loadable, which the bimanual SO101 path
    would otherwise require.
    """
    from evo_rlt.adapters.crp.config_arm_dual import CRPArmDualConfig

    def _ip(side: str) -> str:
        for f in followers:
            if side in f.get("alias", "").lower():
                ip = f.get("ip")
                if not ip:
                    raise ValueError(f"CRP follower {f.get('alias')!r} has no 'ip'")
                return str(ip)
        raise ValueError(
            f"No {side} CRP follower in manifest. Found: {[f.get('alias') for f in followers]}"
        )

    cams: dict = {alias: _build_camera_config(c) for c in cameras if (alias := c["alias"])}

    kwargs: dict = {"ip1": _ip("left"), "ip2": _ip("right"), "cameras": cams}
    # GP (cartesian, what send_action writes) and GJ (joint, seeded on connect) are
    # separate register spaces that happen to share the numbers 10/20 on this cell.
    # Keep the manifest keys distinct so the two can never be cross-assigned.
    for manifest_key, left_key, right_key in (
        ("gp_index", "gp_register_left", "gp_register_right"),
        ("gj_register", "gj_register_left", "gj_register_right"),
    ):
        for side, cfg_key in (("left", left_key), ("right", right_key)):
            for f in followers:
                if side in f.get("alias", "").lower() and manifest_key in f:
                    kwargs[cfg_key] = int(f[manifest_key])

    cfg = CRPArmDualConfig(**kwargs)
    if robot_id:
        cfg.id = robot_id
    logger.info(
        "Loaded CRP dual robot config: ip1=%s ip2=%s cameras=%s",
        cfg.ip1, cfg.ip2, sorted(cams),
    )
    return cfg


def _load_single_arm_config(
    follower: dict, cameras: list[dict], robot_id: str | None,
) -> SOFollowerRobotConfig:
    """Build a single-arm SOFollowerRobotConfig. Camera aliases are used as-is
    since there is only one arm, so no left_/right_ prefix stripping applies."""
    cams: dict[str, OpenCVCameraConfig] = {}
    for cam in cameras:
        alias = cam["alias"]
        cams[alias] = _build_camera_config(cam)
    logger.info(
        "Loaded single-arm robot config: port=%s (%d cams)",
        follower["port"], len(cams),
    )
    cfg = SOFollowerRobotConfig(port=follower["port"], cameras=cams)
    if robot_id:
        cfg.id = robot_id
    return cfg


def load_robot_config_from_json(
    path: str | Path,
) -> "BiSOFollowerConfig | SOFollowerRobotConfig | CRPArmDualConfig":
    """Load a robot config from a roboclaw-compatible setup.json.

    Args:
        path: Path to the JSON config file.

    Returns:
        A `CRPArmDualConfig` when the followers are marked `"kind": "crp"`,
        a `SOFollowerRobotConfig` when the manifest lists exactly one follower
        arm, otherwise a fully populated `BiSOFollowerConfig`.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    arms = data.get("arms", [])
    cameras = data.get("cameras", [])
    robot_id = data.get("robot_id", data.get("id"))

    followers = [a for a in arms if "follower" in a.get("type", "").lower()]

    crp_followers = [f for f in followers if f.get("kind", "").lower() == "crp"]
    if crp_followers:
        if len(crp_followers) != len(followers):
            raise ValueError(
                "Manifest mixes CRP and non-CRP followers; one robot config cannot "
                f"cover both. Followers: {[(f.get('alias'), f.get('kind')) for f in followers]}"
            )
        return _load_crp_dual_config(crp_followers, cameras, robot_id)

    if len(followers) == 1:
        return _load_single_arm_config(followers[0], cameras, robot_id)

    left_port = _find_arm_port(arms, "left")
    right_port = _find_arm_port(arms, "right")

    left_cams: dict[str, OpenCVCameraConfig] = {}
    right_cams: dict[str, OpenCVCameraConfig] = {}

    for cam in cameras:
        alias = cam["alias"]
        cam_cfg = _build_camera_config(cam)
        _assign_camera_to_arm(alias, cam_cfg, left_cams, right_cams)

    logger.info(
        "Loaded robot config from %s: left_port=%s (%d cams), right_port=%s (%d cams)",
        path, left_port, len(left_cams), right_port, len(right_cams),
    )

    cfg = BiSOFollowerConfig(
        left_arm_config=SOFollowerConfig(port=left_port, cameras=left_cams),
        right_arm_config=SOFollowerConfig(port=right_port, cameras=right_cams),
    )
    if robot_id:
        cfg.id = robot_id
    return cfg
