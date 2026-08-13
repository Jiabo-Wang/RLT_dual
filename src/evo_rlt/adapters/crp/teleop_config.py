"""
Teleop-only config for dual SO101 leader -> dual CRP follower.

Trimmed from the CRP LeRobot fork's ``scripts/crp_gp/config.py``: the dataset /
camera parts are dropped so this module imports nothing beyond vanilla LeRobot.
Recording config lands here later, once teleop is signed off on hardware.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from lerobot.teleoperators.bi_so_leader.config_bi_so_leader import BiSOLeaderConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderConfig
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS

from .config_arm_dual import CRPArmDualConfig
from .mappers.so101_urdf import CrpGPAlignState

JOINT_POS_KEYS: tuple[str, ...] = tuple(
    f"{prefix}_j{i}.pos" for prefix in ("left", "right") for i in range(1, 7)
)
ACTION_UI_KEYS: tuple[str, ...] = ("left_ui50", "right_ui50")


# Matches record/common.py TELEOP_ID, so CRP teleop reuses the same leader calibration
# as the existing SO101 recording flow.
LEADER_ID = "bimanual_leader"

# Addressed by USB serial rather than /dev/ttyACM*, whose numbering follows kernel
# enumeration order and therefore swaps when the arms are replugged in a different
# order. A silently swapped pair means the left leader drives the right CRP arm, which
# is both dangerous and easy to miss; a stale by-id path fails loudly at connect instead.
# Re-derive with: ls -l /dev/serial/by-id/
LEADER_PORT_LEFT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B41533735-if00"
LEADER_PORT_RIGHT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AAF220248-if00"


def default_leader_calibration_dir():
    """``bi_so_leader`` calibration lives one directory up from where it is looked up.

    ``BiSOLeader`` nests two ``SOLeader`` children, and each child auto-resolves its
    calibration to ``teleoperators/so_leader/{id}_{side}.json`` — but
    ``lerobot-calibrate --teleop.type=bi_so_leader`` writes to
    ``teleoperators/bi_so_leader/``. Point at the latter explicitly; returning None
    lets LeRobot fall back to its own default (and then prompt to calibrate).
    """
    d = HF_LEROBOT_CALIBRATION / TELEOPERATORS / "bi_so_leader"
    return d if d.is_dir() else None


def _default_teleop() -> BiSOLeaderConfig:
    return BiSOLeaderConfig(
        left_arm_config=SOLeaderConfig(port=LEADER_PORT_LEFT, use_degrees=True),
        right_arm_config=SOLeaderConfig(port=LEADER_PORT_RIGHT, use_degrees=True),
        calibration_dir=default_leader_calibration_dir(),
        id=LEADER_ID,
    )


def _default_robot_for_teleop() -> CRPArmDualConfig:
    """Teleop: SO101 + CRP only (no cameras — faster connect, no USB contention)."""
    return CRPArmDualConfig(
        ip1="192.168.0.100",
        ip2="192.168.0.101",
        id="crobotp",
        cameras={},
    )


@dataclass
class ArmGPConfig:
    """Per-arm GP / wrist / Z parameters (CLI: ``--left.*`` / ``--right.*``)."""

    gp_index: int = 10
    # SO101 wrist_roll measures as a clean -z rotation of the tool frame, hence -1.0.
    wrist_roll_sign: float = -1.0
    # SO101 J4 (wrist_flex) -> CRP orientation. 0.0 disables it: tool orientation stays
    # locked to whatever the CRP TCP had at GP align (`locked_rpy`), and only wrist_roll
    # spins it about the tool z axis. Same as the CRP fork / the ACT setup.
    # Set to ±1.0 to map it; check wrist_flex_axis first.
    wrist_flex_sign: float = 0.0
    # Tool-frame axis the flex delta rotates about — only consulted when
    # wrist_flex_sign != 0. "y" is what the SO101 URDF actually measures; the fork used
    # "x", which is ~87° off and is why enabling flex there never worked. See
    # mappers.so101_urdf. The SO101->CRP mount rotation is still unmeasured, so
    # (axis, sign) has 4 candidate combinations to try on hardware.
    wrist_flex_axis: str = "y"
    wrist_roll_max_step_deg: float = 12.0
    wrist_flex_max_step_deg: float | None = 12.0
    z_floor_mm: float = 42.0
    z_scale: float = 2.0


@dataclass
class RightArmGPConfig(ArmGPConfig):
    """Right arm defaults (``gp_index=20`` survives partial ``--right.*`` CLI overrides)."""

    gp_index: int = 20
    # Inherited from the CRP fork, which capped the right arm an order of magnitude
    # tighter than the left (2°/1° vs 12°/12°) without recording why. Kept as-is until
    # the asymmetry is explained on hardware — do not relax it by analogy with the left.
    wrist_roll_max_step_deg: float = 2.0
    wrist_flex_max_step_deg: float | None = 1.0


@dataclass
class TeleoperateDualCRPConfig:
    """Shared config for dual-arm CRP teleop loop."""

    teleop: BiSOLeaderConfig = field(default_factory=_default_teleop)
    robot: CRPArmDualConfig = field(default_factory=_default_robot_for_teleop)
    left: ArmGPConfig = field(default_factory=ArmGPConfig)
    right: RightArmGPConfig = field(default_factory=RightArmGPConfig)
    fps: int = 80
    gp_position_step_mm: float = 80.0
    gp_send_fps: int = 50
    # 0 = send gripper on every main-loop tick (uses ``fps``); else cap UI50 rate (Hz).
    gripper_send_fps: int = 0
    # Min SO101 gripper.pos change (0–100) before sending UI50 (~0.39 ≈ one UI step).
    gripper_min_pos_delta: float = 0.35
    gripper_ui_probe: bool = True
    gripper_ui_probe_hz: float = 1.0
    # Pause after servo enable before GP align/init (seconds); log countdown in terminal.
    gp_align_delay_s: float = 5.0


@dataclass
class DualGPTiming:
    last_gp: float = 0.0
    last_gr: float = 0.0


@dataclass
class ArmTeleopState:
    """Per-arm teleop context built once at startup (hot loop reads fields only)."""

    label: str
    gp_align: CrpGPAlignState
    gp_index: int
    max_step_roll_per_loop: float | None
    max_step_flex_per_loop: float | None
    send_gps_fn: Callable[[int, list], bool]
    set_ui_fn: Callable[[int, int], None]
    pos_step_xy_per_loop: float = 0.0
    step_z_mm: float | None = None
    last_sent: list[float] = field(default_factory=list)
    command_gp: list[float] = field(default_factory=list)
    gripper_pos: float | None = None
    last_gripper_pos: float | None = None
    last_ui50: int | None = None
