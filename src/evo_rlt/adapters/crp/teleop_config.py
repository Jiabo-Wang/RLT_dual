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


# CRP's own leader identity. Deliberately *not* shared with the SO101 flow's
# ``bimanual_leader``: the same physical arms get posed differently for the two cells,
# and a shared id means whichever project calibrated last silently wins. The id also
# becomes the calibration filename, so it has to be unique on its own.
LEADER_ID = "crp_dual_leader"
# Own directory too, so nothing else writes here even if the ids ever collide.
LEADER_CALIBRATION_DIRNAME = "crp_dual_leader"

# Addressed by USB serial rather than /dev/ttyACM*, whose numbering follows kernel
# enumeration order and therefore swaps when the arms are replugged in a different
# order. A silently swapped pair means the left leader drives the right CRP arm, which
# is both dangerous and easy to miss; a stale by-id path fails loudly at connect instead.
# Re-derive with: ls -l /dev/serial/by-id/
LEADER_PORT_LEFT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B41533735-if00"
LEADER_PORT_RIGHT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AAF220248-if00"


def default_leader_calibration_dir():
    """``teleoperators/crp_dual_leader/`` -- CRP's own directory, always.

    Returns the path unconditionally and creates it if missing. Returning ``None``
    when the directory did not exist (the previous behaviour) handed control back to
    LeRobot's default resolution, and ``BiSOLeader``'s nested ``SOLeader`` children
    resolve that to ``teleoperators/so_leader/{id}_{side}.json`` -- a *different*
    directory that other setups also write to. A calibration then landed there,
    teleop kept reading the intended directory, and the recalibration silently did
    nothing. With no fallback branch that cannot recur: the only file this flow can
    read is the only file it can write.
    """
    d = HF_LEROBOT_CALIBRATION / TELEOPERATORS / LEADER_CALIBRATION_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


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
    # Per GP send, not per second. These are the values that were actually in force
    # before fps moved to 16: `_per_loop_step` scales them by ``gp_send_fps / fps``,
    # which was 50/80 = 0.625, so a configured 12.0 acted as 7.5. Raising fps left the
    # numbers alone but multiplied every cap by 1.6, and the left arm -- whose caps are
    # already an order of magnitude looser than the right's -- stopped executing GP
    # while its gripper kept working. Keep ``fps == gp_send_fps`` so the ratio stays 1
    # and these numbers mean what they say.
    wrist_roll_max_step_deg: float = 7.5
    wrist_flex_max_step_deg: float | None = 7.5
    # Lower bound on commanded Z, in CRP base mm. This is the one that limits downward
    # motion — it stops the tool from being driven into the table.
    z_floor_mm: float = 42.0
    # Gain on leader Z travel: 2.0 means the follower rises twice as far as the leader,
    # to make up for the SO101's smaller reach. Inherited from the CRP fork and never
    # measured on this cell, where it doubles every Z target and so doubles the chance
    # of leaving the reachable workspace — and an unreachable GP target is refused
    # silently, with set_GPs still returning true. Back to 1:1 until the reachable Z
    # range here is actually known; raise it deliberately if leader travel is short.
    z_scale: float = 1.0


@dataclass
class RightArmGPConfig(ArmGPConfig):
    """Right arm defaults (``gp_index=20`` survives partial ``--right.*`` CLI overrides)."""

    gp_index: int = 20
    # A sixth of the left arm's cap, inherited from the CRP fork with no recorded
    # reason. Raising it to 7.5 (matching the left arm) does remove the lag that is
    # otherwise felt in the wrist, but the right controller then faults with "J6 axis
    # joint speed exceeds the maximum 250" -- so the fork's asymmetry was buying
    # something after all, even if it never wrote down what.
    #
    # Why the same number is safe on the left and not on the right is still unknown.
    # The commanded rate is identical and well under the limit (7.5 deg per 62.5 ms =
    # 120 deg/s); what differs is that MoveL with T=0 takes its duration from the
    # translation, so the instantaneous rate explodes on frames that rotate while
    # barely moving -- and the right arm is the one doing the fine, near-stationary
    # alignment in this task. Pinning the segment duration in the teach program
    # (T=0.0625 instead of T=0) would fix it at the source and let this go to 7.5.
    wrist_roll_max_step_deg: float = 1.25
    # Not in force: wrist_flex_sign defaults to 0.0, which disables flex mapping
    # entirely. Left at the fork's value until flex is actually turned on.
    wrist_flex_max_step_deg: float | None = 0.625


@dataclass
class TeleoperateDualCRPConfig:
    """Shared config for dual-arm CRP teleop loop."""

    teleop: BiSOLeaderConfig = field(default_factory=_default_teleop)
    robot: CRPArmDualConfig = field(default_factory=_default_robot_for_teleop)
    left: ArmGPConfig = field(default_factory=ArmGPConfig)
    right: RightArmGPConfig = field(default_factory=RightArmGPConfig)
    # 16 Hz (62.5 ms/frame) is the control rate, set by measurement rather than choice.
    # A dual-arm GP send costs ~46 ms: one SDK call blocks ~43 ms waiting on the
    # controller's communication cycle, and the second arm's call rides the same
    # window nearly free (measured 21.5 Hz loops against a 46 ms floor of 21.7 Hz).
    # That leaves ~16 ms per frame for leader reads, FK, cameras and dataset writes.
    # 20 Hz would leave 4 ms and is not achievable; the previous 80/50 defaults were
    # never reached at all -- the loop ran at 16-21 Hz regardless, which silently made
    # the configured fps a lie and would have written wrong dataset timestamps.
    fps: int = 16
    # Also per GP send; 80.0 with the old 0.625 ratio acted as 50.0.
    gp_position_step_mm: float = 50.0
    # Every main-loop tick sends; at 16 Hz there is no headroom for a slower GP rate.
    gp_send_fps: int = 16
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
