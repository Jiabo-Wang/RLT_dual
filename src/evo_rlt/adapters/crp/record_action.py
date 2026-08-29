"""Map SO101 leader joint actions to CRP GP actions for the recording loop.

``record_loop`` hands whatever the teleoperator produced straight to
``robot.send_action``. For an SO101 leader driving an SO101 follower that works,
because both speak the same joint names. A CRP follower does not: it takes a
cartesian GP pose, and the leader's joints only become one after SO101 forward
kinematics plus the leader-to-CRP frame alignment. That translation exists in
``teleop_loop`` and had no counterpart on the recording path, so the first
recorded frame reached ``send_action`` with fourteen missing keys.

This is a drop-in for ``robot_action_processor`` (default: identity) rather than
logic inside ``CRPArmDual``, so the robot keeps taking exactly the action it
publishes in ``action_features`` and the dataset stores the GP pose that was sent
rather than the leader joints that produced it.

Alignment happens on the first action, not at construction: it needs both the
CRP TCP and a leader pose, and the leader pose is the action. Seeding the offset
from wherever the two happen to be means the follower does not jump on frame one
-- the same guarantee ``teleop_loop`` gets from its GP align countdown.
"""

from __future__ import annotations

import logging
from typing import Any

from .mappers.so101_urdf import (
    SO101_ARM_MOTOR_NAMES,
    get_endpose2Crp_urdf,
    get_wrist_flex_deg,
    get_wrist_roll_deg,
    resolve_so101_urdf_path,
    so101_gripper_pos_to_crp_ui50,
    split_bimanual_so101_action,
)
from .mappers.urdf_fk import UrdfChainFK
from .teleop_config import TeleoperateDualCRPConfig
from .teleop_loop import (
    TARGET_FRAME_NAME,
    _align_arms_at_start,
    _command_gp_for_arm,
    _make_arm_states,
)

logger = logging.getLogger(__name__)

_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


class CrpLeaderActionToGp:
    """Callable pipeline stage: bi_so_leader joint action -> CRP GP action."""

    def __init__(self, robot: Any, cfg: TeleoperateDualCRPConfig | None = None) -> None:
        self.robot = robot
        self.cfg = cfg or TeleoperateDualCRPConfig()
        self.kinematics = UrdfChainFK(
            urdf_path=resolve_so101_urdf_path(),
            target_frame_name=TARGET_FRAME_NAME,
            joint_names=list(SO101_ARM_MOTOR_NAMES),
        )
        self._arms: tuple[Any, Any] | None = None

    def invalidate_alignment(self) -> None:
        """Force a re-align against the follower's TCP on the next leader frame.

        Alignment is what keeps the follower from jumping to the leader's absolute
        pose: the mapping is relative to wherever the follower stood when teleop
        engaged. Anything that moves the follower *outside* this mapper -- go-home
        is the one such path -- leaves that reference pointing at the old pose, and
        the next leader frame commands `stale_reference + leader_delta`, i.e. a jump
        straight back to where the arm was before. Observed on hardware: the arms
        returned home after `f`, then jumped back to the insertion pose with the two
        of them close together.
        """
        if self._arms is not None:
            logger.info("CRP action mapper: alignment invalidated; will re-align "
                        "against the current TCP on the next leader frame.")
        self._arms = None

    def _ensure_aligned(self, arm_actions: tuple[dict, dict]) -> tuple[Any, Any]:
        if self._arms is not None:
            return self._arms
        arms = _make_arm_states(self.cfg, self.robot)
        fk_seeds = tuple(
            get_endpose2Crp_urdf(a, kinematics=self.kinematics) for a in arm_actions
        )
        _align_arms_at_start(self.robot, arms, fk_seeds)
        for arm, act in zip(arms, arm_actions, strict=True):
            arm.gp_align.set_wrist_references(get_wrist_roll_deg(act), get_wrist_flex_deg(act))
        self._arms = arms
        logger.info("CRP action mapper aligned against current TCP; leader will not jump.")
        return arms

    def __call__(self, item: tuple[dict[str, Any], Any]) -> dict[str, Any]:
        action, _obs = item

        # A leader action carries per-joint keys; a GP action is already mapped
        # (replay, or a policy trained on this dataset). Passing the latter through
        # untouched keeps this stage safe to leave installed.
        if any(k.endswith("_x.pos") for k in action):
            return action

        arm_actions = split_bimanual_so101_action(action)
        arms = self._ensure_aligned(arm_actions)

        out: dict[str, Any] = {}
        limit_step = self.cfg.gp_position_step_mm > 0
        for arm, act in zip(arms, arm_actions, strict=True):
            gp6 = _command_gp_for_arm(
                arm,
                get_endpose2Crp_urdf(act, kinematics=self.kinematics),
                act,
                apply_step_cap=limit_step,
            )
            arm.command_gp = list(gp6)
            for axis, value in zip(_AXES, gp6, strict=True):
                out[f"{arm.label}_{axis}.pos"] = float(value)

            gripper = act.get("gripper.pos")
            out[f"{arm.label}_ui50"] = (
                float(so101_gripper_pos_to_crp_ui50(gripper))
                if gripper is not None
                else float(arm.last_ui50 or 0)
            )
        return out
