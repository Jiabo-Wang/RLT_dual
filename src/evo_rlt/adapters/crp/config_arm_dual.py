#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CLI / config for dual CRP over one SDK session (``robot.type=crp_arm_dual``).

``ip1`` / ``ip2``: left and right controller addresses.
Used by ``lerobot-crp-tele-dual`` and ``lerobot-crp-record-dual``.
"""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("crp_arm_dual")
@dataclass
class CRPArmDualConfig(RobotConfig):
    """
    Dual CRP over one CrpRobotPy session: ``connect`` + ``connect_second``; GP via
    ``set_GPs`` / ``set_GPs_second``.
    """

    ip1: str = "192.168.0.100"  # left / set_GPs
    ip2: str = "192.168.0.101"  # right / set_GPs_second
    retry_times: int = 3
    connect_settle_s: float = 0.25  # pause before connect_second (s)
    # When True, ``connect()`` skips ``servo_power_on`` (single enable after UI probe subprocess).
    defer_servo_power_on: bool = False
    # Seed GJ10/GJ20 with current joints after connect.
    #
    # Off: this cell drives the arms through GP (cartesian) only, so the GJ registers
    # have no consumer and writing them is pure side effect on a controller that is
    # running a GP program. It is also the prime suspect for the right teach program
    # exiting the moment it is started -- the countdown warning in teleop_loop already
    # told operators to check "GJ seed ... right GJ20 ok=False" when that happens.
    # Turn on only for a teach-pendant GJ program, and verify on hardware.
    init_gj_on_connect: bool = False
    gj_register_left: int = 10
    gj_register_right: int = 20
    gj_trajectory_group_size: int = 5
    # GP (cartesian) registers ``send_action`` writes to. A separate register space
    # from ``gj_register_*`` above, which happens to use the same numbers on this
    # cell -- teleop drives GP10/GP20 while ``connect`` seeds GJ10/GJ20.
    gp_register_left: int = 10
    gp_register_right: int = 20
    # Seed GP10/GP20 with the current TCP during connect, before the cameras.
    #
    # The GP registers outlive the process. A teach program left running acts on
    # whatever is in them the moment the SDK connects, so an arm drives back to the
    # previous session's pose with no operator input at all -- and camera connect
    # gives it seconds in which to do so. Seeding with the pose the arm already holds
    # cannot itself move anything.
    #
    # This was off for a while, on the theory that an extra GP write regressed teleop
    # to "gripper follows, arms do not". That symptom turned out to be the teach
    # program not running at all (the gripper is driven by UI50 registers, which the
    # controller services regardless of any program), so the write was never the
    # cause.
    init_gp_on_connect: bool = True

    # Go-home target per arm, as a 6-element TCP [x, y, z, roll, pitch, yaw] in the
    # same user frame read_end_pose_user() reports. Recorded by
    # ``evo-rlt-crp-set-home`` from wherever the arms are standing, so it is a pose
    # the arms demonstrably reach rather than a joint zero -- the CRP's J6 in
    # particular raises a speed fault when driven to zero from an arbitrary pose
    # (README_crp_dual.md, section 9). None disables go-home for that arm.
    home_tcp_left: list[float] | None = None
    home_tcp_right: list[float] | None = None
    # Opening (UI50, 0-255, 0 closed) to command each gripper before a run starts.
    # ``None`` leaves the gripper where it is, which costs the first observation.
    #
    # ``get_observation`` reports ``_last_ui50``, the value this process last *wrote*,
    # because reading the real opening (UI56) is a ~43 ms SDK round trip that does not
    # fit the control period. Until something writes, that mirror is ``None`` and the
    # observation falls back to ``0.0`` -- which is not "unknown" to a policy, it is
    # "fully closed". Teleop never notices: the leader's gripper position drives UI50
    # from the very first frame. A policy has no leader, so it reads the fallback,
    # and since the state channel is just the previous action echoed back, a policy
    # trained on that channel copies it and clamps the gripper shut.
    #
    # Seeding by commanding is the cheap half of the fix: it makes the state known
    # because we chose it. Reading it instead would be more faithful -- ``ui_probe``
    # already streams UI50 -- but that subprocess has to connect before
    # ``servo_power_on`` (see ``apply_ui_probe_connect_policy``), so wiring it into
    # the record path means reordering connect, not adding a call.
    #
    # Pick a value the policy saw at episode start in its training data, not simply
    # "wide open": the seed lands in the first observation, and an opening the
    # demonstrations never began from is just a different way to be out of
    # distribution.
    init_gripper_ui50_left: int | None = None
    init_gripper_ui50_right: int | None = None
    gp_trajectory_group_size: int = 5
    # Largest translation ``send_action`` will command in one call, in mm.
    #
    # A GP target is an absolute pose, so a caller that hands over something far from
    # where the arm currently is gets a full-speed move across the workspace. Teleop
    # and the recording action mapper each cap their own output, but a policy writes
    # straight to ``send_action`` with nothing in between -- deployment was the one
    # path with no limit at all. Capping here covers every caller, including replay.
    #
    # Matches the teleop default (``gp_position_step_mm``). Raise it only after
    # measuring what the arm can actually track in one control period; 0 disables the
    # cap and should be reserved for bench work with the workspace clear.
    max_gp_step_mm: float = 50.0
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Cameras already maintain one background capture thread each. Consume those
    # buffers via async_read instead of forcing three sequential fresh hardware
    # reads (~56 ms measured). This timeout only applies when no frame is buffered.
    camera_async_read_timeout_ms: int = 100
    # Retries after transient camera failures (same idea as ``crp_arm``).
    camera_read_retries: int = 2
    # Retries after a transient joint read failure ("getCurrentJoint failed for second
    # robot"). The controller refuses the occasional read while it is busy servicing a
    # motion command, and without a retry that single refusal propagates out of
    # get_observation() and aborts the whole episode -- losing every frame recorded so
    # far. Costs nothing on the normal path; a retry only happens after a failure.
    joint_read_retries: int = 3
