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
    # Seed GP10/GP20 with the current TCP after connect.
    #
    # Off by default: switching it on regressed teleop to gripper-only -- the arms
    # stopped executing GP while the gripper kept following. Teleop already writes a
    # full GP group twice (``lock_gps_to_current_tcp``, then the align burst), and
    # this added a third write *before* the teach program is started, which the
    # controller evidently does not tolerate. The mechanism is not understood, so the
    # default returns to the sequence teleop is known to work with.
    #
    # Recording paths that call ``send_action`` without going through
    # ``prepare_dual_gp_session`` may still need a seeded group; turn it on
    # explicitly there and verify on hardware before trusting it.
    init_gp_on_connect: bool = False
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
    # Retries after transient ``cam.read()`` failures (same idea as ``crp_arm``).
    camera_read_retries: int = 2
    # Retries after a transient joint read failure ("getCurrentJoint failed for second
    # robot"). The controller refuses the occasional read while it is busy servicing a
    # motion command, and without a retry that single refusal propagates out of
    # get_observation() and aborts the whole episode -- losing every frame recorded so
    # far. Costs nothing on the normal path; a retry only happens after a failure.
    joint_read_retries: int = 3
