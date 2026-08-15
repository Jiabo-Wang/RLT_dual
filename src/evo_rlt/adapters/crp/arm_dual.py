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
Dual CRP robot driver for GP teleoperation (``lerobot-crp-tele-dual``).

One ``CrpRobotPy`` session, two controllers:
  - ip1 / ``first``: ``send_GPs_first``, ``read_end_pose_first``, UI via ``_set_ui("first", ...)``
  - ip2 / ``second``: ``send_GPs_second``, ``read_end_pose_second``, UI via ``_set_ui("second", ...)``

SDK calls are serialized with a lock. TCP reads are for startup alignment only; do not poll
``get_observation`` / ``read_end_pose_*`` in a tight teleop loop (dual-session instability).
Gripper: UI50 open; UI51–54 motion limits; avoid reading UI56–58 from Python.
"""

import logging
import math
import threading
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.robots.robot import Robot
from .config_arm_dual import CRPArmDualConfig
from .crp_ui import (
    GRIPPER_UI_ACC,
    GRIPPER_UI_DCC,
    GRIPPER_UI_OPEN,
    GRIPPER_UI_SPEED,
    GRIPPER_UI_TORQUE,
    GRIPPER_UI_TRIGGER,
)
from .sdk import (
    ensure_crp_sdk_ld_path,
    ensure_crp_sdk_loaded,
    register_vendor_dual_process_exit,
)

logger = logging.getLogger(__name__)

_BIMANUAL_JOINT_KEYS: tuple[str, ...] = tuple(
    f"{prefix}_j{i}.pos" for prefix in ("left", "right") for i in range(1, 7)
)
_BIMANUAL_UI_KEYS: tuple[str, ...] = tuple(
    f"{arm}_ui{idx}" for arm in ("left", "right") for idx in (50, 56, 57, 58)
)

# Action space: the cartesian GP pose per arm plus the gripper command.
#
# This mirrors what teleop actually puts on the wire (leader FK -> cartesian -> GP),
# so a recorded action is byte-for-byte the command the controller received. Logging
# joint angles instead would mean storing something the loop never sent, and would
# cost an extra ``read_joints`` per frame -- ~42 ms against a ~16 fps budget.
#
# Units follow the SDK: xyz in millimetres, rpy in degrees.
_GP_COMPONENTS: tuple[str, ...] = ("x", "y", "z", "roll", "pitch", "yaw")
_BIMANUAL_GP_KEYS: tuple[str, ...] = tuple(
    f"{arm}_{axis}.pos" for arm in ("left", "right") for axis in _GP_COMPONENTS
)
# ui50 is the only commandable UI register; 56-58 are gripper feedback and belong
# to the observation, not the action.
_BIMANUAL_UI_ACTION_KEYS: tuple[str, ...] = ("left_ui50", "right_ui50")


class CRPArmDual(Robot):
    """One CrpRobotPy session, two controllers (ip1 + ip2)."""

    config_class = CRPArmDualConfig
    name = "crp_arm_dual"

    def __init__(self, config: CRPArmDualConfig):
        super().__init__(config)
        self.config = config
        ensure_crp_sdk_loaded()
        from CrpRobotPy import CrpRobotPy  # noqa: PLC0415

        self.crp_arm_robot = CrpRobotPy()
        self._second_connected = False
        self._sdk_lock = threading.Lock()
        # Last ui50 actually written per arm; surfaced in the observation (see _ui_ft).
        # None means "never written", which forces the first send_action to issue the
        # write even when the commanded value happens to be 0 -- the gripper's opening
        # at power-on is not known to us, so 0 cannot be assumed as the starting point.
        self._last_ui50: dict[str, float | None] = {"left": None, "right": None}
        # Last GP pose commanded per arm; the reference _cap_gp_step clamps against.
        self._last_gp_sent: dict[str, list[float] | None] = {"left": None, "right": None}
        # How often a joint read had to be retried; reported once at disconnect so a
        # controller that is merely flaky is distinguishable from one that is failing.
        self._joint_read_retry_count = 0
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _joints_ft(self) -> dict[str, type]:
        return dict.fromkeys(_BIMANUAL_JOINT_KEYS, float)

    @property
    def _ui_ft(self) -> dict[str, type]:
        """Gripper state in the observation: the last ui50 command, not a fresh read.

        ui56-58 (position / speed / torque feedback) are the real measurement, but
        each UI read is a ~43 ms SDK round trip -- six of them per frame does not fit
        a 62.5 ms budget. They are reachable through the ``crp_getui_probe``
        subprocess at 1 Hz, which teleop starts and the record path currently does
        not. Until that is wired up, the observation carries the value the gripper
        was last told to go to, which costs nothing and still lets a policy see the
        commanded open/close state.
        """
        return dict.fromkeys(_BIMANUAL_UI_ACTION_KEYS, float)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._joints_ft, **self._ui_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **dict.fromkeys(_BIMANUAL_GP_KEYS, float),
            **dict.fromkeys(_BIMANUAL_UI_ACTION_KEYS, float),
        }

    @property
    def is_connected(self) -> bool:
        crp = self.crp_arm_robot
        if not crp.is_connected():
            return False
        if hasattr(crp, "is_connected_second") and callable(crp.is_connected_second):
            second = bool(crp.is_connected_second())
        else:
            second = self._second_connected
        cameras_ok = all(cam.is_connected for cam in self.cameras.values()) if self.cameras else True
        return second and cameras_ok

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        crp = self.crp_arm_robot
        self._second_connected = False
        self._crp_sdk_released = False

        if not crp.connect(self.config.ip1, self.config.retry_times):
            raise RuntimeError(f"connect(ip1={self.config.ip1}) failed.")

        settle = max(0.0, float(self.config.connect_settle_s))
        if settle > 0:
            time.sleep(settle)

        if not hasattr(crp, "connect_second"):
            try:
                crp.disconnect()
            except Exception:
                pass
            raise RuntimeError(
                "CrpRobotPy missing connect_second; rebuild from ~/python_C++/CrpRobotPy"
            )

        ensure_crp_sdk_ld_path()

        if not crp.connect_second(self.config.ip2, self.config.retry_times):
            try:
                crp.disconnect()
            except Exception:
                pass
            raise RuntimeError(
                f"connect_second(ip2={self.config.ip2}) failed; first arm disconnected"
            )

        self._second_connected = True

        from CrpRobotPy import RobotMode  # noqa: PLC0415

        # Auto mode first: servoPowerOn() in Manual is "servo ready" only; switching to Auto
        # afterward drops enable on the teach pendant (servo on → off).
        if not crp.switch_work_mode(RobotMode.Auto):
            self._disconnect_crp_after_failed_connect(crp)
            raise RuntimeError("switch_work_mode(Auto) failed after dual connect.")

        if self.config.defer_servo_power_on:
            logger.info(
                "%s deferring servo_power_on (UI probe subprocess will connect first).",
                self,
            )
        else:
            try:
                if not crp.servo_power_on():
                    raise RuntimeError("servo_power_on returned false")
            except Exception as exc:
                self._disconnect_crp_after_failed_connect(crp)
                raise RuntimeError(
                    "servo_power_on failed after dual connect (binding powers both arms in Auto)."
                ) from exc

        connected_cams: list = []
        try:
            for cam in self.cameras.values():
                cam.connect()
                connected_cams.append(cam)
        except Exception as exc:
            for cam in connected_cams:
                try:
                    cam.disconnect()
                except Exception:
                    pass
            self._disconnect_crp_after_failed_connect(crp)
            raise ConnectionError(
                f"{self} camera connect failed (arms were disconnected). "
                "Run `lerobot-find-cameras` and update --robot.cameras (Orbbec / RealSense serials)."
            ) from exc

        self.configure()
        if self.config.init_gj_on_connect:
            self.seed_gj_registers_from_current_pose()
        if self.config.init_gp_on_connect:
            self.seed_gp_registers_from_current_tcp()
        logger.info("%s connected.", self)

    def seed_gp_registers_from_current_tcp(self) -> tuple[bool, bool]:
        """Write the current TCP into GP10/GP20 as a full group, so GP writes take effect.

        The controller consumes GP as a group and does not act on a partially written
        one: a lone ``set_GPs`` point is accepted, returns true, and produces no
        motion. That failure is silent and reads exactly like a teach program that was
        never started, so seeding is done once at connect rather than left to callers.

        Seeding with the pose the arm is *already* at means this cannot move anything
        -- it only makes the register live. This mirrors ``lock_gps_to_current_tcp``
        on the teleop path.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        gs = max(1, int(self.config.gp_trajectory_group_size))
        results: list[bool] = []
        for label, register, read_tcp, send in (
            ("left", self.config.gp_register_left, self.read_end_pose_first, self.send_GPs_first),
            (
                "right",
                self.config.gp_register_right,
                self.read_end_pose_second,
                self.send_GPs_second,
            ),
        ):
            try:
                tcp = [float(v) for v in read_tcp()]
            except Exception as exc:
                logger.warning("%s GP seed: TCP read failed for %s (%s)", self, label, exc)
                results.append(False)
                continue
            ok = bool(send(int(register), [list(tcp) for _ in range(gs)]))
            if not ok:
                logger.warning(
                    "%s GP seed: set_GPs(GP%s) rejected for %s at (%.0f, %.0f, %.0f)",
                    self, register, label, tcp[0], tcp[1], tcp[2],
                )
            results.append(ok)

        logger.info(
            "%s GP seed on connect: left GP%s ok=%s | right GP%s ok=%s",
            self,
            self.config.gp_register_left,
            results[0],
            self.config.gp_register_right,
            results[1],
        )
        return results[0], results[1]

    def seed_gj_registers_from_current_pose(
        self, *, seed_left: bool = True, seed_right: bool = True
    ) -> tuple[bool, bool]:
        """Write current joint pose into GJ10/GJ20 (5× duplicate rows) for teach-pendant GJ programs."""
        from evo_rlt.adapters.crp.traj_processor import TrajectoryProcessor

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        gs = int(self.config.gj_trajectory_group_size)
        gj_l = int(self.config.gj_register_left)
        gj_r = int(self.config.gj_register_right)
        traj = TrajectoryProcessor(max_points=1, max_joints=gs)
        crp = self.crp_arm_robot

        ok_l = False
        ok_r = False
        with self._sdk_lock:
            if seed_left:
                logger.info("%s GJ seed: reading left joints ...", self)
                left_raw = crp.read_joints()
                left = [float(left_raw[f"j{i}"]) for i in range(1, 7)]
                left_mat = traj.init_matrix(left, gs)
                logger.info("%s GJ seed: set_GJs(GJ%s) left ...", self, gj_l)
                ok_l = bool(crp.set_GJs(gj_l, left_mat))
            else:
                ok_l = True

            if seed_right:
                try:
                    logger.info("%s GJ seed: reading right joints ...", self)
                    right_raw = crp.read_joints_second()
                    if isinstance(right_raw, list):
                        right = [float(v) for v in right_raw[:6]]
                    else:
                        right = [float(right_raw[f"j{i}"]) for i in range(1, 7)]
                    right_mat = traj.init_matrix(right, gs)
                    logger.info("%s GJ seed: set_GJs_second(GJ%s) right ...", self, gj_r)
                    ok_r = bool(crp.set_GJs_second(gj_r, right_mat))
                    if hasattr(crp, "is_connected_second") and callable(crp.is_connected_second):
                        if not crp.is_connected_second():
                            logger.error(
                                "%s ip2 session lost after set_GJs_second — do not START teach program",
                                self,
                            )
                            ok_r = False
                except Exception as exc:
                    logger.warning(
                        "%s GJ seed right (GJ%s) failed: %s — stop right teach program and retry",
                        self,
                        gj_r,
                        exc,
                    )
            else:
                ok_r = True

        logger.info(
            "%s GJ seed on connect: left GJ%s ok=%s | right GJ%s ok=%s",
            self,
            gj_l,
            ok_l,
            gj_r,
            ok_r,
        )
        return ok_l, ok_r

    def ensure_servo_power_on(self) -> None:
        """Enable servo once after deferred connect or steps that drop cabinet enable."""
        if getattr(self, "_crp_sdk_released", False) or not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        with self._sdk_lock:
            if not self.crp_arm_robot.servo_power_on():
                logger.warning("%s servo_power_on returned false during ensure.", self)
                return
        logger.info("%s servo power on (ensure).", self)

    def is_servo_on(self) -> bool:
        """True when the controller reports servo enable (teach-pendant / SDK)."""
        if getattr(self, "_crp_sdk_released", False) or not self.is_connected:
            return False
        fn = getattr(self.crp_arm_robot, "is_servo_on", None)
        if fn is None:
            return True
        with self._sdk_lock:
            return bool(fn())

    def _disconnect_crp_after_failed_connect(self, crp: Any) -> None:
        if hasattr(crp, "disconnect_second"):
            try:
                crp.disconnect_second()
            except Exception:
                pass
        try:
            crp.servo_power_off()
        except Exception:
            pass
        try:
            if hasattr(crp, "is_connected") and crp.is_connected():
                crp.disconnect()
        except Exception:
            pass
        self._second_connected = False

    def read_crp_joints(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not hasattr(self.crp_arm_robot, "read_joints_second"):
            raise NotImplementedError(
                "read_joints_second missing; rebuild from ~/python_C++/CrpRobotPy"
            )
        n_retries = max(0, int(self.config.joint_read_retries))
        for attempt in range(n_retries + 1):
            try:
                with self._sdk_lock:
                    left = self.crp_arm_robot.read_joints()
                    right = self.crp_arm_robot.read_joints_second()
                break
            except RuntimeError as exc:
                # Both arms are re-read on retry rather than just the one that failed:
                # the pair has to come from the same instant to be a usable state.
                if attempt >= n_retries:
                    raise RuntimeError(
                        f"{self} joint read failed {n_retries + 1}x in a row ({exc}). "
                        "The controller is refusing reads, not just busy."
                    ) from exc
                self._joint_read_retry_count += 1
                logger.warning(
                    "%s joint read failed (%s/%s): %s; retrying",
                    self, attempt + 1, n_retries + 1, exc,
                )
                time.sleep(0.005)
        if isinstance(right, list):
            right = {f"j{i + 1}": float(v) for i, v in enumerate(right)}
        out: dict[str, float] = {}
        out.update({f"left_{joint}.pos": float(val) for joint, val in left.items()})
        out.update({f"right_{joint}.pos": float(val) for joint, val in right.items()})
        return out

    def disconnect(self) -> None:
        if getattr(self, "_crp_sdk_released", False):
            return

        crp = self.crp_arm_robot
        had_second = self._second_connected

        try:
            if hasattr(crp, "is_connected_second") and crp.is_connected_second():
                crp.disconnect_second()
            if hasattr(crp, "is_connected") and crp.is_connected():
                crp.disconnect()
        except Exception as exc:
            logger.warning("%s CRP disconnect: %s", self, exc)

        self._second_connected = False
        self._crp_sdk_released = True

        if had_second:
            register_vendor_dual_process_exit()

        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception as exc:
                logger.warning("camera disconnect: %s", exc)

        if self._joint_read_retry_count:
            logger.info(
                "%s joint reads retried %d time(s) this session.",
                self, self._joint_read_retry_count,
            )
        logger.info("%s disconnected.", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = self.read_crp_joints()
        obs_dict.update(
            {f"{arm}_ui50": (0.0 if v is None else v) for arm, v in self._last_ui50.items()}
        )
        n_retries = max(0, int(self.config.camera_read_retries))
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            for attempt in range(n_retries + 1):
                try:
                    obs_dict[cam_key] = cam.read()
                    break
                except RuntimeError:
                    if attempt >= n_retries:
                        raise
                    logger.warning(
                        "%s camera %r read failed (%s/%s); retrying",
                        self,
                        cam_key,
                        attempt + 1,
                        n_retries + 1,
                    )
                    time.sleep(0.005)
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug("%s read %s: %.1fms", self, cam_key, dt_ms)
        return obs_dict

    def set_GJs_first(self, start_index: int, joints_matrix: list[list[float]]) -> bool:
        with self._sdk_lock:
            return bool(self.crp_arm_robot.set_GJs(int(start_index), joints_matrix))

    def set_GJs_second(self, start_index: int, joints_matrix: list[list[float]]) -> bool:
        """Write GJ on ip2 via native ``CrpRobotPy.set_GJs_second``."""
        fn = getattr(self.crp_arm_robot, "set_GJs_second", None)
        if fn is None:
            raise NotImplementedError(
                "set_GJs_second missing; rebuild CrpRobotPy from ~/python_C++/CrpRobotPy"
            )
        with self._sdk_lock:
            return bool(fn(int(start_index), joints_matrix))

    def movej_first(self, joints: list[float]) -> bool:
        with self._sdk_lock:
            self.crp_arm_robot.movej([float(v) for v in joints])
        return True

    def _cap_gp_step(self, label: str, target: list[float]) -> list[float]:
        """Clamp one arm's commanded translation to ``max_gp_step_mm`` per call.

        The reference is the previous commanded pose, seeded on the first call from
        the arm's measured TCP so an opening command cannot jump either. Orientation
        passes through: a rotation cannot fling the arm across the cell the way a
        position step can, and teleop's own cap (``gp_position_step``) is likewise
        translation-only, so this keeps the two paths behaving the same.

        Clamping rather than rejecting is deliberate: a policy that briefly asks for
        something far away should be dragged toward it at a bounded rate, not have the
        arm freeze mid-episode.
        """
        cap = float(self.config.max_gp_step_mm)
        if cap <= 0:
            return target

        prev = self._last_gp_sent.get(label)
        if prev is None:
            try:
                reader = self.read_end_pose_first if label == "left" else self.read_end_pose_second
                prev = reader()
            except Exception as exc:
                # No reference to clamp against; take the target and let the next call
                # be bounded rather than blocking the episode on a transient read.
                logger.warning("%s [%s] GP cap: TCP read failed (%s); first target uncapped", self, label, exc)
                self._last_gp_sent[label] = list(target)
                return target

        dist = math.dist(prev[:3], target[:3])
        if dist <= cap:
            out = list(target)
        else:
            scale = cap / dist
            out = [prev[i] + (target[i] - prev[i]) * scale for i in range(3)] + list(target[3:])
            logger.warning(
                "%s [%s] GP step %.0f mm exceeds max_gp_step_mm=%.0f; advancing toward "
                "(%.0f, %.0f, %.0f) instead of jumping to (%.0f, %.0f, %.0f)",
                self, label, dist, cap, out[0], out[1], out[2], target[0], target[1], target[2],
            )
        self._last_gp_sent[label] = out
        return out

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Write one cartesian GP point and one gripper command per arm.

        Called once per frame by ``record/loop.py``. Deliberately *not* a movej: the
        teach-pendant program consumes GP registers, and issuing motion through a
        different mechanism would fight it.

        Both arms' GP writes share one SDK lock acquisition so a UI poll cannot land
        between them and split the pair across controller cycles -- the same reason
        ``send_gp_tick`` exists for teleop.

        Missing keys are an error rather than a hold-last-value: a partially
        populated action means the caller and ``action_features`` disagree, and
        silently freezing one arm mid-episode corrupts the recording invisibly.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        missing = [k for k in (*_BIMANUAL_GP_KEYS, *_BIMANUAL_UI_ACTION_KEYS) if k not in action]
        if missing:
            raise ValueError(
                f"{self} send_action missing {len(missing)} key(s): {missing[:6]}"
                f"{'...' if len(missing) > 6 else ''}"
            )

        poses = {
            arm: self._cap_gp_step(
                arm, [float(action[f"{arm}_{axis}.pos"]) for axis in _GP_COMPONENTS]
            )
            for arm in ("left", "right")
        }

        with self._sdk_lock:
            crp = self.crp_arm_robot
            ok_left = bool(crp.set_GPs(self.config.gp_register_left, [poses["left"]]))
            ok_right = bool(crp.set_GPs_second(self.config.gp_register_right, [poses["right"]]))

        for label, ok, pose in (("left", ok_left, poses["left"]), ("right", ok_right, poses["right"])):
            if not ok:
                # Accepted-vs-executed is not observable here; a rejection is the one
                # failure the SDK does report, so do not let it pass silently.
                logger.warning(
                    "%s GP rejected (controller / IK); %s pose=%.1f,%.1f,%.1f",
                    self, label, pose[0], pose[1], pose[2],
                )

        sent: dict[str, Any] = {
            f"{arm}_{axis}.pos": poses[arm][i]
            for arm in ("left", "right")
            for i, axis in enumerate(_GP_COMPONENTS)
        }
        # Only write UI50 when the commanded opening actually changed. Each _set_ui is
        # its own blocking SDK round trip and, unlike the GP pair above, the two do not
        # share a controller communication window -- writing both every frame cost
        # ~137 ms on top of the 46 ms GP write and pinned recording at 5 Hz against a
        # configured 16. A gripper is idle for most of an episode, so this is nearly
        # always free. Teleop has always gated the same way (_gripper_pos_changed);
        # only this path was missing it.
        # The returned value is the commanded one either way: a skipped write means the
        # controller is already at that opening, so the action is still what was sent.
        for arm, target in (("left", "first"), ("right", "second")):
            ui50 = int(round(float(action[f"{arm}_ui50"])))
            last = self._last_ui50[arm]
            if last is None or ui50 != int(last):
                self._set_ui(target, GRIPPER_UI_OPEN, ui50)
                self._last_ui50[arm] = float(ui50)
            sent[f"{arm}_ui50"] = float(ui50)
        return sent

    def read_end_pose_first(self) -> list[float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        with self._sdk_lock:
            x, y, z, roll, pitch, yaw = self.crp_arm_robot.read_end_pose_user()
        return [float(x), float(y), float(z), float(roll), float(pitch), float(yaw)]

    def read_end_pose_second(self) -> list[float]:
        if not self.is_connected or not self._second_connected:
            raise DeviceNotConnectedError(f"{self} second arm is not connected.")
        crp = self.crp_arm_robot
        if not hasattr(crp, "read_end_pose_user_second"):
            raise NotImplementedError(
                "read_end_pose_user_second missing; rebuild from ~/python_C++/CrpRobotPy"
            )
        with self._sdk_lock:
            raw = crp.read_end_pose_user_second()
        if len(raw) != 6:
            raise RuntimeError(f"read_end_pose_user_second returned len {len(raw)}, expected 6")
        return [float(v) for v in raw]

    def send_GPs_first(self, start_index: int, gp_points: list[list[float]]) -> bool:
        with self._sdk_lock:
            return bool(self.crp_arm_robot.set_GPs(start_index, gp_points))

    def send_GPs_second(self, start_index: int, gp_points: list[list[float]]) -> bool:
        with self._sdk_lock:
            return bool(self.crp_arm_robot.set_GPs_second(start_index, gp_points))

    def send_dual_gp_stream(
        self,
        *,
        left_gp: list[float] | None,
        right_gp: list[float] | None,
        gp_index_left: int,
        gp_index_right: int,
    ) -> tuple[bool | None, bool | None]:
        """One GP point per arm per tick (matches ``send_gp_tick`` / teleop loop)."""
        with self._sdk_lock:
            crp = self.crp_arm_robot
            if not crp.is_connected():
                return None, None
            if hasattr(crp, "is_connected_second") and callable(crp.is_connected_second):
                if not crp.is_connected_second():
                    return None, None
            ok_left: bool | None = None
            ok_right: bool | None = None
            if left_gp is not None:
                ok_left = bool(crp.set_GPs(int(gp_index_left), [list(left_gp)]))
            if right_gp is not None:
                ok_right = bool(crp.set_GPs_second(int(gp_index_right), [list(right_gp)]))
            return ok_left, ok_right

    def send_dual_gp_init(
        self,
        *,
        left_gp: list[float],
        right_gp: list[float],
        gp_index_left: int,
        gp_index_right: int,
        point_count: int = 5,
    ) -> tuple[bool, bool]:
        """Prime GP buffers with ``point_count`` identical poses (teleop episode start)."""
        left_pts = [list(left_gp) for _ in range(max(1, int(point_count)))]
        right_pts = [list(right_gp) for _ in range(max(1, int(point_count)))]
        with self._sdk_lock:
            crp = self.crp_arm_robot
            ok_left = bool(crp.set_GPs(int(gp_index_left), left_pts))
            ok_right = bool(crp.set_GPs_second(int(gp_index_right), right_pts))
        return ok_left, ok_right

    def send_gp_tick(
        self,
        left: Any | None,
        right: Any | None,
    ) -> tuple[bool | None, bool | None]:
        """Send one GP point per arm under a single SDK lock (avoids UI poll races).

        A 20 ms gap between the two writes was tried here, copying the CRP fork's
        ``_GJ_DUAL_ARM_GAP_S`` ("controller processes one GJ bank at a time"). It made
        no difference, and the reason it cannot be the problem is that sending *only*
        the right arm -- primed, constant target, five seconds, every register from
        GP1 to GP40 -- also produced no motion, while the identical code shape moved
        the left arm 25 mm every time. Bank contention would not survive that test.
        """
        with self._sdk_lock:
            crp = self.crp_arm_robot
            if not crp.is_connected():
                return None, None
            if hasattr(crp, "is_connected_second") and callable(crp.is_connected_second):
                if not crp.is_connected_second():
                    return None, None
            ok_left: bool | None = None
            ok_right: bool | None = None
            if left is not None:
                ok_left = bool(crp.set_GPs(left.gp_index, [left.command_gp]))
            if right is not None:
                ok_right = bool(crp.set_GPs_second(right.gp_index, [right.command_gp]))
            return ok_left, ok_right

    def get_ui_count_first(self) -> int | None:
        fn = getattr(self.crp_arm_robot, "get_ui_count_first", None)
        if fn is None:
            return None
        try:
            with self._sdk_lock:
                count = int(fn())
            return count if count >= 0 else None
        except Exception as exc:
            logger.debug("get_ui_count_first failed: %s", exc)
            return None

    def get_ui_count_second(self) -> int | None:
        fn = getattr(self.crp_arm_robot, "get_ui_count_second", None)
        if fn is None:
            return None
        try:
            with self._sdk_lock:
                count = int(fn())
            return count if count >= 0 else None
        except Exception as exc:
            logger.debug("get_ui_count_second failed: %s", exc)
            return None

    def _set_ui(self, arm: str, index: int, value: int) -> bool:
        fn = getattr(self.crp_arm_robot, f"set_ui_{arm}", None)
        if fn is None:
            return False
        try:
            with self._sdk_lock:
                fn(int(index), int(value))
            return True
        except Exception as exc:
            logger.warning("set_ui_%s(%s, %s) failed: %s", arm, index, value, exc)
            return False

    def get_ui(self, arm: str, index: int) -> int:
        """Read UI register on ``first`` (left) or ``second`` (right) session."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        fn = getattr(self.crp_arm_robot, f"get_ui_{arm}", None)
        if fn is None:
            raise NotImplementedError(
                f"get_ui_{arm} missing; rebuild CrpRobotPy from ~/python_C++/CrpRobotPy"
            )
        with self._sdk_lock:
            return int(fn(int(index)))

    def setup_gripper_ui(
        self,
        *,
        speed: int = 255,
        torque: int = 100,
        acc: int = 255,
        dcc: int = 255,
    ) -> None:
        fixed = {
            GRIPPER_UI_SPEED: speed,
            GRIPPER_UI_TORQUE: torque,
            GRIPPER_UI_ACC: acc,
            GRIPPER_UI_DCC: dcc,
        }
        for idx, val in fixed.items():
            self._set_ui("first", idx, val)
            self._set_ui("second", idx, val)

    def set_gripper_open_first(self, open_0_255: int) -> bool:
        return self._set_ui("first", GRIPPER_UI_OPEN, int(open_0_255))

    def set_gripper_open_second(self, open_0_255: int) -> bool:
        return self._set_ui("second", GRIPPER_UI_OPEN, int(open_0_255))

    def set_gripper_trigger_first(self, value: int = 1) -> bool:
        return self._set_ui("first", GRIPPER_UI_TRIGGER, int(value))

    def set_gripper_trigger_second(self, value: int = 1) -> bool:
        return self._set_ui("second", GRIPPER_UI_TRIGGER, int(value))
