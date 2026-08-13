"""
CRP register-matrix helper, ported verbatim from the CRP LeRobot fork's
``lerobot/tools/TrajProcessor.py``.

``set_GPs`` / ``set_GJs`` take a *matrix* of rows, not a single vector: the
teach-pendant program reads ``group_size`` consecutive registers. ``init_matrix``
builds the "all rows identical" seed matrix; ``write_point`` / ``write_joint``
maintain the sliding window used when streaming a real trajectory.

Do not confuse this with ``mappers.so101_urdf.TrajectoryProcessor``, which is a
thin back-compat alias exposing only ``trajectory_differential`` and carries no
register state.
"""

from __future__ import annotations

import math


class TrajectoryProcessor:
    def __init__(self, max_points: int = 1, max_joints: int = 5):
        if max_points < 1:
            raise ValueError("max_points must be >= 1")
        self.max_points = int(max_points)
        self.max_joints = int(max_joints)

        self._written_once_point = False
        self._written_once_joint = False

        self.points: list[list[float]] = [[0.0] * 6 for _ in range(self.max_points)]
        self.joints: list[list[float]] = [[0.0] * 6 for _ in range(self.max_joints)]

    def trajectory_differential(
        self,
        current_pose: list[float],
        target_pose: list[float],
        step_length: float = 0.1,
    ) -> list[float]:
        """Compute one incremental pose step from current_pose towards target_pose.

        The translation (x,y,z) moves by at most step_length along the straight line.
        Orientation (roll,pitch,yaw) is taken from the target_pose.
        """
        if not (isinstance(current_pose, (list, tuple)) and isinstance(target_pose, (list, tuple))):
            raise ValueError("current_pose and target_pose must be list or tuple of length 6")
        if len(current_pose) != 6 or len(target_pose) != 6:
            raise ValueError("current_pose and target_pose must have length 6: [x,y,z,roll,pitch,yaw]")
        if step_length < 0:
            raise ValueError("step_length must be non-negative")

        cx, cy, cz = float(current_pose[0]), float(current_pose[1]), float(current_pose[2])
        tx, ty, tz = float(target_pose[0]), float(target_pose[1]), float(target_pose[2])

        dx = tx - cx
        dy = ty - cy
        dz = tz - cz

        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        # If already at or within step_length, return the target pose
        if dist <= step_length or dist == 0.0:
            return [float(x) for x in target_pose]

        ux = dx / dist
        uy = dy / dist
        uz = dz / dist

        new_x = cx + ux * step_length
        new_y = cy + uy * step_length
        new_z = cz + uz * step_length

        # Use target orientation
        new_roll = float(target_pose[3])
        new_pitch = float(target_pose[4])
        new_yaw = float(target_pose[5])

        return [round(new_x, 10), round(new_y, 10), round(new_z, 10), new_roll, new_pitch, new_yaw]

    def write_point(self, vec: list[float]) -> None:
        """Write a single 6-element vector into the points history.

        Keeps at most `max_points` latest entries. The buffer order is oldest->newest.
        """
        if not (isinstance(vec, (list, tuple)) and len(vec) == 6):
            raise ValueError("vec must be a list/tuple of length 6")
        point = [float(x) for x in vec]

        if not self._written_once_point:
            # First real write: fill all slots with this point
            self.points = [point[:] for _ in range(self.max_points)]
            self._written_once_point = True
            return

        self.points.append(point)
        if len(self.points) > self.max_points:
            self.points = self.points[-self.max_points :]

    def read_points(self) -> list[list[float]]:
        """Return a shallow copy of the stored points list."""
        return [p[:] for p in self.points]

    def write_joint(self, vec: list[float]) -> None:
        """Write a single 6-element vector into the joints history.

        Keeps at most `max_joints` latest entries. The buffer order is oldest->newest.
        """
        if not (isinstance(vec, (list, tuple)) and len(vec) == 6):
            raise ValueError("vec must be a list/tuple of length 6")
        joint = [float(x) for x in vec]

        if not self._written_once_joint:
            # First real write: fill all slots with this joint
            self.joints = [joint[:] for _ in range(self.max_joints)]
            self._written_once_joint = True
            return

        self.joints.append(joint)
        if len(self.joints) > self.max_joints:
            self.joints = self.joints[-self.max_joints :]

    def read_joints(self) -> list[list[float]]:
        """Return a shallow copy of the stored joints list."""
        return [j[:] for j in self.joints]

    def init_matrix(self, flat: list[float], group_size: int) -> list[list[float]]:
        """Replicate ``flat`` into ``group_size`` identical rows (register seed matrix)."""
        if not isinstance(flat, (list, tuple)):
            raise ValueError("flat must be a list or tuple of numbers representing a single vector")
        if not isinstance(group_size, int) or group_size < 1:
            raise ValueError("group_size must be an int >= 1")

        if len(flat) == 0:
            return []

        row = [float(x) for x in flat]
        return [row[:] for _ in range(group_size)]


__all__ = ["TrajectoryProcessor"]
