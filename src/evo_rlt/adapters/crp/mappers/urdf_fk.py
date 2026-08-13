"""
Minimal URDF forward kinematics — a drop-in replacement for
``lerobot.model.kinematics.RobotKinematics`` limited to the ``forward_kinematics`` call.

Why not use RobotKinematics: it is backed by placo, which drags in pinocchio, coal,
eigenpy and a dozen cmeel-* native wheels (~38 packages), and whose cmeel-boost pin
forces numpy>=2.3 while LeRobot 0.5.1 requires numpy<2.3. That is a lot of machinery
for a serial chain of revolute joints. The CRP follower does its own IK inside the
controller, so the only kinematics this project needs is SO101 *leader* FK: joint
angles in, tool pose out.

Correctness is not taken on faith — ``tests/crp/test_urdf_fk.py`` checks this against
placo over randomized configurations when placo happens to be installed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF ``rpy`` is fixed-axis XYZ, i.e. R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula."""
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    kx, ky, kz = axis / n
    k = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _homogeneous(rot: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = xyz
    return t


def _floats(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not text:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in text.split()], dtype=float)


@dataclass(frozen=True)
class _Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4, parent -> joint frame at zero position
    axis: np.ndarray  # 3-vector in the joint frame
    is_fixed: bool


def _parse_joints(urdf_path: Path) -> dict[str, _Joint]:
    root = ET.parse(urdf_path).getroot()
    joints: dict[str, _Joint] = {}
    for j in root.findall("joint"):
        name = j.get("name") or ""
        jtype = j.get("type") or "fixed"
        parent_el, child_el = j.find("parent"), j.find("child")
        if parent_el is None or child_el is None:
            continue
        o = j.find("origin")
        xyz = _floats(o.get("xyz") if o is not None else None, (0.0, 0.0, 0.0))
        rpy = _floats(o.get("rpy") if o is not None else None, (0.0, 0.0, 0.0))
        a = j.find("axis")
        axis = _floats(a.get("xyz") if a is not None else None, (0.0, 0.0, 1.0))
        joints[name] = _Joint(
            name=name,
            parent=parent_el.get("link") or "",
            child=child_el.get("link") or "",
            origin=_homogeneous(_rpy_to_matrix(*rpy), xyz),
            axis=axis,
            # "continuous" is revolute without limits; both rotate about `axis`.
            is_fixed=jtype not in ("revolute", "continuous"),
        )
    return joints


class UrdfChainFK:
    """Forward kinematics for the serial chain from the URDF root to ``target_frame_name``.

    ``joint_names`` selects which movable joints along that chain are driven by the
    ``forward_kinematics`` argument, in order. Movable joints on the chain that are not
    listed are held at zero — matching how placo's ``RobotKinematics`` treats them.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.urdf_path = Path(urdf_path)
        self.target_frame_name = target_frame_name
        joints = _parse_joints(self.urdf_path)

        by_child = {j.child: j for j in joints.values()}
        if target_frame_name not in by_child:
            raise ValueError(
                f"{target_frame_name!r} is not a child link in {self.urdf_path}; "
                f"available: {sorted(by_child)[:12]}..."
            )

        # Walk child -> parent from the target back to the root, then reverse.
        chain: list[_Joint] = []
        link = target_frame_name
        seen: set[str] = set()
        while link in by_child:
            if link in seen:
                raise ValueError(f"Cycle in URDF chain at link {link!r}")
            seen.add(link)
            j = by_child[link]
            chain.append(j)
            link = j.parent
        self._chain: tuple[_Joint, ...] = tuple(reversed(chain))
        self.root_link = link

        movable = [j.name for j in self._chain if not j.is_fixed]
        self.joint_names: list[str] = list(joint_names) if joint_names is not None else movable
        unknown = [n for n in self.joint_names if n not in movable]
        if unknown:
            raise ValueError(
                f"joint_names {unknown} are not movable joints on the "
                f"{self.root_link} -> {target_frame_name} chain (movable: {movable})"
            )
        self._index_of = {n: i for i, n in enumerate(self.joint_names)}

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """Joint angles in degrees (ordered like ``joint_names``) -> 4x4 tool pose."""
        q = np.asarray(joint_pos_deg, dtype=float).ravel()
        if q.size < len(self.joint_names):
            raise ValueError(
                f"expected at least {len(self.joint_names)} joint values, got {q.size}"
            )
        q_rad = np.deg2rad(q[: len(self.joint_names)])

        t = np.eye(4)
        for j in self._chain:
            t = t @ j.origin
            if j.is_fixed:
                continue
            idx = self._index_of.get(j.name)
            angle = q_rad[idx] if idx is not None else 0.0
            t = t @ _homogeneous(_axis_angle_to_matrix(j.axis, angle), np.zeros(3))
        return t


__all__ = ["UrdfChainFK"]
