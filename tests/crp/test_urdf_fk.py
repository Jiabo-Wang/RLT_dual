"""Guards for the placo-free SO101 forward kinematics.

The numeric baselines here were captured from placo's ``RobotKinematics`` while it was
still a dependency, so they keep pinning the same geometry now that placo is gone.
``test_matches_placo`` re-derives them from placo directly whenever it is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from evo_rlt.adapters.crp.mappers.so101_urdf import (
    SO101_ARM_MOTOR_NAMES,
    get_endpose2Crp_urdf,
    resolve_so101_urdf_path,
)
from evo_rlt.adapters.crp.mappers.urdf_fk import UrdfChainFK

TARGET_FRAME = "gripper_frame_link"


@pytest.fixture(scope="module")
def fk() -> UrdfChainFK:
    return UrdfChainFK(
        resolve_so101_urdf_path(),
        target_frame_name=TARGET_FRAME,
        joint_names=list(SO101_ARM_MOTOR_NAMES),
    )


def test_chain_structure(fk: UrdfChainFK) -> None:
    assert fk.root_link == "base_link"
    assert fk.target_frame_name == TARGET_FRAME
    assert fk.joint_names == list(SO101_ARM_MOTOR_NAMES)


def test_zero_pose_baseline(fk: UrdfChainFK) -> None:
    """Captured from placo; a change here means the URDF or the chain walk moved."""
    gp = get_endpose2Crp_urdf({f"{m}.pos": 0.0 for m in SO101_ARM_MOTOR_NAMES}, kinematics=fk)
    expected = [391.36147, -0.009212, 226.46971, 89.98922, 87.210718, 89.989798]
    np.testing.assert_allclose(gp, expected, atol=1e-5)


def test_output_is_a_rigid_transform(fk: UrdfChainFK) -> None:
    rng = np.random.default_rng(3)
    for _ in range(64):
        t = fk.forward_kinematics(rng.uniform(-180, 180, 5))
        r = t[:3, :3]
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(t[3, :], [0.0, 0.0, 0.0, 1.0], atol=0)


def test_wrist_roll_rotates_about_tool_minus_z(fk: UrdfChainFK) -> None:
    """Why wrist_roll_sign is -1.0: the roll joint drives tool -z, not +z."""
    scipy_rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    rng = np.random.default_rng(11)
    for _ in range(32):
        q = rng.uniform(-90, 90, 5)
        q1 = q.copy()
        q1[4] += 5.0
        rel = fk.forward_kinematics(q)[:3, :3].T @ fk.forward_kinematics(q1)[:3, :3]
        rotvec = scipy_rotation.from_matrix(rel).as_rotvec(degrees=True)
        axis = rotvec / np.linalg.norm(rotvec)
        np.testing.assert_allclose(axis, [0.0, 0.0, -1.0], atol=1e-5)
        assert np.linalg.norm(rotvec) == pytest.approx(5.0, abs=1e-6)


def test_wrist_flex_axis_lies_in_tool_xy_plane(fk: UrdfChainFK) -> None:
    """Why wrist_flex_axis is "y" and not the fork's "x".

    The flex axis stays in the tool XY plane at azimuth 87.21° + wrist_roll — near +y
    at roll 0, roughly orthogonal to +x.
    """
    scipy_rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    for roll in (-60.0, -20.0, 0.0, 20.0, 60.0):
        q = np.array([10.0, -30.0, 40.0, 15.0, roll])
        q1 = q.copy()
        q1[3] += 5.0
        rel = fk.forward_kinematics(q)[:3, :3].T @ fk.forward_kinematics(q1)[:3, :3]
        rotvec = scipy_rotation.from_matrix(rel).as_rotvec(degrees=True)
        axis = rotvec / np.linalg.norm(rotvec)
        assert abs(axis[2]) < 1e-5, "flex axis must stay in the tool XY plane"
        azimuth = np.degrees(np.arctan2(axis[1], axis[0]))
        assert (azimuth - roll) == pytest.approx(87.21, abs=0.01)


def test_rejects_unknown_target_frame() -> None:
    with pytest.raises(ValueError, match="not a child link"):
        UrdfChainFK(resolve_so101_urdf_path(), target_frame_name="no_such_link")


def test_rejects_joint_not_on_chain() -> None:
    # `gripper` is a real URDF joint but hangs off a different branch than the tool frame.
    with pytest.raises(ValueError, match="not movable joints on the"):
        UrdfChainFK(
            resolve_so101_urdf_path(),
            target_frame_name=TARGET_FRAME,
            joint_names=["shoulder_pan", "gripper"],
        )


def test_rejects_too_few_joint_values(fk: UrdfChainFK) -> None:
    with pytest.raises(ValueError, match="expected at least 5 joint values"):
        fk.forward_kinematics(np.zeros(3))


def test_mesh_and_meshless_urdf_agree(fk: UrdfChainFK) -> None:
    """The default URDF drops <mesh> tags; that must not perturb the joint geometry."""
    from pathlib import Path

    mesh_urdf = Path(resolve_so101_urdf_path()).with_name("so101_new_calib.urdf")
    if not mesh_urdf.is_file():
        pytest.skip("mesh-bearing URDF not checked out")
    other = UrdfChainFK(
        mesh_urdf, target_frame_name=TARGET_FRAME, joint_names=list(SO101_ARM_MOTOR_NAMES)
    )
    rng = np.random.default_rng(5)
    for _ in range(100):
        q = rng.uniform(-180, 180, 5)
        np.testing.assert_allclose(
            fk.forward_kinematics(q), other.forward_kinematics(q), atol=1e-12
        )


def test_matches_placo(fk: UrdfChainFK) -> None:
    """Cross-check against the implementation this one replaced, when it is available."""
    pytest.importorskip("placo")
    from lerobot.model.kinematics import RobotKinematics

    ref = RobotKinematics(
        urdf_path=resolve_so101_urdf_path(),
        target_frame_name=TARGET_FRAME,
        joint_names=list(SO101_ARM_MOTOR_NAMES),
    )
    rng = np.random.default_rng(7)
    for _ in range(500):
        q = rng.uniform(-180, 180, 5)
        np.testing.assert_allclose(fk.forward_kinematics(q), ref.forward_kinematics(q), atol=1e-9)
