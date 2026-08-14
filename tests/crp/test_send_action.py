"""Guards for ``CRPArmDual.send_action`` -- the per-frame write the recorder makes.

``record/loop.py`` calls this once per frame, so a mistake here does not crash: it
writes a whole dataset in which the logged action and the pose the controller
received disagree. The checks below are therefore mostly about *which* value lands
in *which* register, not about whether the call succeeds.

The native SDK is replaced with a recorder object, so no controller is needed.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


class _FakeCrp:
    """Records every SDK call ``send_action`` is expected to make."""

    def __init__(self) -> None:
        self.gp_first: list[tuple[int, list]] = []
        self.gp_second: list[tuple[int, list]] = []
        self.ui: list[tuple[str, int, int]] = []
        self.accept = True

    def set_GPs(self, index: int, points: list) -> bool:
        self.gp_first.append((index, points))
        return self.accept

    def set_GPs_second(self, index: int, points: list) -> bool:
        self.gp_second.append((index, points))
        return self.accept

    def set_ui_first(self, index: int, value: int) -> None:
        self.ui.append(("first", index, value))

    def set_ui_second(self, index: int, value: int) -> None:
        self.ui.append(("second", index, value))

    def read_end_pose_user(self) -> tuple[float, ...]:
        return (494.0, 104.0, 298.0, 1.0, 2.0, 3.0)

    def read_end_pose_user_second(self) -> list[float]:
        return [514.0, 105.0, 270.0, 4.0, 5.0, 6.0]

    def is_connected(self) -> bool:
        return True

    def is_connected_second(self) -> bool:
        return True

    def read_joints(self) -> dict[str, float]:
        return {f"j{i}": float(i) for i in range(1, 7)}

    def read_joints_second(self) -> dict[str, float]:
        return {f"j{i}": float(10 + i) for i in range(1, 7)}


@pytest.fixture
def robot(monkeypatch):
    """A ``CRPArmDual`` whose native session is a ``_FakeCrp``."""
    # Stand in for the licensed native extension before arm_dual imports it.
    monkeypatch.setitem(sys.modules, "CrpRobotPy", types.SimpleNamespace(CrpRobotPy=_FakeCrp))
    from evo_rlt.adapters.crp import arm_dual as ad

    monkeypatch.setattr(ad, "ensure_crp_sdk_loaded", lambda: None)
    from evo_rlt.adapters.crp.config_arm_dual import CRPArmDualConfig

    r = ad.CRPArmDual(CRPArmDualConfig(id="test"))
    r._second_connected = True
    return r


def _action(**overrides: float) -> dict[str, Any]:
    act: dict[str, Any] = {}
    for i, arm in enumerate(("left", "right")):
        for j, axis in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
            act[f"{arm}_{axis}.pos"] = 100.0 * i + j
        act[f"{arm}_ui50"] = 50.0 + i
    act.update(overrides)
    return act


def test_action_features_are_gp_pose_plus_gripper(robot):
    assert list(robot.action_features) == [
        "left_x.pos", "left_y.pos", "left_z.pos",
        "left_roll.pos", "left_pitch.pos", "left_yaw.pos",
        "right_x.pos", "right_y.pos", "right_z.pos",
        "right_roll.pos", "right_pitch.pos", "right_yaw.pos",
        "left_ui50", "right_ui50",
    ]


def test_gripper_feedback_registers_appear_nowhere(robot):
    """ui56-58 are not commandable, and reading them per frame does not fit the budget.

    They were declared in ``observation_features`` while ``get_observation`` never
    produced them, which only surfaced as a KeyError once recording actually ran.
    """
    for key in ("left_ui56", "left_ui57", "left_ui58"):
        assert key not in robot.action_features
        assert key not in robot.observation_features


def test_observation_features_match_get_observation(robot):
    """A declared-but-missing key crashes dataset frame building mid-episode."""
    produced = set(robot.get_observation())
    declared = set(robot.observation_features)
    assert declared == produced, f"missing={declared - produced} extra={produced - declared}"


def test_observation_carries_the_last_gripper_command(robot):
    assert robot.get_observation()["left_ui50"] == 0.0

    robot.send_action(_action(left_ui50=77.0, right_ui50=88.0))
    obs = robot.get_observation()

    assert obs["left_ui50"] == 77.0
    assert obs["right_ui50"] == 88.0


def test_each_arm_gets_its_own_pose_and_register(robot):
    robot.send_action(_action())
    crp = robot.crp_arm_robot

    assert crp.gp_first == [(10, [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])]
    assert crp.gp_second == [(20, [[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]])]


def test_registers_follow_config(robot):
    robot.config.gp_register_left = 30
    robot.config.gp_register_right = 40
    robot.send_action(_action())

    assert robot.crp_arm_robot.gp_first[0][0] == 30
    assert robot.crp_arm_robot.gp_second[0][0] == 40


def test_gripper_goes_to_the_matching_arm(robot):
    robot.send_action(_action(left_ui50=12.4, right_ui50=200.6))
    targets = {arm: value for arm, _, value in robot.crp_arm_robot.ui}

    assert targets == {"first": 12, "second": 201}


def test_returned_action_is_what_was_sent(robot):
    sent = robot.send_action(_action(left_ui50=12.4))

    assert sent["left_x.pos"] == 0.0
    assert sent["right_z.pos"] == 102.0
    # Rounded to the integer the UI register actually took, not the float asked for.
    assert sent["left_ui50"] == 12.0
    assert set(sent) == set(robot.action_features)


def test_partial_action_raises_rather_than_freezing_an_arm(robot):
    act = _action()
    del act["right_z.pos"]
    with pytest.raises(ValueError, match="right_z.pos"):
        robot.send_action(act)
    assert robot.crp_arm_robot.gp_first == []


def test_rejected_pose_is_logged_not_swallowed(robot, caplog):
    robot.crp_arm_robot.accept = False
    with caplog.at_level("WARNING"):
        robot.send_action(_action())
    assert "GP rejected" in caplog.text


class TestGpSeed:
    """The GP group must be written whole once, or every later single point is inert."""

    def test_seeds_current_tcp_into_both_registers(self, robot):
        robot.seed_gp_registers_from_current_tcp()
        crp = robot.crp_arm_robot

        assert crp.gp_first == [(10, [[494.0, 104.0, 298.0, 1.0, 2.0, 3.0]] * 5)]
        assert crp.gp_second == [(20, [[514.0, 105.0, 270.0, 4.0, 5.0, 6.0]] * 5)]

    def test_seed_pose_is_where_the_arm_already_is(self, robot):
        """Seeding must not command motion -- it only makes the register live."""
        robot.seed_gp_registers_from_current_tcp()

        seeded = robot.crp_arm_robot.gp_first[0][1][0]
        assert seeded[:3] == list(robot.read_end_pose_first()[:3])

    def test_group_size_follows_config(self, robot):
        robot.config.gp_trajectory_group_size = 3
        robot.seed_gp_registers_from_current_tcp()

        assert len(robot.crp_arm_robot.gp_first[0][1]) == 3

    def test_rejected_seed_is_reported(self, robot, caplog):
        robot.crp_arm_robot.accept = False
        with caplog.at_level("WARNING"):
            ok_left, ok_right = robot.seed_gp_registers_from_current_tcp()

        assert (ok_left, ok_right) == (False, False)
        assert "GP seed" in caplog.text
