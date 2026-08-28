"""Returning the arms to a recorded pose between episodes.

Home is a TCP the arms were physically standing at when `evo-rlt-crp-set-home`
ran, not a joint zero: driving the CRP's J6 to zero from an arbitrary pose raises
a speed fault (README_crp_dual.md section 9), and a reachable pose avoids that
entirely.

go_home() is the only call in the CRP adapter that moves the robot of its own
accord, so the tests below pin when it must stay silent.
"""

import json

import pytest

from evo_rlt.adapters.crp.arm_dual import CRPArmDual
from evo_rlt.adapters.crp.set_home import write_home_poses

HOME_L = [494.3, 104.4, 298.1, -180.0, 0.0, 90.0]
HOME_R = [494.3, 104.4, 298.1, -180.0, 0.0, 90.0]


class FakeRobot:
    """Enough of CRPArmDual for go_home(); records what would be sent."""

    def __init__(self, left=HOME_L, right=HOME_R, group=5, connected=True):
        self._sdk_connected = connected
        self.sent: list[tuple[str, int, list]] = []
        self.config = type("C", (), {
            "home_tcp_left": left, "home_tcp_right": right,
            "gp_register_left": 10, "gp_register_right": 20,
            "gp_trajectory_group_size": group,
        })()

    def __str__(self):
        return "FakeRobot"

    def send_GPs_first(self, idx, pts):
        self.sent.append(("left", idx, pts))
        return True

    def send_GPs_second(self, idx, pts):
        self.sent.append(("right", idx, pts))
        return True


def go_home(robot):
    return CRPArmDual.go_home(robot)


class TestGoHome:
    def test_both_arms_are_commanded_to_their_own_pose(self):
        robot = FakeRobot(left=HOME_L, right=[1.0, 2, 3, 4, 5, 6])
        assert go_home(robot) is True
        sent = {arm: (idx, pts) for arm, idx, pts in robot.sent}
        assert sent["left"][0] == 10 and sent["left"][1][0] == HOME_L
        assert sent["right"][0] == 20 and sent["right"][1][0] == [1.0, 2, 3, 4, 5, 6]

    def test_the_pose_is_repeated_to_fill_the_gp_group(self):
        """The controller ignores a partially written GP group -- same reason
        seed_gp_registers_from_current_tcp repeats it."""
        robot = FakeRobot(group=5)
        go_home(robot)
        for _arm, _idx, points in robot.sent:
            assert len(points) == 5
            assert all(p == points[0] for p in points)

    def test_a_manifest_without_home_is_a_silent_no_op(self):
        robot = FakeRobot(left=None, right=None)
        assert go_home(robot) is False
        assert robot.sent == []

    def test_one_arm_configured_still_moves_that_arm(self):
        robot = FakeRobot(left=HOME_L, right=None)
        go_home(robot)
        assert [arm for arm, _, _ in robot.sent] == ["left"]

    def test_a_malformed_pose_is_refused_not_sent(self):
        robot = FakeRobot(left=[1.0, 2.0, 3.0], right=HOME_R)   # 3 values, not 6
        assert go_home(robot) is False
        assert [arm for arm, _, _ in robot.sent] == ["right"]

    def test_it_refuses_to_move_a_disconnected_robot(self):
        from lerobot.utils.errors import DeviceNotConnectedError

        robot = FakeRobot(connected=False)
        with pytest.raises(DeviceNotConnectedError):
            go_home(robot)
        assert robot.sent == []


class TestManifestWriting:
    def test_only_home_tcp_changes(self, tmp_path):
        original = {
            "robot_id": "crp_dual",
            "datasets": {"root": "~/x", "fps": 16},
            "arms": [
                {"alias": "left_follower", "type": "follower", "ip": "192.168.0.100"},
                {"alias": "right_follower", "type": "follower", "ip": "192.168.0.101"},
                {"alias": "left_leader", "type": "leader", "port": "/dev/x"},
            ],
        }
        path = tmp_path / "m.json"
        path.write_text(json.dumps(original))
        write_home_poses(path, {"left_follower": HOME_L, "right_follower": HOME_R})

        written = json.loads(path.read_text())
        for arm in written["arms"]:
            if arm["type"] == "follower":
                assert "home_tcp" in arm
            else:
                assert "home_tcp" not in arm  # leaders have no TCP
        for arm in written["arms"]:
            arm.pop("home_tcp", None)
        assert written == original

    def test_rerunning_overwrites_rather_than_appends(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps(
            {"arms": [{"alias": "left_follower", "type": "follower", "home_tcp": [0] * 6}]}
        ))
        write_home_poses(path, {"left_follower": HOME_L})
        arm = json.loads(path.read_text())["arms"][0]
        assert arm["home_tcp"] == [round(v, 3) for v in HOME_L]


class TestShippedManifest:
    def test_the_crp_manifest_has_a_six_value_pose_per_follower(self):
        manifest = json.loads(open("configs/crp_dual_manifest.json").read())
        followers = [a for a in manifest["arms"] if a.get("type") == "follower"]
        assert len(followers) == 2
        for arm in followers:
            assert len(arm["home_tcp"]) == 6, arm["alias"]
            # roll near -180 is this dataset's documented convention.
            assert abs(abs(arm["home_tcp"][3]) - 180) < 1, arm["alias"]
