"""Returning the arms to a recorded pose, one bounded step at a time.

The first version wrote the target straight into GP, which makes the controller
run a single MoveL across the whole distance at its own speed. On an industrial
arm that is a large uncontrolled motion. Motion now goes out the way the teleop
and policy paths send it: one interpolated point per tick, each clamped.

Home is a TCP the arms were physically standing at, not a joint zero -- driving
the CRP's J6 to zero from an arbitrary pose raises a speed fault
(README_crp_dual.md section 9).
"""

import json
import math

import pytest

from evo_rlt.adapters.crp.arm_dual import CRPArmDual
from evo_rlt.adapters.crp.set_home import write_home_poses

HOME = [400.0, 100.0, 300.0, -180.0, 0.0, 90.0]
step = CRPArmDual._home_step


class TestStepIsBounded:
    def test_a_diagonal_move_is_clamped_as_a_vector(self):
        """Clamping each axis separately would let a diagonal travel max_mm*sqrt(3)."""
        nxt, done = step([0.0, 0, 0, 0, 0, 0], [100.0, 100.0, 100.0, 0, 0, 0], 2.0, 1.0)
        assert not done
        assert math.dist([0, 0, 0], nxt[:3]) == pytest.approx(2.0, abs=1e-6)

    def test_rotation_is_clamped_per_axis(self):
        nxt, done = step([0.0] * 6, [0, 0, 0, 45.0, -45.0, 45.0], 2.0, 1.0)
        assert not done
        assert nxt[3:] == pytest.approx([1.0, -1.0, 1.0])

    def test_the_last_step_lands_exactly_and_reports_done(self):
        nxt, done = step([0.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0.5], 2.0, 1.0)
        assert done
        assert nxt == pytest.approx([1.0, 0, 0, 0, 0, 0.5])

    def test_translation_done_but_rotation_not_is_not_done(self):
        _nxt, done = step([0.0] * 6, [0.5, 0, 0, 0, 0, 30.0], 2.0, 1.0)
        assert not done

    def test_rotation_takes_the_short_way_round(self):
        """-179 -> +179 is 2 degrees apart, not 358."""
        nxt, done = step([0, 0, 0, -179.0, 0, 0], [0, 0, 0, 179.0, 0, 0], 2.0, 1.0)
        assert not done
        assert nxt[3] == pytest.approx(-180.0)

    def test_arrival_lands_on_the_recorded_numbers_exactly(self):
        """Stepping the short way from 170 to -180 arrives at +180 -- the same
        orientation, but every pose in this dataset is written as -180."""
        nxt, done = step([0, 0, 0, 179.5, 0, 0], [0, 0, 0, -180.0, 0, 0], 2.0, 1.0)
        assert done
        assert nxt[3] == -180.0

    def test_already_there_is_done_without_moving(self):
        nxt, done = step(list(HOME), list(HOME), 2.0, 1.0)
        assert done and nxt == pytest.approx(HOME)

    def test_a_full_run_converges_and_never_oversteps(self):
        pose = [500.0, 0.0, 100.0, 170.0, 10.0, 0.0]
        for _ in range(2000):
            nxt, done = step(pose, HOME, 2.0, 1.0)
            assert math.dist(pose[:3], nxt[:3]) <= 2.0 + 1e-6
            pose = nxt
            if done:
                break
        assert done
        assert pose == pytest.approx(HOME, abs=1e-6)


class FakeRobot:
    # go_home() reaches the interpolator through self, so the stub needs it too.
    _home_step = staticmethod(CRPArmDual._home_step)

    def __init__(self, left=HOME, right=HOME, start=None, gap_limit=500.0):
        self._sdk_connected = True
        self.sent: list[tuple] = []
        # 50 mm and 5 degrees out, so a 2 mm / 1 deg step needs many ticks --
        # a start already within one step would not exercise the streaming at all.
        self.start = start or [450.0, 100.0, 300.0, -175.0, 0.0, 90.0]
        self.config = type("C", (), {
            "home_tcp_left": left, "home_tcp_right": right,
            "gp_register_left": 10, "gp_register_right": 20,
            "home_max_step_mm": 2.0, "home_max_step_deg": 1.0,
            "home_tick_hz": 10000.0, "home_max_distance_mm": gap_limit,
            "home_timeout_s": 10.0,
        })()
        outer = self

        class CRP:
            def read_end_pose_user(self):
                return list(outer.start)

            def read_end_pose_user_second(self):
                return list(outer.start)

        self.crp_arm_robot = CRP()

    def send_dual_gp_stream(self, *, left_gp, right_gp, gp_index_left, gp_index_right):
        self.sent.append((left_gp, right_gp))
        return (True if left_gp else None), (True if right_gp else None)


def go_home(robot):
    return CRPArmDual.go_home(robot)


class TestGoHome:
    def test_it_streams_many_small_points_not_one_jump(self):
        robot = FakeRobot()
        assert go_home(robot) is True
        assert len(robot.sent) > 1, "a single write is the unguarded MoveL this replaced"
        prev = robot.start
        for left_gp, _ in robot.sent:
            assert math.dist(prev[:3], left_gp[:3]) <= 2.0 + 1e-6
            prev = left_gp
        assert robot.sent[-1][0] == pytest.approx(HOME)

    def test_a_far_start_is_refused_outright(self):
        """Creeping 2 mm at a time toward a pose that does not belong to this setup
        is not a safe way to discover the mistake."""
        robot = FakeRobot(start=[1200.0, 100.0, 300.0, -180.0, 0.0, 90.0])
        assert go_home(robot) is False
        assert robot.sent == [], "must not move at all"

    def test_the_limit_is_configurable(self):
        robot = FakeRobot(start=[1200.0, 100.0, 300.0, -180.0, 0.0, 90.0], gap_limit=2000.0)
        assert go_home(robot) is True

    def test_no_home_recorded_is_a_silent_no_op(self):
        robot = FakeRobot(left=None, right=None)
        assert go_home(robot) is False
        assert robot.sent == []

    def test_a_malformed_pose_refuses_to_move(self):
        robot = FakeRobot(left=[1.0, 2.0, 3.0])
        assert go_home(robot) is False
        assert robot.sent == []

    def test_a_rejected_write_stops_the_run(self):
        robot = FakeRobot()
        robot.send_dual_gp_stream = lambda **kw: (False, False)
        assert go_home(robot) is False

    def test_it_refuses_to_move_a_disconnected_robot(self):
        from lerobot.utils.errors import DeviceNotConnectedError

        robot = FakeRobot()
        robot._sdk_connected = False
        with pytest.raises(DeviceNotConnectedError):
            go_home(robot)
        assert robot.sent == []


class TestManifestWriting:
    def test_only_home_tcp_changes(self, tmp_path):
        original = {
            "robot_id": "crp_dual",
            "arms": [
                {"alias": "left_follower", "type": "follower", "ip": "192.168.0.100"},
                {"alias": "left_leader", "type": "leader", "port": "/dev/x"},
            ],
        }
        path = tmp_path / "m.json"
        path.write_text(json.dumps(original))
        write_home_poses(path, {"left_follower": HOME})
        written = json.loads(path.read_text())
        assert "home_tcp" in written["arms"][0]
        assert "home_tcp" not in written["arms"][1]
        written["arms"][0].pop("home_tcp")
        assert written == original
