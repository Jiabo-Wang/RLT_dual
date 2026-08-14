"""Guards for the argv `evo-rlt-record` builds from a manifest.

``common.py`` is shared with the SO101 recording flow, so these tests carry two
jobs: prove the CRP branch produces a usable robot config, and prove the SO101
branches did not move. Everything CRP-specific is gated on ``kind: "crp"``; a
manifest without it must take the path it took before CRP existed.

Camera resolution is stubbed -- it reads real ``/dev/v4l`` nodes, so otherwise
these would pass or fail on whether the cameras happen to be plugged in.
"""

from __future__ import annotations

import json

import pytest

from evo_rlt.adapters.lerobot.record import common


@pytest.fixture(autouse=True)
def _stub_camera_resolution(monkeypatch):
    monkeypatch.setattr(common, "resolve_camera_port", lambda port: f"/dev/resolved/{port}", raising=False)
    monkeypatch.setattr(
        "evo_rlt.adapters.lerobot.record.camera_resolve.resolve_camera_port",
        lambda port: f"/dev/resolved/{port}",
    )


CAMERAS = [
    {"alias": "top", "port": {"serial": "CP0BB53000A0"}, "fourcc": "YUYV"},
    {"alias": "left_wrist", "port": {"serial": "235123073792"}},
    {"alias": "right_wrist", "port": {"serial": "235123075818"}},
]

CRP_FOLLOWERS = [
    {"alias": "left_follower", "type": "follower", "kind": "crp", "ip": "192.168.0.100", "gp_index": 10},
    {"alias": "right_follower", "type": "follower", "kind": "crp", "ip": "192.168.0.101", "gp_index": 20},
]

SO101_FOLLOWERS = [
    {"alias": "left_follower", "type": "follower", "port": "/dev/ttyACM3"},
    {"alias": "right_follower", "type": "follower", "port": "/dev/ttyACM2"},
]


def _argv(followers, cameras):
    if common.is_crp_setup(followers):
        cams, right = common.build_flat_camera_config(cameras), {}
    else:
        cams, right = common.build_camera_configs(cameras)
    return common.build_robot_argv(followers, cams, right, "/tmp/cal")


def _flag(argv: list[str], name: str) -> str:
    for item in argv:
        if item.startswith(f"{name}="):
            return item.split("=", 1)[1]
    raise AssertionError(f"{name} not in {argv}")


class TestCrpBranch:
    def test_robot_type_and_addresses(self):
        argv = _argv(CRP_FOLLOWERS, CAMERAS)

        assert _flag(argv, "--robot.type") == "crp_arm_dual"
        assert _flag(argv, "--robot.ip1") == "192.168.0.100"
        assert _flag(argv, "--robot.ip2") == "192.168.0.101"

    def test_gp_registers_come_from_the_manifest(self):
        argv = _argv(CRP_FOLLOWERS, CAMERAS)

        assert _flag(argv, "--robot.gp_register_left") == "10"
        assert _flag(argv, "--robot.gp_register_right") == "20"

    def test_top_camera_is_not_dropped(self):
        """The SO101 split table has no entry for ``top``; the CRP path must keep it."""
        argv = _argv(CRP_FOLLOWERS, CAMERAS)
        cams = json.loads(_flag(argv, "--robot.cameras"))

        assert sorted(cams) == ["left_wrist", "right_wrist", "top"]

    def test_camera_ports_are_resolved_not_raw(self):
        argv = _argv(CRP_FOLLOWERS, CAMERAS)
        cams = json.loads(_flag(argv, "--robot.cameras"))

        assert cams["top"]["index_or_path"].startswith("/dev/resolved/")
        assert cams["top"]["fourcc"] == "YUYV"

    def test_no_calibration_dir(self):
        """CRP arms calibrate on the controller, not against a host-side file."""
        argv = _argv(CRP_FOLLOWERS, CAMERAS)

        assert not any(a.startswith("--robot.calibration_dir") for a in argv)

    def test_mixed_followers_raise(self):
        followers = [CRP_FOLLOWERS[0], SO101_FOLLOWERS[1]]
        with pytest.raises(ValueError, match="mixes CRP and non-CRP"):
            common.is_crp_setup(followers)

    def test_robot_type_is_a_registered_choice(self):
        """draccus validates --robot.type against RobotConfig's subclass registry.

        Nothing else on the record path imports the CRP stack, so emitting
        ``--robot.type=crp_arm_dual`` without registering it first is rejected at
        parse time with "invalid choice" -- after the whole session has already
        connected the leaders and staged calibrations.
        """
        from lerobot.robots.config import RobotConfig

        _argv(CRP_FOLLOWERS, CAMERAS)
        assert "crp_arm_dual" in RobotConfig.get_known_choices()


class TestSo101NotRegressed:
    def test_bimanual_argv_unchanged(self):
        argv = _argv(SO101_FOLLOWERS, CAMERAS)

        assert _flag(argv, "--robot.type") == "bi_so_follower"
        assert _flag(argv, "--robot.left_arm_config.port") == "/dev/ttyACM3"
        assert _flag(argv, "--robot.right_arm_config.port") == "/dev/ttyACM2"
        assert _flag(argv, "--robot.calibration_dir") == "/tmp/cal"

    def test_bimanual_still_splits_and_renames_cameras(self):
        argv = _argv(SO101_FOLLOWERS, CAMERAS)
        left = json.loads(_flag(argv, "--robot.left_arm_config.cameras"))
        right = json.loads(_flag(argv, "--robot.right_arm_config.cameras"))

        # ``left_wrist``/``right_wrist`` collapse to ``wrist`` per arm; ``top`` is in
        # neither side's alias table and is dropped -- pre-existing behaviour.
        assert sorted(left) == ["wrist"]
        assert sorted(right) == ["wrist"]

    def test_single_arm_argv_unchanged(self):
        argv = _argv([SO101_FOLLOWERS[0]], CAMERAS)

        assert _flag(argv, "--robot.type") == "so101_follower"
        assert _flag(argv, "--robot.port") == "/dev/ttyACM3"

    def test_setup_without_kind_is_not_crp(self):
        assert common.is_crp_setup(SO101_FOLLOWERS) is False


class TestResolveFps:
    def test_cli_wins(self):
        assert common.resolve_fps(25, {"datasets": {"fps": 16}}) == 25

    def test_manifest_when_cli_omitted(self):
        assert common.resolve_fps(None, {"datasets": {"fps": 16}}) == 16

    def test_falls_back_to_30(self):
        """SO101 manifests carry no datasets.fps and must keep their old default."""
        assert common.resolve_fps(None, {}) == 30
        assert common.resolve_fps(None, {"datasets": {"root": "~/x"}}) == 30
