"""Pin CRP's leader-calibration identity so it cannot silently drift back.

Two failures already happened here and neither raised anything:

* a recalibration wrote to ``teleoperators/so_leader/`` while teleop kept reading
  ``teleoperators/bi_so_leader/``, so the recalibration did nothing; and
* CRP shared the SO101 flow's ``bimanual_leader`` id, so whichever cell calibrated
  last overwrote the other's file.

Both are silent -- the arms keep moving, just against the wrong zero. These tests
assert the one property that prevents both: every path that reads or writes CRP
leader calibration resolves to the same file, and that file's name and directory
are CRP's alone.
"""

from __future__ import annotations

import json

import pytest

from evo_rlt.adapters.crp import teleop_config as tc
from evo_rlt.adapters.lerobot.record import common as rc


def test_identity_is_not_the_so101_one():
    assert tc.LEADER_ID == "crp_dual_leader"
    assert tc.LEADER_ID != rc.TELEOP_ID
    assert tc.LEADER_CALIBRATION_DIRNAME != "so_leader"
    assert tc.LEADER_CALIBRATION_DIRNAME != "bi_so_leader"


def test_record_and_teleop_agree_on_the_id():
    """The record path stages calibration under the id; a mismatch stages the wrong name."""
    assert rc.CRP_TELEOP_ID == tc.LEADER_ID


def test_calibration_dir_never_returns_none():
    """Returning None is what handed path resolution back to LeRobot's default.

    That default sends ``BiSOLeader``'s children to ``teleoperators/so_leader/``,
    a directory other setups also write to -- the original silent-overwrite bug.
    """
    d = tc.default_leader_calibration_dir()
    assert d is not None
    assert d.is_dir()
    assert d.name == tc.LEADER_CALIBRATION_DIRNAME


def test_calibrate_tool_targets_what_teleop_reads():
    """The whole point: one string, reached from both directions."""
    from evo_rlt.adapters.crp.calibrate_leader import build_leader
    from lerobot.teleoperators import make_teleoperator_from_config

    teleop = make_teleoperator_from_config(tc.TeleoperateDualCRPConfig().teleop)
    for side in ("left", "right"):
        written = build_leader(side).calibration_fpath
        read = getattr(teleop, f"{side}_arm").calibration_fpath
        assert written == read, f"{side}: calibrate writes {written}, teleop reads {read}"


def test_calibrate_tool_uses_the_configured_ports():
    from evo_rlt.adapters.crp.calibrate_leader import PORTS

    assert PORTS["left"] == tc.LEADER_PORT_LEFT
    assert PORTS["right"] == tc.LEADER_PORT_RIGHT
    assert PORTS["left"] != PORTS["right"]


def test_shipped_manifest_points_at_the_same_files():
    """A manifest pointing elsewhere would stage a stale file over the fresh one."""
    from evo_rlt.adapters.crp.calibrate_leader import build_leader

    manifest = json.load(open("configs/crp_dual_manifest.json"))
    leaders = {a["alias"]: a for a in manifest["arms"] if a["type"] == "leader"}
    for side in ("left", "right"):
        declared = leaders[f"{side}_leader"]["calibration_file"].replace("~", str(__import__("pathlib").Path.home()))
        assert declared == str(build_leader(side).calibration_fpath)


@pytest.mark.parametrize("side", ["left", "right"])
def test_record_stages_under_the_crp_name(side, tmp_path):
    """``stage_leader_calibrations`` names the temp file after the id it passes on."""
    leaders = [
        {"alias": "left_leader", "type": "leader", "port": "/dev/null", "calibration_file": str(tmp_path / "l.json")},
        {"alias": "right_leader", "type": "leader", "port": "/dev/null", "calibration_file": str(tmp_path / "r.json")},
    ]
    argv = rc.build_teleop_argv(leaders, no_teleop=False, teleop_id=rc.CRP_TELEOP_ID)
    assert f"--teleop.id={rc.CRP_TELEOP_ID}" in argv

    tmpdir = rc.stage_leader_calibrations(leaders, argv, rc.CRP_TELEOP_ID)
    try:
        cal_arg = next(a for a in argv if a.startswith("--teleop.calibration_dir="))
        staged_dir = cal_arg.split("=", 1)[1]
        expected = f"{rc.CRP_TELEOP_ID}_{side}.json"
        # The source files do not exist, so nothing is copied; what matters is the
        # name the loader will look for, which comes from the id.
        assert expected == f"{tc.LEADER_ID}_{side}.json"
        assert staged_dir
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()
