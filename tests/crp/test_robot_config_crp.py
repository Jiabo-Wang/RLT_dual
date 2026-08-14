"""Guards for loading a CRP dual-arm robot config out of a manifest.

The camera resolver is stubbed throughout: it reads real ``/dev/v4l/by-id`` nodes and
issues V4L2 ioctls, so exercising it here would make these tests pass or fail based on
what happens to be plugged into the machine. What is worth pinning is the branching
around it -- which config class comes back, which side gets which IP, and that the
failure modes are loud, since a manifest that loads "successfully" with the arms
transposed produces a dataset nobody can tell is wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evo_rlt.adapters.lerobot.record import robot_config as rc


@pytest.fixture(autouse=True)
def _stub_camera_resolution(monkeypatch):
    """Make ``port`` resolution identity-like so tests do not need real cameras."""
    monkeypatch.setattr(
        rc, "resolve_camera_port", lambda port: Path(f"/dev/null/{port!r}")
    )


def _manifest(tmp_path: Path, data: dict) -> str:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return str(path)


CAMERAS = [
    {"alias": "top", "port": {"serial": "CP0BB53000A0"}},
    {"alias": "left_wrist", "port": {"serial": "235123073792"}},
    {"alias": "right_wrist", "port": {"serial": "235123075818"}},
]

CRP_ARMS = [
    {"alias": "left_follower", "type": "follower", "kind": "crp", "ip": "192.168.0.100"},
    {"alias": "right_follower", "type": "follower", "kind": "crp", "ip": "192.168.0.101"},
]


def test_crp_manifest_yields_crp_config(tmp_path):
    cfg = rc.load_robot_config_from_json(
        _manifest(tmp_path, {"robot_id": "crp_dual", "arms": CRP_ARMS, "cameras": CAMERAS})
    )
    assert cfg.type == "crp_arm_dual"
    assert cfg.id == "crp_dual"
    assert (cfg.ip1, cfg.ip2) == ("192.168.0.100", "192.168.0.101")


def test_cameras_keep_their_aliases(tmp_path):
    """CRPArmDual has one flat camera dict, so no ``left_``/``right_`` stripping."""
    cfg = rc.load_robot_config_from_json(
        _manifest(tmp_path, {"arms": CRP_ARMS, "cameras": CAMERAS})
    )
    assert sorted(cfg.cameras) == ["left_wrist", "right_wrist", "top"]


def test_left_right_are_not_transposed(tmp_path):
    """A manifest listing right before left must still map left->ip1."""
    cfg = rc.load_robot_config_from_json(
        _manifest(tmp_path, {"arms": list(reversed(CRP_ARMS)), "cameras": []})
    )
    assert cfg.ip1 == "192.168.0.100"
    assert cfg.ip2 == "192.168.0.101"


def test_gp_index_never_lands_on_a_gj_register(tmp_path):
    """GP (cartesian) and GJ (joint) are separate register spaces sharing 10/20 here."""
    arms = [dict(a, gp_index=99) for a in CRP_ARMS]
    cfg = rc.load_robot_config_from_json(_manifest(tmp_path, {"arms": arms, "cameras": []}))
    assert (cfg.gp_register_left, cfg.gp_register_right) == (99, 99)
    assert (cfg.gj_register_left, cfg.gj_register_right) == (10, 20)


def test_gj_register_never_lands_on_a_gp_register(tmp_path):
    arms = [dict(CRP_ARMS[0], gj_register=30), dict(CRP_ARMS[1], gj_register=40)]
    cfg = rc.load_robot_config_from_json(_manifest(tmp_path, {"arms": arms, "cameras": []}))
    assert (cfg.gj_register_left, cfg.gj_register_right) == (30, 40)
    assert (cfg.gp_register_left, cfg.gp_register_right) == (10, 20)


def test_shipped_crp_manifest_maps_gp_registers(tmp_path):
    """The checked-in manifest carries gp_index 10/20; they must reach gp_register_*."""
    cfg = rc.load_robot_config_from_json("configs/crp_dual_manifest.json")
    assert (cfg.gp_register_left, cfg.gp_register_right) == (10, 20)
    assert (cfg.ip1, cfg.ip2) == ("192.168.0.100", "192.168.0.101")


def test_missing_ip_raises(tmp_path):
    arms = [{"alias": "left_follower", "type": "follower", "kind": "crp"}, CRP_ARMS[1]]
    with pytest.raises(ValueError, match="no 'ip'"):
        rc.load_robot_config_from_json(_manifest(tmp_path, {"arms": arms, "cameras": []}))


def test_missing_side_raises(tmp_path):
    with pytest.raises(ValueError, match="No right CRP follower"):
        rc.load_robot_config_from_json(
            _manifest(tmp_path, {"arms": [CRP_ARMS[0]], "cameras": []})
        )


def test_mixed_crp_and_so101_followers_raise(tmp_path):
    """Half-CRP manifests must fail rather than silently dropping the SO101 arm."""
    arms = [CRP_ARMS[0], {"alias": "right_follower", "type": "follower", "port": "/dev/ttyACM2"}]
    with pytest.raises(ValueError, match="mixes CRP and non-CRP"):
        rc.load_robot_config_from_json(_manifest(tmp_path, {"arms": arms, "cameras": []}))


def test_so101_manifest_still_loads_bimanual(tmp_path):
    """The pre-existing SO101 path must not regress."""
    arms = [
        {"alias": "left_follower", "type": "follower", "port": "/dev/ttyACM3"},
        {"alias": "right_follower", "type": "follower", "port": "/dev/ttyACM2"},
    ]
    cfg = rc.load_robot_config_from_json(
        _manifest(tmp_path, {"arms": arms, "cameras": CAMERAS})
    )
    assert cfg.type == "bi_so_follower"
    # ``top`` has no side prefix, so it falls to the right arm by convention.
    assert sorted(cfg.left_arm_config.cameras) == ["wrist"]
    assert sorted(cfg.right_arm_config.cameras) == ["top", "wrist"]
