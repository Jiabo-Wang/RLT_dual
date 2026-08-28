"""Camera configs must pin the V4L2 backend, not leave OpenCV to choose.

LeRobot's OpenCVCameraConfig defaults to CAP_ANY. On this host that picks a
backend whose cap.set(CAP_PROP_FRAME_WIDTH) returns False even when the width it
reads back is already the requested one -- and _validate_width_and_height treats
that False as fatal:

    failed to set capture_width=640 (actual_width=640, width_success=False)

Measured 2026-08-28 on the same nodes at the same moment: 0/3 cameras connect
under CAP_ANY, 3/3 under V4L2. These are USB UVC devices on Linux, so V4L2 is
what actually drives them either way.
"""

import pytest

from evo_rlt.adapters.lerobot.record.common import _camera_dict, build_flat_camera_config

CV2_CAP_V4L2 = 200


@pytest.fixture
def fake_resolve(monkeypatch):
    monkeypatch.setattr(
        "evo_rlt.adapters.lerobot.record.camera_resolve.resolve_camera_port",
        lambda port: "/dev/video42",
    )


def test_the_backend_matches_opencvs_own_v4l2_constant():
    cv2 = pytest.importorskip("cv2")
    assert CV2_CAP_V4L2 == cv2.CAP_V4L2


def test_lerobots_enum_accepts_the_value_we_emit():
    from lerobot.cameras.configs import Cv2Backends

    assert Cv2Backends(CV2_CAP_V4L2) is Cv2Backends.V4L2


def test_every_camera_gets_the_backend(fake_resolve):
    cams = [{"alias": a, "port": {"serial": "x"}} for a in ("top", "left_wrist", "right_wrist")]
    built = build_flat_camera_config(cams)
    assert set(built) == {"top", "left_wrist", "right_wrist"}
    for name, cfg in built.items():
        assert cfg["backend"] == CV2_CAP_V4L2, name


def test_a_manifest_can_still_override_it(fake_resolve):
    # A future non-UVC camera might legitimately need a different backend.
    cfg = _camera_dict({"alias": "top", "port": {"serial": "x"}, "backend": 0})
    assert cfg["backend"] == 0


def test_the_config_survives_json_round_trip(fake_resolve):
    # It is serialised into --robot.cameras=, so it has to be plain JSON.
    import json

    built = build_flat_camera_config([{"alias": "top", "port": {"serial": "x"}}])
    assert json.loads(json.dumps(built))["top"]["backend"] == CV2_CAP_V4L2


def test_lerobot_builds_a_camera_from_what_we_emit(fake_resolve):
    """The dict is consumed by LeRobot's config machinery, so the key must exist there."""
    from lerobot.cameras.opencv import OpenCVCameraConfig

    built = build_flat_camera_config([{"alias": "top", "port": {"serial": "x"}}])["top"]
    built.pop("type")
    cfg = OpenCVCameraConfig(**built)
    assert int(cfg.backend) == CV2_CAP_V4L2
