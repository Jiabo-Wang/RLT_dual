"""Camera connect retries instead of aborting the whole session.

A killed predecessor's v4l2 nodes release slowly. While a node is still held,
VIDIOC_S_FMT fails and OpenCVCamera raises "failed to set capture_width=640
(actual_width=640, width_success=False)" -- note the actual width is already
correct; only cap.set()'s return value is False. Measured: the identical connect
succeeded seconds later with nothing changed. Since the arms are already powered
by this point, waiting beats tearing the session down.
"""

import pytest

from evo_rlt.adapters.crp.arm_dual import CRPArmDual


class FlakyCamera:
    """Fails `fail_times` times, then connects."""

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.attempts = 0
        self.disconnects = 0

    def connect(self):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RuntimeError(
                "OpenCVCamera(/dev/video18) failed to set capture_width=640 "
                "(actual_width=640, width_success=False)."
            )

    def disconnect(self):
        self.disconnects += 1


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch):
    monkeypatch.setattr("evo_rlt.adapters.crp.arm_dual.time.sleep", lambda _s: None)


def _connect(name, cam):
    # Pass the class as `self`: the method only reads the two class-level constants,
    # so this keeps the real values instead of duplicating them in the test.
    return CRPArmDual._connect_camera_with_retry(CRPArmDual, name, cam)


class TestRetry:
    def test_a_camera_that_settles_is_accepted(self):
        cam = FlakyCamera(fail_times=2)
        _connect("top", cam)
        assert cam.attempts == 3

    def test_first_try_success_does_not_sleep_or_disconnect(self):
        cam = FlakyCamera(fail_times=0)
        _connect("top", cam)
        assert (cam.attempts, cam.disconnects) == (1, 0)

    def test_a_genuinely_dead_camera_still_fails(self):
        cam = FlakyCamera(fail_times=99)
        with pytest.raises(RuntimeError, match="did not connect after"):
            _connect("top", cam)
        assert cam.attempts == CRPArmDual._CAMERA_CONNECT_ATTEMPTS

    def test_the_failure_names_the_camera_and_keeps_the_cause(self):
        cam = FlakyCamera(fail_times=99)
        with pytest.raises(RuntimeError) as excinfo:
            _connect("left_wrist", cam)
        assert "left_wrist" in str(excinfo.value)
        # The operator needs the v4l2 message, not just "it failed".
        assert "capture_width" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    def test_each_retry_releases_the_handle_first(self):
        """Retrying without disconnecting would leak a half-open capture."""
        cam = FlakyCamera(fail_times=2)
        _connect("top", cam)
        assert cam.disconnects == 2  # once per failed attempt, not after the success

    def test_the_wait_is_long_enough_for_a_v4l2_release(self):
        # camera_resolve tells operators to allow ~15 s; the retry budget must cover it.
        budget = CRPArmDual._CAMERA_RETRY_DELAY_S * (CRPArmDual._CAMERA_CONNECT_ATTEMPTS - 1)
        assert budget >= 15
