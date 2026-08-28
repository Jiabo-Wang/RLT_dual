"""`f` must end a failed episode during RLT recording.

Plain policy-driven recording is an evaluation rollout, so a bad episode is
re-recorded with the left arrow and the failure key stays unbound. RLT recording
inverts that: failures are training data, and the warmup gate counts them
(OnlineRLConfig.min_warmup_failures).

`u` already flushes a failed *critical phase* into the replay buffer, but it
deliberately leaves the episode recording. Without `f` the only ways to end a
failed episode were `s` (which labels it a success) or waiting out
episode_time_s -- and go-home, which fires at episode end, was unreachable on
failure.
"""

import inspect

import pytest

from evo_rlt.adapters.lerobot.record import backend


def binding_source() -> str:
    src = inspect.getsource(backend.record)
    start = src.index("bind_failure_key =")
    return src[start:src.index("\n", start)]


class TestBindFailureKey:
    def test_rlt_recording_binds_it(self):
        assert "rlt_active" in binding_source()

    def test_plain_policy_rollout_still_does_not(self):
        # `policy is None` must survive as the other half of the condition, so an
        # evaluation rollout keeps re-record-with-left-arrow as its only route.
        assert "policy is None" in binding_source()

    @pytest.mark.parametrize(
        "policy_is_none, rlt_active, expected",
        [
            (False, True, True),    # online RL / RLT recording
            (False, False, False),  # plain policy rollout (evaluation)
            (True, False, True),    # pure teleop collection
            (True, True, True),     # teleop with RLT labelling
        ],
    )
    def test_truth_table(self, policy_is_none, rlt_active, expected):
        assert (policy_is_none or rlt_active) is expected


class TestKeyTable:
    """The binding dict is built from duplicate keys, so which event wins matters."""

    @staticmethod
    def _bindings(*, rlt_active=True, policy_is_none=False, enable_outcome=True):
        bind_failure = policy_is_none or rlt_active
        bind_outcome = enable_outcome
        pairs = [
            ("space", "toggle_intervention"),
            ("i", "toggle_left_intervention"),
            ("o", "toggle_right_intervention"),
            (None, "toggle_critical_phase"),
            ("s" if bind_outcome else None, "episode_success"),
            ("f" if bind_outcome and bind_failure else None, "episode_failure"),
            (None, "cp_mark_success"),
            (None, "cp_mark_failure"),
            ("r", "start_rl_phase"),
            ("u", "mark_rl_phase_failure"),
            ("s" if rlt_active else None, "end_phase_success"),
            ("f" if rlt_active and bind_failure else None, "end_phase_failure"),
            ("m", "mark_rl_milestone"),
        ]
        return {key: event for key, event in pairs}  # later duplicates win, as in the patch

    def test_f_ends_the_episode_as_failure(self):
        assert self._bindings()["f"] == "end_phase_failure"

    def test_s_still_ends_it_as_success(self):
        assert self._bindings()["s"] == "end_phase_success"

    def test_u_remains_the_critical_phase_failure_key(self):
        """u and f are different granularities and must not collide: u ends the
        attempt and keeps recording, f ends the whole episode."""
        b = self._bindings()
        assert b["u"] == "mark_rl_phase_failure"
        assert b["u"] != b["f"]

    def test_evaluation_rollouts_keep_f_unbound(self):
        assert "f" not in self._bindings(rlt_active=False)
