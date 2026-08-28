"""Behaviour of the reconstructed critical-phase / intervention trackers.

These modules were lost when LeRobot was reinstalled and were rebuilt from their
call sites, so the inferred semantics are pinned here rather than left implicit.
"""

import json

import pytest

from evo_rlt.adapters.lerobot.record.critical_phase import (
    CriticalPhaseTracker,
    EpisodeIntervalTracker,
)


class TestIntervalBoundaries:
    def test_start_stop_records_a_half_open_interval(self):
        t = EpisodeIntervalTracker(label="X")
        t.on_episode_start(3)
        t.start(10)
        assert t.is_active
        t.stop(25)
        assert not t.is_active
        assert t.get_intervals() == [(3, 10, 25, None)]
        assert len(t) == 1

    def test_a_second_start_is_ignored_not_nested(self):
        t = EpisodeIntervalTracker()
        t.on_episode_start(0)
        t.start(5)
        t.start(9)  # ignored
        t.stop(20)
        assert t.get_intervals() == [(0, 5, 20, None)]

    def test_stop_without_start_is_a_noop(self):
        t = EpisodeIntervalTracker()
        t.on_episode_start(0)
        t.stop(7)
        assert t.get_intervals() == []

    def test_stop_before_start_cannot_produce_negative_length(self):
        t = EpisodeIntervalTracker()
        t.on_episode_start(0)
        t.start(30)
        t.stop(10)
        (_, start, end, _), = t.get_intervals()
        assert end >= start

    def test_multiple_intervals_in_one_episode(self):
        t = EpisodeIntervalTracker()
        t.on_episode_start(1)
        t.start(0); t.stop(5)
        t.start(10); t.stop(12)
        assert t.get_intervals() == [(1, 0, 5, None), (1, 10, 12, None)]


class TestEpisodeLifecycle:
    def test_episode_end_closes_an_open_interval(self):
        t = EpisodeIntervalTracker()
        t.on_episode_start(2)
        t.start(4)
        t.on_episode_end(90)
        assert t.get_intervals() == [(2, 4, 90, None)]

    def test_intervals_are_grouped_by_episode(self):
        t = EpisodeIntervalTracker()
        t.on_episode_start(0); t.start(1); t.stop(2); t.on_episode_end(2)
        t.on_episode_start(1); t.start(3); t.stop(4); t.on_episode_end(4)
        assert t.serialize_episode_intervals(0) == [
            {"start_frame": 1, "end_frame": 2, "outcome": None}
        ]
        assert t.serialize_episode_intervals(1) == [
            {"start_frame": 3, "end_frame": 4, "outcome": None}
        ]

    def test_rerecording_drops_that_episodes_intervals(self):
        t = EpisodeIntervalTracker()
        t.on_episode_start(0); t.start(1); t.stop(9); t.on_episode_end(9)
        t.on_episode_start(1); t.start(2)
        t.discard_episode(1)
        assert t.get_intervals() == [(0, 1, 9, None)]
        assert not t.is_active

    def test_an_interval_left_open_is_never_serialized(self):
        t = EpisodeIntervalTracker()
        t.on_episode_start(0)
        t.start(5)  # never closed -- e.g. the process died
        assert t.get_intervals() == []
        assert t.serialize_episode_intervals(0) == []
        assert len(t) == 1  # still tracked in memory


class TestOutcomeLabelling:
    def test_toggle_opens_then_closes(self):
        t = CriticalPhaseTracker()
        t.on_episode_start(0)
        t.toggle(3)
        assert t.is_active
        t.toggle(11)
        assert not t.is_active
        assert t.get_intervals() == [(0, 3, 11, None)]

    @pytest.mark.parametrize("marker, expected", [("mark_success", "success"),
                                                  ("mark_failure", "failure")])
    def test_marking_while_open_closes_and_labels(self, marker, expected):
        t = CriticalPhaseTracker()
        t.on_episode_start(0)
        t.start(2)
        getattr(t, marker)(8)
        assert t.get_intervals() == [(0, 2, 8, expected)]
        assert not t.is_active

    def test_marking_after_the_phase_closed_still_labels_it(self):
        """loop.py marks the outcome again at episode end, once on_episode_end
        has already closed the phase. That call must not be dropped."""
        t = CriticalPhaseTracker()
        t.on_episode_start(4)
        t.start(1)
        t.on_episode_end(50)
        t.mark_success(50)
        assert t.get_intervals() == [(4, 1, 50, "success")]

    def test_marking_labels_only_the_most_recent_interval(self):
        t = CriticalPhaseTracker()
        t.on_episode_start(0)
        t.start(0); t.stop(5)
        t.start(10); t.stop(20)
        t.mark_failure(20)
        assert t.get_intervals() == [(0, 0, 5, None), (0, 10, 20, "failure")]

    def test_marking_with_nothing_recorded_is_a_noop(self):
        t = CriticalPhaseTracker()
        t.on_episode_start(0)
        t.mark_success(5)
        assert t.get_intervals() == []


class TestAutosave:
    def test_closing_an_interval_writes_the_file(self, tmp_path):
        path = tmp_path / "sub" / "critical_phase_intervals.json"
        t = CriticalPhaseTracker(auto_save_path=path)
        t.on_episode_start(7)
        t.start(3)
        assert not path.exists()  # nothing to save until it closes
        t.mark_success(19)
        assert json.loads(path.read_text()) == [
            {"episode_index": 7, "start_frame": 3, "end_frame": 19, "outcome": "success"}
        ]

    def test_schema_matches_what_backend_writes_at_shutdown(self, tmp_path):
        # backend.py's _save_critical_phase_intervals dumps get_intervals() with
        # exactly these keys; the autosave must not disagree with it.
        path = tmp_path / "x.json"
        t = CriticalPhaseTracker(auto_save_path=path)
        t.on_episode_start(0); t.start(1); t.stop(2)
        from_autosave = json.loads(path.read_text())
        from_backend = [
            {"episode_index": ep, "start_frame": s, "end_frame": e, "outcome": o}
            for ep, s, e, o in t.get_intervals()
        ]
        assert from_autosave == from_backend

    def test_an_unwritable_path_does_not_abort_the_session(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        t = CriticalPhaseTracker(auto_save_path=blocker / "nested" / "x.json")
        t.on_episode_start(0)
        t.start(1)
        t.stop(2)  # must not raise
        assert t.get_intervals() == [(0, 1, 2, None)]


class TestAudioFeedback:
    def test_cues_never_raise_even_without_a_tts_backend(self, monkeypatch):
        import evo_rlt.adapters.lerobot.record.audio_feedback as af

        def boom(*_args, **_kwargs):
            raise RuntimeError("no audio device")

        monkeypatch.setattr("lerobot.utils.utils.log_say", boom)
        monkeypatch.setattr(af, "_warned", False)
        af.say_start()
        af.say_success()
        af.say_failure()

    def test_each_cue_says_something_distinct(self, monkeypatch):
        import evo_rlt.adapters.lerobot.record.audio_feedback as af

        said = []
        monkeypatch.setattr("lerobot.utils.utils.log_say",
                            lambda text, **kw: said.append(text))
        af.say_start(); af.say_success(); af.say_failure()
        assert len(set(said)) == 3
