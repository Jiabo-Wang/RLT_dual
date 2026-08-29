"""Moving the follower outside the leader mapper must invalidate its alignment.

The mapper is what stops the follower jumping to the leader's absolute pose: the
mapping is relative to wherever the follower stood when teleop engaged, and
_ensure_aligned latches that reference for the whole session.

go-home moves the follower behind the mapper's back. Observed on hardware: after
`f` the arms returned home, then jumped back to the insertion pose -- the two arms
close together over the workpiece -- because the reset window's first leader frame
commanded stale_reference + delta.
"""

import inspect
import types

import pytest

from evo_rlt.adapters.crp.record_action import CrpLeaderActionToGp


class TestInvalidate:
    @staticmethod
    def _mapper():
        m = object.__new__(CrpLeaderActionToGp)
        m._arms = ("left_state", "right_state")   # pretend a previous align latched
        return m

    def test_it_clears_the_latched_reference(self):
        m = self._mapper()
        m.invalidate_alignment()
        assert m._arms is None

    def test_ensure_aligned_would_realign_afterwards(self):
        """_ensure_aligned short-circuits on a non-None _arms, so clearing it is
        exactly what makes the next leader frame re-align."""
        src = inspect.getsource(CrpLeaderActionToGp._ensure_aligned)
        assert "if self._arms is not None" in src
        m = self._mapper()
        assert m._ensure_aligned.__self__._arms is not None
        m.invalidate_alignment()
        assert m._arms is None

    def test_calling_it_before_any_alignment_is_harmless(self):
        m = object.__new__(CrpLeaderActionToGp)
        m._arms = None
        m.invalidate_alignment()
        assert m._arms is None

    def test_it_is_idempotent(self):
        m = self._mapper()
        m.invalidate_alignment()
        m.invalidate_alignment()
        assert m._arms is None


class TestBackendCallsIt:
    def test_go_home_invalidates_even_when_it_fails(self):
        """A partial return leaves the arm somewhere new too, so the invalidation
        has to be in a finally, not on the success path."""
        from evo_rlt.adapters.lerobot.record import backend

        src = inspect.getsource(backend.record)
        start = src.index("def _go_home_if_requested")
        body = src[start:src.index("def _run_reset_loop_if_needed", start)]
        assert "finally:" in body
        assert "invalidate_alignment" in body
        assert body.index("finally:") < body.index("invalidate_alignment")

    def test_a_robot_without_the_mapper_is_tolerated(self):
        """Non-CRP robots keep the default action processor, which has no such method."""
        from evo_rlt.adapters.lerobot.record import backend

        src = inspect.getsource(backend.record)
        assert 'getattr(robot_action_processor, "invalidate_alignment", None)' in src


class TestOrdering:
    def test_invalidation_happens_before_the_reset_window(self):
        """The reset window is the pure-teleop loop whose first frame would jump."""
        from evo_rlt.adapters.lerobot.record import backend

        src = inspect.getsource(backend.record)
        assert src.index("invalidate_alignment") < src.index("def _run_reset_loop_if_needed")
