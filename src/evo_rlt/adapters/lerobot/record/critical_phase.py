"""Frame-interval trackers for the critical phase and for human intervention.

Previously ``lerobot.utils.critical_phase_tracker``, a module this project added
to its installed LeRobot copy. Reinstalling LeRobot deleted it and it exists in
no branch of any checkout on this machine, so this is a **reconstruction from the
call sites** rather than a recovered file. It lives under ``evo_rlt`` now: it is
this project's concern, and keeping it inside LeRobot is exactly what let a
reinstall wipe it.

What the call sites require (``record/backend.py`` and ``record/loop.py``):

    CriticalPhaseTracker(auto_save_path=...)      EpisodeIntervalTracker(label=...)
      .toggle(frame)                                .start(frame) / .stop(frame)
      .is_active                                    .on_episode_start(episode_index)
      .mark_success(frame) / .mark_failure(frame)   .on_episode_end(episode_frames)
      .get_intervals()  ->  (ep, start, end, outcome) tuples
      len(tracker)                                  .discard_episode(episode_index)
                                                    .serialize_episode_intervals(ep)

Behaviour that the call sites do not pin down, and what was chosen:

- **Frame indices are episode-local.** Every caller passes
  ``dataset.episode_buffer["size"]`` or ``get_episode_frame_index()``, both of
  which reset per episode, and ``backend.py`` logs intervals as "Episode N:
  frames s-e", so they are stored as given.
- **An interval is half-open ``[start, end)``.** ``backend.py`` reports its
  length as ``end - start``.
- **``mark_success`` / ``mark_failure`` close the open interval and label it.**
  With no interval open they label the episode's most recent one instead --
  ``loop.py`` calls them both from the ``s``/``f`` keys (phase still open) and
  again at episode end from the resolved outcome (phase already closed by
  ``on_episode_end``), and the second call must not be silently dropped.
- **Re-recording drops the episode's intervals entirely** rather than keeping
  them under a stale index, since ``dataset.clear_episode_buffer()`` means the
  frames they point at no longer exist.
- **Only closed intervals are ever serialized.** An interval left open by a
  crash has no end frame, so emitting it would put a null into the dataset
  metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

OUTCOME_SUCCESS = "success"
OUTCOME_FAILURE = "failure"


class EpisodeIntervalTracker:
    """Records ``[start, end)`` frame intervals, grouped by episode.

    One interval may be open at a time. ``label`` only appears in log lines, so
    two trackers running at once (critical phase and human intervention) can be
    told apart.
    """

    def __init__(self, label: str = "Interval") -> None:
        self.label = label
        # each: {"episode_index": int, "start_frame": int,
        #        "end_frame": int | None, "outcome": str | None}
        self._intervals: list[dict] = []
        self._episode_index: int = 0
        self._open: dict | None = None

    # --- episode lifecycle -------------------------------------------------

    def on_episode_start(self, episode_index: int) -> None:
        if self._open is not None:
            # Only reachable if the previous episode ended without on_episode_end;
            # drop the interval rather than let it absorb the next episode.
            logger.warning(
                "%s: episode %d started with an interval still open at frame %d; dropping it.",
                self.label, episode_index, self._open["start_frame"],
            )
            self._intervals.remove(self._open)
            self._open = None
        self._episode_index = episode_index

    def on_episode_end(self, episode_frames: int) -> None:
        if self._open is not None:
            self.stop(episode_frames)

    def discard_episode(self, episode_index: int) -> None:
        dropped = [i for i in self._intervals if i["episode_index"] == episode_index]
        if dropped:
            logger.info("%s: dropping %d interval(s) from re-recorded episode %d.",
                        self.label, len(dropped), episode_index)
        self._intervals = [i for i in self._intervals if i["episode_index"] != episode_index]
        self._open = None

    # --- interval boundaries ----------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._open is not None

    def start(self, frame_index: int) -> None:
        if self._open is not None:
            logger.warning("%s: already open since frame %d; ignoring start at %d.",
                           self.label, self._open["start_frame"], frame_index)
            return
        self._open = {
            "episode_index": self._episode_index,
            "start_frame": int(frame_index),
            "end_frame": None,
            "outcome": None,
        }
        self._intervals.append(self._open)
        logger.info("%s: started at episode %d frame %d.",
                    self.label, self._episode_index, frame_index)

    def stop(self, frame_index: int) -> None:
        if self._open is None:
            logger.debug("%s: stop at frame %d with nothing open; ignoring.",
                         self.label, frame_index)
            return
        # A stop before the start would produce a negative length downstream.
        self._open["end_frame"] = max(int(frame_index), self._open["start_frame"])
        logger.info("%s: ended at episode %d frame %d (%d frames).",
                    self.label, self._open["episode_index"], self._open["end_frame"],
                    self._open["end_frame"] - self._open["start_frame"])
        self._open = None
        self._on_closed()

    def _on_closed(self) -> None:
        """Hook for subclasses; base tracker keeps everything in memory."""

    # --- readback ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._intervals)

    def get_intervals(self) -> list[tuple[int, int, int, str | None]]:
        """Closed intervals as ``(episode_index, start_frame, end_frame, outcome)``."""
        return [
            (i["episode_index"], i["start_frame"], i["end_frame"], i["outcome"])
            for i in self._intervals
            if i["end_frame"] is not None
        ]

    def serialize_episode_intervals(self, episode_index: int) -> list[dict]:
        """One episode's closed intervals, for that episode's dataset metadata."""
        return [
            {"start_frame": i["start_frame"], "end_frame": i["end_frame"], "outcome": i["outcome"]}
            for i in self._intervals
            if i["episode_index"] == episode_index and i["end_frame"] is not None
        ]


class CriticalPhaseTracker(EpisodeIntervalTracker):
    """The critical phase, driven by the ``r`` / ``s`` / ``f`` keys.

    ``auto_save_path`` is written every time an interval closes, so a crash or a
    power cut mid-session does not take the labels with it. ``backend.py`` writes
    the same file again at shutdown; both use the same schema so the rewrite is a
    no-op rather than a format change.
    """

    def __init__(self, auto_save_path: str | Path | None = None,
                 label: str = "Critical phase") -> None:
        super().__init__(label=label)
        self.auto_save_path = Path(auto_save_path) if auto_save_path is not None else None

    def toggle(self, frame_index: int) -> None:
        """First press opens the phase, second closes it."""
        if self.is_active:
            self.stop(frame_index)
        else:
            self.start(frame_index)

    def mark_success(self, frame_index: int) -> None:
        self._close_with_outcome(frame_index, OUTCOME_SUCCESS)

    def mark_failure(self, frame_index: int) -> None:
        self._close_with_outcome(frame_index, OUTCOME_FAILURE)

    def _close_with_outcome(self, frame_index: int, outcome: str) -> None:
        target = self._open
        if target is not None:
            self.stop(frame_index)
        else:
            # Called again at episode end, after on_episode_end already closed the
            # phase. Label the episode's most recent interval instead of dropping it.
            closed = [i for i in self._intervals
                      if i["episode_index"] == self._episode_index and i["end_frame"] is not None]
            if not closed:
                logger.debug("%s: %s at frame %d with no interval to label; ignoring.",
                             self.label, outcome, frame_index)
                return
            target = closed[-1]
        target["outcome"] = outcome
        logger.info("%s: episode %d frames %d-%d labelled %s.",
                    self.label, target["episode_index"], target["start_frame"],
                    target["end_frame"], outcome)
        self._save()

    def _on_closed(self) -> None:
        self._save()

    def _save(self) -> None:
        if self.auto_save_path is None:
            return
        try:
            self.auto_save_path.parent.mkdir(parents=True, exist_ok=True)
            payload = [
                {"episode_index": ep, "start_frame": s, "end_frame": e, "outcome": o}
                for ep, s, e, o in self.get_intervals()
            ]
            self.auto_save_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            # An unwritable path must not abort a recording session.
            logger.warning("%s: could not autosave to %s (%s).",
                           self.label, self.auto_save_path, exc)
