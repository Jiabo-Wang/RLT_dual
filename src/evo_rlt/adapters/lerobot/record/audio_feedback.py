"""Spoken cues for the critical-phase keys.

Previously ``lerobot.utils.audio_feedback``, a module this project added to its
installed LeRobot copy. Reinstalling LeRobot deleted it, and because the imports
sat inside the key handlers the failure only surfaced when the operator pressed
a key -- with the robot moving. It lives here now so a LeRobot reinstall cannot
take it out again.

The operator's hands are on the leader arms and their eyes are on the workpiece,
not the terminal, which is the whole reason these are spoken rather than logged.
Never raise: a missing TTS backend must not interrupt an episode in progress.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_warned = False


def _say(text: str) -> None:
    global _warned
    try:
        from lerobot.utils.utils import log_say

        log_say(text, play_sounds=True, blocking=False)
    except Exception as exc:  # TTS backend missing, no audio device, ...
        if not _warned:
            logger.warning(
                "Spoken cues unavailable (%s); phase changes will only be logged.", exc
            )
            _warned = True
        logger.info(text)


def say_start() -> None:
    """The critical phase just opened."""
    _say("Critical phase start")


def say_success() -> None:
    """The critical phase closed and was labelled a success."""
    _say("Success")


def say_failure() -> None:
    """The critical phase closed and was labelled a failure."""
    _say("Failure")
