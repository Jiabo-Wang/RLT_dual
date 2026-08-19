"""Canonical CRP action-angle representation.

The screw-cell tool orientation is locked at the Euler roll branch cut. The CRP
controller and scipy may therefore spell the same pose as either +180 or -180
degrees. Those values are physically identical but are disastrous regression
targets, so use one exact value everywhere actions enter the robot or a dataset.
"""

from __future__ import annotations

import math


CRP_LOCKED_ROLL_DEG = -180.0
CRP_LOCKED_ROLL_TOLERANCE_DEG = 1.0


def canonicalize_crp_roll_deg(
    value: float,
    *,
    locked_roll_deg: float = CRP_LOCKED_ROLL_DEG,
    tolerance_deg: float = CRP_LOCKED_ROLL_TOLERANCE_DEG,
) -> float:
    """Snap either spelling of the locked roll to exactly ``-180``.

    Values away from the +/-180 branch cut are left untouched so this helper is
    safe for diagnostic poses and for future tasks that intentionally rotate the
    tool away from the screw-cell's locked orientation.
    """

    angle = float(value)
    if not math.isfinite(angle):
        return angle
    distance_to_branch = abs(abs(angle) - abs(float(locked_roll_deg)))
    if distance_to_branch <= float(tolerance_deg):
        return float(locked_roll_deg)
    return angle
