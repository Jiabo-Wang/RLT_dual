"""Import helpers for symbols LeRobot has moved between releases.

The project pins LeRobot v0.5.1 (see pyproject), but the deployment box also
carries a v0.5.2 checkout, and installing that editable silently replaces the
pinned copy. Two module paths changed between the two, both imported at module
scope, so the whole record/online-RL path stopped importing:

    build_dataset_frame, combine_feature_dicts
        v0.5.1  lerobot.datasets.feature_utils
        v0.5.2  lerobot.utils.feature_utils
    predict_action, init_keyboard_listener, sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility
        v0.5.1  lerobot.utils.control_utils
        v0.5.2  lerobot.common.control_utils

Resolving them here keeps the fallbacks in one place instead of scattering
try/except across four call sites, and follows the convention hil.py already
uses for ``DeviceDroppedConnectionError``. The error raised when a symbol is
missing from *every* known location names the release, because the useful next
question is "which LeRobot is installed", not "which import failed".
"""

from __future__ import annotations

import importlib
from typing import Any

_CONTROL_UTILS = ("lerobot.utils.control_utils", "lerobot.common.control_utils")

# symbol -> module paths to try, newest-known layout last
_LOCATIONS: dict[str, tuple[str, ...]] = {
    "build_dataset_frame": ("lerobot.datasets.feature_utils", "lerobot.utils.feature_utils"),
    "combine_feature_dicts": ("lerobot.datasets.feature_utils", "lerobot.utils.feature_utils"),
    "predict_action": _CONTROL_UTILS,
    "init_keyboard_listener": _CONTROL_UTILS,
    "sanity_check_dataset_name": _CONTROL_UTILS,
    "sanity_check_dataset_robot_compatibility": _CONTROL_UTILS,
}


def _resolve(name: str) -> Any:
    modules = _LOCATIONS[name]
    for module_path in modules:
        try:
            return getattr(importlib.import_module(module_path), name)
        except (ImportError, AttributeError):
            continue

    try:
        import lerobot

        version = getattr(lerobot, "__version__", "unknown")
        where = getattr(lerobot, "__file__", "unknown")
    except ImportError:  # pragma: no cover - lerobot is a hard dependency
        version = where = "not installed"
    raise ImportError(
        f"Could not find {name!r} in any known LeRobot location ({', '.join(modules)}).\n"
        f"  Installed LeRobot: {version} at {where}\n"
        "  This project is pinned to v0.5.1 (see pyproject.toml). An editable install of a "
        "different checkout replaces that copy without warning.\n"
        f"  If {name!r} moved again, add its new module path to evo_rlt.adapters.lerobot._compat."
    )


build_dataset_frame = _resolve("build_dataset_frame")
combine_feature_dicts = _resolve("combine_feature_dicts")
predict_action = _resolve("predict_action")
init_keyboard_listener = _resolve("init_keyboard_listener")
sanity_check_dataset_name = _resolve("sanity_check_dataset_name")
sanity_check_dataset_robot_compatibility = _resolve("sanity_check_dataset_robot_compatibility")

__all__ = [
    "build_dataset_frame",
    "combine_feature_dicts",
    "init_keyboard_listener",
    "predict_action",
    "sanity_check_dataset_name",
    "sanity_check_dataset_robot_compatibility",
]
