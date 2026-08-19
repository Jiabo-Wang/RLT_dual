import pytest
import rerun.blueprint as rrb

from evo_rlt.cli.dataset_viz import build_blueprint, parse_episode_spec


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("all", [0, 1, 2, 3, 4]),
        ("*", [0, 1, 2, 3, 4]),
        ("", [0, 1, 2, 3, 4]),
        ("2", [2]),
        ("1-3", [1, 2, 3]),
        ("3-", [3, 4]),
        ("-2", [0, 1, 2]),
        ("0-1,4", [0, 1, 4]),
        ("4,0-1", [0, 1, 4]),  # sorted, deduplicated
        ("1-2,2-3", [1, 2, 3]),
        (" 1 , 3 ", [1, 3]),
    ],
)
def test_parse_episode_spec(spec, expected):
    assert parse_episode_spec(spec, total=5) == expected


@pytest.mark.parametrize("spec", ["5", "0-5", "-9", "3-1"])
def test_parse_episode_spec_rejects(spec):
    with pytest.raises(ValueError):
        parse_episode_spec(spec, total=5)


_FEATURES = {
    "action": {"dtype": "float32", "shape": [14]},
    "observation.state": {"dtype": "float32", "shape": [14]},
    "observation.images.top": {"dtype": "video", "shape": [480, 640, 3]},
    "complementary_info.phase": {"dtype": "float32", "shape": [1]},
    "timestamp": {"dtype": "float32", "shape": [1]},
    "episode_index": {"dtype": "int64", "shape": [1]},
}


def test_build_blueprint_puts_every_camera_in_its_own_view():
    blueprint = build_blueprint(_FEATURES, ["observation.images.top", "observation.images.left"])
    origins = [view.origin for view in _views(blueprint)]
    assert origins.count("observation.images.top") == 1
    assert origins.count("observation.images.left") == 1
    # Vector features get an axis each; the 1-wide flags share the "/" view.
    assert "action" in origins and "observation.state" in origins
    assert "timestamp" not in origins


def test_build_blueprint_without_cameras():
    origins = [view.origin for view in _views(build_blueprint(_FEATURES, []))]
    assert not any(str(origin).startswith("observation.images") for origin in origins)
    assert "action" in origins


def _views(blueprint):
    """Depth-first walk of a blueprint tree, yielding leaf views."""
    stack = [blueprint.root_container]
    while stack:
        node = stack.pop()
        children = getattr(node, "contents", None)
        if isinstance(node, rrb.Container):
            stack.extend(children or [])
        elif isinstance(node, rrb.View):
            yield node
