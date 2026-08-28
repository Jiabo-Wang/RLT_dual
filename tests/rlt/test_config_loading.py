"""A --config file's shape fields must actually take effect, and must be checked.

Both halves used to be missing: load_training_config() overwrote action_dim /
proprio_dim / vla_horizon / chunk_length / cameras with SO101 constants right
after parsing the YAML, so a CRP dual-arm config silently came back as the
SO101 one and nothing downstream noticed (slicing a 14-dim tensor to 12 is
valid Python).
"""

import json
import textwrap

import pytest

from evo_rlt.cli.common import (
    DEFAULT_ACTION_DIM,
    DEFAULT_CAMERAS,
    assert_config_matches_dataset,
    load_training_config,
)

CRP_YAML = textwrap.dedent(
    """
    control_hz: 16
    action_dim: 14
    proprio_dim: 14
    vla_horizon: 50
    chunk_length: 10
    cameras:
      - top
      - left_wrist
      - right_wrist
    """
)


def _write_config(tmp_path, body=CRP_YAML):
    path = tmp_path / "cfg.yaml"
    path.write_text(body)
    return str(path)


def _write_dataset(tmp_path, *, action_dim=14, state_dim=14, fps=16,
                   cameras=("top", "left_wrist", "right_wrist")):
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    features = {
        "action": {"shape": [action_dim]},
        "observation.state": {"shape": [state_dim]},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {"shape": [480, 640, 3]}
    (root / "meta" / "info.json").write_text(json.dumps({"fps": fps, "features": features}))
    return str(root)


class TestShapeFieldsFromYaml:
    def test_yaml_shape_fields_win(self, tmp_path):
        cfg = load_training_config(_write_config(tmp_path))
        assert cfg.action_dim == 14
        assert cfg.proprio_dim == 14
        assert cfg.control_hz == 16
        assert cfg.cameras == ["top", "left_wrist", "right_wrist"]

    def test_no_config_path_keeps_legacy_defaults(self):
        cfg = load_training_config(None)
        assert cfg.action_dim == DEFAULT_ACTION_DIM
        assert cfg.cameras == list(DEFAULT_CAMERAS)

    def test_chunk_length_must_be_below_vla_horizon(self, tmp_path):
        body = CRP_YAML.replace("chunk_length: 10", "chunk_length: 50")
        with pytest.raises(ValueError, match="chunk_length"):
            load_training_config(_write_config(tmp_path, body))


class TestDatasetCrossCheck:
    def test_matching_dataset_passes(self, tmp_path):
        cfg = load_training_config(_write_config(tmp_path))
        assert_config_matches_dataset(cfg, _write_dataset(tmp_path))

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"action_dim": 12}, "action_dim"),
            ({"state_dim": 12}, "proprio_dim"),
            ({"fps": 30}, "control_hz"),
            ({"cameras": ("left_wrist", "right_wrist", "right_front")}, "cameras"),
        ],
    )
    def test_each_mismatch_is_reported(self, tmp_path, kwargs, expected):
        cfg = load_training_config(_write_config(tmp_path))
        with pytest.raises(ValueError, match=expected):
            assert_config_matches_dataset(cfg, _write_dataset(tmp_path, **kwargs))


class TestShippedConfigs:
    def test_crp_config_matches_its_robot(self, tmp_path):
        cfg = load_training_config("src/evo_rlt/core/configs/crp_dual_rlt.yaml")
        assert_config_matches_dataset(cfg, _write_dataset(tmp_path))

    def test_so101_config_is_rejected_against_a_crp_dataset(self, tmp_path):
        cfg = load_training_config("src/evo_rlt/core/configs/pi05_rlt.yaml")
        with pytest.raises(ValueError):
            assert_config_matches_dataset(cfg, _write_dataset(tmp_path))
