from __future__ import annotations

import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_CAMERAS = ["left_wrist", "right_wrist", "right_front"]
DEFAULT_ACTION_DIM = 12
DEFAULT_PROPRIO_DIM = 12
DEFAULT_VLA_HORIZON = 50
DEFAULT_CHUNK_LENGTH = 10

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def configure_logging(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger(name)


# Fields a --config file is allowed to set. Historically these five were
# overwritten with the DEFAULT_* constants immediately after parsing the YAML,
# so writing `action_dim: 14` in a config did nothing and said nothing -- a
# CRP dual-arm config (14 dims, top/left_wrist/right_wrist) came back as the
# SO101 one (12 dims, left_wrist/right_wrist/right_front) and build_pi05_policy
# then constructed the adapter against the wrong action space.
SHAPE_FIELDS = ("action_dim", "proprio_dim", "vla_horizon", "chunk_length", "cameras")


def load_training_config(config_path: str | None):
    """Load an RLT config. A --config file's shape fields now win.

    Without a config path the DEFAULT_* constants still apply, so existing
    no-config invocations are unchanged.
    """
    from evo_rlt.core.config import RLTConfig

    if not config_path:
        config = RLTConfig()
        config.action_dim = DEFAULT_ACTION_DIM
        config.proprio_dim = DEFAULT_PROPRIO_DIM
        config.vla_horizon = DEFAULT_VLA_HORIZON
        config.chunk_length = DEFAULT_CHUNK_LENGTH
        config.cameras = list(DEFAULT_CAMERAS)
        return config

    config = RLTConfig.from_yaml(config_path)
    if config.chunk_length >= config.vla_horizon:
        raise ValueError(
            f"{config_path}: chunk_length ({config.chunk_length}) must be smaller than "
            f"vla_horizon ({config.vla_horizon}) -- the RL chunk is a prefix of the VLA "
            "horizon, and the paper states C < H."
        )
    return config


def assert_config_matches_dataset(config, dataset_root: str | Path) -> None:
    """Cross-check a config's shape fields against a LeRobot dataset's meta/info.json.

    A mismatch here used to surface as a shape error deep inside the first
    training batch, or not at all -- ``proprio[..., :12]`` on a 14-dim tensor is
    valid Python that silently drops two dimensions.
    """
    import json

    info_path = Path(dataset_root) / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    features = info["features"]

    problems: list[str] = []
    for field_name, feature_key in (("action_dim", "action"), ("proprio_dim", "observation.state")):
        expected = features[feature_key]["shape"][0]
        actual = getattr(config, field_name)
        if actual != expected:
            problems.append(f"{field_name}={actual} but dataset {feature_key} has {expected} dims")

    ds_cameras = sorted(
        k.removeprefix("observation.images.") for k in features if k.startswith("observation.images.")
    )
    if sorted(config.cameras) != ds_cameras:
        problems.append(f"cameras={sorted(config.cameras)} but dataset has {ds_cameras}")

    if config.control_hz != info["fps"]:
        problems.append(f"control_hz={config.control_hz} but dataset fps={info['fps']}")

    if problems:
        raise ValueError(
            f"Config does not match dataset at {dataset_root}:\n  "
            + "\n  ".join(problems)
        )


def build_pi05_policy(
    config,
    model_path: str,
    task_instruction: str,
    device: str,
    token_pool_size: int,
    dtype: str,
    rl_token_checkpoint: str | None = None,
    vla_cache_dir: str | None = None,
    image_only: bool = False,
    active_cameras: list[str] | None = None,
    tokenizer_path: str | None = None,
):
    from evo_rlt.adapters.lerobot.pi05_adapter import Pi05VLAAdapter
    from evo_rlt.core.policy import RLTPolicy
    import torch

    vla = Pi05VLAAdapter(
        model_path=model_path,
        actual_action_dim=config.action_dim,
        actual_proprio_dim=config.proprio_dim,
        task_instruction=task_instruction,
        dtype=dtype,
        device=device,
        cache_dir=vla_cache_dir,
        token_pool_size=token_pool_size,
        image_only=image_only,
        active_cameras=active_cameras,
        tokenizer_path=tokenizer_path,
    )
    policy = RLTPolicy(config, vla).to(device)
    if rl_token_checkpoint is not None:
        checkpoint = torch.load(rl_token_checkpoint, map_location=device, weights_only=False)
        policy.rl_token.load_state_dict(checkpoint["rl_token_state_dict"], strict=False)
    return policy
