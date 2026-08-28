from __future__ import annotations

import logging
import sys
from pathlib import Path


# .../src/evo_rlt/cli/common.py -> parents[1]=evo_rlt, [2]=src, [3]=repo root.
# These were parents[2] and REPO_ROOT/"src", i.e. the src dir and a non-existent
# src/src; the sys.path insert below was a no-op so nothing ever noticed.
_PKG_ROOT = Path(__file__).resolve().parents[1]      # .../src/evo_rlt
SRC_ROOT = _PKG_ROOT.parent                          # .../src
REPO_ROOT = SRC_ROOT.parent                          # .../RLT_dual (editable installs)
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


# Derived from the package itself, so it is right for editable and site-packages
# installs alike (pyproject ships core/configs/*.yaml as package data).
PACKAGED_CONFIG_DIR = _PKG_ROOT / "core" / "configs"

_log = logging.getLogger(__name__)


def resolve_config_path(config_path: str | Path) -> Path:
    """Find a --config file without depending on the caller's cwd.

    Two workspaces coexist on the training box (``~/crp-rlt`` holds the VLA run,
    ``~/RLT_dual`` holds this package), so ``--config src/evo_rlt/core/configs/x.yaml``
    resolves or not purely by which one you happened to cd into. Tries, in order:
    the path as given, the same path under the repo root, and the bare filename
    under the packaged config directory -- so ``--config crp_dual_rlt.yaml`` works
    from anywhere.
    """
    given = Path(config_path).expanduser()
    candidates = [given]
    if not given.is_absolute():
        candidates.append(REPO_ROOT / given)
    candidates.append(PACKAGED_CONFIG_DIR / given.name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    available = sorted(p.name for p in PACKAGED_CONFIG_DIR.glob("*.yaml"))
    raise FileNotFoundError(
        f"Config {config_path!r} not found. Looked in:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + f"\nAvailable packaged configs: {', '.join(available) or '(none)'}"
        + "\nA bare filename (e.g. --config crp_dual_rlt.yaml) resolves regardless of cwd."
    )


def resolve_model_path(model_path: str) -> str:
    """Disambiguate a local checkpoint directory from a Hugging Face repo id.

    A missing local path used to reach transformers, which treated it as a repo
    id and reported ``Repo id must be in the form 'repo_name' or
    'namespace/repo_name'`` -- an error about naming that says nothing about the
    directory being absent. Anything that looks like a filesystem path is
    checked here instead; bare ``org/name`` ids pass through untouched.
    """
    looks_local = (
        model_path.startswith(("~", ".", "/"))
        or Path(model_path).expanduser().exists()
        or (model_path.count("/") > 1)
    )
    if not looks_local:
        return model_path

    resolved = Path(model_path).expanduser()
    if resolved.is_dir():
        return str(resolved.resolve())

    hint = ""
    parent = resolved.parent
    while parent != parent.parent and not parent.is_dir():
        parent = parent.parent
    if parent.is_dir():
        siblings = sorted(p.name for p in parent.iterdir() if p.is_dir())[:12]
        if siblings:
            hint = f"\n  {parent} contains: {', '.join(siblings)}"
    raise FileNotFoundError(
        f"--model-path {model_path!r} looks like a local checkpoint directory but does "
        f"not exist.\n  Resolved to: {resolved}{hint}\n"
        "  A pi0.5 checkpoint directory holds config.json + model.safetensors.\n"
        "  (Pass a bare Hugging Face id such as lerobot/pi05_base to load from the Hub.)"
    )


def resolve_artifact_path(path: str | Path, *, must_exist: bool = False, label: str = "path") -> Path:
    """Make an artifact path absolute and log it, so cwd never silently decides it.

    ``--output outputs/crp_token_std.pt`` written from ``~/crp-rlt`` and read back
    from ``~/RLT_dual`` are two different files, and nothing complained.
    """
    resolved = Path(path).expanduser().resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(
            f"{label} {str(path)!r} does not exist.\n  Resolved to: {resolved}\n"
            "  Relative paths resolve against the current directory -- pass an absolute "
            "path if the file was produced from a different workspace."
        )
    _log.info("%s -> %s", label, resolved)
    return resolved


def warn_on_shadowing_cuda_libs() -> list[str]:
    """Warn when LD_LIBRARY_PATH puts a system CUDA ahead of the one torch ships.

    The failure this catches surfaces far from its cause: a system libcublas
    loaded in place of the bundled one runs until the first batched GEMM and
    then raises ``CUBLAS_STATUS_INVALID_VALUE``, which reads like a shape bug.
    Clearing LD_LIBRARY_PATH for the launch is the usual fix. Returns the
    offending directories (empty when clean) rather than raising -- some setups
    legitimately need a system CUDA.
    """
    import os

    raw = os.environ.get("LD_LIBRARY_PATH", "")
    if not raw:
        return []

    try:
        import torch

        bundled_root = Path(torch.__file__).resolve().parent.parent / "nvidia"
    except Exception:  # torch not importable yet; nothing to compare against
        return []

    offenders = []
    for entry in raw.split(os.pathsep):
        if not entry:
            continue
        directory = Path(entry).expanduser()
        if not directory.is_dir():
            continue
        if bundled_root in directory.parents or directory == bundled_root:
            continue  # this IS torch's own bundle
        if any(directory.glob("libcublas.so*")) or any(directory.glob("libcudart.so*")):
            offenders.append(str(directory))

    if offenders:
        _log.warning(
            "LD_LIBRARY_PATH contains CUDA libraries that may shadow the ones torch "
            "ships: %s. If this run dies with CUBLAS_STATUS_INVALID_VALUE inside a "
            "GEMM, relaunch with an empty LD_LIBRARY_PATH:\n"
            "    LD_LIBRARY_PATH= python -m evo_rlt.cli...",
            ", ".join(offenders),
        )
    return offenders


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

    resolved = resolve_config_path(config_path)
    _log.info("config -> %s", resolved)
    config = RLTConfig.from_yaml(resolved)
    if config.chunk_length >= config.vla_horizon:
        raise ValueError(
            f"{resolved}: chunk_length ({config.chunk_length}) must be smaller than "
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
    from evo_rlt.adapters.lerobot.pi05_low_mem_load import install as install_pi05_low_mem_loader
    from evo_rlt.core.policy import RLTPolicy
    import torch

    # A missing local checkpoint reaches transformers as a repo id and comes back
    # as "Repo id must be in the form ...", which points at naming rather than at
    # the absent directory. Fail here with the path that was actually checked.
    model_path = resolve_model_path(model_path)
    _log.info("model -> %s", model_path)
    warn_on_shadowing_cuda_libs()

    # Pi05VLAAdapter goes through PI05Policy.from_pretrained, whose stock path
    # fp32-random-inits 4.14B params before overwriting them from the checkpoint:
    # a 23.7 GiB transient and 116 s per load. Same fix the deploy path installs.
    install_pi05_low_mem_loader()

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
