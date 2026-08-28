from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

DEFAULT_SETUP_PATH = Path.home() / ".roboclaw/workspace/embodied/manifest.json"
DEFAULT_DATASET_ROOT = Path.home() / ".roboclaw/workspace/embodied/datasets"


def load_setup_json(path: str | None = None) -> dict[str, Any]:
    setup_path = Path(path).expanduser() if path else DEFAULT_SETUP_PATH
    with open(setup_path) as fh:
        return json.load(fh)


def resolve_dataset_root(setup: dict[str, Any]) -> Path:
    dataset_root = setup.get("datasets", {}).get("root", "")
    if not dataset_root:
        return DEFAULT_DATASET_ROOT
    return Path(dataset_root).expanduser()


def resolve_fps(cli_fps: int | float | None, setup: dict[str, Any]) -> int | float:
    """CLI ``--fps`` wins; otherwise the manifest's ``datasets.fps``; otherwise 30.

    The control rate is a property of the cell, not of the run: CRP tops out near
    16 Hz because a dual-arm GP send costs ~46 ms. Carrying it in the manifest keeps
    every recording on that cell consistent, and keeps the dataset timestamps honest
    -- a dataset labelled 30 fps that was captured at 16 is silently wrong.
    Manifests without ``datasets.fps`` keep the previous default of 30.
    """
    if cli_fps is not None:
        return cli_fps
    return setup.get("datasets", {}).get("fps", 30)


def get_sorted_followers(setup: dict[str, Any]) -> list[dict[str, Any]]:
    followers = [arm for arm in setup["arms"] if "follower" in arm["type"]]
    followers.sort(key=lambda arm: 0 if "left" in arm.get("alias", "") else 1)
    return followers


def get_sorted_leaders(setup: dict[str, Any]) -> list[dict[str, Any]]:
    leaders = [arm for arm in setup["arms"] if "leader" in arm["type"]]
    leaders.sort(key=lambda arm: 0 if "left" in arm.get("alias", "") else 1)
    return leaders

log = logging.getLogger(__name__)

CAMERA_RENAME = {"left_wrist": "wrist", "right_wrist": "wrist", "right_front": "front"}
LEFT_CAMERA_ALIASES = {"left_wrist"}
RIGHT_CAMERA_ALIASES = {"right_wrist", "right_front"}
TELEOP_ID = "bimanual_leader"
# CRP keeps its own leader identity so the two cells can never overwrite each
# other's calibration -- the id is also the calibration filename. Mirrors
# evo_rlt.adapters.crp.teleop_config.LEADER_ID.
CRP_TELEOP_ID = "crp_dual_leader"
# Single-arm (so101_follower / so101_leader) ids. Kept separate from the
# bimanual ids above so calibration staging never collides between modes.
FOLLOWER_ID_SINGLE = "so_follower"
TELEOP_ID_SINGLE = "so_leader"


def install_safe_follower_torque_enable(robot: Any) -> None:
    """Make follower torque-enable hold the measured pose, not a stale goal.

    STS3215 ``Goal_Position`` is RAM-backed and can read as zero after a
    power cycle while the arm is physically near the middle of its range.
    LeRobot's SOFollower.configure() enables torque before the first normal
    action is sent.  If that stale zero remains, all joints immediately try
    to chase it; the resulting inrush can trip the servo's input-voltage or
    overload protection while ``enable_torque()`` is writing the following
    ``Lock=1`` register.

    Wrap each follower bus instance so every future disabled->enabled
    transition first copies raw Present_Position to raw Goal_Position and
    verifies the write.  This is installed before connect/configure, and is
    deliberately follower-only: leader arms remain torque-disabled.
    """

    arms = [
        arm
        for arm in (getattr(robot, "left_arm", None), getattr(robot, "right_arm", None))
        if arm is not None
    ]
    if not arms and hasattr(robot, "bus"):
        arms = [robot]

    for arm in arms:
        bus = getattr(arm, "bus", None)
        if bus is None or getattr(bus, "_evo_rlt_safe_torque_enable", False):
            continue
        original_enable_torque = bus.enable_torque

        def safe_enable_torque(
            motors=None,
            num_retry: int = 0,
            *,
            _bus=bus,
            _original=original_enable_torque,
        ) -> None:
            present = _bus.sync_read(
                "Present_Position", motors=motors, normalize=False, num_retry=num_retry
            )
            _bus.sync_write(
                "Goal_Position", present, normalize=False, num_retry=num_retry
            )
            written = _bus.sync_read(
                "Goal_Position", motors=motors, normalize=False, num_retry=num_retry
            )
            if written != present:
                raise RuntimeError(
                    "Refusing to enable follower torque: failed to synchronize "
                    f"Goal_Position to Present_Position (present={present}, goal={written})"
                )
            log.info("Primed follower Goal_Position from Present_Position before torque enable")
            _original(motors, num_retry=num_retry)

        bus.enable_torque = safe_enable_torque
        bus._evo_rlt_safe_torque_enable = True


@dataclass(frozen=True)
class RobotSetup:
    setup: dict[str, Any]
    followers: list[dict[str, Any]]
    leaders: list[dict[str, Any]]
    left_cameras: dict[str, Any]
    right_cameras: dict[str, Any]
    # Leader calibration identity: CRP's own, or the shared SO101 one.
    teleop_id: str = TELEOP_ID


@dataclass(frozen=True)
class RunPaths:
    dataset_name: str
    dataset_root: Path
    day_dir: Path
    log_file: Path


def is_crp_setup(followers: list[dict[str, Any]]) -> bool:
    """True when the manifest's followers are CRP arms rather than SO101 ones.

    Gated on an explicit ``kind`` so a manifest without it takes exactly the code
    path it took before CRP existed.
    """
    kinds = {str(f.get("kind", "")).lower() for f in followers}
    if kinds == {"crp"}:
        return True
    if "crp" in kinds:
        raise ValueError(
            "Manifest mixes CRP and non-CRP followers; one robot cannot cover both. "
            f"Followers: {[(f.get('alias'), f.get('kind')) for f in followers]}"
        )
    return False


def load_robot_setup(setup_json: str | None) -> RobotSetup:
    setup = load_setup_json(setup_json)
    followers = get_sorted_followers(setup)
    leaders = get_sorted_leaders(setup)
    if is_crp_setup(followers):
        # CRPArmDual owns one flat camera dict and does not re-prefix per arm, so
        # aliases survive verbatim -- `top` is not forced onto one of the arms, and
        # a camera whose alias is in neither side's table is no longer dropped.
        return RobotSetup(
            setup,
            followers,
            leaders,
            build_flat_camera_config(setup.get("cameras", [])),
            {},
            teleop_id=CRP_TELEOP_ID,
        )
    if len(followers) == 1:
        # Single-arm: cameras are not split left/right, so they are all
        # stored under `left_cameras`; `right_cameras` stays empty and
        # downstream single-arm branches never read it.
        cameras = build_single_arm_camera_config(setup.get("cameras", []))
        return RobotSetup(setup, followers, leaders, cameras, {})
    if len(followers) < 2:
        raise ValueError(
            f"Need 1 follower arm (single-arm) or 2 follower arms (bimanual), got {len(followers)}"
        )
    left_cameras, right_cameras = build_camera_configs(setup.get("cameras", []))
    return RobotSetup(setup, followers, leaders, left_cameras, right_cameras)


def _camera_dict(camera: dict[str, Any]) -> dict[str, Any]:
    """One camera's config dict, with ``port`` resolved to a stable device path.

    Serialised into a ``--robot.cameras=`` CLI argument, so the resolved path is
    stringified: ``json.dumps`` cannot encode a ``Path``.
    """
    from evo_rlt.adapters.lerobot.record.camera_resolve import resolve_camera_port

    resolved = resolve_camera_port(camera["port"])
    return {
        "type": "opencv",
        "index_or_path": str(resolved) if isinstance(resolved, Path) else resolved,
        "width": camera.get("width", 640),
        "height": camera.get("height", 480),
        "fps": camera.get("fps", 30),
        # 200 = cv2.CAP_V4L2. LeRobot defaults to CAP_ANY, which on this host picks a
        # backend whose cap.set(CAP_PROP_FRAME_WIDTH) returns False even when the
        # width it reports back is already the requested one -- and OpenCVCamera
        # treats that False as fatal, so all three cameras failed to connect with
        # "failed to set capture_width=640 (actual_width=640, width_success=False)".
        # Measured on 2026-08-28: 0/3 cameras connect under CAP_ANY, 3/3 under V4L2,
        # same nodes, same moment. These are USB UVC devices on Linux; V4L2 is the
        # backend they are actually driven through either way.
        "backend": camera.get("backend", 200),
    }


def build_flat_camera_config(cameras: list[dict[str, Any]]) -> dict[str, Any]:
    """Cameras keyed by their manifest alias, with no left/right split.

    Used by the single-arm and CRP paths. The bimanual SO101 path cannot use this:
    ``BiSOFollower`` owns one camera set per arm and re-prefixes the feature names,
    so its cameras have to be split and un-prefixed first.
    """
    out: dict[str, Any] = {}
    for camera in cameras:
        alias = camera["alias"]
        camera_config = _camera_dict(camera)
        if camera.get("fourcc"):
            camera_config["fourcc"] = camera["fourcc"]
        out[alias] = camera_config
    return out


# Pre-existing name kept so nothing outside this module has to change.
build_single_arm_camera_config = build_flat_camera_config


def build_camera_configs(cameras: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    left_cameras: dict[str, Any] = {}
    right_cameras: dict[str, Any] = {}
    for camera in cameras:
        alias = camera["alias"]
        camera_config = _camera_dict(camera)
        if camera.get("fourcc"):
            camera_config["fourcc"] = camera["fourcc"]
        target_name = CAMERA_RENAME.get(alias, alias)
        if alias in LEFT_CAMERA_ALIASES:
            left_cameras[target_name] = camera_config
        elif alias in RIGHT_CAMERA_ALIASES:
            right_cameras[target_name] = camera_config
    return left_cameras, right_cameras


def resolve_run_paths(setup: dict[str, Any], dataset_tag: str, dataset_prefix: str) -> RunPaths:
    now = datetime.now()
    date_folder = f"{now:%m%d}_{dataset_tag}"
    dataset_leaf = f"{dataset_prefix}_{now:%H%M%S}"
    day_dir = resolve_dataset_root(setup) / date_folder
    dataset_root = day_dir / dataset_leaf
    return RunPaths(
        dataset_name=f"local/{dataset_leaf}",
        dataset_root=dataset_root,
        day_dir=day_dir,
        log_file=day_dir / f"{dataset_leaf}.log",
    )


def configure_logging(log_file: Path, log_level: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)


def remove_existing_dataset(dataset_root: Path) -> None:
    if dataset_root.exists():
        log.info("Removing existing dataset dir: %s", dataset_root)
        shutil.rmtree(dataset_root)


def stage_arm_calibration(arm: dict[str, Any], dst: Path) -> None:
    calibration_file = arm.get("calibration_file")
    if calibration_file:
        src = Path(calibration_file).expanduser()
    else:
        serial = Path(arm["calibration_dir"]).name
        src = Path(arm["calibration_dir"]).expanduser() / f"{serial}.json"
    if src.exists():
        shutil.copy2(src, dst)
        log.info("Calibration staged: %s -> %s", src, dst)
        return
    log.warning("Calibration file not found: %s", src)


def stage_follower_calibrations(followers: list[dict[str, Any]], cal_dir: str) -> None:
    # CRP arms are calibrated on their own controller; there is no host-side file to
    # stage, and the manifest carries no ``calibration_dir`` for them. Guarding here
    # rather than at each caller keeps every record subcommand covered.
    if is_crp_setup(followers):
        log.info("CRP followers: calibration lives on the controller, nothing to stage.")
        return
    if len(followers) == 1:
        stage_arm_calibration(followers[0], Path(cal_dir) / f"{FOLLOWER_ID_SINGLE}.json")
        return
    for side, arm in (("left", followers[0]), ("right", followers[1])):
        stage_arm_calibration(arm, Path(cal_dir) / f"bimanual_{side}.json")


def build_teleop_argv(
    leaders: list[dict[str, Any]], no_teleop: bool, teleop_id: str = TELEOP_ID
) -> list[str]:
    if no_teleop:
        log.warning("Teleop disabled by --no-teleop")
        return []
    if not leaders:
        log.warning("Teleop disabled: no leader arms configured")
        return []
    if len(leaders) == 1:
        log.info("Teleop enabled (single-arm): leader=%s", leaders[0]["port"])
        return [
            "--teleop.type=so101_leader",
            f"--teleop.port={leaders[0]['port']}",
            f"--teleop.id={TELEOP_ID_SINGLE}",
        ]
    if len(leaders) < 2:
        log.warning("Teleop disabled: need 2 leader arms, got %d", len(leaders))
        return []
    log.info("Teleop enabled: left=%s, right=%s", leaders[0]["port"], leaders[1]["port"])
    return [
        "--teleop.type=bi_so_leader",
        f"--teleop.left_arm_config.port={leaders[0]['port']}",
        "--teleop.left_arm_config.use_degrees=true",
        f"--teleop.right_arm_config.port={leaders[1]['port']}",
        "--teleop.right_arm_config.use_degrees=true",
        f"--teleop.id={teleop_id}",
    ]


def stage_leader_calibrations(
    leaders: list[dict[str, Any]], teleop_argv: list[str], teleop_id: str = TELEOP_ID
) -> TemporaryDirectory[str] | None:
    if not teleop_argv:
        return None
    leader_cal_dir = TemporaryDirectory(prefix="record-leader-cal-")
    if len(leaders) == 1:
        stage_arm_calibration(leaders[0], Path(leader_cal_dir.name) / f"{TELEOP_ID_SINGLE}.json")
    else:
        for side, arm in (("left", leaders[0]), ("right", leaders[1])):
            stage_arm_calibration(arm, Path(leader_cal_dir.name) / f"{teleop_id}_{side}.json")
    teleop_argv.append(f"--teleop.calibration_dir={leader_cal_dir.name}")
    return leader_cal_dir


def build_robot_argv(
    followers: list[dict[str, Any]],
    left_cameras: dict[str, Any],
    right_cameras: dict[str, Any],
    cal_dir: str,
) -> list[str]:
    if is_crp_setup(followers):
        left, right = followers[0], followers[1]
        # Importing the config registers ``crp_arm_dual`` as a RobotConfig subclass.
        # draccus builds --robot.type's choices from that registry at parse time, and
        # nothing else on the record path imports the CRP stack, so without this the
        # argv below is rejected as an invalid choice.
        from evo_rlt.adapters.crp.config_arm_dual import CRPArmDualConfig  # noqa: F401

        # No --robot.calibration_dir: CRP arms are calibrated on the controller, not
        # against a host-side file the way STS3215 servos are.
        return [
            "--robot.type=crp_arm_dual",
            "--robot.id=crp_dual",
            f"--robot.ip1={left['ip']}",
            f"--robot.ip2={right['ip']}",
            f"--robot.gp_register_left={left.get('gp_index', 10)}",
            f"--robot.gp_register_right={right.get('gp_index', 20)}",
            # Opening each gripper is driven to before the run, so the first
            # observation carries a real value instead of the "fully closed"
            # fallback. Per-cell, so it lives in the manifest.
            *(
                [f"--robot.init_gripper_ui50_left={left['gripper_init_ui50']}"]
                if "gripper_init_ui50" in left
                else []
            ),
            *(
                [f"--robot.init_gripper_ui50_right={right['gripper_init_ui50']}"]
                if "gripper_init_ui50" in right
                else []
            ),
            # Recorded by evo-rlt-crp-set-home. Absent from a manifest that never ran
            # it, in which case go-home is a no-op and behaviour is unchanged.
            *(
                [f"--robot.home_tcp_left={json.dumps(left['home_tcp'])}"]
                if "home_tcp" in left
                else []
            ),
            *(
                [f"--robot.home_tcp_right={json.dumps(right['home_tcp'])}"]
                if "home_tcp" in right
                else []
            ),
            f"--robot.cameras={json.dumps(left_cameras)}",
        ]

    if len(followers) == 1:
        return [
            "--robot.type=so101_follower",
            f"--robot.id={FOLLOWER_ID_SINGLE}",
            f"--robot.calibration_dir={cal_dir}",
            f"--robot.port={followers[0]['port']}",
            f"--robot.cameras={json.dumps(left_cameras)}",
        ]
    return [
        "--robot.type=bi_so_follower",
        "--robot.id=bimanual",
        f"--robot.calibration_dir={cal_dir}",
        f"--robot.left_arm_config.port={followers[0]['port']}",
        "--robot.left_arm_config.use_degrees=true",
        f"--robot.left_arm_config.cameras={json.dumps(left_cameras)}",
        f"--robot.right_arm_config.port={followers[1]['port']}",
        "--robot.right_arm_config.use_degrees=true",
        f"--robot.right_arm_config.cameras={json.dumps(right_cameras)}",
    ]


def build_dataset_argv(
    *,
    dataset_name: str,
    dataset_root: Path,
    task: str,
    num_episodes: int,
    episode_time_s: int,
    fps: int,
    vcodec: str,
    rename_map: str | None = None,
) -> list[str]:
    argv = [
        f"--dataset.repo_id={dataset_name}",
        f"--dataset.root={dataset_root}",
        f"--dataset.single_task={task}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.fps={fps}",
        f"--dataset.vcodec={vcodec}",
        "--dataset.push_to_hub=false",
        f"--dataset.video_encoding_batch_size={num_episodes + 1}",
        "--dataset.streaming_encoding=true",
    ]
    if rename_map:
        argv.append(f"--dataset.rename_map={rename_map}")
    return argv


def build_policy_overrides(
    *,
    policy_path: str | None,
    vla_path: str | None,
    rl_token_path: str | None,
    phase_mode: str | None = None,
    chunk_exec_steps: int | None = None,
    n_action_steps: int | None = None,
) -> list[str]:
    if policy_path is None:
        return []
    overrides = [f"--policy.path={policy_path}"]
    if phase_mode is not None:
        overrides.append(f"--policy.phase_mode={phase_mode}")
    if chunk_exec_steps is not None:
        overrides.append(f"--policy.chunk_exec_steps={chunk_exec_steps}")
    if n_action_steps is not None:
        if n_action_steps <= 0:
            raise ValueError("policy n_action_steps must be positive")
        overrides.append(f"--policy.n_action_steps={n_action_steps}")
    if vla_path is not None:
        overrides.append(f"--policy.vla_pretrained_path={vla_path}")
    if rl_token_path is not None:
        overrides.append(f"--policy.rl_token_pretrained_path={rl_token_path}")
    return overrides


def build_rtc_argv(
    *,
    enabled: bool,
    execution_horizon: int,
    max_guidance_weight: float,
    prefix_attention_schedule: str,
    vla_execution_horizon: int | None,
    action_queue_size_to_get_new_actions: int | None,
) -> list[str]:
    argv = [
        f"--rlt.rtc_enabled={'true' if enabled else 'false'}",
        f"--rlt.rtc_execution_horizon={execution_horizon}",
        f"--rlt.rtc_max_guidance_weight={max_guidance_weight}",
        f"--rlt.rtc_prefix_attention_schedule={prefix_attention_schedule}",
    ]
    if vla_execution_horizon is not None:
        argv.append(f"--rlt.vla_rtc_execution_horizon={vla_execution_horizon}")
    if action_queue_size_to_get_new_actions is not None:
        argv.append(
            "--rlt.rtc_action_queue_size_to_get_new_actions="
            f"{action_queue_size_to_get_new_actions}"
        )
    return argv


def preflight_motor_connections(
    followers: list[dict[str, Any]],
    leaders: list[dict[str, Any]],
    cal_dir: str,
    leader_cal_dir: str | None,
) -> None:
    from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
    from lerobot.robots.so_follower import SOFollower, SOFollowerConfig, SOFollowerRobotConfig
    from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderConfig, SOLeaderTeleopConfig

    def disconnect(device: Any) -> None:
        for arm_name in ("left_arm", "right_arm"):
            arm = getattr(device, arm_name, None)
            if arm is not None and arm.is_connected:
                arm.disconnect()
        if getattr(device, "is_connected", False):
            device.disconnect()

    # CRP followers have no host-side motor bus to check: the controller owns the
    # joints and the SDK reaches it over Ethernet, so the manifest carries
    # `ip`/`gp_index` and no `port`. Building an SOFollower from one raised
    # KeyError: 'port'. Same guard stage_follower_calibrations already applies --
    # and only the follower half is skipped, since the leaders below are SO101 on
    # serial for the CRP setup too.
    if is_crp_setup(followers):
        log.info(
            "CRP followers: no host-side motor bus to preflight "
            "(the SDK checks reachability when it connects)."
        )
    else:
        log.info("Preflight checking follower motor connections before loading policy")
        if len(followers) == 1:
            robot = SOFollower(
                SOFollowerRobotConfig(
                    id=FOLLOWER_ID_SINGLE,
                    calibration_dir=Path(cal_dir),
                    port=followers[0]["port"],
                )
            )
        else:
            robot = BiSOFollower(
                BiSOFollowerConfig(
                    id="bimanual",
                    calibration_dir=Path(cal_dir),
                    left_arm_config=SOFollowerConfig(port=followers[0]["port"], use_degrees=True),
                    right_arm_config=SOFollowerConfig(port=followers[1]["port"], use_degrees=True),
                )
            )
        install_safe_follower_torque_enable(robot)
        try:
            robot.connect(calibrate=True)
            log.info("Preflight follower motor check passed")
        finally:
            disconnect(robot)

    if not leaders or leader_cal_dir is None:
        return

    log.info("Preflight checking leader motor connections before loading policy")
    if len(leaders) == 1:
        teleop = SOLeader(
            SOLeaderTeleopConfig(
                id=TELEOP_ID_SINGLE,
                calibration_dir=Path(leader_cal_dir),
                port=leaders[0]["port"],
            )
        )
    else:
        teleop = BiSOLeader(
            BiSOLeaderConfig(
                id=TELEOP_ID,
                calibration_dir=Path(leader_cal_dir),
                left_arm_config=SOLeaderConfig(port=leaders[0]["port"], use_degrees=True),
                right_arm_config=SOLeaderConfig(port=leaders[1]["port"], use_degrees=True),
            )
        )
    try:
        teleop.connect(calibrate=True)
        log.info("Preflight leader motor check passed")
    finally:
        disconnect(teleop)


def load_dataset_stats_from_pretrained(pretrained_path: str | Path) -> dict[str, dict[str, Any]] | None:
    """Load the (feature -> {stat_name: tensor}) dataset_stats dict bundled
    with a saved lerobot policy checkpoint's own preprocessor pipeline --
    i.e. the normalization the model was actually TRAINED with.

    For online RL (rlt_ac), the outer ChunkACPolicy wrapper is built fresh
    every session (no --policy.path of its own) against a brand-new,
    zero-episode dataset, so `make_pre_post_processors()`'s usual
    `dataset_stats` source (the recording dataset's own stats) is always
    empty. Without this, the frozen VLA -- which DOES expect properly
    normalized state/action, per its own saved
    `policy_preprocessor.json`/`*_normalizer_processor.safetensors` -- ends
    up fed effectively un-normalized (or default-normalized) observations
    despite loading the right weights, which reads as the VLA "acting
    randomly" even outside any RL involvement. This loads that checkpoint's
    real stats so they can be passed through instead.

    Returns None if the checkpoint has no normalizer_processor step (e.g. a
    non-lerobot-standard checkpoint, or one saved without normalization).
    """
    from safetensors.torch import load_file

    pretrained_path = Path(pretrained_path)
    preprocessor_json = pretrained_path / "policy_preprocessor.json"
    if not preprocessor_json.is_file():
        return None
    with open(preprocessor_json) as fh:
        spec = json.load(fh)
    state_file = next(
        (
            step.get("state_file")
            for step in spec.get("steps", [])
            if step.get("registry_name") == "normalizer_processor"
        ),
        None,
    )
    if not state_file:
        return None
    flat = load_file(str(pretrained_path / state_file))
    stats: dict[str, dict[str, Any]] = {}
    for key, tensor in flat.items():
        # Feature names themselves contain dots (observation.state,
        # observation.images.left_wrist); stat names (mean/std/min/max/
        # q01.../q99) never do, so the LAST dot always separates them.
        feature_name, stat_name = key.rsplit(".", 1)
        stats.setdefault(feature_name, {})[stat_name] = tensor
    return stats


def set_offline_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    _quiet_video_encoder_logs()


def _quiet_video_encoder_logs() -> None:
    """Suppress libx264's per-container startup banner (cpu caps, codec info)
    that streaming video encoding prints once per camera per episode chunk.
    PyAV's log callback is process-global, so setting it once here covers
    encoders created later in background threads too."""
    try:
        import av

        av.logging.set_level(av.logging.ERROR)
    except ImportError:
        pass
