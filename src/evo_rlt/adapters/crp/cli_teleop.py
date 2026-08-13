#!/usr/bin/env python

"""
Dual SO101 leader teleoperation for dual CRP arms (thin CLI wrapper).

    evo-rlt-crp-teleop --robot.ip1=192.168.0.100 --robot.ip2=192.168.0.101

Core logic lives in ``teleop_config`` / ``teleop_loop`` / ``mappers`` /
``ui_probe``. ``CRPArmDual`` is constructed directly rather than through
``make_robot_from_config``: vanilla LeRobot 0.5.1 dispatches that on a fixed
isinstance chain and does not know about CRP.
"""

import logging
from dataclasses import asdict
from pprint import pformat

from lerobot.configs import parser
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.utils import init_logging

from .arm_dual import CRPArmDual
from .mappers.so101_urdf import SO101_ARM_MOTOR_NAMES, resolve_so101_urdf_path
from .mappers.urdf_fk import UrdfChainFK
from .sdk import exit_process_after_dual_disconnect
from .teleop_config import TeleoperateDualCRPConfig
from .teleop_loop import TARGET_FRAME_NAME, lock_gps_to_current_tcp, teleop_loop_crp_dual
from .ui_probe import GripperUiProbeManager, apply_ui_probe_connect_policy

logger = logging.getLogger(__name__)


@parser.wrap()
def teleoperate_dual(cfg: TeleoperateDualCRPConfig) -> None:
    init_logging()
    logger.info(pformat(asdict(cfg)))

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = CRPArmDual(cfg.robot)

    ui_probe: GripperUiProbeManager | None = None
    if cfg.gripper_ui_probe:
        ui_probe = GripperUiProbeManager(hz=cfg.gripper_ui_probe_hz)
    apply_ui_probe_connect_policy(robot, ui_probe)

    teleop.connect()
    try:
        robot.connect()
    except Exception:
        teleop.disconnect()
        raise

    urdf_path = resolve_so101_urdf_path()
    kinematics = UrdfChainFK(
        urdf_path=urdf_path,
        target_frame_name=TARGET_FRAME_NAME,
        joint_names=list(SO101_ARM_MOTOR_NAMES),
    )
    logger.debug("URDF FK path=%s frame=%s", urdf_path, TARGET_FRAME_NAME)

    try:
        teleop_loop_crp_dual(teleop, robot, kinematics, cfg, ui_probe=ui_probe)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, stopping dual teleoperation.")
    finally:
        if ui_probe is not None:
            ui_probe.stop_all()
        try:
            lock_gps_to_current_tcp(robot, cfg)
        except Exception as exc:
            logger.warning("GP lock on exit failed: %s", exc)
        robot.disconnect()
        teleop.disconnect()
        exit_process_after_dual_disconnect()


def main() -> None:
    teleoperate_dual()


if __name__ == "__main__":
    main()
