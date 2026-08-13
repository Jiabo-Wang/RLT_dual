"""
CRP dual-arm follower support for Evo-RLT.

Hardware pairing: dual SO101 leader (``bi_so_leader``) drives two CRP arms over
one CrpRobotPy session. The leader's joints are run through SO101 FK and sent to
CRP as GP (Cartesian) increments — the two arms are not kinematically identical,
so this is not the joint-passthrough pairing that ACT-style dual SO101 uses.

Native runtime (``third_party/CrpRobotPy``: ``CrpRobotPy.so``,
``libRobotService.so``, ``license.key``) is machine-licensed and stays out of
git; see ``docs/CRP_SETUP.md``.
"""

from .config_arm_dual import CRPArmDualConfig

__all__ = ["CRPArmDualConfig"]
