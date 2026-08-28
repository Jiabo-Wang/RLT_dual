"""Preflight for a pi0.5 deployment: load the real weights, grab the real cameras.

``--dry-run`` on ``evo-rlt-record`` only echoes the argv it would pass to
``lerobot-record`` -- it never touches the checkpoint or the hardware. This does
the parts that actually fail: resolve the three cameras, load the checkpoint
through the same path ``backend.py`` uses, push one real frame through the
preprocessor -> policy -> postprocessor chain, and time the result against the
16 fps control budget.

It pings the arms but never commands them, so it is safe to run at any time --
with the robots powered off you just get a failed reachability check.

    conda activate crp_rlt_small && cd ~/RLT_dual
    python diagnostics/pi05_preflight.py                       # 相机 + 权重 + 推理
    python diagnostics/pi05_preflight.py --skip-cameras        # 只验权重和推理
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = REPO / "pretrained_model"
DEFAULT_DATASET = REPO / "data/crp_dual/crp_rlt_dataset"
DEFAULT_MANIFEST = REPO / "configs/crp_dual_manifest.json"
FPS = 16

# pi0.5 expects openpi camera names; the robot emits top/left_wrist/right_wrist.
RENAME_MAP = {
    "observation.images.top": "observation.images.base_0_rgb",
    "observation.images.left_wrist": "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist": "observation.images.right_wrist_0_rgb",
}
TASK = (
    "Pick up the workpiece and place it on the platform. Remove the pin from the holder "
    "and insert it into the hole on top of the workpiece. Then place the assembled object "
    "in the target area."
)


def _peak_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def check_cameras(manifest: Path) -> list[str]:
    """Confirm every camera the manifest names resolves to a colour node by serial."""
    from evo_rlt.adapters.lerobot.record.camera_resolve import color_node_for_serial

    wanted = {c["alias"]: c["port"]["serial"] for c in json.loads(manifest.read_text())["cameras"]}
    problems = []
    for alias, serial in wanted.items():
        try:
            node = color_node_for_serial(serial)
        except ValueError as exc:
            problems.append(f"相机 {alias} (SN {serial}): {exc}")
            print(f"  相机 {alias:12s} SN {serial:14s} -> 找不到")
        else:
            print(f"  相机 {alias:12s} SN {serial:14s} -> {node}")
    return problems


def _interface_on_subnet(ip: str) -> str | None:
    """The interface holding an address in ``ip``'s own /24, if any.

    A plain ping is not enough here. This host's WiFi carries 192.168.5.49/21,
    a mask wide enough to swallow 192.168.0.0/24, so the kernel routes the arms'
    addresses out over WiFi and an unrelated device answering there reads as
    "arm reachable" while the wired port is down. Measured on 2026-08-28:
    192.168.0.100 replied from 18:93:41:59:c4:b9 over wlp13s0 while the arm's
    own ARP entry on enp12s0 was INCOMPLETE.
    """
    prefix = ip.rsplit(".", 1)[0] + "."
    out = subprocess.run(
        ["ip", "-o", "-4", "addr", "show"], capture_output=True, text=True
    ).stdout
    for line in out.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        name, cidr = fields[1], fields[3]
        addr, _, mask = cidr.partition("/")
        # Only an interface configured *on that /24* counts; a /21 that merely
        # contains it is the false positive this exists to reject.
        if addr.startswith(prefix) and int(mask or 32) >= 24:
            return name
    return None


def check_arms(manifest: Path) -> list[str]:
    """Ping every CRP follower, bound to the interface on its subnet.

    Read-only -- no SDK session, so it is safe with the arms powered off.
    """
    arms = [
        a for a in json.loads(manifest.read_text())["arms"]
        if a.get("type") == "follower" and a.get("ip")
    ]
    problems = []
    for arm in arms:
        ip, alias = arm["ip"], arm["alias"]
        iface = _interface_on_subnet(ip)
        if iface is None:
            print(f"  臂 {alias:16s} {ip:16s} -> 本机没有接口在这个网段上")
            problems.append(
                f"臂 {alias} ({ip}): 没有任何接口配置在 {ip.rsplit('.', 1)[0]}.0/24 上。"
                f"检查有线口是否插好（cat /sys/class/net/enp12s0/carrier 应为 1）"
            )
            continue
        ok = subprocess.run(
            ["ping", "-c2", "-W2", "-I", iface, ip],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
        print(f"  臂 {alias:16s} {ip:16s} -> {'通' if ok else '不通'}  (经 {iface})")
        if not ok:
            problems.append(f"臂 {alias} ({ip}) 经 {iface} 不通 —— 检查上电与网线")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--skip-cameras", action="store_true")
    ap.add_argument("--skip-arms", action="store_true")
    args = ap.parse_args()

    problems: list[str] = []

    print("== 相机 ==")
    if args.skip_cameras:
        print("  (--skip-cameras)")
    else:
        try:
            problems += check_cameras(args.manifest)
        except Exception as exc:  # resolve_cameras shells out to v4l2; be loud but keep going
            problems.append(f"相机解析失败: {type(exc).__name__}: {exc}")
            print(f"  失败: {exc}")

    print("\n== 机械臂 ==")
    if args.skip_arms:
        print("  (--skip-arms)")
    else:
        problems += check_arms(args.manifest)

    import numpy as np
    import torch

    from evo_rlt.adapters.lerobot.pi05_low_mem_load import install
    from lerobot.configs.policies import PreTrainedConfig
    from evo_rlt.adapters.lerobot._compat import build_dataset_frame, predict_action
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.device_utils import get_safe_torch_device

    print("\n== CUDA 库 ==")
    from evo_rlt.cli.common import warn_on_shadowing_cuda_libs
    shadowing = warn_on_shadowing_cuda_libs()
    if shadowing:
        # Not fatal: some setups need a system CUDA. But this failure otherwise
        # surfaces as CUBLAS_STATUS_INVALID_VALUE inside a GEMM, far from its cause.
        print(f"  LD_LIBRARY_PATH 中有可能顶掉 torch 自带 CUDA 的目录: {', '.join(shadowing)}")
        print("  若之后报 CUBLAS_STATUS_INVALID_VALUE, 用 LD_LIBRARY_PATH= 重跑")
    else:
        print("  没有会顶掉 torch 自带 CUDA 的目录")

    print("\n== 权重 ==")
    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.pretrained_path = str(args.policy_path)
    if cfg.type != "pi05":
        problems.append(f"checkpoint 不是 pi05, 是 {cfg.type}")
    install()

    t = time.time()
    policy = get_policy_class(cfg.type).from_pretrained(args.policy_path, config=cfg).eval()
    torch.cuda.synchronize()
    print(f"  加载 {time.time() - t:.1f}s | CPU 峰值 {_peak_gib():.1f} GiB "
          f"| 显存 {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    print(f"  chunk={cfg.chunk_size} n_action_steps={cfg.n_action_steps} "
          f"去噪步数={cfg.num_inference_steps} dtype={cfg.dtype}")

    print("\n== 推理链 ==")
    ds = LeRobotDataset(repo_id="local/crp_rlt_dataset", root=args.dataset_root)
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(args.policy_path),
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )

    sample = ds[0]
    raw_obs: dict = {}
    for cam in ("top", "left_wrist", "right_wrist"):
        img = sample[f"observation.images.{cam}"]
        img = img.numpy() if torch.is_tensor(img) else np.asarray(img)
        if img.ndim == 3 and img.shape[0] == 3:  # CHW float -> HWC uint8
            img = (img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        raw_obs[cam] = img
    state = sample["observation.state"]
    state = state.numpy() if torch.is_tensor(state) else np.asarray(state)
    for name, val in zip(ds.features["observation.state"]["names"], state):
        raw_obs[name] = float(val)

    device = get_safe_torch_device(cfg.device)

    def infer():
        frame = build_dataset_frame(ds.features, raw_obs, prefix=OBS_STR)
        return predict_action(
            observation=frame, policy=policy, device=device,
            preprocessor=pre, postprocessor=post,
            use_amp=cfg.use_amp, task=TASK, robot_type="crp_arm_dual",
        )

    action = infer()
    torch.cuda.synchronize()
    names = cfg.action_feature_names
    vals = action.flatten().float().cpu().tolist()
    print(f"  action ({len(vals)} 维):")
    for n, v in zip(names, vals):
        print(f"    {n:16s} {v:9.2f}")

    lat = []
    for _ in range(5):
        policy.reset(); pre.reset(); post.reset()
        torch.cuda.synchronize(); t = time.time(); infer(); torch.cuda.synchronize()
        lat.append(time.time() - t)
    med = sorted(lat)[len(lat) // 2]
    budget = cfg.n_action_steps / FPS
    print(f"\n  重规划中位 {med * 1000:.0f} ms | 预算 {budget:.2f}s "
          f"({cfg.n_action_steps} 步 @ {FPS} fps)")
    print(f"  峰值显存 {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    if med >= budget:
        problems.append(f"重规划 {med:.2f}s 超过 {budget:.2f}s 预算")

    print()
    if problems:
        print("preflight 未通过:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("preflight 通过 — 相机、机械臂、权重、推理链都就绪。")
    print("开跑前记得：示教器上清报警 -> Reset -> 伺服使能 -> Auto，")
    print("然后在 GP alignment 倒计时**之后**按绿色 START。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
