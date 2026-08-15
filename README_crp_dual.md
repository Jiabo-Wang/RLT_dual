# CRP 双臂从臂 + SO101 双主臂

| 部件 | 地址 / 序列号 |
| --- | --- |
| CRP 左臂 | `192.168.0.100`（GP10） |
| CRP 右臂 | `192.168.0.101`（GP20） |
| SO101 左主臂 | USB `5B41533735` |
| SO101 右主臂 | USB `5AAF220248` |
| top 相机 | Orbbec Gemini 335 `CP0BB53000A0` |
| left_wrist | RealSense D405 `235123075818` |
| right_wrist | RealSense D405 `235123073792` |

## 0. 环境（一次性）

```bash
conda create -n rlt_dual --clone evo-rlt -y
conda activate rlt_dual
cd ~/intern/RLT_dual
pip install -e ".[lerobot,crp]"
```
CRP 原生 SDK（闭源，不入 git）：

```bash
cd ~/intern/RLT_dual
SDK=~/intern/CrobotpOSSDK-1.4.6-Linux-x86_64-ubuntu22/bin
FORK=~/intern/Crp/lerobot

mkdir -p third_party/CrpRobotPy
cp $SDK/{libRobotService.so,license.key,identifier.id} third_party/CrpRobotPy/
cp $FORK/third_party/CrpRobotPy/CrpRobotPy.so          third_party/CrpRobotPy/
cp $SDK/license.key .

PROBE=src/evo_rlt/adapters/crp/getui_probe
cp $SDK/{libRobotService.so,license.key} $PROBE/
cp $FORK/src/lerobot/robots/crp_arm_dual/getui_probe/crp_getui_probe $PROBE/
chmod +x $PROBE/crp_getui_probe
```

`license.key` 必须在仓库根目录（SDK 按进程 CWD 找），否则 `unlicensed`。

换机器时重新生成指纹，找厂家换 license：

```bash
cd $SDK && LD_LIBRARY_PATH=$SDK ./identifier && cat identifier.id
```
改了 `crp_getui_probe.cpp` 才需要重编：
```bash
CRP_OSSDK_ROOT=~/intern/CrobotpOSSDK-1.4.6-Linux-x86_64-ubuntu22 \
  src/evo_rlt/adapters/crp/getui_probe/build.sh
```

## 1. 硬件检查

```bash
conda activate rlt_dual && cd ~/intern/RLT_dual
ping -c3 192.168.0.100
ping -c3 192.168.0.101
ls -l /dev/serial/by-id/
# lisencse
ls -l license.key
# 应列出 crp_dual_leader_left.json / crp_dual_leader_right.json
ls ~/.cache/huggingface/lerobot/calibration/teleoperators/crp_dual_leader/
```
```bash
ffplay -fflags nobuffer -flags low_delay /dev/video8
```

### 主臂标定
```bash
python -m evo_rlt.adapters.crp.calibrate_leader --side left
python -m evo_rlt.adapters.crp.calibrate_leader --side right
```
### 查看标定文件使用路径
```bash
python -m evo_rlt.adapters.crp.calibrate_leader --side right --show-path
```

## 2. 遥操

```bash
conda activate rlt_dual && cd ~/intern/RLT_dual
evo-rlt-crp-teleop
```

**启动前先把两臂的 J6 摇回 0°**（示教器关节坐标模式）。J6 不在 0 时，示教器程序一启动
就可能报 `J6轴关节速度超过最大允许值` —— `MoveL` 拿到笛卡尔位姿后自己解 IK，6 轴机器人
同一位姿有多组关节解，J6 已经转开时控制器可能解到另一个分支，第一个动作就是把 J6 全速
甩过去。实测：右臂 J6=27° 必报，J6=0° 从不报。程序**启动那一刻**才有这个风险，跑起来
之后随便转。

启动时若 J6 偏离 0 超过 5°，日志会在倒计时前警告：

```
[right] J6 is at 27.1 deg, not near 0. ...
```

按 START 的时机也要对——等这两行打印出来**之后**再按：

```
[left]  GP locked to current TCP (...)
[right] GP locked to current TCP (...)
```

这两行之前 GP 寄存器里还是上次退出时锁的旧坐标，提前按会让机器人冲向那个旧位置。
## 3. 采数据
```bash
conda activate rlt_dual && cd ~/intern/RLT_dual

evo-rlt-record full --initial-source teleop \
  --setup-json configs/crp_dual_manifest.json \
  --dataset-tag screw_demo_v1 \
  --task "Pick up the workpiece and place it on the platform. Remove the pin from the holder and insert it into the hole on top of the workpiece. Then place the assembled object in the target area." \
  --num-episodes 3 \
  --episode-time-s 300 \
  --reset-time-s 10 \
  --vcodec h264 \
  --discard-unlabeled-episodes
```

### 每条 episode 的流程

```
录制（最长 --episode-time-s）→ 按 s/f 标注 → 复位窗口（--reset-time-s）→ 下一条
```

VLA 阶段（`--initial-source vla`，有策略在跑）**不绑定 `f`**：跑坏的一条用 ← 重录，
而不是标记失败。`s` 和方向键照常。副作用是失去了标注失败的唯一手段，而在线 RL 的
warmup 会统计失败条数（`min_warmup_failures`，默认 3）—— 要跑在线 RL 得把这个阈值
调下来，否则 warmup 永远结束不了。

`--discard-unlabeled-episodes` 会丢掉没按 s/f 就超时结束的条目；不想手动标注就换成 `--default-episode-success success`。

### 采完必须先查实际帧率

```bash
python -c "
import csv, statistics as st
rows=list(csv.reader(open('/tmp/frame_timing.csv'))); h=rows[0]
c=list(zip(*[[float(x) for x in r] for r in rows[1:] if len(r)==len(h)]))
print(f'实际 {1000/st.median(c[1]):.1f} Hz | send {st.median(c[4]):.0f}ms obs {st.median(c[2]):.0f}ms')"
```

到 14–16 Hz 才算对。**低于这个值的数据不要用来训练** —— 数据集会按 16 fps 存时间戳，实际采得慢就等于时间戳造假，回放速度和 action chunk 的时间尺度都是错的。

## 4. 回放

### 只看录像（不动机器人）

```bash
ls -dt data/crp_dual/*/*/          # 看有哪些，改下面的路径

HF_HUB_OFFLINE=1 lerobot-dataset-viz \
  --root data/crp_dual/0815_screw_demo_v1/record_teleop_full_154118 \
  --repo-id local/record_teleop_full_154118 \
  --episode-index 0 --mode local
```

目录名里的 `0815_screw_demo_v1` 来自 `--dataset-tag`，不是固定的 `teleop_full`。
`repo-id` = 末级目录名前加 `local/`。`HF_HUB_OFFLINE=1` 不能省：`--root` 写错时
lerobot 会去查 HuggingFace Hub，报一个和本地路径无关的 404。

### 真机回放

```bash
evo-rlt-crp-replay \
  --root data/crp_dual/0815_screw_demo_v1/record_teleop_full_154118 \
  --repo-id local/record_teleop_full_154118 \
  --episode 0
```

同样在倒计时期间按绿色 START。数据集存的是**绝对** GP 位姿，所以起播前会检查第 0 帧离当前 TCP 多远，超过 50mm 直接拒绝（避免全速横穿工作空间）：

```
[left] frame 0 is 12 mm from the current TCP
```

超了就先把臂摇到起始位姿附近；确认路径无障碍再用 `--allow-jump` 跳过检查。

| 参数 | 默认 |
| --- | --- |
| `--episode` | 0 |
| `--fps` | 数据集自带的 |
| `--max-jump-mm` | 50 |
| `--allow-jump` | 关 |
| `--start-delay-s` | 5 |

用的是 `lerobot-replay` 之外的独立入口，因为 CRP 需要 START 倒计时和第 0 帧跳跃保护，那两样它没有。动作向量按录制原样下发，不做任何重映射。

### 数据管理

```bash
# 看有哪些 session
for d in data/crp_dual/*/*/; do python -c "
import json;i=json.load(open('$d/meta/info.json'))
print(f\"{'$d'}: {i['total_episodes']} ep / {i['total_frames']} 帧 @ {i['fps']}fps\")"; done

# 删某条 episode
lerobot-edit-dataset --repo_id local/xxx --root data/crp_dual/.../xxx \
  --new_repo_id local/xxx --new_root data/crp_dual/.../xxx \
  --operation.type delete_episodes --operation.episode_indices "[3]"
```

失败的运行会留下 0 帧的空目录，直接 `rm -rf` 掉。

## 5. 排障

**`unlicensed` / `Failed to obtain IRobotService`**
仓库根目录缺 `license.key`。SDK 按进程 CWD 找。

**`set_GPs` 全返回 true，机器人不动**
示教器程序没在跑。查报警 → Reset → 伺服使能 → Auto → 绿色 START（常亮）。
连接时的 `switch_work_mode` + `servo_power_on` 会停掉已在运行的程序，所以要在
倒计时**之后**按 START。

**`J6轴关节速度超过最大允许值250`，按 START 瞬间就报**
J6 没归零。示教器上把 J6 摇回 0° 再启动，见第 2 节。跟遥操的转速参数无关 —— 报错发生在
程序执行第一条 `MoveL` 时，主循环里的限幅一个都还没生效。

**每次连接都提示重新标定**
舵机 EEPROM 和标定文件对不上。按 **ENTER** 用文件里的值写回舵机；按 `c` 是全量重标。

**相机报 `No camera with serial ...`**
相机没插，或序列号写错。报错会列出实际连接的序列号。

## 6. 实测数据（2026-08-14，左臂）

| 项 | 值 |
| --- | --- |
| ping 控制器 | 1.4 ms |
| SDK 单次调用 | 43 ms |
| 双臂 GP 下发 | 46 ms |
| 控制器死区 | 178 ms |
| GP 拒绝率 | 0% |
| 从臂速度（调倍率前 / 后） | 28 / ~222 mm/s |
| 三相机并发 | 23.8 fps @640x480 |

主循环上限 16–21 Hz，所以 `--fps` 定 16。

## 7. 待办

- [ ] **上机验证采集与回放**。以下四处修复只过了单元测试和离线数据，未经真机：
  leader 关节→GP 映射接线（`record_action.py` 原本零引用）、`send_action` 的 UI50
  写入门控（实测 send 183ms→预期 46ms）、`_last_ui50` 初始化、connect 后的 START 倒计时
- [ ] 相机三台并发不稳定：同一 USB3 控制器上 isoc 带宽预留冲突，降分辨率/帧率/MJPG/
  backend/warmup 均无效。正解是装 `pyrealsense2` 改走 lerobot 原生 realsense 后端
- [ ] `record/hil.py` 干预释放靠写主臂 `Goal_Position`，CRP 做不到，改成重新对齐 GP
- [ ] （可选）放开 wrist_flex：标定 `(axis, sign)`；查清右臂限幅为何比左臂紧一个数量级
