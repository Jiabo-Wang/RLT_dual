# CRP 双臂从臂 + SO101 双主臂遥操

| 部件 | 地址 / 序列号 |
| --- | --- |
| CRP 左臂 | `192.168.0.100`（GP10） |
| CRP 右臂 | `192.168.0.101`（GP20） |
| SO101 左主臂 | USB `5B41533735` |
| SO101 右主臂 | USB `5AAF220248` |
| top 相机 | Orbbec Gemini 335 `CP0BB53000A0` |
| left_wrist | RealSense D405 `235123073792` |
| right_wrist | RealSense D405 `235123075818` |

> **第 0 节是一次性配置，这台机器已经做完，日常不用跑。**
> 每次上机只跑第 1 节（硬件检查）和第 2 节（遥操）。

## 0. 环境（一次性）
```bash
conda create -n rlt_dual --clone evo-rlt -y
conda activate rlt_dual
cd ~/intern/RLT_dual
pip install -e ".[lerobot,crp]"
```

### CRP 原生 SDK（一次性）
```bash
cd ~/intern/RLT_dual
SDK=~/intern/CrobotpOSSDK-1.4.6-Linux-x86_64-ubuntu22/bin
FORK=~/intern/Crp/lerobot

mkdir -p third_party/CrpRobotPy
cp $SDK/{libRobotService.so,license.key,identifier.id} third_party/CrpRobotPy/
cp $FORK/third_party/CrpRobotPy/CrpRobotPy.so          third_party/CrpRobotPy/

PROBE=src/evo_rlt/adapters/crp/getui_probe
cp $SDK/{libRobotService.so,license.key} $PROBE/
cp $FORK/src/lerobot/robots/crp_arm_dual/getui_probe/crp_getui_probe $PROBE/
chmod +x $PROBE/crp_getui_probe
```
`crp_getui_probe`（读示教器 UI56–58 夹爪反馈的子进程）上面已经拷了编好的二进制，
**不用重编**。只有改了 `crp_getui_probe.cpp` 才需要：
```bash
CRP_OSSDK_ROOT=~/intern/CrobotpOSSDK-1.4.6-Linux-x86_64-ubuntu22 \
  src/evo_rlt/adapters/crp/getui_probe/build.sh
```
## 1. 硬件检查

```bash
conda activate rlt_dual && cd ~/intern/RLT_dual

ping -c3 192.168.0.100
ping -c3 192.168.0.101

sudo chmod 666 /dev/ttyACM*

# 主臂：按 USB 序列号寻址，不用 /dev/ttyACM*（编号会随插拔顺序变，反了会左右臂对调）
# 应列出 5B41533735(左) 和 5AAF220248(右)
ls -l /dev/serial/by-id/

# 主臂标定：应列出 bimanual_leader_left.json / bimanual_leader_right.json
ls ~/.cache/huggingface/lerobot/calibration/teleoperators/bi_so_leader/
```

换过主臂 / 不确定哪条是左时，用这个确认（手动扳动**左**臂，`shoulder_pan.pos` 应该变）：

```bash
python -c "
import time
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
from evo_rlt.adapters.crp.teleop_config import LEADER_PORT_LEFT, LEADER_ID, default_leader_calibration_dir
t = SOLeader(SOLeaderTeleopConfig(port=LEADER_PORT_LEFT, use_degrees=True,
                                  id=f'{LEADER_ID}_left',
                                  calibration_dir=default_leader_calibration_dir()))
t.connect(calibrate=False)
for _ in range(30):
    print({k: round(v,1) for k,v in t.get_action().items()}); time.sleep(0.3)
t.disconnect()"
```

反了就把 `teleop_config.py` 里 `LEADER_PORT_LEFT` / `LEADER_PORT_RIGHT` 两个序列号对调。

标定文件缺了才需要重新标定（会要求手动摆姿态）：

```bash
lerobot-calibrate --teleop.type=bi_so_leader --teleop.id=bimanual_leader \
  --teleop.left_arm_config.port=$(python -c "from evo_rlt.adapters.crp.teleop_config import LEADER_PORT_LEFT as p; print(p)") \
  --teleop.right_arm_config.port=$(python -c "from evo_rlt.adapters.crp.teleop_config import LEADER_PORT_RIGHT as p; print(p)")
```

### 相机

```bash
# 三台是否都在（8086:0b5b=D405, 2bc5:0800=Gemini335）
lsusb | grep -E "8086:0b5b|2bc5:0800"

# 序列号
for d in /sys/bus/usb/devices/*/; do
  v=$(cat $d/idVendor 2>/dev/null); p=$(cat $d/idProduct 2>/dev/null)
  case "$v:$p" in
    8086:0b5b) echo "D405       $(cat $d/serial 2>/dev/null)";;
    2bc5:0800) echo "Gemini335  $(cat $d/serial 2>/dev/null)";;
  esac
done

# v4l2 节点：一台深度相机占 6~8 个节点（depth / 左右 IR / color），
# 节点号按内核枚举顺序分配，重新插拔就会变，所以按序列号反查 color 节点
for d in /sys/class/video4linux/video*; do
  n=$(basename $d); p=$(readlink -f $d/device)
  while [ "$p" != "/" ] && [ ! -f "$p/serial" ]; do p=$(dirname $p); done
  v4l2-ctl -d /dev/$n --list-formats 2>/dev/null | grep -qE "'(YUYV|MJPG)'" \
    && echo "/dev/$n  color  $(cat $p/serial 2>/dev/null)"
done
```

本机当次的结果（`0001` 是笔记本自带摄像头，忽略）：

| 挂载位 | 序列号 | depth | IR | color |
| --- | --- | --- | --- | --- |
| top | `CP0BB53000A0` | video14 | video16 / video18 | **video20** |
| left_wrist | `235123073792` | video2 | video4 | **video6** |
| right_wrist | `235123075818` | video8 | video10 | **video12** |

```bash
# 看画面：必须用上面查出来的 color 节点，挑到 depth 节点只会出一片灰白噪声
ffplay -fflags nobuffer -flags low_delay /dev/video20   # top        Orbbec
ffplay -fflags nobuffer -flags low_delay /dev/video6    # left_wrist D405
ffplay -fflags nobuffer -flags low_delay /dev/video12   # right_wrist D405
```

`v4l2-ctl` 来自 `v4l-utils`（`sudo apt install v4l-utils`）。

遥操不用相机，采数据才用。

## 2. 遥操

上机顺序：

1. 电柜伺服使能
2. 两个示教器切 **Auto**，装载 GP 程序（左读 GP10、右读 GP20）
3. 跑命令
4. 看到 `GP alignment in 5s...` 倒计时时，在两个示教器按**绿色 START**（常亮=运行，闪烁=没启动）

```bash
conda activate rlt_dual && cd ~/intern/RLT_dual
evo-rlt-crp-teleop
```

`Ctrl-C` 退出（会锁 GP 到当前 TCP 再断开）。心跳日志每秒一行：

```
loop=80
	L_GP=[512,-133,318] L_UI50=204, L_POSIT=201, L_SPEED=0, L_TORQUE=12
	R_GP=[498, 121,305] R_UI50=0,   R_POSIT=3,   R_SPEED=0, R_TORQUE=8
```

### 参数

| 参数 | 默认 |
| --- | --- |
| `--fps` | 80 |
| `--gp_send_fps` | 50 |
| `--gp_position_step_mm` | 80 |
| `--gp_align_delay_s` | 5 |
| `--left.z_scale` / `--right.z_scale` | 2.0 |
| `--left.z_floor_mm` / `--right.z_floor_mm` | 42 |
| `--left.wrist_roll_sign` | -1.0 |
| `--left.wrist_flex_sign` | 0.0（关闭，同 ACT） |
| `--left.wrist_flex_axis` | `y`（仅 sign≠0 时生效） |
| `--left.wrist_flex_max_step_deg` | 12.0（左）/ 1.0（右） |
| `--robot.init_gj_on_connect` | true |

### wrist_flex（默认关闭，同 ACT）

`wrist_flex_sign=0.0`：主臂 J4 不映射到 CRP 姿态。工具姿态锁在 **GP 对齐那一刻的
CRP TCP 姿态**，只有 `wrist_roll` 能绕工具 z 轴自转它。对齐时工具垂直，整个 episode
就保持垂直。所以主臂实际控制 **xyz + roll 四个自由度**。

要放开的话，`(axis, sign)` 有 4 种组合要上机试——SO101 末端系到 CRP 工具系的安装
旋转没建模，只能测。判据：主臂腕部低头，CRP 工具应同向低头；轴选错的表现是绕错误的
轴转（该低头却侧摆），不只是方向相反。先压小限幅：

```bash
evo-rlt-crp-teleop --left.wrist_flex_sign=1.0  --left.wrist_flex_max_step_deg=2.0  # (y, +1)
evo-rlt-crp-teleop --left.wrist_flex_sign=-1.0 --left.wrist_flex_max_step_deg=2.0  # (y, -1)
evo-rlt-crp-teleop --left.wrist_flex_sign=1.0  --left.wrist_flex_axis=x --left.wrist_flex_max_step_deg=2.0
evo-rlt-crp-teleop --left.wrist_flex_sign=-1.0 --left.wrist_flex_axis=x --left.wrist_flex_max_step_deg=2.0
```

默认 `wrist_flex_axis=y`（从 SO101 URDF 实测：flex 轴恒在工具 xy 平面内，方位角
`87.21° + roll`，roll=0 时几乎正好 +y）。fork 里写的 `x` 差了约 87°，基本正交——
这才是 fork 里 flex 一直用不了的原因，不是符号反了。

## 3. 待办

- [ ] 上机量 **GP 拒绝率** 和 **`set_GPs` 到实际运动的延迟**（决定 GP 闭环带宽够不够）
- [ ] （可选）放开 wrist_flex：标定 `(axis, sign)` 4 种组合；查清右臂限幅为何比左臂紧一个数量级
- [ ] 移植 Orbbec 相机后端（vanilla lerobot 0.5.1 没有）+ 装 `pyorbbecsdk` / `pyrealsense2`
- [ ] 实现 `CRPArmDual.send_action()`（现在直接 raise），action = 左右 GP6 + 左右 ui50
- [ ] `record/robot_config.py` 只认 `OpenCVCameraConfig`，要扩展 Orbbec / RealSense
- [ ] `record/hil.py` 干预释放靠写主臂 `Goal_Position`，CRP 做不到，改成重新对齐 GP
- [ ] 帧率：RLT 30fps vs CRP fork 16fps，先按 16 跑通
