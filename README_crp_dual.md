# CRP 双臂从臂 + SO101 双主臂

| 部件 | 地址 / 序列号 |
| --- | --- |
| CRP 左臂 | `192.168.0.100`（GP10） |
| CRP 右臂 | `192.168.0.101`（GP20） |
| SO101 左主臂 | USB `5B41533735` |
| SO101 右主臂 | USB `5AAF220248` |
| top 相机 | Orbbec Gemini 335 `CP0BB53000A0` |
| left_wrist | RealSense D405 `235123073792` |
| right_wrist | RealSense D405 `235123075818` |

日常只跑第 1、2 节。第 0 节一次性，已做完。

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

# 应列出 5B41533735(左) 5AAF220248(右)
ls -l /dev/serial/by-id/

# 必须存在
ls -l license.key

# 应列出 crp_dual_leader_left.json / crp_dual_leader_right.json
ls ~/.cache/huggingface/lerobot/calibration/teleoperators/crp_dual_leader/

# 三台相机（8086:0b5b=D405, 2bc5:0800=Gemini335）
lsusb | grep -E "8086:0b5b|2bc5:0800"

# 序列号 -> color 节点
python -m evo_rlt.adapters.lerobot.record.camera_resolve
```

`camera_resolve` 输出（`0001` 是笔记本自带摄像头）：

```
235123073792   video6    left_wrist
235123075818   video12   right_wrist
CP0BB53000A0   video20   top
```
看画面：

```bash
ffplay -fflags nobuffer -flags low_delay /dev/video20   # top
ffplay -fflags nobuffer -flags low_delay /dev/video6    # left_wrist
ffplay -fflags nobuffer -flags low_delay /dev/video12   # right_wrist
```

### 主臂标定
```bash
python -m evo_rlt.adapters.crp.calibrate_leader --side left
python -m evo_rlt.adapters.crp.calibrate_leader --side right
```

只看路径不动机械臂：

```bash
python -m evo_rlt.adapters.crp.calibrate_leader --side right --show-path
```

## 2. 遥操

```bash
conda activate rlt_dual && cd ~/intern/RLT_dual
evo-rlt-crp-teleop
```
| 参数 | 默认 |
| --- | --- |
| `--fps` | 16 |
| `--gp_send_fps` | 16 |
| `--gp_position_step_mm` | 80 |
| `--gp_align_delay_s` | 5 |
| `--left.z_scale` / `--right.z_scale` | 2.0 |
| `--left.z_floor_mm` / `--right.z_floor_mm` | 42 |
| `--left.wrist_roll_sign` | -1.0 |
| `--left.wrist_flex_sign` | 0.0（关闭） |
| `--left.wrist_flex_axis` | `y` |
| `--left.wrist_flex_max_step_deg` | 12.0（左）/ 1.0（右） |
| `--robot.init_gj_on_connect` | true |
| `--robot.init_gp_on_connect` | true |

`wrist_flex_sign=0.0` 时主臂控制 xyz + roll 四个自由度，工具姿态锁在 GP 对齐那一刻。

放开 flex 要试 4 种组合（先压小限幅）：

```bash
evo-rlt-crp-teleop --left.wrist_flex_sign=1.0  --left.wrist_flex_max_step_deg=2.0
evo-rlt-crp-teleop --left.wrist_flex_sign=-1.0 --left.wrist_flex_max_step_deg=2.0
evo-rlt-crp-teleop --left.wrist_flex_sign=1.0  --left.wrist_flex_axis=x --left.wrist_flex_max_step_deg=2.0
evo-rlt-crp-teleop --left.wrist_flex_sign=-1.0 --left.wrist_flex_axis=x --left.wrist_flex_max_step_deg=2.0
```

## 3. 采数据

前置同第 2 节，外加三台相机插上。

```bash
conda activate rlt_dual && cd ~/intern/RLT_dual

evo-rlt-record full --initial-source teleop \
  --setup-json configs/crp_dual_manifest.json \
  --task "Insert the copper screw into the black sleeve." \
  --num-episodes 5 --dry-run
```

`--dry-run` 打印最终参数，确认：

```
--robot.type=crp_arm_dual --robot.ip1=192.168.0.100 --robot.ip2=192.168.0.101
--robot.cameras={"top": ..., "left_wrist": ..., "right_wrist": ...}
--dataset.fps=16
```

`index_or_path` 应是 `/dev/v4l/by-id/...` 不是数字。确认后去掉 `--dry-run`。

数据落在 `~/lerobot_data/crp_dual/<MMDD>_teleop_full/record_teleop_full_<HHMMSS>/`。

| 参数 | 说明 |
| --- | --- |
| `--num-episodes` | 采几条 |
| `--episode-time-s` | 单条上限秒数，默认 3000 |
| `--task` | 任务描述 |
| `--fps` | 不用传，默认取 manifest 的 16 |
| `--dataset-tag` | 目录名后缀，默认 `teleop_full` |
| `--default-episode-success` | 跳过手动标注成功/失败 |

## 4. 排障

**`unlicensed` / `Failed to obtain IRobotService`**
仓库根目录缺 `license.key`。SDK 按进程 CWD 找。

**`set_GPs` 全返回 true，机器人不动**
示教器程序没在跑。查报警 → Reset → 伺服使能 → Auto → 绿色 START（常亮）。
连接时的 `switch_work_mode` + `servo_power_on` 会停掉已在运行的程序，所以要在
倒计时**之后**按 START。

**每次连接都提示重新标定**
舵机 EEPROM 和标定文件对不上。按 **ENTER** 用文件里的值写回舵机；按 `c` 是全量重标。

**相机报 `No camera with serial ...`**
相机没插，或序列号写错。报错会列出实际连接的序列号。

## 5. 实测数据（2026-08-14，左臂）

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

## 6. 待办

- [ ] 上机跑一次真正的采集（第 3 节）。目前只有 `--dry-run` 和单元测试
- [ ] `record/hil.py` 干预释放靠写主臂 `Goal_Position`，CRP 做不到，改成重新对齐 GP
- [ ] （可选）放开 wrist_flex：标定 `(axis, sign)`；查清右臂限幅为何比左臂紧一个数量级
