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
conda activate rlt_dual && cd ~/intern/RLT_dual
python -m evo_rlt.adapters.lerobot.record.camera_resolve --snapshot
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
## 3. 采数据
```bash
conda activate rlt_dual && cd ~/intern/RLT_dual

evo-rlt-record full --initial-source teleop \
  --setup-json configs/crp_dual_manifest.json \
  --dataset-tag screw_demo_v1 \
  --task "Pick up the workpiece and place it on the platform. Remove the pin from the holder and insert it into the hole on top of the workpiece. Then place the assembled object in the target area." \
  --num-episodes 90 \
  --episode-time-s 300 \
  --reset-time-s 7 \
  --vcodec h264 \
  --discard-unlabeled-episodes

```
# 继续采集数据
conda activate rlt_dual && cd ~/intern/RLT_dual

evo-rlt-record full --initial-source teleop \
  --setup-json configs/crp_dual_manifest.json \
  --resume-dir data/crp_dual/0818_screw_demo_v1/record_teleop_full_110819  \
  --task "Pick up the workpiece and place it on the platform. Remove the pin from the holder and insert it into the hole on top of the workpiece. Then place the assembled object in the target area." \
  --num-episodes 50 \
  --episode-time-s 300 \
  --reset-time-s 10 \
  --vcodec h264 \
  --discard-unlabeled-episodes


### 采完必须先查实际帧率

```bash
python -c "
import csv, statistics as st
rows=list(csv.reader(open('/tmp/frame_timing.csv'))); h=rows[0]
c=list(zip(*[[float(x) for x in r] for r in rows[1:] if len(r)==len(h)]))
print(f'实际 {1000/st.median(c[1]):.1f} Hz | send {st.median(c[4]):.0f}ms obs {st.median(c[2]):.0f}ms')"
```

三路相机使用后台采集，控制循环只取各自最新的新帧；结束时还会输出
`/tmp/frame_timing_summary.json`。目标是中位实际频率至少 15.2 Hz，且
`slow_fraction`（超过目标周期 1.2 倍的帧）不高于 10%。不满足时终端会打印
`[Timing]` warning，当前 episode 不应并入训练集。数据集按 16 fps 写时间戳，
实际采得慢会让回放速度和 action chunk 的时间尺度都出错。

## 4. 回放

### 只看录像（不动机器人）

```bash
conda activate rlt_dual && cd ~/intern/RLT_dual

ls -dt data/crp_dual/*/*/          # 看有哪些，改下面的路径

evo-rlt-dataset-viz \
  --root data/crp_dual/0819_screw_demo_v1/record_teleop_full_111237 \
  --repo-id local/record_teleop_full_111237 \
  --episodes all \
  --stride 4 \
  --jpeg-quality 85 \
  --batch-size 32 \
  --num-workers 4 \
  --memory-limit 75% \
  --tolerance-s 1e-4

# 看合并后的所有
evo-rlt-dataset-viz \
  --root data/crp_dual/crp_merged_screw_v1 \
  --episodes all \
  --stride 4 \
  --jpeg-quality 85 \
  --batch-size 32 \
  --num-workers 4 \
  --memory-limit 75% \
  --tolerance-s 1e-4

```

```bash
# 1. 先扫曲线找可疑的几条：不碰 mp4，72 条 / 3 万帧约 25 秒
evo-rlt-dataset-viz --root data/crp_dual/0817_screw_demo_v1/record_teleop_full_172919 --no-images

# 2. 再对着那几条细看，全帧不抽
evo-rlt-dataset-viz --root data/crp_dual/0818_screw_demo_v1/record_teleop_full_154945 \
  --episodes 79-82 --stride 1

# 3. 数据在远端机器上
evo-rlt-dataset-viz --root data/crp_dual/0817_screw_demo_v1/record_teleop_full_172919 \
  --episodes all --stride 8 --serve --grpc-port 9876
```

### 真机回放

```bash
evo-rlt-crp-replay \
  --root data/crp_dual/0817_screw_demo_v1/record_teleop_full_140733 \
  --repo-id local/record_teleop_full_140733 \
  --episode 0
```

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

## 5. 合并数据

### 合并 session

```bash
conda activate rlt_dual && cd ~/intern/RLT_dual

HF_HUB_OFFLINE=1 python -m evo_rlt.cli.merge_lerobot_datasets \
  --input-parent data/crp_dual/0817_screw_demo_v1 \
  --output-repo-id local/crp_merged_screw_v1 \
  --output-root data/crp_dual/crp_merged_screw_v1 \
  --repo-id-prefix local/crp_0817_
```
conda activate rlt_dual && cd ~/intern/RLT_dual

HF_HUB_OFFLINE=1 python -m evo_rlt.cli.merge_lerobot_datasets \
  --input-parent data/crp_dual/0817_screw_demo_v1 \
  --input-parent data/crp_dual/0818_screw_demo_v1 \
  --output-repo-id local/crp_merged_screw_v2 \
  --output-root data/crp_dual/crp_merged_screw_v2 \
  --repo-id-prefix local/crp_

### 合并多天的 session

`--input-parent` 可以重复给，按给出的顺序拼接。**不要覆盖正在被训练读取的目录**，换一个新的 `--output-root`。

```bash
conda activate rlt_dual && cd ~/intern/RLT_dual

HF_HUB_OFFLINE=1 python -m evo_rlt.cli.merge_lerobot_datasets \
  --input-parent data/crp_dual/0817_screw_demo_v1 \
  --input-parent data/crp_dual/0818_screw_demo_v1 \
  --output-repo-id local/crp_merged_screw_v2 \
  --output-root data/crp_dual/crp_merged_screw_v2 \
  --repo-id-prefix local/crp_
```

合并前要求各 session 的 `fps` / `robot_type` / 相机 key 与分辨率完全一致，脚本最后会核对 episode 数和帧数是否等于各输入之和，不一致直接报错。

### 核对合并结果

```bash
python -c "
import json;i=json.load(open('data/crp_dual/crp_merged_screw_v1/meta/info.json'))
print(i['total_episodes'],'ep /',i['total_frames'],'帧 @',i['fps'],'fps')"
```

### 回放合并后的数据

```bash
HF_HUB_OFFLINE=1 lerobot-dataset-viz \
  --root data/crp_dual/crp_merged_screw_v2 \
  --repo-id local/crp_merged_screw_v2 \
  --episode-index 463 --mode local
```

### 删除坏 episode

```bash
lerobot-edit-dataset \
  --repo_id local/crp_merged_screw_v1 \
  --root data/crp_dual/crp_merged_screw_v1 \
  --new_repo_id local/crp_merged_screw_v1 \
  --new_root data/crp_dual/crp_merged_screw_v1 \
  --operation.type delete_episodes \
  --operation.episode_indices "[3]"
```

## 6. 训练 ACT

### 先统一 CRP roll 表示（只需执行一次）

CRP 锁定姿态位于欧拉角的 `-180/+180` 分界，同一姿态在原始数据里有两种数值。
只处理合并后的 v3；`0817/0818/0819_screw_demo_v1` 三天原始 session 不会被修改。

```bash
python -m evo_rlt.cli.canonicalize_crp_rolls \
  --root data/crp_dual/crp_merged_screw_v3
```

清洗后 `left_roll.pos` / `right_roll.pos` 应当都是 `-180`，必须重新训练，不能续训
使用旧归一化统计量的 checkpoint。

### 训练

直接使用 v3 的全部 648 条 episode 训练，不留验证集。

```bash
conda activate rlt_dual && cd ~/intern/RLT_dual

HF_HUB_OFFLINE=1 lerobot-train \
  --dataset.repo_id=local/crp_merged_screw_v3 \
  --dataset.root=data/crp_dual/crp_merged_screw_v3 \
  --policy.type=act \
  --policy.chunk_size=32 \
  --policy.n_action_steps=8 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --steps=100000 \
  --save_freq=10000 \
  --eval_freq=0 \
  --num_workers=2 \
  --wandb.enable=true \
  --wandb.project=crp_dual_act \
  --wandb.disable_artifact=true \
  --output_dir=outputs/act_crp_screw_v3 \
  --job_name=act_crp_screw_v3

```

当前没有仿真评估环境，因此 `eval_freq` 保持为 `0`。通过定期保存 checkpoint，并在固定
真机工况下比较成功率来选择部署模型。

### 续训

```bash
HF_HUB_OFFLINE=1 lerobot-train \
  --config_path=outputs/act_crp_screw_v3/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

### 部署 / 采 ACT baseline

```bash

conda activate rlt_dual && cd ~/intern/RLT_dual

evo-rlt-record full --initial-source vla \
  --setup-json configs/crp_dual_manifest.json \
  --policy-path outputs/act_crp_screw_v3/checkpoints/080000/pretrained_model \
  --policy-n-action-steps 8 \
  --task "Pick up the workpiece and place it on the platform. Remove the pin from the holder and insert it into the hole on top of the workpiece. Then place the assembled object in the target area." \
  --dataset-tag act_eval_080000 \
  --num-episodes 10 \
  --episode-time-s 300 \
  --reset-time-s 10 \
  --vcodec h264 \
  --no-teleop
```

## 7. 训练 smolVLA

### 环境补充（一次性）

```bash
conda activate rlt_dual
pip install 'num2words>=0.5.14,<0.6.0'
```

### 下载权重（一次性）

```bash
conda activate rlt_dual && cd ~/intern/RLT_dual

python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('lerobot/smolvla_base'))"

python -c "
from transformers import AutoProcessor
AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM2-500M-Video-Instruct')
print('processor ok')"
```

### 训练

```bash
conda activate rlt_dual && cd ~/intern/RLT_dual

RM='{"observation.images.top": "observation.images.camera1", "observation.images.left_wrist": "observation.images.camera2", "observation.images.right_wrist": "observation.images.camera3"}'

HF_HUB_OFFLINE=1 lerobot-train \
  --dataset.repo_id=local/crp_merged_screw_v2 \
  --dataset.root=data/crp_dual/crp_merged_screw_v2 \
  --policy.path=lerobot/smolvla_base \
  --policy.load_vlm_weights=false \
  --policy.chunk_size=32 \
  --policy.n_action_steps=32 \
  --policy.scheduler_decay_steps=30000 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --rename_map="$RM" \
  --batch_size=16 \
  --steps=30000 \
  --save_freq=5000 \
  --eval_freq=0 \
  --num_workers=2 \
  --wandb.enable=true \
  --wandb.project=crp_dual_smolvla \
  --wandb.disable_artifact=true \
  --output_dir=outputs/smolvla_crp_screw_v1 \
  --job_name=smolvla_crp_screw_v1
```

### 续训

```bash
conda activate rlt_dual && cd ~/intern/RLT_dual

HF_HUB_OFFLINE=1 lerobot-train \
  --config_path=outputs/smolvla_crp_screw_v1/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

### 部署前 dry-run

```bash
conda activate rlt_dual && cd ~/intern/RLT_dual

RM='{"observation.images.top": "observation.images.camera1", "observation.images.left_wrist": "observation.images.camera2", "observation.images.right_wrist": "observation.images.camera3"}'

evo-rlt-record full --initial-source vla \
  --setup-json configs/crp_dual_manifest.json \
  --policy-path outputs/smolvla_crp_screw_v1/checkpoints/last/pretrained_model \
  --rename-map "$RM" \
  --task "Pick up the workpiece and place it on the platform. Remove the pin from the holder and insert it into the hole on top of the workpiece. Then place the assembled object in the target area." \
  --dataset-tag smolvla_eval \
  --num-episodes 10 \
  --episode-time-s 300 \
  --reset-time-s 10 \
  --vcodec h264 \
  --no-teleop \
  --dry-run
```

### 部署 / 采 smolVLA baseline

```bash
conda activate rlt_dual && cd ~/intern/RLT_dual

RM='{"observation.images.top": "observation.images.camera1", "observation.images.left_wrist": "observation.images.camera2", "observation.images.right_wrist": "observation.images.camera3"}'

evo-rlt-record full --initial-source vla \
  --setup-json configs/crp_dual_manifest.json \
  --policy-path outputs/smolvla_crp_screw_v1/checkpoints/last/pretrained_model \
  --rename-map "$RM" \
  --task "Pick up the workpiece and place it on the platform. Remove the pin from the holder and insert it into the hole on top of the workpiece. Then place the assembled object in the target area." \
  --dataset-tag smolvla_eval \
  --num-episodes 10 \
  --episode-time-s 300 \
  --reset-time-s 10 \
  --vcodec h264 \
  --no-teleop
```

## 8. 训练 pi0.5

### SFT

```bash
export TORCHDYNAMO_DISABLE=1

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/crp_merged_screw_v1 \
  --dataset.root=data/crp_dual/crp_merged_screw_v1 \
  --policy.path=<PI05_BASE_DIR> \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --steps=100000 \
  --save_freq=10000 \
  --eval_freq=0 \
  --num_workers=2 \
  --tolerance_s=1e-4 \
  --output_dir=outputs/pi05_crp_ft \
  --job_name=pi05_crp_ft
```

### 续训

```bash
export TORCHDYNAMO_DISABLE=1

python -m lerobot.scripts.lerobot_train \
  --config_path=outputs/pi05_crp_ft/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

### 接着已有 ckpt 再训

```bash
export TORCHDYNAMO_DISABLE=1

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/crp_merged_screw_v1 \
  --dataset.root=data/crp_dual/crp_merged_screw_v1 \
  --policy.path=outputs/pi05_crp_ft/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --steps=100000 \
  --save_freq=10000 \
  --eval_freq=0 \
  --num_workers=2 \
  --tolerance_s=1e-4 \
  --output_dir=outputs/pi05_crp_ft_stage2 \
  --job_name=pi05_crp_ft_stage2
```

### 部署 / 采 pi0.5 baseline

```bash
conda activate rlt_dual && cd ~/intern/RLT_dual

evo-rlt-record full --initial-source vla \
  --setup-json configs/crp_dual_manifest.json \
  --policy-path outputs/pi05_crp_ft/checkpoints/last/pretrained_model \
  --task "Pick up the workpiece and place it on the platform. Remove the pin from the holder and insert it into the hole on top of the workpiece. Then place the assembled object in the target area." \
  --dataset-tag pi05_baseline_eval \
  --num-episodes 30 \
  --episode-time-s 300 \
  --reset-time-s 10 \
  --vcodec h264
```

## 9. 排障

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
