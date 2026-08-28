# 在 A6000 上训 CRP 的 RL token（阶段 3）

写于 2026-08-28。命令已在采集机（RTX 5070 Ti / 30 GiB 内存）上跑通 20 步 smoke test，
所以到 A6000 之后不该再有环境层面的意外。本文只写这一个阶段。

---

## 0. A6000 上已经有什么

从 pi0.5 SFT 那次运行的 wandb 元数据还原（`wandb/run-20260819_211931-biy4hbcy/files/wandb-metadata.json`）：

| 东西 | 路径 |
| --- | --- |
| 主机 | `lenovo-ThinkStation-P920`，2× RTX A6000 48 GiB，503 GiB 内存 |
| conda 环境 | `/home/lenovo/miniforge3/envs/crp-rlt` |
| 演示数据集 | `/home/lenovo/桌面/crp_rlt_dataset` |
| pi0.5 SFT 输出 | `/home/lenovo/crp-rlt/outputs/vla_ft` |
| pi0.5 base | `/home/lenovo/models/pi05_base_a538eb27` |

**数据集和 SFT 权重都已经在那台机器上，不需要用移动硬盘搬。** 先核对一遍再动手：

```bash
ls ~/桌面/crp_rlt_dataset/meta/info.json
ls ~/crp-rlt/outputs/vla_ft/checkpoints/
python -c "
import json; i=json.load(open('$HOME/桌面/crp_rlt_dataset/meta/info.json'))
print(i['total_episodes'],'ep /',i['total_frames'],'帧 @',i['fps'],'fps |',i['robot_type'])"
# 期望: 622 ep / 220967 帧 @ 16 fps | crp_arm_dual
```

## 1. A6000 上缺什么

**`evo_rlt` 这个包。** pi0.5 SFT 走的是 `lerobot.scripts.lerobot_train`，没用到本仓库；
而 `Jiabo-Wang/crp-rlt` 是个纯文档仓库，没有 `src/`。RL token 训练需要
`evo_rlt.cli.train_rl_token`，所以要把代码拉过去：

```bash
cd ~ && git clone https://github.com/Jiabo-Wang/RLT_dual.git
cd RLT_dual

# 装进已有的 crp-rlt 环境（它已经有 lerobot + pi05 + torch）
conda activate crp-rlt
pip install -e . --no-deps          # --no-deps: 不要让它重新解析 lerobot 的 pin
python -c "import evo_rlt; print('evo_rlt OK')"
```

`--no-deps` 是有意的：`pyproject.toml` 的 `lerobot` extra 会去拉 v0.5.1 的 tarball，
而那台机器上的 lerobot 是训 pi0.5 时装好的、能跑的版本，不要动它。装完确认一下：

```bash
python -c "
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from evo_rlt.cli.common import load_training_config, assert_config_matches_dataset
cfg = load_training_config('src/evo_rlt/core/configs/crp_dual_rlt.yaml')
assert_config_matches_dataset(cfg, '$HOME/桌面/crp_rlt_dataset')
print('配置与数据集一致:', cfg.action_dim, '维,', cfg.control_hz, 'fps,', cfg.cameras)"
```

这一步会自己报错，不用肉眼比对 —— `assert_config_matches_dataset` 会核对维度、相机、fps。

## 2. Smoke test（先跑这个）

```bash
conda activate crp-rlt && cd ~/RLT_dual

HF_HUB_OFFLINE=1 python -m evo_rlt.cli.train_rl_token \
  --model-path ~/crp-rlt/outputs/vla_ft/checkpoints/last/pretrained_model \
  --demo-dataset-path ~/桌面/crp_rlt_dataset \
  --config src/evo_rlt/core/configs/crp_dual_rlt.yaml \
  --output-dir outputs/crp_rl_token_smoke \
  --steps 20 --save-every 20 --num-workers 0
```

采集机上这一步的实际输出，用来对照：

```
pi05 low-memory checkpoint loader installed
pi05 checkpoint loaded (low-memory path), all keys matched
Policy ready. RL token params: 100.7M
RLTDemoDataset: 220967 samples,
  cameras=['observation.images.top', 'observation.images.left_wrist',
           'observation.images.right_wrist'], chunk=50, normalize_actions=False
Step 0/20 loss=3.5962 avg100=3.5962 lr=0.00e+00
RL token training finished in 14.6s. final=2.8450 avg100=3.2996
```

三个必须对上的地方：**相机是 top/left_wrist/right_wrist**（不是 SO101 那套）、
**220967 samples**、**loss 在下降**。

## 3. 量 token 方差，决定 --norm-gamma

`reconstruction_loss` 重建的是 VLA 的 **2048 维 token 表征**（不是动作），
所以加权用的是每个 token 维度的 std。采集机上用 20 个 batch 量到的分布：

| | |
| --- | --- |
| std 范围 | 0.28 – 26.46（**93×**） |
| **方差最大的单个维度占 MSE** | **42.6%** |
| 前 10 维 | 57.0% |
| 中位 std | 0.54 |

**一个维度占掉 42.6%**，意味着用论文的原始 MSE（`gamma=0`，Eq. 2）时，RL token 的
瓶颈主要在编码那一个数。`--norm-gamma 0.5` 就是为这种情况准备的
（`rl_token.py` 的 docstring：*"partially compensates the few high-variance dims
that otherwise dominate the gradient"*）。

⚠️ **上面这组数字只用了 20 batch × 2 = 40 个样本**，对 2048 维统计量偏少。集中度
这么极端不太可能纯属噪声，但到 A6000 上用默认的 100 batch 重量一遍再决定：

```bash
HF_HUB_OFFLINE=1 python -m evo_rlt.cli.compute_token_variance \
  --model-path ~/crp-rlt/outputs/vla_ft/checkpoints/last/pretrained_model \
  --demo-dataset-path ~/桌面/crp_rlt_dataset \
  --config src/evo_rlt/core/configs/crp_dual_rlt.yaml \
  --output outputs/crp_token_std.pt \
  --num-batches 100 --batch-size 2
```

它会同时写一份 `.summary.json`，里面有按 std 排序的维度列表。

## 4. 正式训练

放行条件（照 rlt-single）：**2000–10000 步，重建 loss 收敛**。

```bash
conda activate crp-rlt && cd ~/RLT_dual

HF_HUB_OFFLINE=1 python -m evo_rlt.cli.train_rl_token \
  --model-path ~/crp-rlt/outputs/vla_ft/checkpoints/last/pretrained_model \
  --demo-dataset-path ~/桌面/crp_rlt_dataset \
  --config src/evo_rlt/core/configs/crp_dual_rlt.yaml \
  --output-dir outputs/crp_rl_token \
  --steps 10000 \
  --save-every 2000 \
  --num-workers 8 \
  --norm-stats outputs/crp_token_std.pt \
  --norm-gamma 0.5
```

A6000 有 48 GiB 显存，`--num-workers 8` 和更大的 `--batch-size` 都开得起
（配置里是 2，那是按 5070 Ti 的 16 GiB 定的）。但**改 batch_size 要同步看 lr**，
配置里的 `2.0e-4` 是配 batch 2 的。

### 关于 loss 曲线和中间 checkpoint

`train_rl_token` 没有 wandb 也没有 `--log-file`，但**曲线不会丢** —— 每次保存都会写
`losses.json`（完整历史）并把 `losses` 一并存进 checkpoint。rlt-single 那个"训练 loss
曲线永久丢失"的坑这边不存在，已在 smoke test 上核过（20 个点，3.5962 → 2.8450）。

⚠️ **但 checkpoint 是单个固定文件名 `demo_adapt_checkpoint.pt`，每次保存直接覆盖，
不留中间版本。** 训到 10000 步时，2000/4000/6000/8000 步的权重都没了。如果中途发散
就只能从头再来。想留中间版本得自己在旁边拷：

```bash
# 另开一个终端，每 10 分钟快照一次
while true; do
  f=outputs/crp_rl_token/demo_adapt_checkpoint.pt
  [ -f "$f" ] && cp "$f" "outputs/crp_rl_token/snap_$(date +%H%M%S).pt"
  sleep 600
done
```

## 5. 训完带回部署机什么

- `outputs/crp_rl_token/demo_adapt_checkpoint.pt`（RL token 权重，实测 977 MB —— 100.7M 参数按 fp32 存 + 优化器无关项）
- `outputs/crp_rl_token/losses.json`（完整 loss 曲线，用来判断是否收敛）
- `outputs/crp_token_std.pt` + 用的 `--norm-gamma` 值（不记下来事后无法复现）

阶段 3 之后 **VLA 和 RL token 全程冻结**，在线只训 actor 和 critic 两个小 MLP ——
这是它能在几小时内收敛的根本原因。所以带回去的这两个 checkpoint 就是全部。

---

## 已知的、还没解决的

- **`gamma = 0.999` 是估的。** `crp_dual_rlt.yaml` 里有推导：critic 按 `gamma**C` 自举，
  16 fps / C=10 下 0.99 的半衰期只有 4.3 秒，盖不住这个装配任务的关键段。0.999 给到
  43 秒。但**关键段的实际时长还没量过**，回来要量了再复核。这不影响阶段 3。
- **动作空间的量纲问题**（14 维里 `right_y` 占原始动作 MSE 的 47%，
  `left/right_roll` 和 `left/right_pitch` 四个维度实际恒定）**是阶段 4–6 的事**，
  影响 `action_clip_delta` 和 critic，不影响 RL token。细节见 `RLT_PLAN.md §4.1`。
- **关键段标注全是 0.0**（220,967 帧从未标注）。阶段 3 不需要它；它挡的是离线
  transition cache，而那条不在论文主路径上。
