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
| **pi0.5 SFT checkpoint** | **`/home/lenovo/crp-rlt/outputs/vla_crp_rlt/checkpoints/last/pretrained_model`** |
| pi0.5 base | `/home/lenovo/models/pi05_base_a538eb27` |

⚠️ **checkpoint 路径变过。** 早先本文写的是 `outputs/vla_ft/...`，那是 wandb 元数据里
记录的**当时**的 `output_dir`，现在已不存在。实际可加载的是 `outputs/vla_crp_rlt`，
其中 `checkpoints/last` 是指向 `030000_crp——rlt` 的软链接。

`vla_ft` 那个路径不存在时，transformers 会把它当成 Hugging Face repo id，于是报
`Repo id must be in the form 'repo_name' or 'namespace/repo_name'` —— 一个关于命名的
错误，完全没提目录不存在。**这已经在代码里修掉了**：`--model-path` 只要长得像本地路径
就先做存在性检查，报错会写明查了哪个路径、父目录里实际有什么。不要去动 `repo_type`。

### 先把这四个变量定死

后面所有命令都用它们，避免任何相对路径。**在 A6000 上先跑这段并确认全绿：**

```bash
export RLT_REPO=$HOME/RLT_dual
export VLA_CKPT=$HOME/crp-rlt/outputs/vla_crp_rlt/checkpoints/last/pretrained_model
export DEMO_DS=$HOME/桌面/crp_rlt_dataset
export RLT_OUT=$RLT_REPO/outputs          # 所有产物统一放这里, 不跟着 cwd 跑

for p in "$VLA_CKPT/config.json" "$VLA_CKPT/model.safetensors" "$DEMO_DS/meta/info.json"; do
  [ -e "$p" ] && echo "OK   $p" || echo "缺失 $p"
done
mkdir -p "$RLT_OUT"
python -c "
import json,os; i=json.load(open(os.environ['DEMO_DS']+'/meta/info.json'))
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
import os
from evo_rlt.cli.common import (load_training_config, assert_config_matches_dataset,
                                resolve_model_path, warn_on_shadowing_cuda_libs)
cfg = load_training_config('crp_dual_rlt.yaml')          # 裸文件名, 不依赖 cwd
assert_config_matches_dataset(cfg, os.environ['DEMO_DS'])
print('VLA ->', resolve_model_path(os.environ['VLA_CKPT']))
warn_on_shadowing_cuda_libs()
print('配置与数据集一致:', cfg.action_dim, '维,', cfg.control_hz, 'fps,', cfg.cameras)"
```

这一步是完整的 preflight，四件事都会自己报错，不用肉眼比对：配置能否找到、维度/相机/fps
是否与数据集一致、VLA checkpoint 是否真的存在、`LD_LIBRARY_PATH` 有没有 CUDA 冲突。

**`--config` 现在只写文件名就行。** 之前 `--config src/evo_rlt/core/configs/crp_dual_rlt.yaml`
在 `~/crp-rlt` 下会 `FileNotFoundError`，因为那是 `~/RLT_dual` 的相对路径。现在按
「原样 → 仓库根 → 包内配置目录」依次查找，`--config crp_dual_rlt.yaml` 在任何目录下都能解析
（找不到时会列出所有可用配置名）。老写法仍然有效。

## 1b. LD_LIBRARY_PATH 与 cuBLAS 冲突

这台机器上出现过：

```
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling
cublasSgemmStridedBatched(...)
```

根因是系统 CUDA 的 `libcublas` 排在 torch 自带的那份之前被加载。它不会在启动时报错，
要等到第一个批量 GEMM 才炸，看起来像 shape bug。**清空 `LD_LIBRARY_PATH` 即可**：

```bash
LD_LIBRARY_PATH= python -m evo_rlt.cli...
```

`build_pi05_policy` 现在会在加载模型前自动检查并 warning，不用等到 GEMM。但它只是
**警告不是拦截** —— 有些环境确实需要系统 CUDA，所以由你决定。下面的命令都带上了
`LD_LIBRARY_PATH=`。

## 2. Smoke test（先跑这个）

```bash
conda activate crp-rlt && cd "$RLT_REPO"

HF_HUB_OFFLINE=1 LD_LIBRARY_PATH= python -m evo_rlt.cli.train_rl_token \
  --model-path "$VLA_CKPT" \
  --demo-dataset-path "$DEMO_DS" \
  --config crp_dual_rlt.yaml \
  --output-dir "$RLT_OUT/crp_rl_token_smoke" \
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
conda activate crp-rlt && cd "$RLT_REPO"

HF_HUB_OFFLINE=1 LD_LIBRARY_PATH= python -m evo_rlt.cli.compute_token_variance \
  --model-path "$VLA_CKPT" \
  --demo-dataset-path "$DEMO_DS" \
  --config crp_dual_rlt.yaml \
  --output "$RLT_OUT/crp_token_std.pt" \
  --num-batches 100 --batch-size 2
```

它会同时写一份 `.summary.json`，里面有按 std 排序的维度列表。

⚠️ **`--output` 写绝对路径（`$RLT_OUT/...`）不是偶然。** 两个 workspace 并存时，
在 `~/crp-rlt` 下写 `--output outputs/crp_token_std.pt` 产生的是
`~/crp-rlt/outputs/crp_token_std.pt`，而之后在 `~/RLT_dual` 下用
`--norm-stats outputs/crp_token_std.pt` 读的是另一个文件 —— 以前这会静默地退化成
不加权训练。现在两端都会把解析后的绝对路径打进日志，`--norm-stats` 找不到文件直接报错。
跑完对一下两条日志里的路径是否同一个。

## 4. 正式训练

放行条件（照 rlt-single）：**2000–10000 步，重建 loss 收敛**。

```bash
conda activate crp-rlt && cd "$RLT_REPO"

HF_HUB_OFFLINE=1 LD_LIBRARY_PATH= python -m evo_rlt.cli.train_rl_token \
  --model-path "$VLA_CKPT" \
  --demo-dataset-path "$DEMO_DS" \
  --config crp_dual_rlt.yaml \
  --output-dir "$RLT_OUT/crp_rl_token" \
  --steps 10000 \
  --save-every 2000 \
  --num-workers 8 \
  --norm-stats "$RLT_OUT/crp_token_std.pt" \
  --norm-gamma 0.5 \
  2>&1 | tee "$RLT_OUT/crp_rl_token_train.log"
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
  f="$RLT_OUT/crp_rl_token/demo_adapt_checkpoint.pt"
  [ -f "$f" ] && cp "$f" "$RLT_OUT/crp_rl_token/snap_$(date +%H%M%S).pt"
  sleep 600
done
```

## 5. 训完带回部署机什么

全部在 `$RLT_OUT`（= `~/RLT_dual/outputs`）下：

- `crp_rl_token/demo_adapt_checkpoint.pt`（RL token 权重，实测 977 MB —— 100.7M 参数按 fp32 存）
- `crp_rl_token/losses.json`（完整 loss 曲线，用来判断是否收敛）
- `crp_rl_token_train.log`（`tee` 出来的完整日志，开头几行有解析后的绝对路径，存档用）
- `crp_token_std.pt` + 实际用的 `--norm-gamma` 值（不记下来事后无法复现）

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
