# CRP 双臂上复现 RLT —— 现状与计划

写于 2026-08-28，起因是把单臂姊妹项目
[Jiabo-Wang/rlt-single-so101](https://github.com/Jiabo-Wang/rlt-single-so101)
的经验接过来。那个项目和本仓库同源于 `MINT-SJTU/Evo-RLT`，它把单臂路径走完了七个阶段
并做了 7 轮对照评测，本文的判断大量依赖它的 `docs/notes/RESULTS.md` 和
`DIFF_FROM_UPSTREAM.md`。

**先说清楚一件事：rlt-single 明确不宣称复现了 RLT。**
7 轮评测里 1 轮有效、1 轮探索性、4 轮作废、1 轮中止。最强的一轮（易位置）主判据显著
（惩罚化完成时间 4.02s vs 8.00s，p=0.0003），但没有任何一轮同时满足它自己全部的预注册
有效性条件。作废原因主要是硬件退化（3D 打印夹爪开裂）和方法问题（盲化被破坏、成功率
分母定义不一致）。

它最大的产出不是"跑通了"，而是把会让结论作废的坑趟了一遍。本文按那份清单来排。

---

## 1. CRP 双臂现在在哪一阶段

对照 rlt-single 的七阶段划分：

| # | 阶段 | 状态 | 依据 |
| --- | --- | --- | --- |
| 0 | 环境与设备 | ✅ | `diagnostics/pi05_preflight.py` 全绿：三路相机按 serial 解析、两臂 ping 通 |
| 1 | 演示采集 | ✅ | 622 ep / 220,967 帧 @ 16 fps，`crp_arm_dual` |
| 2 | VLA 微调（SFT） | ✅ | pi0.5 全参 30k 步，loss 0.0189，见 `docs/pi05_vla_ft_report/` |
| — | VLA 部署验证 | ✅ | 显存 8.71 GiB，重规划 178 ms（预算 3.12 s） |
| 3 | **RL-token 离线训练** | ❌ **下一步** | `outputs/` 下无产物 |
| 4 | warmup | ❌ | 代码有，未跑 |
| 5 | critic-only | ❌ | 代码有，未跑 |
| 6 | 在线强化 | ❌ | 代码有，未跑 |
| 7 | 对照评测 | ❌ | 未开始 |

**关键段标注（`complementary_info.phase`）全部 220,967 帧都是 0.0，即从未标注过。**
这不阻塞阶段 3 —— `train_rl_token` 只吃原始演示数据做重建，不需要 phase。它阻塞的是
离线 transition cache / RLPD 双 buffer，而按论文主路径那条不在必经之路上
（actor 和 critic 全程在线从零开始）。

---

## 2. rlt-single 的 bug 在本仓库存不存在

逐条核过。结论和直觉相反：**本仓库在算法正确性上普遍领先 rlt-single 分叉时的状态。**

| rlt-single 的发现 | 本仓库 |
| --- | --- |
| 动作量纲不一致 —— critic 训机器人单位、actor 查归一化，"在线 RL 一直在做纯 BC" | ✅ 已修，`loop.py` 的 `_normalize_executed_action` 拿不到 unnormalizer 直接拒绝录制 |
| 人工接管把 BC 锚点污染成机器人单位 | ✅ 不存在 —— 但见下，这是个**有意的反向设计** |
| `transition_cache_v2` 拿 `ref_chunk` 当 `exec_chunk`，critic 退化成 V(s) | ✅ 已修，`_compose_exec_chunk` 取真实示范动作并带断言 |
| chunk 级 TD 时序错位（`next_i = i + 1`） | ✅ 已修，`next_frame = start_frame + C` |
| actor 探索噪声从未生效（`sample()` 没被调用） | ✅ 已修，`core/policy.py` 调了，`fixed_std=0.05` |
| TD3 target policy smoothing 缺失 | ✅ 已有 `target_noise_std` / `target_noise_clip` |
| 分层采样有放回，把小池子放大 7–12 倍 | ⚠️ 见 §3 |
| `skip_prefix_recording` 硬编码 | ⚠️ 是配置项，但 `online_cli.py` / `runner.py` 三处仍硬编码 `true` |

rlt-single 自己的 `DIFF_FROM_UPSTREAM §B2` 印证了方向：它把 transition cache v2、
双 buffer、RankQ 都列为"上游有、本项目没有"。

**所以要搬的不是算法，是工程护栏和方法论。**

### 一处需要留意的算法分歧

论文（和 rlt-single）在人工接管时把 BC 锚点换成人的动作；本仓库**故意保持 `ref` 为
VLA 参考**，理由写在 `online_collector.py` 的注释里：actor 是残差头 `mu = ref + delta`，
`ref == exec` 会把"VLA 本来会做 X，修正是 delta=Y"这个最有价值的信号抹成恒零。

这个推理成立，但它依赖残差 actor。而 rlt-single 在 2026-08-24 恰恰**放弃了残差 actor**
（`actor_residual_to_ref` 默认改回 false），因为实测 `|mu − ref|` 从 0.032 单调收缩到
0.0136，"练出来的只是个 VLA" —— 残差头 + BC 锚合起来把上限锁死了。

本仓库 `configuration_rlt_ac.py` 的默认值已经是 `actor_residual_to_ref = False`（论文侧），
但 `online_cli.py` 硬编码传 `true`。**这两个选择是耦合的**：若哪天改成非残差，
`ref` 的取法要一并重新论证。

---

## 3. 本次已改

- **分层采样补齐方式做成开关。** `sample_stratified(allow_resample=...)` /
  `--stratified-allow-resample`，默认无放回。rlt-single 实测有放回让 33 条接管数据
  占住每批 20%，BC 项跳 3.5 倍持续 36 集；本仓库原测试则刻意锁死有放回，理由（稀疏
  奖励下别放弃配额）同样成立。CRP 还没在线跑过，没有本机数据可判，所以默认取有实测
  依据的一侧，另一侧完整保留。

- **`--config` 的形状字段现在真的生效了。** `cli/common.py` 原来在解析完 YAML 后
  无条件用 SO101 常量覆盖 `action_dim` / `proprio_dim` / `vla_horizon` /
  `chunk_length` / `cameras`。也就是说**在此之前根本无法为 CRP 配置 RLT** ——
  写 `action_dim: 14` 不报错也不生效，`build_pi05_policy` 会拿 12 维去构造 14 维机器人的
  适配器。新增 `assert_config_matches_dataset()` 对着数据集 `meta/info.json` 交叉校验，
  并接到四个吃 `--config` 的 CLI 上。

- **新增 `configs/crp_dual_rlt.yaml`**，每个值都注明来源（实测 / 论文）。

- 顺带修了 `online_cli.py` 四个 help 字符串里的裸 `%`，它让
  `evo-rlt-online-train --help` 直接抛 `TypeError` 崩掉。

---

## 4. 两个 CRP 特有的问题，rlt-single 遇不到

### 4.1 state 和 action 不同构

| | CRP 双臂 | SO101（rlt-single） |
| --- | --- | --- |
| `observation.state` | 关节角 `left_j1..j6.pos`, `right_j1..j6.pos` + 2 夹爪 | 关节位置 6 维 |
| `action` | **笛卡尔位姿** `left_x/y/z/roll/pitch/yaw.pos`, `right_*` + 2 夹爪 | 关节位置 6 维 |

SO101 上两者同构，CRP 上不是（CRP 控制器在 GP 通道跑自己的 IK）。影响两处：

1. **actor 残差的语义。** `mu = ref + delta` 里的 `delta` 是位姿增量，不是关节增量。
   本身没问题，但任何"把 state 和 action 当同一空间"的推理都不成立。
2. **`action_clip_delta` 是单个标量。** 14 维里混着毫米（x/y/z）和角度（roll/pitch/yaw）
   和夹爪开度（ui50）。同一个 δ 对三者含义完全不同。动作已归一化到 QUANTILES 空间
   （所以 δ 名义上是无量纲的），但各维的 `q99−q01` 差异会让等效物理幅度差出量级 ——
   rlt-single 就踩过这个：`wrist_roll` 的 `q99−q01` 只有 0.558，归一化对它极度敏感。
   **上机前先把 14 维的 `q99−q01` 打出来看一眼。**

### 4.2 gamma 必须重算，不能沿用 0.99

critic 按 chunk 自举，等效每 chunk 折扣是 `gamma**C`。半衰期：

```
半衰期(秒) = ln(0.5) / (C · ln gamma) · (C / fps)
```

C=10、16 fps 下：

| gamma | 半衰期 |
| --- | --- |
| 0.99 | 4.3 s |
| 0.995 | 8.6 s |
| 0.998 | 21.6 s |
| **0.999** | **43.3 s** |

rlt-single 在 30 fps 上遇到同一问题（0.99 只有 2.3 s），改用 0.999 并设为必传。
本仓库三处默认值仍是 0.99。CRP 的关键段（对齐 + 插销）比它计划的 5–8 s 长，
`crp_dual_rlt.yaml` 暂取 0.999，**但这个值要在量到关键段实际时长之后复核**。

---

## 5. 接下来的顺序

1. **跑阶段 3：训 RL token。** 冻结 pi0.5，训 encoder/decoder 重建瓶颈，2000–10000 步，
   放行条件是重建 loss 收敛。用新配置：

   ```bash
   conda activate crp_rlt_small && cd ~/RLT_dual
   evo-rlt-train-rl-token \
     --model-path pretrained_model \
     --demo-dataset-path data/crp_dual/crp_rlt_dataset \
     --config src/evo_rlt/core/configs/crp_dual_rlt.yaml \
     --output-dir outputs/crp_rl_token
   ```

   ⚠️ **这条命令还没跑过。** 配置校验已通过，但 `train_rl_token` 在 CRP 数据上
   端到端没验证过；第一次跑先用 `--steps 20` 做 smoke test（rlt-single 的做法）。
   另外它默认 `--model-path lerobot/pi05_base`，我们要指向自己 SFT 后的权重。

2. **量关键段时长**，回填 gamma。可以从演示数据里估：标几条 episode 的对齐+插销区间。

3. **打印 14 维的 `q99−q01`**，确认 `action_clip_delta` 对每一维都是合理的物理幅度。

4. **补 rlt-single 的工程护栏**（`DIFF_FROM_UPSTREAM §B1`，按性价比排）：
   - 相机 alias 未知/缺失时报错，而不是静默丢弃
   - go-home 关节名校验（匹配不上会退化成"全 0 度"，有撞机风险）
   - `critic-health` 诊断（判断 critic 是否退化成 V(s)）
   - `shape_contract.py` 式的单一事实来源（本次的配置校验是它的第一步）

5. **阶段 4–6 在线 RL。** 这是同一条命令的三个连续窗口，不是三条命令。

6. **阶段 7 之前先立规矩**（rlt-single §6，每条都是被一次具体失败逼出来的）：
   - 预注册 + 偏离记录，开跑前提交
   - 交错分块（20 集一块 A/B 严格交替），否则无法区分效应与时段漂移
   - 盲化用解析后的屏幕输出，不用日志文本白名单
   - 硬件监控进预注册（夹爪入口值漂移 >0.5 停机）
   - 不看中途成绩，只报有效性
   - 判据先在已知真值上验证
   - **成功率分母必须写清楚**是"全部尝试"还是"进入过关键段的集"
     （`--rlt.skip_prefix_recording` 决定这件事；评测时应关掉）

7. **不要做的两件事**（rlt-single 的教训）：
   - 不要投入 "RL-token 到训练质心的距离" 这个分布外判据，轮内 AUC ≈ 0.55，几乎无区分力
   - 不要加 offline BC warm start —— 离线数据全是成功演示，critic 会外推高估；
     论文主路径是 actor/critic 全程在线从零开始

---

## 6. 硬件

rlt-single 有 110 集评测因为 3D 打印夹爪开裂而白跑，领臂欠压故障出现 5 次并最终导致收尾。
CRP 是工业臂，这两类具体故障不适用，但教训通用：**把会漂移的硬件量化指标写进预注册，
每块记录一次，超阈值停机。** CRP 这边对应的候选是夹爪 `ui50` 的入口值
（数据集里就是 `left_ui50` / `right_ui50` 两维）。
