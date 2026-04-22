# ECE567 Final Project — 分工与项目计划

## 一、项目概述

### 1.1 项目目标

在 Phase 1 复现 MAPPO 和 IPPO baseline 的基础上，提出一个新的算法 **HybridPPO**：通过价值混合（Value Mixing）将 MAPPO 的集中式 critic 和 IPPO 的独立 critic 线性融合，探索两种 critic 之间的最优平衡点。

### 1.2 核心思想

对每个 agent 同时维护两个 critic 子网络：

- **Centralized critic** V_cent：MAPPO 风格，输入全局状态/所有 agent 拼接观测
- **Independent critic** V_ind：IPPO 风格，仅输入本地观测

最终 value 是二者的凸组合：

> **V_hybrid = α · V_cent + (1 − α) · V_ind**

其中 α ∈ [0, 1] 是超参数：

| α 值 | 等价于 |
|------|--------|
| 1.0 | 纯 MAPPO（全集中） |
| 0.0 | 纯 IPPO（全独立） |
| (0, 1) | Hybrid 混合 |

**关键数学性质**：由于 GAE 对 V 是线性的，"混 value" 和 "混 advantage" 数学上等价，因此只需修改 critic 输出，ClipPPOLoss 内部完全不用改。

### 1.3 为什么选这个方案

1. **实现简单**：只需新增 ~150 行代码，核心就是一个并联 critic + mixing module
2. **不修改 BenchMARL 内部**：只在 `algorithms/` 目录下新增文件
3. **故事清晰**：α-sweep 曲线本身就有意义——揭示中心化信号的任务依赖性
4. **下限有保障**：即使 Hybrid 没显著优于 MAPPO/IPPO，α-sweep 曲线 + 理论分析也足以写报告

---

## 二、人员分工

### 角色定义

| 人员 | 说明 |
|------|------|
| mz | 算法实现，跑hybridppo实验 |
| phn | 跑baseline实验，跑hybridppo实验 |
| rlt | 跑baseline实验，跑hybridppo实验 |
| zy | 负责写 report |
| hwt | 负责写 report |
| lyh | 负责写 report |

---

## 三、任务清单与详细说明

### 阶段一：Baseline 补跑 + 代码开发

#### 任务 1：实现 HybridPPO 核心算法 — mz

**目标**：在 BenchMARL 中注册一个新的 HybridPPO 算法。

**需要创建/修改的文件**：

1. **新建** `MARL/third_party/BenchMARL/benchmarl/algorithms/hybridppo.py`（~150 行）

   继承 `Mappo` 类（因为 MAPPO 的 actor、loss、process_batch 都可以复用），核心改动：
   
   - `__init__`：新增 `alpha` 参数
   - 重写 `get_critic()`：返回一个 `TensorDictSequential`，依次包含：
     - global_critic（centralised=True，输出 `state_value_cent`）
     - local_critic（centralised=False，输出 `state_value_ind`）
     - mixing module：`alpha * V_cent + (1-alpha) * V_ind → state_value`
   - 重写 `_get_parameters()`：返回 actor + global_critic + local_critic 三组参数
   - `has_centralized_critic()` 返回 False，`has_independent_critic()` 返回 False，新增 `has_critic()` 返回 True

2. **新建** `MARL/third_party/BenchMARL/benchmarl/conf/algorithm/hybridppo.yaml`

   ```yaml
   defaults:
     - hybridppo_config
     - _self_
   
   share_param_critic: True
   clip_epsilon: 0.2
   entropy_coef: 0.0
   critic_coef: 1.0
   loss_critic_type: "l2"
   lmbda: 0.9
   scale_mapping: "biased_softplus_1.0"
   use_tanh_normal: True
   minibatch_advantage: False
   alpha: 0.5
   ```

3. **修改** `MARL/third_party/BenchMARL/benchmarl/algorithms/__init__.py`（加 2 行）

   ```python
   from .hybridppo import Hybridppo, HybridppoConfig
   ```

4. **新建** `MARL/src/hybridppo_vmas.py`（~40 行）

   实验 runner 脚本，支持命令行参数 `--task`、`--alpha`、`--seed`、`--max_frames`。

**验收标准**：`alpha=1.0` 在 VMAS Balance 上 12K smoke test 结果与 MAPPO 一致；`alpha=0.0` 结果与 IPPO 一致。

---

#### 任务 2：跑 MAPPO 和 IPPO 的完整 baseline — 【phn + rlt】

**背景**：Phase 1 只跑了一部分：IPPO 只完整跑了 Navigation 和 Simple Tag（3M frames），MAPPO 只做了 smoke test（12K frames）。现在需要完整的 baseline 数据做对比。

**需要跑的实验**（共 10 个 runs）：

| # | 算法 | 环境 | Seed | Frames | 预估时间 | 负责人 |
|---|------|------|------|--------|----------|--------|
| 1 | MAPPO | VMAS Balance | 5 | 3M | ~3h | phn |
| 2 | MAPPO | VMAS Balance | 56 | 3M | ~3h | phn |
| 3 | MAPPO | VMAS Balance | 567 | 3M | ~3h | phn |
| 4 | MAPPO | VMAS Navigation | 567 | 3M | ~3h | phn |
| 5 | IPPO | VMAS Balance | 5 | 3M | ~3h | rlt    |
| 6 | IPPO | VMAS Balance | 56 | 3M | ~3h | rlt |
| 7 | IPPO | VMAS Balance | 567 | 3M | ~3h | rlt |
| 8 | IPPO | VMAS Navigation | 567 | 3M | ~3h | rlt |

> 注：IPPO + VMAS Navigation 在 Phase 1 已跑过 seed=0，但为了确保环境一致性建议重跑。

**运行方式**：

```bash
cd MARL
source third_party/BenchMARL/.venv/bin/activate
export WANDB_MODE=disabled

# MAPPO + VMAS Balance, seed=0
python src/mappo_vmas_balance.py

# MAPPO + VMAS Navigation（需要新建脚本或修改现有脚本指定 task）
# 方法：直接用 CLI
cd third_party/BenchMARL
python benchmarl/run.py algorithm=mappo task=vmas/navigation seed=0 experiment.max_n_frames=3000000 experiment.loggers='[csv]'

# IPPO + VMAS Balance
python benchmarl/run.py algorithm=ippo task=vmas/balance seed=0 experiment.max_n_frames=3000000 experiment.loggers='[csv]'
```

**注意**：跑之前先做 12K smoke test 确认环境正常（约 3 分钟）。

**验收标准**：每个 run 在 `collection_reward_episode_reward_mean.csv` 中有完整的 25 个 iteration 数据点。

**潜在阻力**：
- PettingZoo 环境可能需要 `pygame`；VMAS 不需要
- 如果 CPU 跑太慢，可以尝试把 `experiment.sampling_device` 改为 CPU（默认已经是）
- 如果出现 NaN，检查 `on_policy_n_envs_per_worker` 是否需要从 10 降到 4

---

### 阶段二：主实验 α-sweep

#### 任务 3：跑 HybridPPO α-sweep 主实验 — 【mz + phn + rlt】

**前置条件**：任务 1 完成（hybridppo.py 可运行），任务 2 完成（有 baseline 对比数据）。

**实验矩阵**：

| α | VMAS Balance seed=5 | seed=56 | seed=567 | VMAS Navigation seed=567 |
|---|---------------------|--------|--------|------------------------|
| 0.0 (IPPO) | ✅ 已有 | ✅ 已有 | ✅ 已有 | ✅ 已有 |
| 0.25 | mz | —       | — | — |
| 0.5 | mz | rlt | phn | phn |
| 0.75 | mz | — | — | — |
| 1.0 (MAPPO) | ✅ 已有 | ✅ 已有 | ✅ 已有 | ✅ 已有 |

**合计**：约 15-17 个新 runs，每人 5-6 个 runs。

**运行方式**：

```bash
cd MARL
source third_party/BenchMARL/.venv/bin/activate
export WANDB_MODE=disabled

# 示例：alpha=0.5, VMAS Balance, seed=0
python src/hybridppo_vmas.py --task balance --alpha 0.5 --seed 0 --max_frames 3000000
```

**数据收集**：每个 run 的 CSV 文件按 `{algo}_{task}_alpha{X}_seed{Y}` 命名，放在统一目录下。

**验收标准**：所有 15-17 个 runs 完成，每个有完整 25 个 evaluation 点。

---

### 阶段三：消融实验（可选）

#### 任务 4：α 退火实验 — 【时间允许的话】

**实验内容**：α 从 1.0 线性退火到 0.0，在 VMAS Balance 上 × 1 seed，与固定最优 α 对比。

**实现方式**：在 `hybridppo.py` 中添加 `alpha_schedule: "fixed" | "linear_anneal"` 选项。

**价值**：作为报告中的一个额外分析点，展示动态 α 策略的效果。

---

### 阶段四：Report 撰写

#### 任务 5-10：Report 撰写 — 【D + E + F】

见下方第四节"Report 详细大纲"。

---

## 四、Report 详细大纲（10 页，11pt single-column single-spaced）

> 以下大纲供D、E、F 参考撰写。标注了每节的目标页数、负责人建议、以及需要包含的内容。

### Report 结构总览

| 章节 | 目标页数 | 负责人 | 内容 |
|------|----------|--------|------|
| 1. Introduction | 0.7 页 | D | 动机与贡献 |
| 2. Background | 1.0 页 | D | 技术背景 |
| 3. Phase 1 Recap | 0.5 页 | D | Phase 1 结果复述 |
| 4. **Method** | **1.5 页** | E | HybridPPO 方法描述 |
| 5. Experimental Setup | 0.5 页 | E | 实验设置 |
| 6. **Results** | **2.0 页** | F | 主实验结果 |
| 7. **Analysis** | **1.5 页** | F | 分析与讨论 |
| 8. Discussion & Limitations | 0.5 页 | D | 局限性与未来工作 |
| 9. Conclusion | 0.3 页 | D | 总结 |
| References | 1.5 页 | 全员 | 参考文献 + 图表 |

---

### 第 1 节：Introduction（0.7 页）

**需要写的内容**：

1. **开篇**：多智能体强化学习（MARL）在合作/竞争场景中的重要性
2. **问题**：MAPPO（集中式 critic）和 IPPO（独立 critic）是两种主流 PPO 变体，但经验上哪个更好是任务相关的
   - 引用 Yu et al. 2022（MAPPO paper）：MAPPO 在多数任务上更好
   - 引用 de Witt et al. 2020：IPPO 在 SMAC 若干任务上胜过 MAPPO
   - 这意味着**中心化信号的价值是任务相关的**，存在一个介于两者之间的最优点
3. **我们的贡献**：
   - 提出 HybridPPO：用 α ∈ [0,1] 线性融合集中式和独立式 critic 的 value 估计
   - 在 VMAS Balance 和 Navigation 上进行系统性的 α-sweep 实验
   - 分析最优 α 的任务依赖性，并给出理论解释

**需要引用的论文**：
- Schulman et al. 2017 (PPO)
- Yu et al. 2022 (MAPPO)
- de Witt et al. 2020 (IPPO)
- Rolic et al. 2024 (BenchMARL)

---

### 第 2 节：Background（1.0 页）

**需要写的内容**：

1. **多智能体强化学习（MARL）基础**（~0.2 页）
   - Dec-POMDP 形式化：状态 s，每个 agent 有局部观测 o_i，动作 a_i，共享/独立奖励
   - 非平稳性问题：每个 agent 的策略在变化，从一个 agent 的视角看环境是非平稳的

2. **CTDE 范式**（~0.2 页）
   - Centralized Training with Decentralized Execution
   - 训练时利用全局信息，执行时只用局部信息

3. **MAPPO**（~0.2 页）
   - 分散 actor + 集中 critic
   - Critic 输入：全局状态 s（或所有 agent 观测的拼接 [o_1, o_2, ..., o_n]）
   - 使用 GAE (λ-return) 计算 advantage
   - Clipped surrogate objective

4. **IPPO**（~0.2 页）
   - 每个 agent 独立运行 PPO
   - Actor 和 critic 都只看局部观测 o_i
   - 简单但面临 non-stationarity

**公式**：
- PPO clipped objective
- GAE advantage

---

### 第 3 节：Phase 1 Recap（0.5 页）

**需要写的内容**：

简述 Phase 1 的实验环境设置和 baseline 复现结果。

1. **环境**：VMAS Balance（4 agents 合作搬运）、VMAS Navigation（3 agents 避碰导航）
2. **框架**：BenchMARL v1.5.2 + TorchRL v0.11.1
3. **超参数表**（直接搬 Phase 1 report 的表格）
4. **Baseline 结果表**：MAPPO 和 IPPO 在两个环境上的 mean ± std return

> 这部分直接从 Phase 1 的 2 页 report 中的 Environment 和 Baselines 部分改编即可。

---

### 第 4 节：Method — HybridPPO（1.5 页）

> 这是报告的核心部分之一，请重点写清楚。

**需要写的内容**：

1. **动机**（~0.3 页）
   - MAPPO 和 IPPO 各有优劣，不存在绝对赢家
   - 核心直觉：中心化信号的价值取决于任务的结构——有些任务需要全局协调，有些任务的局部信息就足够
   - 与其硬选一个，不如用权重 α 线性融合

2. **Value Mixing 方法**（~0.5 页）
   - 形式化定义：
     
     > V_hybrid(s, o_i) = α · V_cent(s) + (1 − α) · V_ind(o_i)
   
   - 两个 critic 子网络的架构：
     - V_cent：centralised=True，输入所有 agent 拼接观测，输出单一 value（shared params）再 expand 到 n_agents
     - V_ind：centralised=False，输入单 agent 观测，输出 per-agent value
   - Actor 完全不变（分散 MLP，只看局部观测）

3. **与 Advantage Mixing 的等价性**（~0.3 页）
   - 由于 GAE 对 V 是线性的：A^GAE(V_hybrid) = α · A^GAE(V_cent) + (1-α) · A^GAE(V_ind)
   - 因此"混 value"和"混 advantage"数学上等价
   - 选择混 value 是因为实现更简单——只需修改 critic 输出，loss 模块原封不动

4. **梯度分配性质**（~0.2 页）
   - Critic loss: L = E[(V_hybrid − V_target)²]
   - 对两个子网络求梯度：
     - ∂L/∂θ_cent ∝ α · (V_hybrid − V_target)
     - ∂L/∂θ_ind ∝ (1−α) · (V_hybrid − V_target)
   - 当 α=1 时 V_ind 无梯度（退化为 MAPPO）；α=0 时 V_cent 无梯度（退化为 IPPO）

5. **架构图**（~0.2 页）
   - 画一张图：左侧是两个 critic 分支（global/local），中间是 mixing（α 权重），右侧输出 V_hybrid 给 GAE

**关键图表**：
- Figure 1: HybridPPO 架构图（双 critic + mixing）

---

### 第 5 节：Experimental Setup（0.5 页）

**需要写的内容**：

1. **环境**
   - VMAS Balance：4 agents 合作搬运重物到目标位置
   - VMAS Navigation：3 agents 导航到各自目标并避免碰撞（LIDAR range 0.35）

2. **超参数**
   
   | 参数 | 值 |
   |------|-----|
   | Learning rate | 5e-5 |
   | Discount (γ) | 0.99 |
   | GAE λ | 0.9 |
   | Clip ε | 0.2 |
   | Frames per batch | 6,000 |
   | Minibatch size | 400 |
   | Minibatch iters | 45 |
   | Parallel envs | 10 |
   | Max frames | 3M |
   | MLP layers | [256, 256] |
   | Activation | Tanh |
   | Seeds | {0, 1, 2} for Balance; {0} for Navigation |
   | α values | {0.0, 0.25, 0.5, 0.75, 1.0} |

3. **Baseline**
   - MAPPO（α=1.0）和 IPPO（α=0.0）作为两端 baseline
   - 使用完全相同的超参数，唯一区别是 α

4. **评估**
   
   - 每 120K frames 评估一次，每次 10 episodes
   - 报告 mean episode return

---

### 第 6 节：Results（2.0 页）

> 这是报告的另一个核心部分。需要等实验数据出来后填充。

**需要包含的图表**：

1. **Figure 2: VMAS Balance 上的训练曲线**（~0.5 页）
   - x 轴：frames（0-3M）；y 轴：mean episode return
   - 5 条曲线（α=0.0, 0.25, 0.5, 0.75, 1.0），每条是 3 seeds 的 mean ± std（阴影）
   - 一张图，大且清晰

2. **Figure 3: VMAS Navigation 上的训练曲线**（~0.3 页）
   - 同上，但只有 1 seed，所以没有阴影

3. **Table 1: 主结果表**（~0.3 页）
   
   | Method | α | VMAS Balance | VMAS Navigation |
   |--------|---|-------------|-----------------|
   | IPPO | 0.0 | mean ± std | mean |
   | HybridPPO | 0.25 | mean ± std | — |
   | HybridPPO | 0.5 | mean ± std | mean |
   | HybridPPO | 0.75 | mean ± std | — |
   | MAPPO | 1.0 | mean ± std | mean |

4. **Figure 4: α vs Final Return 图**（~0.5 页）
   - **这是最重要的图**
   - x 轴：α（0, 0.25, 0.5, 0.75, 1.0）；y 轴：final mean episode return
   - 两条线：VMAS Balance 和 VMAS Navigation
   - VMAS Balance 用 error bar（3 seeds），Navigation 用点
   - 这张图揭示：最优 α 是否存在？是否任务相关？

5. **文字分析**（~0.4 页）
   - 描述训练曲线的趋势（哪个 α 收敛最快？哪个最终性能最好？）
   - 描述 α-sweep 的形状：单调？倒 U 形？U 形？平坦？
   - 与 baseline 对比：HybridPPO 是否优于两端？

---

### 第 7 节：Analysis（1.5 页）

**需要写的内容**：

1. **最优 α 的任务依赖性**（~0.5 页）
   - 如果 Balance 上最优 α ≠ Navigation 上最优 α → 说明中心化信号的价值确实是任务相关的
   - 分析原因：Balance 需要高度协调（4 agents 推同一个物体），所以可能更偏 MAPPO；Navigation 主要是避碰，局部信息可能就够了，所以可能更偏 IPPO

2. **α-sweep 曲线形状的含义**（~0.3 页）
   - **倒 U 形**：说明 Hybrid 确实比两端都好 → 中心化和独立信号各有贡献，融合后互补
   - **单调**：说明一端始终优于另一端 → 但 α-sweep 本身仍然有价值，因为它量化了差距
   - **平坦**：说明在这个任务上 α 不敏感 → 鲁棒性好的信号

3. **样本效率分析**（~0.3 页）
   - 达到相同 return 所需的 frames 数对比（从训练曲线上读）
   - 如果某个 α 收敛更快，说明混合信号提供了更好的 learning signal

4. **理论解释**（~0.4 页）
   - 回到梯度分配性质：α 控制了两个 critic 的有效学习率比例
   - 最优 α 反映了"全局信息 vs 局部信息的信噪比"
   - 与 de Witt et al. 2020 的发现一致：某些任务上独立学习就够了，中心化反而引入噪声

---

### 第 8 节：Discussion & Limitations（0.5 页）

**需要写的内容**：

1. **局限性**：
   - 只在 2 个 VMAS 环境上测试（合作类），未测试竞争类环境
   - α 是固定超参数，需要手动选择
   - 双 critic 带来 ~2x 的 critic 前向计算开销（在 MLP 规模下可忽略，但大规模场景需考虑）

2. **未来工作**：
   - α 退火策略（从 MAPPO 渐进过渡到 IPPO）
   - 自适应 α（基于两个 critic 的 TD-error 方差做 inverse-variance weighting）
   - 在更多环境上验证（PettingZoo, SMACv2）
   - 将 Value Mixing 思想推广到其他算法（如 MADDPG, MASAC）

---

### 第 9 节：Conclusion（0.3 页）

1-2 段话总结：
- 我们提出 HybridPPO，用 α 线性融合 MAPPO 和 IPPO 的 critic
- 在 VMAS 环境上的实验表明 [填入关键发现]
- 这一发现支持了"中心化信号的价值是任务相关的"这一假设

---

### References

确保引用以下论文：
1. Schulman et al. 2017 — PPO
2. Yu et al. 2022 — MAPPO (NeurIPS)
3. de Witt et al. 2020 — IPPO (SMAC)
4. Rolic et al. 2024 — BenchMARL (JMLR)
5. Schulman et al. 2016 — GAE
6. 如果引用了其他论文请一并加入
