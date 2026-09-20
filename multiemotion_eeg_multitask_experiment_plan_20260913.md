# EEG-Aligned 结构 × 多情绪回归矩阵计划

> 状态：第一条独立单标签路线1254/1254、第二条共享多头路线114/114均已完成，分别输出两组结构 × 11情绪表  
> 更新：2026-09-20
> 目标：以 0814 窗口结构和 0906 EMA-bag 结构变体为行、11 个情绪标签为列，分别形成“标签专属 EEGPT＋独立标量”和“11 标签共同监督 EEGPT＋共享多头”两组结果矩阵；两组共用 split、下游 seeds 与 EMA-event 评价单位，并保留可复用的标签专属 EEGPT partial-FT embedding 库。

## 执行状态（2026-09-20）

- Phase 0 已通过：`28,819` 行、`1,253` events、每 event `23` windows；两个主协议在 window/event/subject-day 三层均零重叠，全部 token 与 canonical `sample_id` 同序。
- Phase 1 已通过：带正式 EEGPT checkpoint 的 `fatigue/cross_day/seed240800` 精确复现历史结果（RMSE `0.9272`、raw r `0.2741`、centered r `0.1005`）；repaired `within_subject_day` 为 RMSE `1.0202`、raw r `0.1553`、centered r `0.0387`。历史复现保留旧 trainability 语义，正式 ST/MT 使用严格 last-2-block 选择器。
- Phase 2A 已完成 `72/72` runs。validation-only route 为 `cross_day/B0_Wphysio_full`（macro sRMSE `0.9244 +/- 0.0010`）和 `within_subject_day/A1_Wphysio_no_audio`（`0.9283 +/- 0.0010`）。
- Phase 2B 已完成 `12/12` matched runs。H1-H0 在 `cross_day` 的 mean delta sRMSE/raw r 为 `+0.0004/+0.0026`，在 `within_subject_day` 为 `+0.0024/+0.0241`；Gate 2 决策为 `promote_multihead`。
- 两组矩阵均将0814收敛为 `window_attention_regression_full_mean` 一行，0906保留18个结构变体。共享多头组固定 `A1_Wphysio_no_audio__eeg_eegpt_partial_ft_multitask_11label_v1`；独立标量组每列读取对应标签专属 EEGPT token，A1/Wphysio/no_audio 与 split 保持一致。
- Phase 3 的 `11 labels × 2 protocols × 1 upstream seed` 已完成22/22，全部 `metrics.status=ok`。2026-09-16 使用 `100_audit_eegpt_single_label_bank.py` 复核22份 token 的 `(28819,256)` shape、有限值、canonical sample 顺序、split indices、protocol/label/seed 与监督边界，22/22 通过；证据保存为 `outputs/multiemotion_20260913/phase3_single_task_eeg/bank_audit.json`。额外的 `cross_day/inspired/240801` 仅保留审计。这些标签专属 token 用于第一条独立标量路线。
- 2026-09-20，`94` 以 canonical 11 标签共同监督 EEGPT，完成 `cross_day` 与 `within_subject_day` 两份上游 `seed_240800` token；EEG-only event-level macro sRMSE/raw r 分别为 `0.9712/0.3731` 与 `0.8990/0.1844`。`98` 随后在原 `structure_matrix_A1/` 根清除 fatigue-supervised runs/summary，使用 MT11 token 重跑 `19 × 2 × 3 = 114` runs；preflight、token profile、11标签顺序、split indices 和 val/test event/targets 审计均通过。`99` 以 MT11 结果覆盖本地 `structure_matrix_summary/`，两个协议的 validation macro sRMSE 均选中0814窗口结构。
- 第一条独立路线由 `101` 将每个标签的 EEGPT partial-FT token 接回同一 A1/Wphysio/no_audio 的0814/0906标量回归结构；22/22 bag 与 split 来源审计通过，`11 labels × 19 structures × 2 protocols × 3 downstream seeds = 1254` runs 全部完成。`102` 严格汇总 `missing=0`，生成[独立的六张11标签表](outputs/server_sync/multiemotion_20260913/single_task_structure_matrix_summary/README.md)。固定0814结构下，MT11 共享多头的 test macro raw r 为 `cross_day 0.3733`、`within_subject_day 0.2429`；三seed筛查不作单模块因果或显著性主张。

当前轻量证据副本：`outputs/server_sync/multiemotion_20260913/`；完整远端根：`outputs/multiemotion_20260913/`。

## 1. 核心结论与执行顺序

本计划有两条完整实验路线，按相同的 A1/Wphysio/no_audio、canonical event 与 repaired split 并列评估：

1. **独立单标签路线**：对 11 个标签分别执行 EEGPT partial FT（上游固定 `seed_240800`），导出 `(28,819,256)` 标签专属 token；每个标签独立训练0814单一窗口结构和0906的18个结构变体，各用3个下游 seed，形成第一组11列表。token 库也保留供复用。
2. **共享多头路线**：固定一套 11 标签共同监督的 EEGPT embedding（上游 `seed_240800`），0814只保留一个窗口级结构，0906保留18个结构变体；每行都接两层共享 MLP与11个两层专属回归头，一次输出11列，形成第二组11列表。两路线独立呈现输入与监督边界。

正式顺序为：

```text
数据与 repaired split 审计
        ↓
共同 A1 + Wphysio + no_audio、0814 单行 + 0906 18结构
        ├─ 11标签专属 EEGPT partial FT → 11套独立标量回归 → 第一组表
        └─ 11标签共同监督 EEGPT token → 共享两层 MLP + 11个两层 head → 第二组表
```

两组矩阵下游回归均使用 3 个 matched seeds；同一标签、协议与路线内的结构行固定输入。label-specific token bank 每个 `protocol × label` 只生成上游 `seed_240800`，不把上游随机性与下游结构稳定性混在同一轮预算中。

### 1.1 主矩阵的固定行列

列顺序固定为：

```text
inspired, alert, determined, attentive, active,
hostile, nervous, upset, afraid, ashamed, fatigue
```

行由以下结构组成：

```text
0814:
  window_attention_regression_full_mean

0906:
  bag_static_reg × {uniform,last_10s,last_30s,last_60s,first_30s,kernel_short,kernel_medium,kernel_long}
  state_uniform_reg
  prior_uniform_reg
  prior_regD_uniform_reg
  global_kernel_no_prior_reg
  dynamic_kernel_no_prior_reg
  dynamic_kernel_prior_uniform_reg
  dynamic_kernel_reg
  dynamic_fixed_short_reg
  dynamic_fixed_medium_reg
  dynamic_fixed_long_reg
```

`cross_day` 与 repaired `within_subject_day` 分表报告；raw r、standardized RMSE 和 within-subject centered r 各自成表，避免在单元格中混合多个量纲。

## 2. 已确认基础与本计划新增内容

### 2.1 已确认基础

- Canonical 数据包含 `28,819` 个 10 秒窗口、`1,253` 个 EMA event，每个 event 有 `23` 个窗口，`event_window_id=0..22`。
- 标签顺序固定为：

```text
inspired, alert, determined, attentive, active,
hostile, nervous, upset, afraid, ashamed, fatigue
```

- 11 个标签均为有限值，当前取值范围均为 1--5。
- `hostile`、`nervous`、`afraid`、`ashamed` 存在稀疏高分尾部，逐标签结果和高分段诊断必须保留。
- 已有 `fixed_tokens_multitask_fusion_v1` 完成过单 seed、两协议、12 条固定 token 路线，共 24 runs。它使用一个直接输出 11 维的共享 head，属于低成本基线，尚不能承担多 seed 晋级结论。
- 当前 EEG `eegpt_partial_ft_v1` 的 256D token 由 fatigue 监督训练；`eegpt_frozen_v1` 才是 label-free EEG 对照。

### 2.2 本计划新增内容

- 11 个标签各自独立的 EEGPT 局部微调与融合回归。
- 一套 11 标签共同监督的 EEGPT 局部微调。
- 两层共享 MLP + 11 个两层标签专属回归头。
- 共享输出层、专属多头、独立单任务三者的同数据配对比较。
- EMA event 级主评估、subject-day 配对 bootstrap、5-seed 稳定性与 Holm 多重比较校正。

## 3. 数据、监督单位与协议合同

### 3.1 数据单位

每次 EMA 评分构成一个监督 event：

```text
1 EMA event
└── 23 个重叠的 10 秒窗口
    ├── EEG 256D token
    ├── Wear 256D token
    ├── Video 256D token
    └── Audio 256D token
```

训练阶段可沿用窗口级前向计算，但必须满足：

- 同一 event 的 23 个窗口始终进入同一个 train、validation 或 test split。
- 每个 event 的训练总权重相同；实现采用 `1 / event有效窗口数` 的窗口权重，兼容未来缺窗情况。
- validation 与 test 的主预测先在 event 内对有效窗口做简单均值，再计算 event-level 指标。
- 窗口级指标只作为与历史 fatigue 结果兼容的诊断项。
- bootstrap 的最小重采样块为 `subject-day`，不能将 23 个窗口当作 23 个独立统计样本。

### 3.2 正式协议

| 级别 | Protocol | 用途 |
| --- | --- | --- |
| Primary | `cross_day` | 检验跨日期泛化 |
| Primary | `outputs/splits/within_subject_day` | 检验同被试的 held-out-day 泛化；按 `subject-day` 整体划分；修复后的唯一正式名称 |
| Diagnostic | `cross_subject` | 检验跨被试迁移，只在主阶段通过后扩展 |
| Historical only | `/vePFS-0x0d/DailyEEG/splits_new/within_subject_day` | 原宽松窗口级划分；已从所有当前入口禁用，旧结果只用于 provenance |

从 2026-09-13 起，`within_subject_day` 统一指 `<aligned-root>/outputs/splits/within_subject_day`：同一 `subject-day` 不跨 train/validation/test。`within_subject_day_strict` 只作为旧产物中的历史名称，不再用于新命令、目录或报告。实验 manifest 仍记录 split root，以便审计数据来源。

每个 protocol 独立训练：

```text
train = pretrain + finetune
validation = early stopping、route/head 选择、阈值锁定
test = 锁定配置后的最终评估
```

任何 route、head、loss 权重和 epoch 选择都只能读取 validation 结果。

## 4. 两条主路线

### 4.1 路线 A：11 套独立单任务模型 `ST-11`

对每个标签 `k` 独立执行：

```text
raw EEG
  → EEGPT
  → 解冻最后 2 个 transformer blocks + final norm/projection
  → 256D label-specific EEG embedding
  → 与 fixed Wear / Video / Audio token 融合
  → 两层 task trunk + 一个两层 scalar regression head
  → y_k
```

标签特定 token 命名：

```text
eegpt_partial_ft_single_inspired_v1
eegpt_partial_ft_single_alert_v1
...
eegpt_partial_ft_single_fatigue_v1
```

统一约束：

- 11 个模型从同一 EEGPT 初始 checkpoint 出发。
- 11 个标签使用同一组 split、seed、局部解冻深度和优化器配置。
- 每个标签只用自己的 train label 更新 encoder、projection 和 scalar head。
- 每个标签分别用 train split 的均值和标准差标准化目标。
- 每个标签分别导出 `(28,819, 256)` 的监督 EEG token，并记录 `target_label`、protocol、seed 和 supervision boundary。
- 正式 ST fusion 使用与 MT 模型单个分支相同的“两层 trunk + 两层 head”；11 套模型分别训练，用于减少 ST/MT 架构深度差异。
- Phase 1 另保留当前 legacy scalar head 的 fatigue 复现，它只承担入口一致性检查。

这一组结果回答：标签专门化的 EEG 表征能达到什么水平，以及哪些标签真正从局部微调获益。

### 4.2 路线 B：共享多任务模型 `MT-11`

第一版采用两阶段训练，保持现有离线 token 工作流：

```text
阶段 B1：raw EEG
  → 一套 EEGPT partial FT
  → 11-label EEG loss
  → 一套 256D multi-task EEG token

阶段 B2：[EEG, Wear, Video, Audio] 256D tokens
  → modality-token attention + query pooling
  → 两层共享 MLP
  → 11 个两层专属回归头
  → 11 个连续预测
```

EEG encoder 在 B1 完成后导出并冻结；B2 暂不向 EEG encoder 反传。这样可以分别判断“EEG 多标签监督”与“融合多头结构”的贡献，并控制显存和工程复杂度。

推荐命名：

```text
eegpt_partial_ft_multitask_11label_v1
multitask_shared2_head2_11label_v1
```

## 5. 多头回归网络的精确定义

设融合后的 pooled representation 为 `z ∈ R^H`，第一版固定 `H=128`。

### 5.1 两层共享 MLP

“两层”按两个线性层定义：

```text
shared(z):
  LayerNorm(H)
  Linear(H, H) → GELU → Dropout(0.1)
  Linear(H, H) → GELU → Dropout(0.1)
```

得到所有标签共享的 `h_shared ∈ R^H`。

### 5.2 11 个两层专属回归头

每个标签拥有独立参数：

```text
head_k(h_shared):
  Linear(H, H/2) → GELU → Dropout(0.1)
  Linear(H/2, 1)
```

模型使用 `ModuleDict[label_name]` 保存 11 个 head，最终按固定标签顺序拼为 `(batch, 11)`。

第一版不对输出做 `[1,5]` clamp；主指标使用连续原始预测。clamp 后结果可以作为诊断，不能替代主结果。

### 5.3 必需 head 对照

| Head ID | 结构 | 回答的问题 |
| --- | --- | --- |
| `E0_existing_11out` | 现有 pooled representation → 共享 MLP → `Linear(H,11)` | 历史 fixed-token 基线 |
| `H0_shared2_linear11` | 两层共享 MLP → 单个 `Linear(H,11)` | 控制共享 trunk 深度 |
| `H1_shared2_11xhead2` | 两层共享 MLP → 11 个两层专属 head | 用户提出的正式多头结构 |
| `H0_param_matched` | 加宽共享 trunk → `Linear(H,11)`，总参数量匹配 H1 | 判断收益是否只来自参数量增加；仅在 H1 晋级后运行 |

核心 head 机制比较为 `H1 - H0_shared2_linear11`。输入 token、fusion、dropout、训练 seed 和数据顺序必须完全匹配。

## 6. 损失函数

### 6.1 第一版正式损失

对每个标签只用 train split 统计量标准化：

```text
z_ik = (y_ik - mean_train,k) / std_train,k
```

有标签掩码 `m_ik` 时：

```text
L_k = sum_i m_ik * (pred_ik - z_ik)^2 / sum_i m_ik
L_total = (1 / 11) * sum_k L_k
```

当前 11 个标签均完整，仍保留 mask 接口以固定数据合同。等权的 per-label standardized MSE 可以防止高方差标签主导总 loss。

### 6.2 暂缓的损失扩展

以下机制只在正式基线显示明确失败模式后进入单因素消融：

- 高分尾部加权 MSE 或 Huber；
- ordinal cumulative 辅助损失；
- uncertainty weighting；
- GradNorm、PCGrad 等梯度冲突处理。

第一轮同时改变 head、loss 和 encoder 会削弱结论可解释性，因此第一版只使用等权标准化 MSE。

## 7. 公平对照矩阵

| Route ID | EEG 表征 | Fusion head | 训练套数 | 科学角色 |
| --- | --- | --- | ---: | --- |
| `B_mean` | 无模型 | train global mean / train-only subject mean | 0 | 防止低方差标签的伪提升 |
| `F0_H0` | `eegpt_frozen_v1` | `H0_shared2_linear11` | 每 protocol/seed 1 套 | label-free 共享输出对照 |
| `F0_H1` | `eegpt_frozen_v1` | `H1_shared2_11xhead2` | 每 protocol/seed 1 套 | 纯 head 专门化效应 |
| `FTfat_H1` | fatigue-supervised `eegpt_partial_ft_single_fatigue_v1` | `H1` | 每 protocol/seed 1 套 | fatigue 表征向其他标签迁移 |
| `ST11_F0` | `eegpt_frozen_v1` | 11 套独立两层 trunk + 两层 scalar head | 每标签独立 | ST 架构下的 frozen EEG 对照 |
| `ST11_PFT` | 11 套 label-specific EEGPT partial FT | 11 套独立两层 trunk + 两层 scalar head | 每标签独立 | 标签专门化上界 |
| `MT11_H1` | 一套 11-label EEGPT partial FT | `H1` | 每 protocol/seed 1 套 | 统一多情绪主模型 |

必须给出的配对比较：

1. `F0_H1 - F0_H0`：专属 head 的作用。
2. `FTfat_H1 - F0_H1`：fatigue 监督表征的跨标签迁移。
3. `MT11_H1 - F0_H1`：多标签 EEG 监督的总体价值。
4. `MT11_H1 - FTfat_H1`：多标签监督相对 fatigue-only 监督的价值。
5. `ST11_PFT - ST11_F0`：独立模型中标签专属 EEG 微调的价值。
6. `MT11_H1 - ST11_PFT`：统一模型相对 11 套专门模型的性能与成本平衡。

## 8. 分阶段执行计划与停止门槛

### Phase 0：数据与 split 审计

检查项：

- 28,819 行、1,253 events、每 event 23 个窗口；
- 11 标签存在、顺序固定、全部有限；
- 每个 protocol 的 train/validation/test 在 window、event、subject-day 三层均无违规重叠；
- manifest 中记录 split root；`within_subject_day` 只允许解析到 aligned `outputs/splits/within_subject_day`；
- 所有 token 的 `sample_id` 与 canonical index 完全同序；
- `cross_day` 和 `within_subject_day` 的 frozen、fatigue partial-FT 与后续 multi-task token 路径独立；
- 重新核验历史上以 strict 名称记录的 fatigue-supervised EEG token，并在复用时迁移到 canonical 名称，不能直接假定该阻塞仍存在或已经消失。

输出：

```text
phase0_contract.json
phase0_label_distribution.csv
phase0_split_overlap.json
phase0_token_inventory.json
```

**Gate 0：** 行数、event 完整性、标签有限性、split 隔离或 sample 对齐任一失败，停止全部训练并修复合同。

### Phase 1：fatigue 复现与模型 smoke

任务：

1. 用新单任务入口在 `fatigue / cross_day / seed 240800` 复现当前 EEGPT partial-FT 配置。
2. 检查 train-only normalization、最后 2 blocks 解冻、AMP、梯度裁剪和 best validation checkpoint。
3. 对 `H0`、`H1` 做前向、反向、save/load、缺失模态 mask 和输出标签顺序测试。
4. 在一个小 split 上验证 `(batch,11)` prediction、逐标签 loss 和反标准化。

复现阶段使用 legacy 的窗口级 checkpoint selector；通过后，ST 与 MT 的正式实验统一切换为 event-level validation selector。相同软件和确定性设置下，保存预测的最大绝对差要求不超过 `1e-5`；底层 CUDA 算子无法完全确定时，RMSE/raw r/centered r 的绝对差均要求不超过 `0.005`，并记录运行环境。

**Gate 1：** fatigue 的数据行、split、可训练参数集合必须与当前流程一致；相同 checkpoint/seed 的差异超出上述容差时停止并定位。H1 任一标签无梯度或保存后预测不一致时停止。

### Phase 2：fixed-token head 验证（已完成的历史筛查）

#### Phase 2A：旧 embedding-route 筛查

使用 label-free `eegpt_frozen_v1` 和现有 `E0_existing_11out`：

- 候选仍为现有 12 条 `B0/A1/A2 × Wphysio/Wdeep × full/no_audio` 路线；
- 使用 seeds `240800, 240801, 240802`；
- 每个 protocol 根据 validation macro standardized RMSE 选一条共同 route；
- fatigue 和 positive/negative 两组 validation 指标作为 guardrail；
- 历史 test winner 只作背景，不能参与新 route 锁定。

如果已有 seed `240800` 产物通过路径、配置和预测审计，可以复用，只补另外两个 seeds。

#### Phase 2B：head 配对

在锁定 route 上，用完全相同输入比较 `H0_shared2_linear11` 与 `H1_shared2_11xhead2`。

3-seed 晋级门槛：

- 两个主协议的 mean delta macro standardized RMSE 均不高于 `+0.01`；
- 至少一个主协议达到 `delta macro standardized RMSE <= -0.02` 或 `delta macro raw r >= +0.02`；
- 改善方向至少 `2/3` seeds 一致；
- fatigue、positive activation、negative distress 三组均未同时出现 RMSE 与 raw r 的实质退化。

该阶段已经证明 H1 入口可运行并完成 matched 3-seed 检查。12 条 embedding route 不作为最终结果表的12行；它们只承担历史筛查依据。当前正式输入为 `A1_Wphysio_no_audio__eeg_eegpt_partial_ft_multitask_11label_v1`，所有结构行读取同一套上游 `seed_240800` token。

**Gate 2：** 已通过。主矩阵固定使用 H1，并冻结 embedding route。

### Phase 3：标签专属 EEGPT embedding 资产库（已完成并审计）

对 11 个标签执行相同任务清单：

1. `eegpt_frozen_v1` EEG-only baseline；
2. `eegpt_partial_ft_single_<label>_v1`；
3. train-only channel/target normalization；
4. validation event-level RMSE early stopping；
5. 导出完整 256D EEG token；
6. 保存 window prediction 与 event-mean prediction作为上游质量审计；
7. token 产物按标签和协议保存，供第一条独立标量矩阵读取；第二条共享多头矩阵读取一套 11 标签共同监督 EEG token。

第一轮规模：

```text
11 labels × 2 primary protocols × 1 upstream seed = 22 EEG partial-FT runs
```

该阶段承担资产生成和上游质量审计，不用单 seed 预测指标作性能晋级结论。标签专属 embedding 已进入1254-run独立标量矩阵；两条路线同时改变 EEG 监督与回归头合同，其差值不能单独归因于 embedding。确有稳定收益时，才补上游 `240801/240802` 做 encoder-seed robustness，并安排隔离因素的机制对照。

**Gate 3：已通过。** 22 个 EEG runs 全部完成并通过 shape、sample order、split provenance、有限值和监督边界审计；embedding bank 已用于独立单标签结构矩阵，并保留为可复用资产。单 seed EEG-only 指标保留为上游审计，不据此筛除 token。

### Phase 4：`MT-11` 共同 EEGPT 局部微调

配置与 ST-11 对齐：

| 参数 | 固定值 |
| --- | ---: |
| EEGPT 解冻深度 | 最后 2 blocks + norm/projection/head |
| Encoder LR | `1e-5` |
| Shared/head LR | `1e-3` |
| Weight decay | `1e-4` |
| Dropout | `0.1` |
| Batch size | `256`，OOM 时保持等效 batch |
| Epoch 上限 | `80` |
| Patience | `15` |
| Gradient clip | `1.0` |
| EEG embedding seed | `240800` |
| Downstream regression seeds | `240800,240801,240802` |

任务：

1. EEG-only `MT11` 训练并导出一套 256D token；
2. 使用 Phase 2 锁定 route 训练 `MT11_H1`；
3. 同 seed 运行 `F0_H1` 和 `FTfat_H1`；
4. 保存每个标签、每个 event 的 prediction 供严格配对。

3-seed 晋级门槛：

- 相对 `F0_H1`，至少一个主协议达到 `delta macro standardized RMSE <= -0.02` 或 `delta macro raw r >= +0.02`，另一个指标保持非劣；
- 11 个标签中至少 6 个在 RMSE 或 raw r 上呈改善方向；
- positive activation 和 negative distress 两组各至少 2 个标签呈改善方向；
- fatigue 相对独立 fatigue 模型满足：`delta standardized RMSE <= +0.03`、`delta raw r >= -0.02`、`delta centered r >= -0.02`；
- 至少 `2/3` seeds 方向一致。

**Gate 4：** 通过则补到 5 seeds。未通过则保留 `ST-11` 与 fixed-token 方案，停止端到端 EEG+fusion 联训和其他模态解冻。

### Phase 5：5-seed 正式配对确认

补 seeds `240803,240804`，最终比较：

```text
F0_H0
F0_H1
FTfat_H1
ST11_F0
ST11_PFT
MT11_H1
```

每个 seed 内先对相同 test events 做 subject-day paired bootstrap `2,000` 次，再汇总 seed-level delta：

- mean ± std；
- 正向 seed 数；
- 每 seed paired 95% CI；
- pooled 结论前保留每 seed 数值；
- 11 个标签的正式显著性检验使用 Holm correction。

统一模型 `MT11_H1` 的正式晋级条件：

1. 相对 `F0_H1`，两个主协议均通过 3-seed 阶段的总体改善/守门条件；
2. 相对 `ST11_PFT`，macro standardized RMSE 不超过 `+0.02`，macro raw r 不低于 `-0.02`；
3. fatigue 的非劣阈值继续成立；
4. 关键 delta 至少 `4/5` seeds 方向一致；
5. 每个标签逐项呈现，平均值不能掩盖稀疏负性标签失败。

达到上述条件后，决策写为：

```text
promote_unified_multitask_11label
```

若统一模型整体非劣、少数标签稳定落后，决策写为：

```text
promote_multitask_with_specialist_subset
```

若 `ST11_PFT` 明显领先，决策写为：

```text
retain_single_task_11model_bank
```

若两类监督 EEG 均未超过 frozen baseline，决策写为：

```text
retain_fixed_tokens_multilabel
```

### Phase 6：只在主门槛通过后的机制消融

按单因素顺序运行：

1. `H1` vs `H0_param_matched`；
2. shared trunk 一层 vs 两层；
3. 专属 head 一层 vs 两层；
4. EEGPT 解冻最后 1、2、4 blocks；
5. 等权 MSE vs tail-aware Huber/ordinal auxiliary；
6. 检测到稳定梯度冲突后，再比较 PCGrad/GradNorm。

每个消融只改变一个因素，并继续使用相同 protocol、route、seed 和 test events。

## 9. 两条结构矩阵的报告口径

最终报告分成两组独立表格：

1. **共享多头表（已完成）**：行是0814单一窗口结构和0906的18个结构变体，列是11个情绪；每行使用同一 `A1_Wphysio_no_audio__eeg_eegpt_partial_ft_multitask_11label_v1` 输入和 H1 head 合同。
2. **独立单标签表（已完成）**：相同的19行和11列；每列用该标签专属 EEGPT partial-FT token 与独立的标量回归模型，Wear/Video/无音频合同和 repaired split 保持一致。结果不与共享多头表混排。

两条路线分别按 validation 指标选择结构，test 只用于锁定后报告。标签专属 EEG 由标签监督训练，跨列的输入表征不同；分析时明确这一监督边界。

## 10. 指标与结果表

### 10.1 每标签主指标

| 指标 | 层级 | 角色 |
| --- | --- | --- |
| RMSE | EMA event | 主误差指标 |
| standardized RMSE | EMA event | 跨标签等权汇总与门槛 |
| MAE | EMA event | 稳健误差补充 |
| raw Pearson r | EMA event | 跨 event 排序/动态范围 |
| within-subject centered r | EMA event | 个体内变化诊断 |
| label std / prediction std | EMA event | 常数预测和动态范围压缩诊断 |
| tail MAE (`y>=3`, 同时报 n) | EMA event | 稀疏负性标签高分尾部诊断 |

### 10.2 必须保留的汇总

- 11 标签逐项表；
- positive activation 五标签均值；
- negative distress 五标签均值；
- fatigue 单列；
- macro equal-label average；
- 每 label 的最佳 route、监督类型、seed wins 和 bootstrap CI；
- 参数量、训练 GPU-hours、token 存储和单次推理成本。

### 10.3 基线要求

- `global_train_mean`：每标签只用 train 均值；
- `subject_train_mean`：只用该被试 train events，未知被试回退 global train mean；
- `eegpt_frozen_v1`；
- 现有 fixed-token 11-output baseline；
- fatigue-supervised EEG transfer；
- `ST11_F0`、`ST11_PFT` 与 `MT11_H1`。

## 11. 计算预算与运行规模

首轮上游单 seed、下游 3-seed 预算：

| 模块 | 主要 runs | 说明 |
| --- | ---: | --- |
| Fixed-token route 筛查 | 已完成 | 只作历史筛查，不进入最终行轴 |
| H0/H1 head 配对 | 12 | 2 heads × 2 protocols × 3 seeds |
| ST11 EEG partial FT | 22 | 11 labels × 2 protocols × 1 upstream seed (`240800`) |
| 共享多头结构 × 情绪矩阵 | 114/114 | 19结构 × 2协议 × 3下游 seeds；每 run 用 H1 一次输出11标签 |
| 独立单标签结构 × 情绪矩阵 | 1254/1254 | 11标签 × 19结构 × 2协议 × 3下游 seeds；每标签用自己的 EEGPT token |
| 额外 frozen/PFT 固定结构 fusion | 暂缓 | 与已完成的两组结构矩阵分开，作为后续机制对照 |
| MT11 EEG partial FT | 2 | 2 protocols × 1 upstream seed (`240800`) |
| MT11/Frozen/Fatigue-transfer fusion | 18 | 3 representations × 2 protocols × 3 seeds |

下游若进入5-seed扩展，只补新的 downstream seeds；上游 embedding 仍固定 `seed_240800`。只有专门的 encoder-seed robustness 审计才生成额外 EEG seeds。运行 manifest 必须分别记录 `embedding_seed` 与 `downstream_seed`，以及开始/结束时间、GPU、峰值显存、epoch、best validation metric、失败类型和产物路径。

## 12. 实现落点

保持现有 `scripts/multilabel/48_run_fixed_tokens_multilabel_fusion.py` 作为历史基线，不改写其结果语义。建议新增：

| 文件 | 职责 |
| --- | --- |
| `scripts/multilabel/93_run_single_task_11label_eeg.py` | 循环 11 标签调用 EEG encoder 流程，写独立 token 与 manifest |
| `src/daily_multimodal/training/eeg_multitask.py` | 11-label EEGPT partial-FT、共享 loss、event-level validation |
| `scripts/multilabel/94_run_eegpt_multitask_11label.py` | MT11 EEG-only 训练与 256D token 导出 |
| `src/daily_multimodal/training/multihead_regression.py` | `H0/H1` 共享 trunk 与 11 个专属 head |
| `scripts/multilabel/95_run_multiemotion_fusion.py` | frozen/fatigue/ST/MT token 的统一融合入口 |
| `scripts/multilabel/96_summarize_multiemotion.py` | event 聚合、matched seed delta、bootstrap、Holm 和决策门槛 |

建议产物树：

```text
outputs/multiemotion_20260913/
├── phase0_contract/
├── head_screen/
├── single_task_eeg/{protocol}/{label}/seed_{seed}/
├── multitask_eeg/{protocol}/seed_{seed}/
├── fusion/{route_id}/{protocol}/seed_{seed}/
├── predictions/
└── summary/
```

远端 EEG token 建议单独存放：

```text
/vePFS-0x0d/DailyEEG_multimodal/embeddings/
└── eeg_encoder_256d_tokens_multiemotion_v1/
    ├── single_task/{protocol}/{label}/seed_{seed}.npz
    └── multitask_11label/{protocol}/seed_{seed}.npz
```

每个 token 文件至少包含：

```text
eeg_emb, eeg_mask, sample_id, protocol, seed,
label_names, target_label or multitask_target_set,
supervision_boundary, train_index, val_index, test_index
```

## 13. 最小测试清单

- 11 标签顺序从读取、loss、prediction 到保存全程一致；
- 单标签 runner 的 `target_label` 与导出目录一致；
- H1 恰好有 11 个相互独立的参数集合；
- 改动某个 head 参数只影响对应标签输出；
- 每个 head 都能收到非零梯度；
- target normalization 只由 train split 拟合；
- 缺失一个模态时 mask 后预测有限；
- event 聚合不跨 event；
- train/validation/test 的 event 与 subject-day 合同通过；
- prediction NPZ 可以完整复算 Markdown 表；
- subject-day bootstrap 在 seed 内进行，再汇总 seeds；
- fatigue 新入口复现通过后才允许批量运行 11 标签。

## 14. 预期论文论证结构

这组实验最终可支持三个层次的结论：

1. **能力扩展**：同一 EEG-aligned 四模态系统能够从 fatigue 扩展到 11 个日常情绪维度。
2. **共享机制**：共享 EEG/融合表示与标签专属回归头能够利用跨标签结构，同时为每个标签保留预测专门化。
3. **效率权衡**：一套 `MT11_H1` 模型与 11 套 `ST11_PFT` 模型在性能、训练成本、token 存储和推理成本上的可量化平衡。

论文主张由最终门槛决定：统一模型达到预设非劣与稳定性门槛时，主线强调共享多任务建模；独立模型稳定领先时，主线强调情绪特定生理表征与选择性专门化；监督 EEG 未超过 frozen token 时，主线保留 fixed-token 多情绪基线并停止扩大 encoder 复杂度。

## 15. 执行检查表

- [x] Phase 0 数据、event、split、token 合同通过
- [x] Phase 1 fatigue 复现与 H0/H1 smoke 通过
- [x] Phase 2 validation-only route 锁定完成
- [x] Phase 2 H1 多头结构通过 3-seed gate
- [x] 主矩阵锁定 `A1_Wphysio_no_audio__eeg_eegpt_partial_ft_multitask_11label_v1`，并将0814收敛为单一窗口结构
- [x] 完成 A1 输入的0814单行 + 0906结构变体 × 11情绪的两协议三指标表
- [x] Phase 3 完成 11 标签 × 2 协议 × 1 upstream seed 的22个 ST EEG runs
- [x] Phase 3 完成全部 token 的 shape/order/provenance 审计
- [x] Phase 4 完成 MT11 EEG 与固定输入19结构的3-seed screen
- [ ] Phase 4 fatigue、标签组和总体门槛通过
- [ ] Phase 5 补齐 5 seeds、paired bootstrap 与 Holm correction
- [x] 输出 A1 固定 embedding 的结构主表
- [x] 完成1254-run独立单标签矩阵并输出独立11列表
- [ ] 记录最终 `promote / retain / stop` 决策
