# Multi-label Embedding Route Handoff

## 接手目标

本 handoff 面向当前 EEG-aligned 四模态疲劳预测路线的多标签扩展。核心问题是：当前技术路线是否应该继续只围绕 `fatigue` 优化 embedding，还是改为利用 11 个 PANAS 风格标签共同训练一套更通用的 supervised representation。

当前可用标签为：

```text
inspired, alert, determined, attentive, active,
hostile, nervous, upset, afraid, ashamed, fatigue
```

需要先区分三种表征类型：

| 类型 | 定义 | 换目标标签后 embedding 是否变化 |
| --- | --- | --- |
| Label-free / frozen embedding | 提取器不看任何下游标签 | 不应变化 |
| Single-task supervised embedding | 用单个目标标签训练 encoder/projection | 会随目标标签变化 |
| Multi-task supervised embedding | 用 11 个标签共同训练 encoder/projection | 对所有标签共享一套 embedding |

## 当前路线的监督边界

当前四模态路线中，监督边界并不一致：

| 模态 / route | 当前状态 | 说明 |
| --- | --- | --- |
| EEG `eegpt_frozen_v1` | label-free / frozen | 可作为共享 fixed EEG token baseline |
| EEG `eegpt_partial_ft_v1` | `fatigue` supervised | 当前最强 EEG route；换标签重新训练后 embedding 会变化 |
| EEG `cbramod_frozen_v1` | encoder frozen，projection/head supervised | encoder 本体不变，但 256D projection 受目标标签影响 |
| EEG `cbramod_partial_ft_v1` | supervised | 换标签后 embedding 会变化 |
| EEG `eeg_de_5band_1s_avg_v1` | DE 特征固定，MLP projection supervised | 原始 DE 不变，256D projection 会变化 |
| Wear `Wphysio` | fixed | 生理/运动手工特征和投影路线当前不依赖目标标签 |
| Wear `Wdeep` | fixed | 当前是固定随机 1D conv / TCN-like sequence extractor，不是监督训练 encoder |
| Video `B0/A1/A2` | frozen | 2x 主脸 ROI + DINOv2 frozen embedding |
| Audio openSMILE | fixed | eGeMAPS 是固定特征提取；openSMILE 本身不可微调 |

准确表述：当前最强 EEG embedding 是 `fatigue-supervised representation`，但项目里也保留了 `eegpt_frozen_v1` 这条 label-free baseline。Video、Audio、Wear 当前基本是 fixed/frozen embedding 路线。

## 两大路线

### 路线 1：每个标签各自训练一套 embedding

每个标签单独训练 encoder/projection：

```text
fatigue -> eeg_emb_fatigue
alert   -> eeg_emb_alert
active  -> eeg_emb_active
...
```

优点：

- 每个标签的 representation 最专门。
- 可以测试不同标签需要的 EEG/Wear/Video/Audio 表征是否不同。

缺点：

- 需要维护 11 套 embedding，训练和存储成本高。
- 不利于构建统一多标签系统。
- 标签间相关性没有被显式利用。

适合用途：

- 作为上限实验，判断 single-task specialization 能带来多少收益。
- 对 `fatigue`、`alert`、`active` 等重点标签做少量对比。

### 路线 2：11 个标签共同训练一套 embedding

统一训练目标：

```text
raw signal -> shared embedding -> 11-label prediction head
```

基本损失：

```text
L_total = mean_k w_k * MSE(y_pred[k], y_true[k])
```

其中 `k` 遍历 11 个标签，`w_k` 是可选标签权重。第一版建议先使用等权 MSE；后续再考虑 centered loss、标签组权重或不确定性加权。

优点：

- 只生成一套 embedding。
- 利用标签间结构，例如 `alert`、`attentive`、`active` 与 `fatigue` 的关系。
- 更符合最终多标签预测系统。

缺点：

- 这不再是 label-free embedding，而是 multi-label-supervised embedding。
- 如果不同标签梯度冲突，需要处理标签权重或分组训练。
- 评估时必须逐标签报告，不应只看平均 loss。

## 路线 2 的三种实现

### A. Fixed embedding + multi-task fusion

token 生成方式：

```text
eeg_emb   = EEGPT frozen 或其他预生成 fixed EEG token
wear_emb  = Wphysio / Wdeep fixed token
video_emb = DINOv2 B0/A1/A2 frozen token
audio_emb = openSMILE eGeMAPS fixed token

[eeg_emb, wear_emb, video_emb, audio_emb]
-> fusion encoder
-> 11-label head
```

训练内容：

```text
只训练 fusion encoder + 11-label prediction head
```

这里的 `4 modality tokens` 来自已有 `.npz` embedding 文件，不在线更新。换标签或多标签训练时，底层 embedding 不变。

价值：

- 最干净地回答“当前 fixed/frozen 多模态表征对其他标签有没有用”。
- 成本最低，适合先跑完整 11 标签矩阵。
- 结果可作为所有 supervised embedding 方案的 baseline。

限制：

- 如果 fixed token 对某些标签信息不足，fusion head 不能补回底层表征缺失。

### B. Per-modality multi-task embedding + fusion

每个模态先单独用 11 个标签训练一套 256D embedding：

```text
EEG raw   -> EEG encoder   -> eeg_emb   -> 11-label EEG head
Wear raw  -> Wear encoder  -> wear_emb  -> 11-label Wear head
Video raw -> Video encoder -> video_emb -> 11-label Video head
Audio raw -> Audio encoder -> audio_emb -> 11-label Audio head
```

训练完后丢弃或保留单模态 head，取中间 embedding 做融合：

```text
[eeg_emb, wear_emb, video_emb, audio_emb]
-> fusion encoder
-> 11-label fusion head
```

token 生成方式：

```text
4 modality tokens 由各自已经 multi-task 训练好的单模态 encoder 预生成。
```

价值：

- 每个模态都学习“自己能解释的多标签状态信息”。
- 单模态贡献清楚，便于定位：例如 Wear 是否主要解释 `active/alert`，Audio 是否主要解释 `upset/nervous`。
- 比端到端训练更容易调试。

限制：

- 每个模态都被标签监督塑形，不能再称为 label-free。
- 单模态最优目标可能不等于融合最优目标。
- Video/Audio/Wear 的可训练 encoder 需要重新设计，不能直接把当前 fixed 特征当作已经可微调的 encoder。

### C. End-to-end multi-task multimodal training

raw signal 在线进入各模态 encoder，再由最终 fusion loss 统一反传：

```text
EEG raw   -> trainable EEG encoder   -> eeg token
Wear raw  -> trainable Wear encoder  -> wear token
Video raw -> trainable Video encoder -> video token
Audio raw -> trainable Audio encoder -> audio token

[eeg token, wear token, video token, audio token]
-> fusion encoder
-> 11-label head
```

token 生成方式：

```text
4 modality tokens 在线生成，并受最终 11-label fusion loss 更新。
```

训练内容：

```text
encoder + fusion + 11-label head 一起训练
```

价值：

- 目标最直接，优化最终多模态多标签预测。
- fusion 可以自己学习哪个模态对哪个标签重要。

限制：

- 算力和工程成本最高。
- 最容易过拟合，尤其 Video/Audio encoder 解冻时。
- 缺失模态、不同模态采样率和 batch 组织会显著复杂化训练。
- 单模态贡献解释更困难，需要 ablation 和 mask sensitivity 分析。

## 各模态改成可训练路线的可行性

| 模态 | 当前路线 | 可训练改造建议 |
| --- | --- | --- |
| EEG | EEGPT/CBraMod/DE 已有 supervised 路线 | 先做 `eegpt_multitask_11label_v1`，与 `eegpt_frozen_v1` 和 `eegpt_partial_ft_fatigue_v1` 对照 |
| Wear | `Wphysio` / `Wdeep` 当前 fixed | 优先做 `wear_tcn_multitask_v1`；比 Video/Audio 成本低，且与 `active/alert/fatigue` 生理关系更直接 |
| Video | DINOv2 frozen B0/A1/A2 | 先做 adapter/LoRA 或最后若干 block partial FT，不建议第一版全量 fine-tune |
| Audio | openSMILE fixed | openSMILE 本身不可微调；若要 fine-tune，需要换成 wav2vec2/HuBERT/emotion2vec 等 neural encoder |

建议不要同时解冻四个模态。先固定大部分变量，逐个验证收益。

## 推荐实验顺序

### Step 0：标签审计

先确认 11 个标签在 canonical 28,819 行 window index 中全部存在且数值分布合理：

```text
n, mean, std, min, median, max
```

如果某些负性标签极端稀疏或方差很低，应提前标记为低信号标签。

### Step 1：Fixed embedding + multi-task fusion baseline

目的：

```text
评估当前 fixed/frozen 四模态 token 对 11 个标签的可预测性。
```

建议输入：

```text
EEG:   eegpt_frozen_v1
Wear:  Wphysio / Wdeep
Video: B0/A1/A2
Audio: openSMILE
```

关键边界：

- `fixed_tokens_multitask_fusion_v1` 这条 baseline 的 EEG 必须优先使用 `eegpt_frozen_v1`，保持四路 token 都是 fixed/frozen 输入。
- `eegpt_partial_ft_v1` 已经带有 `fatigue` 监督，适合另设 `fatigue_supervised_eeg_transfer_11label_v1`，用于评估 fatigue-supervised EEG token 对其他标签的迁移能力。
- 这两条结果表应分开命名和汇报，避免把 label-free fixed baseline 与 fatigue-supervised transfer 混在同一个 baseline 结论里。

输出：

```text
每个 protocol x route 的 11-label metrics:
RMSE, MAE, raw r, centered r
```

实现要求：

- 当前 `scripts/32_run_eegpt_centered_loss.py` 是 `--target-label` 单目标入口，`AttentionRegressor` 输出标量预测；它可以作为结构参考，但不能直接视为 11-label training head。
- 第一版需要新增或扩展 multi-output 入口：target 从 `(N,)` 扩展为 `(N, 11)`，head 输出 11 维，并支持逐标签 loss mask、逐标签反标准化和逐标签 metrics。
- 等权 MSE 的归一化应按标签分别使用 train split 统计量，防止高方差标签主导总 loss。

重点协议：

```text
cross_day
within_subject_day_strict
```

`within_subject_day_strict` 是正式 held-out-day 口径；原始 `within_subject_day` 可保留为窗口级诊断。`cross_subject` 保留为诊断，不作为第一轮主结论。

### Step 2：EEG multi-task supervised embedding

目的：

```text
判断一套 11-label EEG embedding 是否比 fatigue-only EEG embedding 更适合整体任务。
```

对照：

| Route | 含义 |
| --- | --- |
| `eegpt_frozen_v1` | label-free EEG baseline |
| `eegpt_partial_ft_fatigue_v1` | 当前 fatigue single-task supervised EEG |
| `eegpt_multitask_11label_v1` | 11-label supervised EEG |

判断标准：

- `eegpt_multitask_11label_v1` 在 `fatigue` 上接近 single-task fatigue。
- 同时在 `alert/attentive/active` 等其他标签上优于 frozen 或 fatigue-only transfer。
- 不显著牺牲 `cross_day` 和 `within_subject_day_strict` 的稳定性。

进入下一阶段的硬门槛：

- 至少使用多 seed paired comparison，而不是单 seed 最优值。
- `fatigue` 相对 `eegpt_partial_ft_fatigue_v1` 的 RMSE / raw r / centered r 不能出现实质性退化；允许的退化阈值应在实验前写入 config 或 report header。
- `alert/attentive/active` 等 activation 标签相对 `eegpt_frozen_v1` 或 `fatigue_supervised_eeg_transfer_11label_v1` 应有稳定增益，并报告正向 seed 数。
- `cross_day` 与 `within_subject_day_strict` 两个主协议都需要通过稳定性检查后，再推进 Wear multi-task 或端到端方案。

### Step 3：Wear multi-task supervised embedding

优先做 Wear，而不是 Video/Audio：

- 训练成本低。
- 与 `active`、`alert`、`fatigue` 的生理联系更直接。
- 当前 `Wdeep` 只是 fixed random sequence baseline，替换为可训练 TCN/Transformer 的收益更有实验价值。

建议路线：

```text
wear_physio_fixed
wear_deep_fixed
wear_tcn_multitask_v1
wear_transformer_multitask_v1
```

### Step 4：Video / Audio adapter

Video：

```text
video_dinov2_frozen_B0/A1/A2
video_dinov2_adapter_multitask_v1
video_dinov2_lora_multitask_v1
```

Audio：

```text
audio_opensmile_fixed
audio_wav2vec2_adapter_multitask_v1
audio_emotion2vec_partial_ft_v1
```

openSMILE 保留为 fixed baseline；可训练 audio 需要 neural encoder。

### Step 5：端到端 multi-task fusion

只有在 Step 1-4 明确显示多标签 supervised embedding 有稳定收益后，再考虑端到端训练。第一版端到端应限制解冻范围：

```text
EEG partial FT + Wear TCN trainable + Video/Audio frozen
```

全四模态 encoder 同时解冻应作为后续高成本实验。

## 评估和汇报要求

所有实验必须按 protocol 独立训练：

```text
train = pretrain + finetune
val   = early stopping / model selection
test  = final metrics only
```

正式汇报协议：

```text
primary:    cross_day, within_subject_day_strict
diagnostic: cross_subject, 原始 within_subject_day
```

原始 `within_subject_day` 只能作为同一 subject-day 内窗口级诊断，不进入 held-out-day 主结论。

必须逐标签报告：

```text
RMSE
MAE
raw r
centered r
test label std
prediction std
```

不要只报告 11 标签平均值。平均值可以作为摘要，但主表必须保留逐标签结果。

建议增加两类汇总：

1. 标签组汇总：

```text
positive/activation: inspired, alert, determined, attentive, active
negative/distress: hostile, nervous, upset, afraid, ashamed
fatigue: fatigue
```

2. route 胜负表：

```text
每个标签上哪个 route 的 raw r / centered r / RMSE 最优
```

## 关键决策点

### 是否继续单标签 fatigue-supervised EEG

保留，但作为 strong fatigue baseline。它不适合作为统一多标签 embedding 的唯一主线，因为它的监督目标只来自 `fatigue`。

### 是否直接为每个标签训练 11 套 EEG embedding

暂不作为主线。可以选 2-3 个代表标签做上限对照：

```text
fatigue, alert, active
```

如果 single-task 明显优于 multi-task，再扩大到所有标签。

### 是否让所有模态都 multi-task supervised

可以，但应分阶段推进。推荐顺序：

```text
fixed embedding + multi-task fusion
-> EEG multi-task embedding
-> Wear multi-task embedding
-> Video/Audio adapter
-> constrained end-to-end fusion
```

### 如何解释 route 2B 中的 `4 modality tokens`

如果是 fixed embedding + multi-task fusion：

```text
4 tokens = 预生成 fixed/frozen .npz embedding
```

如果是 end-to-end multi-task fusion：

```text
4 tokens = raw signal 在线经过 trainable/frozen encoder 后生成
```

这两个实验含义不同，报告中必须分开命名。

## 建议命名

| 名称 | 含义 |
| --- | --- |
| `fixed_tokens_multitask_fusion_v1` | 当前 fixed/frozen token，只训练 fusion + 11-label head |
| `eegpt_multitask_11label_v1` | EEGPT partial FT，由 11-label loss 训练 |
| `wear_tcn_multitask_11label_v1` | Wear 序列 encoder，由 11-label loss 训练 |
| `video_dinov2_adapter_11label_v1` | DINOv2 adapter/LoRA，由 11-label loss 训练 |
| `audio_wav2vec2_adapter_11label_v1` | neural audio encoder adapter，由 11-label loss 训练 |
| `end2end_multitask_fusion_v1` | 多模态 encoder + fusion 端到端训练 |

## 当前结论

当前项目可以自然扩展到多标签监督表征学习，但需要明确监督边界：

- 当前 Video、Audio、Wear 主要是 fixed/frozen token，换标签后 embedding 不应变化。
- 当前最强 EEG route 是 `fatigue` supervised，换标签或改成 11-label loss 后 embedding 会变化。
- 多标签路线的第一步应是 `fixed_tokens_multitask_fusion_v1`，因为它成本最低、解释最干净。
- 真正的统一多标签 embedding 应优先从 EEG 做起，再做 Wear；Video/Audio 建议用 adapter/LoRA 或 neural encoder adapter，暂不做全量解冻。

接手时优先避免把三件事混在一起：

```text
label-free fixed embedding evaluation
single-task supervised embedding
multi-task supervised embedding
```

三者实验含义不同，命名、结果表和结论必须分开。
