# Fatigue Prediction Calibration Handoff 2026-08-16

## 目标

本 handoff 指导新对话沿着当前主线继续实验：`10s EEG/Wear/Video/Audio aligned windows -> 每模态 256D embedding -> modality-token cross-attention -> fatigue regression`。

当前可视化和原始预测显示，模型已经捕捉到一部分 fatigue 趋势，主要瓶颈集中在输出分布校准：

- 预测动态范围明显小于真实标签。抽样 subject-day 上，预测标准差通常只有真实标准差的 `10%-26%`；全测试集 Top 3 平均约为 `38%-63%`。
- 高 fatigue 段被低估。cross_day 与 cross_subject 的高值段局部 bias 常达到 `-2.4` 到 `-3.2`。
- 低 fatigue 段被抬高。within_subject_day 的低值段局部 bias 约为 `+1.4` 到 `+1.8`。
- within_subject_day 的趋势最好，说明个体内相对变化已有信号；当前更需要改进幅度校准、极端值建模和时间结构建模。

本轮实验的核心问题是：在固定当前 256D embedding 主线的前提下，如何让 fusion 输出从“保守均值预测”变成“有足够动态范围且极端值更准的 fatigue 预测”。

## 固定口径

### 数据与协议

- 数据：`28819` 个 EEG-aligned 10s windows。
- EEG 原始数据：`/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new/X.npy`，shape `(28819, 2000, 59)`，200 Hz，59 通道。
- 任务标签：`fatigue`。
- 主协议：`cross_day`、`within_subject_day`。
- 诊断协议：`cross_subject`。
- 训练集：`pretrain + finetune`。
- 验证集：`val`，用于早停、模型选择、进入下一阶段 gate。
- 测试集：`test`，只在阶段冻结后评估；调参过程不能用 test 做选择。

### 当前工程入口

- 技术路线文档：`technical_route_20260814.md`
- 当前脚本入口说明：`scripts/README.md`
- EEG 256D token 生成：`scripts/embeddings/34_run_eeg_encoder_matrix.py`
- 当前 fusion 训练入口：`scripts/window_fatigue/32_run_eegpt_centered_loss.py`
- 当前 fusion report：`outputs/server_sync/eeg_encoder_256d_5route_20260814/reports/eeg_encoder_256d_5route_fusion_video_only_seed240800_raw.json`
- 当前 prediction NPZ：`outputs/server_sync/eeg_encoder_256d_5route_20260814/predictions/eeg_encoder_256d_5route_fusion_video_only_seed240800_raw/`
- 可视化参考：
  - `fatigue_top3_routes_by_protocol.png`
  - `fatigue_top3_centered_routes_by_protocol.png`

### 当前主线基线

优先以 `EEGPT partial FT` 的 EEG 256D token 为主，因为它在两个主协议上稳定领先。Phase 1-4 先聚焦四条代表路线：

| protocol | baseline role | fusion route | val RMSE | val raw r | val centered r | test RMSE | test raw r | test centered r |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cross_day` | 最高 raw r | `B0_Wphysio_full` | 0.7895 | 0.4269 | 0.3005 | 0.9481 | 0.3504 | 0.1176 |
| `cross_day` | 最低 RMSE / centered r 较强 | `B0_Wphysio_no_audio` | 0.7814 | 0.3750 | 0.2146 | 0.9189 | 0.3301 | 0.1526 |
| `within_subject_day` | 最高 raw r / centered r | `B0_Wdeep_no_audio` | 0.8715 | 0.5366 | 0.3693 | 0.9337 | 0.4252 | 0.2122 |
| `within_subject_day` | 最低 RMSE | `A2_Wdeep_full` | 0.8768 | 0.4984 | 0.3004 | 0.9138 | 0.4046 | 0.1736 |

阶段通过后，再扩展到完整 `EEGPT partial FT × B0/A1/A2 × Wphysio/Wdeep × full/no_audio` 矩阵。CBraMod 和 DE+MLP 保留为对照或诊断，不作为第一轮调参主线。

## 统一指标

每个阶段必须输出以下指标，且 train/val/test 分开：

- `RMSE`
- `MAE`
- `raw Pearson r`
- `within-subject centered r`
- `per-subject r mean/std`
- `pred_std / true_std`
- `low fatigue bias`：低分位或 fatigue=1/2 区间的 `mean(pred - y)`
- `high fatigue bias`：高分位或 fatigue=4/5 区间的 `mean(pred - y)`
- `extreme MAE`：低疲劳与高疲劳样本上的 MAE
- `calibration slope/intercept`，如适用
- 每个 run 的 `loss history`、最佳 epoch、早停原因、配置快照

推荐输出目录：

```text
outputs/server_sync/fatigue_calibration_20260816/
  phase0_preflight/
  phase1_posthoc_calibration/
  phase2_loss_sampler/
  phase3_ordinal_head/
  phase4_temporal_fusion/
  phase5_formal_paired/
```

## Phase 0：Preflight 与诊断复现

### 要做什么

1. 读取当前 report 和 prediction NPZ。
2. 校验 `28819` 行、`sample_id`、`event_id`、`subject_id`、`test_index` 对齐。
3. 校验 train/val/test index 无重叠。
4. 复现两张可视化图对应的 subject-day 统计。
5. 生成 baseline diagnostic markdown，至少包含四条代表路线的全测试集和抽样 subject-day 的动态范围、low/high bias。

### 产物

- `phase0_preflight/preflight.json`
- `phase0_preflight/baseline_prediction_diagnostics.md`
- `phase0_preflight/baseline_distribution_stats.csv`

### 进入 Phase 1 的门槛

- `preflight.ok == true`
- 所有代表路线 prediction NPZ 可读，index 与 target 对齐。
- train/val/test 无重叠。
- 诊断能复现当前现象：预测动态范围显著低于真实标签，且高值低估/低值抬高成立。

若未通过，先修复数据读取和 prediction 对齐；不能进入模型调参。

## Phase 1：Post-hoc Calibration

### 要验证的判断

当前 fusion 输出可能已经包含排序和趋势信息，主要缺口是尺度和偏置校准。先用低成本、leakage-safe 的校准器检验这个判断。

### 实验设置

对四条代表路线分别做：

1. `linear_calibration`：在 val 上拟合 `y_cal = a * pred + b`。
2. `variance_calibration`：在 val 上对预测均值和标准差做匹配。
3. `clipped_variance_calibration`：校准后裁剪到 train label 范围。
4. `isotonic_calibration`：可选，只作为诊断，必须记录过拟合风险。

校准器只能使用 train/val split 的 label。test label 只能用于阶段冻结后的最终评估。

### 产物

- 每条路线的 calibrated predictions NPZ。
- `phase1_posthoc_calibration/metrics_val.json`
- `phase1_posthoc_calibration/metrics_test_frozen.json`
- `phase1_posthoc_calibration/calibration_summary.md`

### 进入 Phase 2 的门槛

基于 val，与未校准 baseline 配对比较。任一主协议至少一条代表路线满足：

- `pred_std / true_std` 提升 `>= +0.15`，或 high fatigue absolute bias 降低 `>= 15%`。
- `RMSE` 不恶化超过 `+0.010`。
- `MAE` 不恶化超过 `+0.010`。
- `raw r` 下降不超过 `-0.010`。
- centered r 下降不超过 `-0.015`。

若校准显著改善幅度但 RMSE 略微恶化，保留为“校准分析结果”，继续 Phase 2；若校准无法改善动态范围，先检查标签分布、归一化和预测保存逻辑。

## Phase 2：Loss 与 Sampler 改造

### 要验证的判断

MSE 训练让模型偏向条件均值，导致预测范围压缩。通过 loss weighting、variance regularization 和 label-balanced sampling，提高极端值和动态范围。

### 实验设置

在 `scripts/window_fatigue/32_run_eegpt_centered_loss.py` 中新增或分支实现以下模式：

1. `weighted_mse_label_bins`
   - 只用 train label 统计分箱。
   - 对低频 label bin 提高权重。
   - 初始权重建议限制在 `[0.5, 3.0]`。

2. `huber_extreme_weight`
   - Huber loss 作为主项。
   - fatigue 低/高分位样本加权。

3. `mse_variance_reg`
   - `loss = MSE + lambda_var * max(0, target_std - pred_std)^2`
   - std 统计建议在 batch 内按 subject 或 subject-day 聚合；样本数不足时跳过该项。
   - 初始 `lambda_var`: `0.05, 0.1, 0.2`。

4. `label_subject_balanced_sampler`
   - train sampler 同时考虑 label bin 和 subject。
   - 保持 val/test 不变。

第一轮只跑 seed `240800` 和四条代表路线；通过后跑 3 seeds。

### 产物

- `phase2_loss_sampler/configs/*.json`
- `phase2_loss_sampler/metrics_val.json`
- `phase2_loss_sampler/predictions/*.npz`
- `phase2_loss_sampler/loss_history/*.json`
- `phase2_loss_sampler/paired_delta_summary.md`

### 进入 Phase 3 的门槛

先看 3 seeds 的 val paired delta。进入下一阶段需满足：

- 至少一个主协议满足 `mean delta raw r >= +0.015` 或 `mean delta centered r >= +0.020`。
- `mean delta RMSE <= +0.010`。
- `mean delta MAE <= +0.010`。
- `pred_std / true_std` 平均提升 `>= +0.10`。
- high fatigue absolute bias 或 extreme MAE 降低 `>= 15%`。
- 至少 `2/3` seeds 在主选择指标上为正向。

若 Phase 2 已达到 test 上稳定提升，可以跳过 Phase 3，直接进入 Phase 5 formal paired。

## Phase 3：Ordinal / Distributional Head

### 要验证的判断

`fatigue` 是 1-5 阶梯标签，纯连续回归头容易输出均值。Ordinal head 可能更适合恢复低/高等级。

### 实验设置

在 fusion pooled representation 后替换或并联 head：

1. `ordinal_cumulative_head`
   - 输出 4 个 cumulative logits。
   - 训练 ordinal binary targets。
   - 推理时将等级概率转为期望 fatigue。

2. `classification_expectation_head`
   - 输出 5 类概率。
   - 预测值为 `sum(p_k * label_value_k)`。
   - 可与 MSE 辅助项联合训练。

3. `hybrid_regression_ordinal_head`
   - `loss = MSE + lambda_ord * ordinal_loss`
   - 初始 `lambda_ord`: `0.1, 0.3, 1.0`。

第一轮沿用 Phase 2 的最佳 loss/sampler 设置；如果 Phase 2 没有通过，则使用当前 raw MSE baseline 设置。

### 产物

- `phase3_ordinal_head/metrics_val.json`
- `phase3_ordinal_head/predictions/*.npz`
- `phase3_ordinal_head/confusion_or_bin_summary.md`

### 进入 Phase 4 的门槛

基于 3 seeds 的 val paired delta：

- `pred_std / true_std` 平均提升 `>= +0.10`。
- high fatigue absolute bias 降低 `>= 20%`，或 high fatigue MAE 降低 `>= 10%`。
- `raw r` 下降不超过 `-0.010`。
- centered r 下降不超过 `-0.015`。
- `RMSE` 不恶化超过 `+0.015`。
- 至少一个主协议通过，另一个主协议没有明显退化。

若 ordinal head 主要改善极端值但 RMSE 不占优，保留为“极端疲劳识别友好”的候选，并在 Phase 5 做 paired 检验。

## Phase 4：Temporal Fusion Head

### 要验证的判断

真实 fatigue 在 subject-day 内呈现平台和转折，单窗口 fusion 忽略了时间上下文。轻量 temporal module 可能提升个体内趋势和转折处预测。

### 实验设置

1. 每个 10s window 先通过当前四模态 cross-attention 得到 fused representation。
2. 按 `subject_id + day` 或 `event_id + event_window_id` 组成时间序列。
3. 候选 temporal module：
   - 小型 TCN：2-3 层，kernel size `3/5`。
   - GRU：1 层，hidden `128`。
   - Transformer encoder：1-2 层，2 heads。
4. 训练与 batch 构造必须保持 split 安全：
   - train sequence 只包含 train windows。
   - val/test sequence 不能从 train label 获取未来上下文。
   - 若使用邻近窗口上下文，必须记录上下文来源和是否跨 split。

第一轮跑 seed `240800`，只跑两个主协议和两条最强 baseline route：

- `cross_day + B0_Wphysio_no_audio`
- `within_subject_day + B0_Wdeep_no_audio`

### 产物

- `phase4_temporal_fusion/sequence_preflight.json`
- `phase4_temporal_fusion/metrics_val.json`
- `phase4_temporal_fusion/predictions/*.npz`
- `phase4_temporal_fusion/temporal_visual_checks/`

### 进入 Phase 5 的门槛

基于 val，任一主协议满足：

- `raw r >= baseline + 0.020`，或 centered r `>= baseline + 0.030`。
- `RMSE <= baseline + 0.015`。
- high fatigue bias 降低 `>= 15%`。
- 转折附近 MAE 不高于 baseline。
- sequence preflight 无 split leakage。

若 temporal module 只有 within_subject_day 提升，可作为个体内建模路线进入 Phase 5；cross_day 单独保留 calibration/loss/ordinal 候选。

## Phase 5：Formal 5-Seed Paired Evaluation

### 要验证的判断

从 Phase 1-4 选出的候选是否在主协议上稳定优于当前 baseline。

### 实验设置

候选数量控制在 `<= 4` 条：

- 最佳 post-hoc calibration。
- 最佳 loss/sampler。
- 最佳 ordinal/hybrid head。
- 最佳 temporal head，如 Phase 4 通过。

每条候选跑：

- protocols：`cross_day`、`within_subject_day`；`cross_subject` 只作诊断。
- seeds：沿用项目固定 5 seeds，若没有统一列表，使用 `240800-240804`。
- routes：先跑 Phase 0 的四条代表路线；候选通过后扩展完整 EEGPT partial FT fusion matrix。

### 产物

- `phase5_formal_paired/all_metrics.json`
- `phase5_formal_paired/paired_delta_summary.md`
- `phase5_formal_paired/prediction_distribution_summary.md`
- `phase5_formal_paired/top_route_visualizations/`
- 每个 run 的 predictions、loss history、config snapshot。

### 作为正式结论的门槛

相对同 seed baseline，正式候选需要满足：

- 两个主协议中至少一个满足：
  - `mean delta raw r >= +0.020`，或
  - `mean delta centered r >= +0.030`，或
  - `mean delta RMSE <= -0.015`。
- 另一个主协议不能明显退化：
  - `mean delta RMSE <= +0.015`
  - `mean delta MAE <= +0.010`
  - `mean delta raw r >= -0.010`
- `pred_std / true_std` 平均提升 `>= +0.10`。
- high fatigue absolute bias 降低 `>= 15%`。
- 至少 `4/5` seeds 在主选择指标上非负。
- 没有新增 split leakage、NaN、prediction shape mismatch 或训练不稳定。

若只改善校准指标而主指标变化小，结论写作应定位为“calibration improvement / extreme fatigue correction”，避免声明整体预测性能显著提升。

## Phase 6：扩展矩阵与论文表述

### 触发条件

Phase 5 至少一个候选通过正式门槛。

### 要做什么

1. 扩展到完整 `EEGPT partial FT × B0/A1/A2 × Wphysio/Wdeep × full/no_audio`。
2. 对 CBraMod frozen、CBraMod partial FT、DE+MLP 只跑最强候选设置下的小规模诊断，确认改进来自 fusion head 机制还是 EEGPT partial FT 特有。
3. 重新生成：
   - protocol Top 3 by raw r
   - protocol Top 3 by centered r
   - full result appendix
   - prediction-vs-true visualizations
4. 更新 `technical_route_YYYYMMDD.md`，保留监督边界：`EEGPT partial FT` 和非 frozen EEG tokens 属于 fatigue-supervised embedding。

### 完成门槛

- 完整矩阵无失败 run。
- 结果表与 predictions 可复核。
- 所有正式表只用 test split。
- 文档明确区分：
  - label-free/frozen embedding
  - fatigue-supervised EEG embedding
  - post-hoc calibration
  - temporal or ordinal fusion head

## 实现建议

### 最小代码改动路径

优先扩展当前 fusion 脚本，而不是新建一批互相平行的实验脚本：

- 在 `scripts/window_fatigue/32_run_eegpt_centered_loss.py` 增加：
  - `--loss-mode weighted_mse_label_bins|huber_extreme_weight|mse_variance_reg|hybrid_ordinal`
  - `--calibration none|linear|variance|clipped_variance|isotonic`
  - `--sampler default|label_balanced|label_subject_balanced`
  - `--head regression|ordinal|classification_expectation|hybrid`
  - `--temporal-head none|tcn|gru|transformer`
- 单独增加 diagnostics helper 可接受：
  - metrics JSON
  - predictions NPZ
  - split/index 文件
  - 输出 distribution stats 与 subject-day visual checks

### 训练默认值

保持当前 baseline 默认值，避免引入额外混杂：

- AdamW
- lr `1e-3`
- weight decay `1e-4`
- batch size `256`
- max epochs `80`
- patience `15`
- dropout `0.1`
- hidden_dim `128`

只有对应阶段明确需要时，才调整 loss、head、sampler 或 temporal module。

### 记录要求

每个 run 必须记录：

- git commit 或当前 worktree 状态摘要。
- 命令行参数。
- seed。
- protocol。
- EEG route。
- fusion route。
- loss/head/sampler/calibration/temporal 配置。
- train/val/test metrics。
- loss history。
- prediction NPZ。
- split leakage audit。
- GPU 型号、可见 GPU 数、运行时长。

## 停止条件

出现以下情况时停止扩展矩阵，先回到诊断：

- 校准后 `pred_std / true_std` 仍低于 `0.35`，且 high fatigue bias 没有改善。
- train/val 差距扩大，表现为 val RMSE 恶化 `> +0.03` 或 early stop 过早。
- CBraMod partial FT 出现明显过拟合时，不继续扩大 CBraMod partial sweep。
- temporal module 检查发现 sequence 构造跨 split 或使用未来 label 信息。
- 新候选只在单 seed 上提升，3-seed 不稳定。

## 新对话第一条执行指令建议

可直接复制给新对话：

```text
请读取 prediction_calibration_handoff_20260816.md、technical_route_20260814.md、scripts/README.md 和当前 fusion report，然后从 Phase 0 开始执行。每完成一个 Phase，先输出该阶段产物路径、val gate 结果和是否进入下一阶段；不要用 test 指标做调参选择。正式结论只在 Phase 5 的 5-seed paired evaluation 后给出。
```
