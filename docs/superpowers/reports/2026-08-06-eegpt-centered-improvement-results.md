# EEGPT Centered-r 改进与 strict within-subject-day 重跑结果

## 结论摘要

本轮先审计原始 split，再修复 `within_subject_day` 的 subject-day pair 重叠，并在新构造的 `within_subject_day_strict` 上重跑 16 个 B0/A1/A2 route-aware 组合。更新后的主结论如下：

- 原始 `within_subject_day` 的 train/val/test window index 两两不重叠，但三组共享全部 `150` 个 subject-day pair；它只能作为窗口级诊断，不能支撑严格“同一被试内跨日期泛化”结论。
- 新的 `within_subject_day_strict` 以 subject-day pair 为划分单位：train `90` 个 subject-day、val `30` 个、test `30` 个；`subject_day_overlap_train_val = 0`、`subject_day_overlap_train_test = 0`、`subject_day_overlap_val_test = 0`。
- strict 重跑后，test raw r 从旧协议的 `0.3032–0.3161` 降至最高 `0.1115`，test centered r 从旧协议最高 `0.1026` 降至最高 `0.0682`；严格 held-out day 下，当前 cross-attention 组合对同一被试内部跨日期 fatigue 波动的可泛化预测仍较弱。
- strict split 的最低 RMSE 为 `B0_Wdeep_bio_only`：RMSE `0.9678`、raw r `0.0665`、centered r `0.0452`。最高 raw r 为 `A2_Wdeep_no_audio`：raw r `0.1115`、RMSE `0.9783`、centered r `0.0196`。最高 centered r 为 `B0_Wphysio_no_video`：centered r `0.0682`、RMSE `0.9937`、raw r `0.0598`。
- 因此，论文口径应保留 `cross_day` 作为当前可用的严格跨日期证据；原始 `within_subject_day` 结果保留为窗口级诊断；strict within-subject-day 作为更严格的改进目标，提示需要 residual head、日期级标签建模、窗口聚合或更强的个体内动态目标。

实验 1 显示，subject mean 已经覆盖当前模型 raw fatigue 排序信号的主要量级：

- `cross_day`：subject-mean baseline raw r = `0.2828`，当前 A1 主候选 raw r = `0.2594`。
- `within_subject_day`：subject-mean baseline raw r = `0.4448`，当前最佳 raw r = `0.3161`。
- subject-mean baseline 的 centered prediction 在每个被试内部为常数，因此 centered r 数学上未定义；它没有提供个体内动态信号。

实验 2 共完成 27 个 run，覆盖 3 个协议/路线组合、raw baseline、centered MSE 和 centered correlation 两类目标。centered loss 没有达到预设的“centered r 至少提升 `+0.02` 且 RMSE 恶化不超过 `+0.015`”标准：

- `cross_day/A1_Wphysio_no_audio` 的最高 centered r = `0.0454`，低于现有最佳 `0.1107`。
- `within_subject_day/A1_Wphysio_no_audio` 的最高 centered r = `0.1088`，只比现有最佳 `0.1026` 高 `0.0062`；对应 RMSE = `0.9652`，比当前主候选 `0.9348` 高 `0.0304`。
- `within_subject_day/B0_Wdeep_no_audio` 的 raw baseline centered r = `0.0960`，centered loss 没有带来提升。

当前不建议直接开展实验 3。应先修复严格的日期级 split，再对 `within_subject_day/A1_Wphysio_no_audio + raw_centered_corr, λ=1.0` 和 `raw_centered_mse, λ=0.3` 做多 seed 复核。复核后若 centered r 的提升方向稳定，再开展 residual head；若提升消失，优先检查日期级标签与窗口定义。

## 实验 1：Split audit + subject-mean baseline

### Split audit

| protocol | train | val | test | train/test subject overlap | train/test subject-day overlap | train/test index overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cross_subject` | 17641 | 5106 | 6072 | 0 | 0 | 0 |
| `cross_day` | 16813 | 6187 | 5819 | 15 | 0 | 0 |
| `within_subject_day` | 17243 | 5708 | 5868 | 15 | 150 | 0 |

`cross_day` 的 15 个被试跨越 train/test，但 subject-day pair 完全分离，符合跨日期评估口径。

`within_subject_day` 的 train、val、test 各包含 150 个 subject-day pairs，且三组两两重叠。当前 split 在窗口层面互斥，在 subject-day 层面存在同一天窗口跨 split 的情况；它支持窗口级泛化诊断，不能支撑严格的“使用被试早期日期预测后续日期”结论。

### Subject-mean baseline

subject mean 只使用 `pretrain + finetune` 的 fatigue 标签估计；测试被试均已在训练集出现，coverage = `100%`。

| protocol | RMSE | MAE | raw r | centered r | current RMSE winner |
| --- | ---: | ---: | ---: | ---: | --- |
| `cross_day` | 0.9492 | 0.7406 | 0.2828 | undefined (zero within-subject prediction variance) | A1: RMSE 0.9298, raw r 0.2594, centered r 0.1107 |
| `within_subject_day` | 0.8965 | 0.6921 | 0.4448 | undefined (zero within-subject prediction variance) | A1: RMSE 0.9348, raw r 0.3032, centered r 0.0778 |

subject mean 的 centered r 为 undefined 是预期行为：对每个被试输出一个固定均值，去除被试均值后预测向量全为零。该 baseline 直接显示了稳定被试差异对 raw r 的贡献，并未证明任何个体内波动追踪能力。

## 实验 2：Raw + centered multi-task loss

训练保持原 cross-attention 结构、train-only token/target normalization、`pretrain + finetune` train 规则和固定 test 指标定义。centered loss 在 batch 内对至少包含两个窗口的 subject 计算，使用 subject-balanced batches。每个组合包含 raw baseline、`raw_centered_mse` λ ∈ {0.1, 0.3, 0.5, 1.0} 和 `raw_centered_corr` λ ∈ {0.1, 0.3, 0.5, 1.0}。

| protocol / experiment | run | RMSE | raw r | centered r | 相对当前中心指标 | 判定 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `cross_day/A1_Wphysio_no_audio` | raw baseline | 0.9685 | 0.1689 | 0.0404 | -0.0703 | 保留对照 |
| `cross_day/A1_Wphysio_no_audio` | MSE λ=0.3 | 0.9559 | 0.1849 | 0.0454 | -0.0653 | 无效 |
| `cross_day/A1_Wphysio_no_audio` | Corr λ=0.3 | 0.9470 | 0.0730 | -0.0519 | -0.1626 | 无效 |
| `within_subject_day/A1_Wphysio_no_audio` | raw baseline | 0.9511 | 0.2748 | 0.0654 | -0.0124 vs A1 | 保留对照 |
| `within_subject_day/A1_Wphysio_no_audio` | MSE λ=0.3 | 0.9498 | 0.2671 | 0.0708 | -0.0070 vs A1 | 无效 |
| `within_subject_day/A1_Wphysio_no_audio` | Corr λ=1.0 | 0.9652 | 0.2221 | 0.1088 | +0.0062 vs current best | 诊断候选 |
| `within_subject_day/B0_Wdeep_no_audio` | raw baseline | 0.9565 | 0.2739 | 0.0960 | -0.0066 vs current best | 保留对照 |
| `within_subject_day/B0_Wdeep_no_audio` | MSE λ=0.1 | 0.9485 | 0.2601 | 0.0589 | -0.0437 vs current best | 无效 |

实验 2 是单 seed λ sweep，结果用于方向筛选。当前 `within_subject_day/A1` 的 correlation loss λ=1.0 只产生小幅 centered r 增益，并伴随明显 RMSE 代价；同时该协议的 subject-day split 存在重叠，因此这条结果不适合作为结构改造的直接依据。

## 实验 0：strict within-subject-day split 修复与重跑

### 修复后的划分

新的 `within_subject_day_strict` 直接使用 canonical EEG window index 的 `subject_id + day_id` 作为最小划分单位。同一个 subject-day pair 的所有 23 个窗口必须进入同一个 split；每个被试按日期顺序划分为早期 train、中期 val、后期 test。训练时仍然使用 `pretrain + finetune` 作为 train，`val` 用于早停和模型选择，`test` 只用于最终报告。

| split | windows | events | subjects | days | subject-day pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pretrain` | 5957 | 259 | 15 | 30 | 30 |
| `finetune` | 11178 | 486 | 15 | 60 | 60 |
| `train` | 17135 | 745 | 15 | 90 | 90 |
| `val` | 5658 | 246 | 15 | 30 | 30 |
| `test` | 6026 | 262 | 15 | 30 | 30 |

| overlap check | count |
| --- | ---: |
| `index_train_val` | 0 |
| `index_train_test` | 0 |
| `index_val_test` | 0 |
| `subject_day_overlap_train_val` | 0 |
| `subject_day_overlap_train_test` | 0 |
| `subject_day_overlap_val_test` | 0 |

该修复保留“同一批被试均出现在 train/val/test”的 within-subject 设定，同时保证验证集和测试集来自未参与训练的日期。它回答的问题变为：给定每个被试较早日期的训练数据，模型能否预测同一被试后续日期的 fatigue。

### strict test 结果

| 选择标准 | 最优实验 | RMSE | MAE | Raw r | Centered r | Per-subject r mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 最低 RMSE | `B0_Wdeep_bio_only` | 0.9678 | 0.7183 | 0.0665 | 0.0452 | 0.0222 |
| 最高 raw r | `A2_Wdeep_no_audio` | 0.9783 | 0.7464 | 0.1115 | 0.0196 | 0.0548 |
| 最高 centered r | `B0_Wphysio_no_video` | 0.9937 | 0.7595 | 0.0598 | 0.0682 | 0.0892 |

| Experiment | RMSE | MAE | Raw r | Centered r | Per-subject r mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B0_Wphysio_full` | 0.9970 | 0.7559 | 0.0510 | 0.0389 | 0.0511 |
| `B0_Wphysio_no_audio` | 1.0003 | 0.7574 | 0.0213 | -0.0604 | -0.0245 |
| `B0_Wphysio_no_video` | 0.9937 | 0.7595 | 0.0598 | 0.0682 | 0.0892 |
| `B0_Wphysio_bio_only` | 0.9706 | 0.6881 | 0.0603 | 0.0274 | 0.0353 |
| `B0_Wdeep_full` | 0.9741 | 0.7402 | 0.0853 | 0.0458 | 0.0800 |
| `B0_Wdeep_no_audio` | 0.9768 | 0.7238 | 0.0633 | 0.0363 | 0.0331 |
| `B0_Wdeep_no_video` | 0.9843 | 0.7521 | 0.0880 | 0.0434 | 0.0409 |
| `B0_Wdeep_bio_only` | 0.9678 | 0.7183 | 0.0665 | 0.0452 | 0.0222 |
| `A1_Wphysio_full` | 0.9847 | 0.7484 | 0.0256 | 0.0337 | 0.0716 |
| `A1_Wphysio_no_audio` | 0.9805 | 0.7390 | 0.0684 | -0.0259 | -0.0245 |
| `A1_Wdeep_full` | 0.9989 | 0.7680 | 0.0630 | 0.0454 | 0.0735 |
| `A1_Wdeep_no_audio` | 1.0025 | 0.7672 | 0.0291 | -0.0324 | 0.0233 |
| `A2_Wphysio_full` | 0.9941 | 0.7511 | 0.0429 | 0.0433 | 0.0913 |
| `A2_Wphysio_no_audio` | 0.9978 | 0.7553 | 0.0470 | -0.0496 | 0.0133 |
| `A2_Wdeep_full` | 0.9802 | 0.7462 | 0.0822 | 0.0467 | 0.0705 |
| `A2_Wdeep_no_audio` | 0.9783 | 0.7464 | 0.1115 | 0.0196 | 0.0548 |

### 更新后的解释

strict split 下，旧 `within_subject_day` 的高 raw r 没有保留下来。最强 raw r 只有 `0.1115`，最强 centered r 只有 `0.0682`，而且两者来自不同组合；这说明旧结果中的相当一部分相关性来自同一 subject-day 内窗口相似性或日期级分布共享。当前模型可以在 `cross_day` 中利用所有被试早期日期学习总体跨日期规律，但在“每个被试只用早期日期训练、预测该被试后期日期”的严格设置下，个体内 fatigue 波动仍没有形成稳定可泛化信号。

因此，可以写成：当前模型学到了一部分被试间和日期间的总体 fatigue 水平差异；在严格 within-subject held-out-day 设定下，被试内部跨日期 fatigue 波动仍较弱，尚不足以作为主结论。后续改进应把 `within_subject_day_strict` 作为主要检验场，而不是继续依赖原始 `within_subject_day`。

## 实验 3 是否值得做

### 当前决定

暂不直接做实验 3 的正式 residual-head 结论实验。

### 已完成与先做的两项工作

1. 已重建严格的 `within_subject_day_strict`：以完整 subject-day pair 为 split 单位，保证 train/val/test 的 subject-day pair 两两不重叠；保留同一被试跨日期的训练/测试关系。
2. 下一步先在 strict split 上对两个候选做多 seed：
   - `A1_Wphysio_no_audio + raw_centered_corr, λ=1.0`
   - `A1_Wphysio_no_audio + raw_centered_mse, λ=0.3`

strict 单 seed route matrix 显示 raw r 和 centered r 均明显下降，因此 residual head 仍值得作为机制实验，但不宜直接作为主线结论实验。若多 seed 后 centered r 稳定提升且 RMSE 代价受控，再扩大到 residual head；若 centered r 仍接近零或提升不稳定，应转向日期级标签、窗口尺度、事件级聚合和动态状态定义。

## 产物

- [split audit JSON](../../outputs/server_sync/eegpt_centered_improvement/split_audit_subject_day.json)
- [split audit Markdown](../../outputs/server_sync/eegpt_centered_improvement/split_audit_subject_day.md)
- [subject-mean baseline JSON](../../outputs/server_sync/eegpt_centered_improvement/subject_mean_baseline.json)
- [subject-mean baseline Markdown](../../outputs/server_sync/eegpt_centered_improvement/subject_mean_baseline.md)
- [multi-task loss matrix JSON](../../outputs/server_sync/eegpt_centered_improvement/multitask_loss_matrix.json)
- [multi-task loss matrix Markdown](../../outputs/server_sync/eegpt_centered_improvement/multitask_loss_matrix.md)
- [strict within-subject-day split audit JSON](../../outputs/server_sync/eegpt_centered_improvement/within_subject_day_strict_split_audit.json)
- [strict within-subject-day split audit Markdown](../../outputs/server_sync/eegpt_centered_improvement/within_subject_day_strict_split_audit.md)
- [strict within-subject-day route matrix JSON](../../outputs/server_sync/eegpt_centered_improvement/within_subject_day_strict_route_matrix.json)
- [strict within-subject-day route matrix Markdown](../../outputs/server_sync/eegpt_centered_improvement/within_subject_day_strict_route_matrix.md)
