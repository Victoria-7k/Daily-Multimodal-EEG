# EEGPT Centered-r 改进实验 1–2 结果

## 结论摘要

本轮将 `within_subject_day` 统一定义为 subject-day 级日期顺序划分：每个 subject-day 的全部窗口只进入一个集合；每个被试的早期日期进入 train，中期日期进入 val，后期日期进入 test。`cross_day` 与 `within_subject_day` 的 train/val/test 均保持 window index 和 subject-day pair 两两不重叠。

实验 1、2 已按这一划分重新完成，结果文件直接覆盖上一版同名产物。

- 实验 1 的 `within_subject_day` 规模为 train `17135`、val `5658`、test `6026`，subject-day pair 为 `90/30/30`，三组两两 overlap 均为 `0`。
- subject-mean baseline 的 raw r 为 `cross_day=0.2828`、`within_subject_day=0.2806`；centered r 数学上未定义，因为该 baseline 对每个被试输出固定均值。
- 实验 2 共完成 `27` 个 run。centered r 的最高值为 `cross_day/A1_Wphysio_no_audio + raw_centered_mse, λ=0.3` 的 `0.0454`；`within_subject_day` 中最高值为 `B0_Wdeep_no_audio + raw_centered_mse, λ=0.5` 的 `0.0386`。
- centered loss 没有形成稳定的个体内动态增益。预设的 centered r 绝对提升 `+0.02` 且 RMSE 恶化不超过 `+0.015` 的进入标准未被可靠满足。

当前结论：subject mean 已经解释了主要 raw fatigue 排序量级；在日期隔离的 `within_subject_day` 上，centered loss 尚未显示出稳定可泛化的个体内动态信号。实验 3 可以作为小规模机制验证继续推进，正式 residual head 结论实验应等待多 seed 证据。

## 评估口径

| protocol | 评估方式 | train | val | test | train/val subject-day overlap | train/test subject-day overlap | val/test subject-day overlap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cross_day` | 同一被试跨日期泛化 | 16813 | 6187 | 5819 | 0 | 0 | 0 |
| `within_subject_day` | 同一被试按日期顺序 held-out-day 泛化 | 17135 | 5658 | 6026 | 0 | 0 | 0 |

两个协议的 window index overlap 也均为 `0`。`within_subject_day` 中 15 个被试均出现在 train、val、test，测试的是同一被试后续日期的预测能力。

## 实验 1：Split audit + subject-mean baseline

### Split audit

| protocol | pretrain | finetune | train | val | test | train subject-day pairs | val subject-day pairs | test subject-day pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cross_day` | 6122 | 10691 | 16813 | 6187 | 5819 | 90 | 30 | 30 |
| `within_subject_day` | 5957 | 11178 | 17135 | 5658 | 6026 | 90 | 30 | 30 |

`cross_day` 和 `within_subject_day` 的 split audit 均通过：index overlap、subject-day overlap 全部为 `0`。`cross_day` 允许同一被试跨日期出现在 train/test；`within_subject_day` 进一步按每个被试的日期顺序构造 train、val、test。

### Subject-mean baseline

subject mean 只使用 `pretrain + finetune` 的 fatigue 标签估计；测试被试均在训练集出现，coverage 为 `100%`。

| protocol | RMSE | MAE | raw r | centered r |
| --- | ---: | ---: | ---: | ---: |
| `cross_day` | 0.9492 | 0.7406 | 0.2828 | undefined |
| `within_subject_day` | 0.9424 | 0.7151 | 0.2806 | undefined |

centered r 为 undefined 是预期行为：对每个被试输出一个固定均值，按被试去均值后预测向量为零。这个 baseline 直接量化了稳定被试差异对 raw r 的贡献，不提供个体内动态预测能力。

## 实验 2：Raw + centered multi-task loss

训练保持原 cross-attention 结构、train-only token/target normalization、`pretrain + finetune` train 规则和固定 test 指标定义。centered loss 在 batch 内对至少包含两个窗口的 subject 计算，并使用 subject-balanced batches。每个组合包含 raw baseline、`raw_centered_mse` λ ∈ `{0.1, 0.3, 0.5, 1.0}` 和 `raw_centered_corr` λ ∈ `{0.1, 0.3, 0.5, 1.0}`。

### 代表性结果

| protocol / route | loss | λ | RMSE | raw r | centered r | 选择含义 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `cross_day/A1_Wphysio_no_audio` | raw | 0 | 0.9685 | 0.1689 | 0.0404 | raw reference |
| `cross_day/A1_Wphysio_no_audio` | raw_centered_mse | 0.3 | 0.9559 | 0.1849 | 0.0454 | centered r 最高 |
| `cross_day/A1_Wphysio_no_audio` | raw_centered_corr | 0.1 | 0.9532 | 0.2018 | 0.0263 | raw r 最高 |
| `within_subject_day/A1_Wphysio_no_audio` | raw | 0 | 0.9959 | 0.0668 | -0.0137 | raw reference |
| `within_subject_day/A1_Wphysio_no_audio` | raw_centered_corr | 0.1 | 1.0007 | 0.0673 | 0.0061 | centered r 最高 |
| `within_subject_day/B0_Wdeep_no_audio` | raw | 0 | 1.0035 | 0.0745 | 0.0344 | raw reference |
| `within_subject_day/B0_Wdeep_no_audio` | raw_centered_mse | 0.5 | 0.9687 | 0.1506 | 0.0386 | RMSE、raw r、centered r 均为本路线最高 |

全量矩阵共 `27` 个 run，完整数值保存在同步 JSON 和 Markdown 中。不同 loss run 使用连续 seed；λ sweep 先用于方向筛选，最终机制结论仍需多 seed 复核。

### 结果解释

- `cross_day/A1` 的 centered r 从 raw reference 的 `0.0404` 提升到 `0.0454`，提升幅度很小。
- `within_subject_day/A1` 的最高 centered r 为 `0.0061`，RMSE 为 `1.0007`；该路线没有形成稳定收益。
- `within_subject_day/B0` 的 `raw_centered_mse, λ=0.5` 得到本轮较好的单次结果，但 centered r 仅为 `0.0386`，需要多 seed 验证稳定性。
- 所有路线的 centered r 仍处于较低水平，当前证据支持“总体 fatigue 水平信号强于个体内跨日期动态信号”。

## 实验 3：初步多 seed 验证

在 `within_subject_day` 上完成两个候选与对应 raw baseline 的五 seed 配对验证。每对实验共享 seed、模型结构、数据划分、优化器和训练预算。进入 residual head 正式实验的门槛为：mean Δcentered r ≥ `+0.0200`、mean ΔRMSE ≤ `+0.0150`，且至少 `4/5` 个 seed 的 centered r 为正增益。

| Candidate | Raw centered r | Candidate centered r | Mean Δcentered r | Positive seeds | Mean ΔRMSE | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `B0_Wdeep_no_audio + raw_centered_mse, λ=0.5` | 0.0210 ± 0.0209 | 0.0147 ± 0.0214 | -0.0063 ± 0.0261 | 3/5 | -0.0154 ± 0.0275 | 未通过 |
| `A1_Wphysio_no_audio + raw_centered_corr, λ=0.1` | -0.0252 ± 0.0205 | -0.0173 ± 0.0078 | +0.0079 ± 0.0135 | 4/5 | +0.0127 ± 0.0230 | 未通过 |

`B0_Wdeep_no_audio` 的 centered r 平均下降；`A1_Wphysio_no_audio` 的 centered r 在 4 个 seed 中提高，但平均增益仅 `+0.0079`，未达到 `+0.0200` 门槛。两个候选都不进入 residual head 正式实验。

当前应优先研究日期级标签、事件级聚合、窗口尺度和动态状态定义；residual head 可在这些证据更新后重新评估。

## 产物

- [split audit JSON](../../outputs/server_sync/eegpt_centered_improvement/split_audit_subject_day.json)
- [split audit Markdown](../../outputs/server_sync/eegpt_centered_improvement/split_audit_subject_day.md)
- [subject-mean baseline JSON](../../outputs/server_sync/eegpt_centered_improvement/subject_mean_baseline.json)
- [subject-mean baseline Markdown](../../outputs/server_sync/eegpt_centered_improvement/subject_mean_baseline.md)
- [multi-task loss matrix JSON](../../outputs/server_sync/eegpt_centered_improvement/multitask_loss_matrix.json)
- [multi-task loss matrix Markdown](../../outputs/server_sync/eegpt_centered_improvement/multitask_loss_matrix.md)
- [experiment 3 multiseed summary JSON](../../outputs/server_sync/eegpt_centered_improvement/experiment3_multiseed/experiment3_multiseed_summary.json)
- [experiment 3 multiseed summary Markdown](../../outputs/server_sync/eegpt_centered_improvement/experiment3_multiseed/experiment3_multiseed_summary.md)
