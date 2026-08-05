# EEGPT Multimodal Centered-r Improvement Handoff

## 接手目标

本 handoff 面向 EEGPT EEG 对齐后的多模态 cross-attention 下一轮实验。当前结果已经能支撑一个清楚判断：模型在 `cross_day` 中具备一定总体 fatigue 排序能力；原始 `within_subject_day` 只能作为窗口级诊断，因为 split audit 发现 train/val/test 两两共享全部 150 个 subject-day pair。2026-08-06 已补齐 `within_subject_day_strict`，并在 strict split 上重跑 16 个 B0/A1/A2 route-aware 组合；strict 结果显示 raw r 和 centered r 均明显下降，因此下一轮目标应从“哪个 route 最优”推进到“如何建模同一被试内部的跨日期 fatigue 波动”。

优先完成两类工作：

1. 做 split audit 和 subject-mean baseline，量化 raw r 有多少来自被试稳定均值。
2. 围绕 centered r 做目标函数和结构改造实验，判断当前模态 embedding 是否包含可学习的个体内动态信号。

## 当前依据和产物

本地主要依据：

- 完整结果报告：`docs/superpowers/reports/2026-08-01-eegpt-b0-a1-a2-cross-attention-results.md`
- HTML 报告：`docs/superpowers/reports/html/2026-08-01-eegpt-b0-a1-a2-cross-attention/report.html`
- PPT 风格汇报：`docs/superpowers/reports/html/2026-08-01-eegpt-b0-a1-a2-cross-attention/slides.html`
- 分窗示意图：`docs/superpowers/reports/html/2026-08-01-eegpt-b0-a1-a2-cross-attention/figures/windowing-method.svg`
- 三协议示意图：`docs/superpowers/reports/html/2026-08-01-eegpt-b0-a1-a2-cross-attention/figures/split-protocols-schematic.svg`

远端结果来源：

- merged Markdown：`/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/reports/eegpt_allvideo_fusion_matrix_all_protocols_summary.md`
- merged JSON：`/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/reports/eegpt_allvideo_fusion_matrix_all_protocols_summary.json`
- preflight：`/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/reports/eegpt_allvideo_alignment_preflight.json`
- strict split audit：`/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/reports/eegpt_centered_improvement/within_subject_day_strict_split_audit.md`
- strict route matrix：`/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/reports/eegpt_centered_improvement/within_subject_day_strict_route_matrix.md`

## 已确认的数据和划分口径

本轮实验使用固定 EEG canonical window index：

| 项目 | 当前取值 |
| --- | ---: |
| 评分事件数 | 1253 |
| 每事件窗口数 | 23 |
| 总窗口数 | 28819 |
| 窗口长度 | 10s |
| stride | 5s |
| 目标标签 | `fatigue` |

三种协议均来自 `/vePFS-0x0d/DailyEEG/splits_new/`。每个协议包含 `pretrain.json`、`finetune.json`、`val.json`、`test.json`、`split_info.json`。监督训练时使用 `pretrain + finetune` 作为 train，`val` 用于早停和模型选择，`test` 用于最终指标。

| 协议 | 评估重点 | Pretrain | Finetune | Train 合计 | Val | Test |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `cross_subject` | 跨被试泛化 | 6122 | 11519 | 17641 | 5106 | 6072 |
| `cross_day` | 跨日期泛化 | 6122 | 10691 | 16813 | 6187 | 5819 |
| `within_subject_day` | 当前同一被试内窗口级诊断 | 6122 | 11121 | 17243 | 5708 | 5868 |

当前解释口径：

- `cross_subject`：按被试划分，测试被试整体保留。
- `cross_day`：按日期维度划分，train/val/test 来自同一总体被试池；预测 `sub1 day5` 时，训练池包含所有被试的训练日期数据，例如 `sub1-sub5 day1-day3`。
- `within_subject_day`：当前 split audit 显示 train/val/test 的 window index 两两不重叠，但 subject-day pair 两两重叠均为 `150`。因此它只能作为同一批 subject-day 内的窗口级诊断；严格“同一被试内跨日期泛化”需要重新构造 subject-day pair 不重叠的 held-out-day split。

新增 split audit 结论：

| 协议 | index train/val/test 重叠 | subject-day train-val | subject-day train-test | subject-day val-test | 当前证据等级 |
| --- | ---: | ---: | ---: | ---: | --- |
| `cross_subject` | 0 | 0 | 0 | 0 | 可用于跨被试泛化 |
| `cross_day` | 0 | 0 | 0 | 0 | 可用于跨日期泛化 |
| `within_subject_day` | 0 | 150 | 150 | 150 | 只能用于窗口级诊断 |
| `within_subject_day_strict` | 0 | 0 | 0 | 0 | 可用于同一被试 held-out-day 诊断 |

`within_subject_day_strict` 的 split 规模为 train `17135`、val `5658`、test `6026`；对应 subject-day pair 为 train `90`、val `30`、test `30`。strict test 上最低 RMSE 为 `B0_Wdeep_bio_only`（RMSE `0.9678`、raw r `0.0665`、centered r `0.0452`），最高 raw r 为 `A2_Wdeep_no_audio`（raw r `0.1115`），最高 centered r 为 `B0_Wphysio_no_video`（centered r `0.0682`）。

## 关键结果

完成矩阵为 3 个协议 x 16 个实验，共 48 个 metrics 文件。16 个实验由 B0 的 8 个控制组合，加上 A1/A2 各 4 个真实使用视频的组合构成；`no_video` 和 `bio_only` 是路线无关控制项，只在 B0 名下保留一次。

| 协议 | RMSE 最优实验 | RMSE | MAE | Raw r | Centered r | Raw r 最优实验 | Raw r |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `cross_subject` | `B0_Wphysio_bio_only` | 0.9012 | 0.6861 | -0.0207 | -0.0415 | `B0_Wdeep_bio_only` | 0.0835 |
| `cross_day` | `A1_Wphysio_no_audio` | 0.9298 | 0.7055 | 0.2594 | 0.1107 | `A1_Wphysio_no_audio` | 0.2594 |
| `within_subject_day` | `A1_Wphysio_no_audio` | 0.9348 | 0.7255 | 0.3032 | 0.0778 | `A2_Wdeep_full` | 0.3161 |

需要保留的主结论：

- `A1_Wphysio_no_audio` 是当前 `cross_day` 和 `within_subject_day` 的低误差主候选。
- `A2_Wdeep_full` 在 `within_subject_day` 给出最高 raw r，适合作为排序指标对照。
- 音频在当前 openSMILE token 和 fusion 设置下整体拉高误差：18 个 full-vs-no-audio 配对中，full 的平均 ΔRMSE 为 `+0.0151`，平均 Δraw r 为 `-0.0271`。
- centered r 最高值仍低：`cross_day` 最优 `0.1107`，`within_subject_day` 最优 `0.1026`。这说明当前模型更擅长总体 fatigue 水平排序，对同一被试内部相对波动的追踪仍是主要改进空间。
- `cross_subject` 的最低 RMSE 行 raw r 接近 0，适合作为跨被试误差基线，不适合直接支撑跨被试 fatigue ranking 已经有效。

## Raw r 与 centered r 的指标公式

对 test split 中样本 `i`，真实标签为 `y_i`，预测为 `ŷ_i`，被试为 `s_i`。

Raw r：

```text
raw_r = corr({ŷ_i}, {y_i}), i in test
```

Within-subject centered r：

```text
ŷ_i_centered = ŷ_i - mean(ŷ_j | s_j = s_i, j in test)
y_i_centered = y_i - mean(y_j | s_j = s_i, j in test)

within_subject_centered_r =
  corr({ŷ_i_centered}, {y_i_centered}), i in test
```

解释：

- raw r 衡量全部测试窗口混在一起后的总体预测排序。
- centered r 先扣除每个被试自己的测试集均值，再衡量同一被试内部的相对升降是否被模型排对。
- raw r 高、centered r 低时，结论应写成：模型已经捕捉到被试间稳定疲劳水平差异和总体排序信号，但个体内跨日期疲劳波动仍较弱。

## 下一轮实验总览

建议按以下顺序执行：

0. 构造严格 within-subject held-out-day split
1. Split audit + subject-mean baseline
2. Raw + centered multi-task loss
3. Residual prediction head
4. Modality-specific centered ablation
5. Subject-adversarial / subject-invariant training
6. Audio quality gate 和 missing-modality robustness
7. 多 seed 稳定性验证

## 实验 0：严格 within-subject held-out-day split

### 原因

现有 `within_subject_day` 的 train/val/test window index 互不重叠，但 150 个 subject-day pair 在三集合中全部重叠。该协议可以检查同一 subject-day 内不同窗口的预测稳定性，不能支撑“训练日期和测试日期隔离”的严格结论。后续若要写“同一被试内跨日期泛化”，需要先修复划分单位。

### 预期结果

新的 strict split 应满足：

- 同一 subject 的 train/val/test 均有样本。
- 任意 subject-day pair 只出现在一个 split 中。
- `subject_day_overlap_train_val = 0`
- `subject_day_overlap_train_test = 0`
- `subject_day_overlap_val_test = 0`
- window index 继续两两不重叠。

### 操作指南

新增脚本：

```text
scripts/33_build_strict_within_subject_day_split.py
scripts/30_audit_eegpt_splits.py
```

构造逻辑：

```text
for each subject:
    group rows by subject_id + day_id
    sort subject-day groups by day/order
    assign early groups to train, middle groups to val, late groups to test
    keep all windows from the same subject-day group in one split
```

推荐比例：

```text
train 60%
val 20%
test 20%
```

输出目录建议：

```text
/vePFS-0x0d/DailyEEG/splits_new/within_subject_day_strict/
  pretrain.json
  finetune.json
  val.json
  test.json
  split_info.json
```

验证后再跑当前主候选：

```text
A1_Wphysio_no_audio
B0_Wdeep_no_audio
A2_Wdeep_full
```

判定标准：

- 如果 strict split 下 raw r 和 centered r 仍为正，才能支撑同一被试内跨日期泛化。
- 如果 strict split 下指标显著下降，论文口径应保留 `cross_day` 的跨日期结论，把旧 `within_subject_day` 作为窗口级诊断附录或补充实验。

## 实验 1：Split Audit + Subject-Mean Baseline

### 原因

当前 raw r 和 centered r 差距明显。第一步需要确认 raw r 能否被一个极简单的 subject mean baseline 解释。如果只用训练集中每个被试的平均 fatigue 就能得到接近模型的 raw r，而 centered r 接近 0，则说明当前性能主要来自被试稳定均值。

### 预期结果

合理预期：

- subject-mean baseline 的 raw r 在 `cross_day` 和 `within_subject_day` 上不会太低。
- subject-mean baseline 的 centered r 应接近 0。
- 若模型 raw r 只比 subject-mean baseline 略高，下一轮主攻 residual/centered objective。
- 若模型 centered r 明显高于 baseline，说明当前 embedding 已含有一部分个体内动态信号。

### 操作指南

新增脚本：

```text
scripts/30_audit_eegpt_splits.py
scripts/31_run_subject_mean_baseline.py
src/daily_multimodal/training/centered_metrics.py
tests/test_centered_metrics.py
```

`centered_metrics.py` 至少提供：

```text
safe_pearsonr(y_true, y_pred) -> float | None
within_subject_centered_arrays(y_true, y_pred, subject_ids) -> tuple
evaluate_regression_with_centered(y_true, y_pred, subject_ids) -> dict
predict_subject_train_mean(train_y, train_subjects, test_subjects) -> ndarray
```

`30_audit_eegpt_splits.py` 输出字段：

```text
protocol
split
window_count
event_count
subject_count
day_count
subject_ids
day_ids
subject_overlap_train_val
subject_overlap_train_test
day_overlap_train_val
day_overlap_train_test
subject_day_overlap_train_test
```

`31_run_subject_mean_baseline.py` 逻辑：

```text
for protocol in ["cross_day", "within_subject_day"]:
    train = pretrain + finetune
    for each subject:
        subject_mean = mean(train fatigue for this subject)
    test prediction:
        y_hat_i = subject_mean[s_i]
        fallback = train global mean if subject unseen
    compute rmse, mae, raw_r, centered_r, per_subject_r_mean
```

建议远端运行：

```bash
python scripts/30_audit_eegpt_splits.py \
  --window-index /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/data/canonical_window_index.jsonl \
  --splits-root /vePFS-0x0d/DailyEEG/splits_new \
  --out-json reports/eegpt_split_audit_subject_day.json \
  --out-md reports/eegpt_split_audit_subject_day.md

python scripts/31_run_subject_mean_baseline.py \
  --window-index /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/data/canonical_window_index.jsonl \
  --splits-root /vePFS-0x0d/DailyEEG/splits_new \
  --target fatigue \
  --protocols cross_day within_subject_day \
  --out-json reports/eegpt_subject_mean_baseline.json \
  --out-md reports/eegpt_subject_mean_baseline.md
```

验收标准：

- audit 报告明确每个协议 train/val/test 的被试数、日期数、event 数、window 数。
- subject-day pair 在 train/test 是否重叠必须明确。
- subject-mean baseline 和当前 cross-attention 模型在 raw r、centered r 上并列表达。

## 实验 2：Raw + Centered Multi-Task Loss

### 原因

当前模型优化 raw target MSE，训练信号自然鼓励预测绝对 fatigue 水平。加入 centered loss 可以额外奖励模型学同一被试内部的相对变化。这个实验只改训练目标，不改主要模型结构，是最轻量的 centered r 改进尝试。

### 和 Residual Head 的区别

Raw + centered multi-task loss 保持单一输出：

```text
ŷ = f(x)
```

Residual prediction head 改模型结构：

```text
ŷ = b_subject + δ_day
```

前者用 loss 约束同一个预测值同时服务 raw 和 centered 目标；后者把“被试稳定水平”和“当天相对偏移”显式拆成两个组成部分。

### 推荐公式

每个 batch 内按 subject 去均值，仅对 batch 内样本数不少于 2 的 subject 计算 centered loss：

```text
raw_loss = MSE(ŷ, y)

ŷ_i^c = ŷ_i - mean(ŷ_j | s_j = s_i, j in batch)
y_i^c = y_i - mean(y_j | s_j = s_i, j in batch)

centered_loss = MSE(ŷ^c, y^c)
loss = raw_loss + λ * centered_loss
```

可选第二版本直接优化相关性：

```text
centered_corr_loss = 1 - corr(ŷ^c, y^c)
loss = raw_loss + λ * centered_corr_loss
```

### 预期结果

理想结果：

- `cross_day` centered r 从 `0.1107` 继续上升，同时 RMSE 保持在 `A1_Wphysio_no_audio` 当前 `0.9298` 附近。
- `within_subject_day` centered r 超过当前最优 `0.1026`。
- 若 centered r 上升伴随 raw r 小幅下降，可以把它解释为总体排序和个体内波动之间的合理取舍。
- 若 centered r 不上升，说明现有 embedding 或 batch 组织方式对个体内动态信号不足，需要进入 residual head 或 modality-specific 分析。

### 操作指南

新增训练参数：

```text
--loss-mode raw
--loss-mode raw_centered_mse
--loss-mode raw_centered_corr
--centered-lambda 0.1
--centered-lambda 0.3
--centered-lambda 0.5
--centered-lambda 1.0
--subject-balanced-batches
```

实现要点：

1. 在 dataset 中保证每个 sample 返回 `subject_id`。
2. batch sampler 尽量让每个 batch 包含每个 subject 的多个窗口；如果无法做到，至少跳过 batch 内单样本 subject 的 centered loss。
3. centered loss 只用于 `cross_day` 和 `within_subject_day`；`cross_subject` 保持 raw loss 对照。
4. metrics JSON 中新增：

```text
loss_mode
centered_lambda
batch_centered_subject_count_mean
train_raw_loss
train_centered_loss
test.raw_r
test.within_subject_centered_r
test.per_subject_r_mean
```

最小运行矩阵：

| 协议 | 基础实验 | λ |
| --- | --- | --- |
| `cross_day` | `A1_Wphysio_no_audio` | 0.1, 0.3, 0.5, 1.0 |
| `within_subject_day` | `A1_Wphysio_no_audio` | 0.1, 0.3, 0.5, 1.0 |
| `within_subject_day` | `B0_Wdeep_no_audio` | 0.1, 0.3, 0.5 |

判定标准：

- centered r 绝对提升至少 `+0.02`，且 RMSE 恶化不超过 `+0.015`，进入多 seed。
- centered r 提升但 RMSE 恶化超过 `+0.03`，作为诊断结果保留，不进入主模型。
- centered r 无提升时，不继续扩大 λ sweep，转向 residual head。

## 实验 3：Residual Prediction Head

### 原因

当前科学问题可以自然拆成两部分：

```text
fatigue = subject-level baseline + within-subject residual
```

Residual head 把这个假设写进结构中，更适合解释 raw r 和 centered r 的差距，也能单独报告 residual 分支是否真正学到了同一被试内部波动。

### 推荐结构

优先实现固定 subject baseline 版本，避免 subject embedding 直接记忆测试身份：

```text
b_s = mean(train fatigue | subject=s)
δ_hat = residual_head(fused_multimodal_embedding)
ŷ = b_s + δ_hat
```

训练目标：

```text
raw_loss = MSE(ŷ, y)
residual_target = y - b_s
residual_loss = MSE(δ_hat, residual_target)
loss = raw_loss + α * residual_loss
```

推荐 sweep：

```text
α = 0.3, 0.5, 1.0
```

### 预期结果

理想结果：

- raw r 至少接近当前 `A1_Wphysio_no_audio`。
- centered r 明显超过 subject-mean baseline。
- residual-only prediction `δ_hat` 与 `y - b_s` 的相关性为正，并且跨 seed 稳定。

可能结果和解释：

- `b_s` 分支贡献大、`δ_hat` 接近 0：当前模型主要靠被试均值。
- `δ_hat` 在 Wdeep 或 EEG-rich 设置中更强：个体内动态信号可能来自序列生理或 EEG。
- raw r 下降但 centered r 上升：模型从总体排序转向个体内波动，适合作为机制实验，不一定作为主结果。

### 操作指南

新增模型模式：

```text
--model-mode standard_cross_attention
--model-mode residual_head_fixed_subject_mean
```

训练前计算：

```text
subject_train_mean[subject] = mean(y_train for subject)
global_train_mean = mean(y_train)
```

数据载入时为每个样本附加：

```text
subject_baseline = subject_train_mean.get(subject_id, global_train_mean)
residual_target = y - subject_baseline
```

metrics JSON 中新增：

```text
model_mode
residual_alpha
subject_mean_baseline_metrics
residual_only_metrics
combined_prediction_metrics
subject_baseline_coverage
```

最小运行矩阵：

| 协议 | 实验 | α |
| --- | --- | --- |
| `cross_day` | `A1_Wphysio_no_audio` | 0.3, 0.5, 1.0 |
| `within_subject_day` | `A1_Wphysio_no_audio` | 0.3, 0.5, 1.0 |
| `within_subject_day` | `B0_Wdeep_no_audio` | 0.3, 0.5, 1.0 |
| `within_subject_day` | `A2_Wdeep_no_audio` | 0.5 |

判定标准：

- `combined_prediction_metrics.within_subject_centered_r` 超过当前最佳至少 `+0.02`。
- `residual_only_metrics.raw_r` 可以较低，但 `residual_only_metrics.within_subject_centered_r` 应为正且稳定。
- 若 `subject_mean_baseline_metrics.raw_r` 已接近 combined prediction，而 residual-only 无贡献，应将后续重心转向更强模态表征或更细时间标签。

## 实验 4：Modality-Specific Centered Ablation

### 原因

当前主候选是 `A1_Wphysio_no_audio`，但 `within_subject_day` 中 centered r 最优来自 `B0_Wdeep_no_audio`。这提示低误差和个体内波动可能由不同模态路线支撑。需要判断 EEG、Wphysio、Wdeep、视频、音频分别贡献 raw r 还是 centered r。

### 预期结果

可能出现的可解释模式：

- EEG-only centered r 最稳：短时脑电状态对疲劳波动最敏感。
- Wdeep centered r 高于 Wphysio：序列型 wearable 表征更能捕捉个体内动态。
- 视频提升 raw r 但不提升 centered r：视频主要增强总体状态或被试相关特征。
- 音频持续拖累 centered r：优先做音频质量门控。

### 操作指南

在当前 route-aware 矩阵基础上新增单模态和双模态运行：

```text
eeg_only
wear_only_physio
wear_only_deep
video_only_A1
video_only_A2
eeg_wear_physio
eeg_wear_deep
eeg_video_A1
eeg_video_A2
wear_video_A1
wear_video_A2
```

重点协议：

```text
cross_day
within_subject_day
```

每个结果必须报告：

```text
rmse
mae
raw_r
within_subject_centered_r
per_subject_r_mean
mask_coverage_by_modality
```

判定标准：

- 找到 centered r 最高的单模态或双模态组合。
- 若某个模态 centered r 稳定高于 fusion，下一轮 fusion 需要给该模态更高权重或单独 residual branch。
- 若所有单模态 centered r 都接近 0，模型结构改造空间有限，应回到标签质量、窗口尺度或日级状态定义。

## 实验 5：Subject-Adversarial / Subject-Invariant Training

### 原因

raw r 明显高于 centered r，说明 embedding 中可能包含较强 subject identity 信息。Subject-adversarial training 通过降低 fused embedding 对 subject ID 的可辨识度，检查 subject identity 是否压制了 fatigue dynamic signal。

### 推荐结构

```text
h = fusion_encoder(tokens)
fatigue_hat = fatigue_head(h)
subject_logits = subject_classifier(GradientReverse(h))

loss = fatigue_loss + β * subject_adv_loss
```

其中 `GradientReverse` 在反向传播中让 fusion encoder 难以编码 subject identity。

推荐 sweep：

```text
β = 0.01, 0.03, 0.1
```

### 预期结果

理想结果：

- subject probe accuracy 下降。
- centered r 上升。
- raw r 小幅下降可接受。

若 subject accuracy 下降但 centered r 不升，说明 subject identity 并非主要瓶颈，或动态 fatigue 信息未被当前 embedding 捕捉。

### 操作指南

新增输出：

```text
subject_classifier_train_acc
subject_classifier_val_acc
posthoc_subject_probe_acc
fatigue_rmse
fatigue_raw_r
fatigue_centered_r
```

最小运行矩阵：

| 协议 | 实验 | β |
| --- | --- | --- |
| `cross_day` | `A1_Wphysio_no_audio` | 0.01, 0.03, 0.1 |
| `within_subject_day` | `A1_Wphysio_no_audio` | 0.01, 0.03, 0.1 |
| `within_subject_day` | `B0_Wdeep_no_audio` | 0.03 |

判定标准：

- centered r 提升且 subject probe accuracy 下降，说明 subject-invariant 表征有效。
- centered r 不变或下降时，停止 adversarial sweep，保留为诊断。

## 实验 6：Audio Quality Gate 和 Missing-Modality Robustness

### 原因

当前 full 相对 no-audio 的平均 ΔRMSE 为 `+0.0151`，平均 Δraw r 为 `-0.0271`。音频分支目前拖累主候选，但这可能来自音频质量、mask 缺失、openSMILE 特征和 fusion 权重处理，而非音频模态没有价值。

### 预期结果

合理预期：

- 加质量门控后，full 的 RMSE 与 no-audio 差距缩小。
- 高质量音频子集可能提升 raw r 或 centered r。
- 如果质量门控无效，后续报告可更有力地采用 no-audio 主线。

### 操作指南

新增设置：

```text
--audio-mode disabled
--audio-mode opensmile_all
--audio-mode opensmile_quality_gated
--audio-min-quality A
--audio-min-quality AB
--modality-dropout 0.1
--modality-dropout 0.3
```

评估时分桶：

```text
audio_present
audio_missing
audio_quality_A
audio_quality_B
audio_quality_low
```

判定标准：

- quality-gated full 比 no-audio 的 RMSE 恶化小于 `+0.005`，且 raw r 或 centered r 有提升，保留音频门控路线。
- full 仍稳定低于 no-audio，主报告继续使用 no-audio，音频作为独立待优化分支。

## 实验 7：多 Seed 稳定性验证

### 原因

当前 48 个结果主要用于 route selection。下一轮若要支撑论文结论或模型选择，需要验证 best candidates 在随机种子下稳定。

### 推荐 seed

```text
240729
240730
240731
240801
240805
```

### 最小矩阵

| 协议 | 实验 | 模式 |
| --- | --- | --- |
| `cross_day` | `A1_Wphysio_no_audio` | baseline, best multi-task, best residual |
| `within_subject_day` | `A1_Wphysio_no_audio` | baseline, best multi-task, best residual |
| `within_subject_day` | `B0_Wdeep_no_audio` | baseline, best residual |
| `within_subject_day` | `A2_Wdeep_full` | baseline |

### 汇总方式

每个组合报告：

```text
rmse mean ± std
raw_r mean ± std
centered_r mean ± std
per_subject_r_mean mean ± std
best_seed
```

进入主结论的最低要求：

- centered r 的提升方向在多数 seed 一致。
- RMSE 均值恶化在可解释范围内。
- 最优 seed 不能作为唯一证据。

## 推荐执行顺序

第一轮，诊断：

```text
split audit
subject-mean baseline
current best candidate reproduction
```

第二轮，轻量改造：

```text
raw_centered_mse λ sweep
raw_centered_corr λ sweep
```

第三轮，结构改造：

```text
residual_head_fixed_subject_mean α sweep
```

第四轮，归因：

```text
modality-specific centered ablation
subject-adversarial diagnostic
audio quality gate
```

第五轮，稳定性：

```text
multi-seed rerun for accepted candidates
summary report and HTML update
```

## 建议目录结构

远端建议输出：

```text
reports/eegpt_centered_improvement/
  split_audit_subject_day.json
  split_audit_subject_day.md
  subject_mean_baseline.json
  subject_mean_baseline.md
  multitask_loss_matrix.json
  residual_head_matrix.json
  modality_centered_ablation.json
  subject_adversarial_matrix.json
  audio_quality_gate_matrix.json
  multiseed_summary.json
  multiseed_summary.md
  centered_improvement_final_report.md
```

本地同步建议输出：

```text
outputs/server_sync/eegpt_centered_improvement/
docs/superpowers/reports/2026-08-xx-eegpt-centered-r-improvement-results.md
docs/superpowers/reports/html/2026-08-xx-eegpt-centered-r-improvement/
```

## 最终报告应回答的问题

下一轮结束后，报告至少回答：

1. `cross_day` 和 `within_subject_day` 的 raw r 有多少能由 subject mean baseline 解释？
2. centered loss 是否能提升个体内波动追踪？
3. residual head 的 `δ_day` 分支是否学到了超过 subject mean 的信息？
4. 哪个模态或模态组合最能解释 centered r？
5. 降低 subject identity 信息是否提升 centered r？
6. 音频在质量门控后是否仍拖累主候选？
7. 改进是否跨 seed 稳定？

## 推荐论文/汇报口径

当前主线可以写成：

```text
The current EEG-aligned multimodal model captures protocol-dependent global fatigue ranking, especially in cross-day and within-subject-day settings. However, within-subject centered evaluation shows that individual fatigue fluctuations remain substantially harder than global ordering. We therefore introduce subject-mean diagnostics and residual/centered objectives to separate stable subject-level fatigue differences from within-subject dynamic changes.
```

中文汇报口径：

```text
当前模型已经能在跨天和被试内跨天协议中捕捉一定总体疲劳排序，A1_Wphysio_no_audio 是低误差主候选；但 centered r 明显低于 raw r，说明同一被试内部跨日期疲劳波动仍是主要瓶颈。下一步将通过 subject-mean baseline、centered loss、residual head 和模态级 centered ablation，区分被试稳定差异与个体内动态信号，并寻找真正能提升 centered r 的模态和结构。
```

## 接手检查清单

- [ ] 先运行 split audit，确认 `cross_day` 和 `within_subject_day` 的 subject/day 交叠口径。
- [ ] 先实现 centered metrics 单元测试，再接入训练脚本。
- [ ] 先跑 subject-mean baseline，再改模型。
- [ ] 所有新模型都同时报告 RMSE、MAE、raw r、centered r、per-subject r mean/std。
- [ ] 对 `cross_day` 和 `within_subject_day` 分别解释 raw r 与 centered r；不要把二者混成一个“模型相关性”。
- [ ] 改进候选进入多 seed 前，先用单 seed λ/α sweep 确认方向。
- [ ] 最终同步远端 JSON/Markdown 到本地，并更新 Markdown 报告和 HTML/PPT 汇报。
