# Daily-affect 标签置换相关性真实性检查计划

> 版本：2026-09-07  
> 适用路线：`technical_route_20260906.md` 中的 Daily-affect EMA-bag ordinal 路线  
> 核心目标：检验 held-out test 上的序数预测和相关性是否依赖 EMA event 输入与真实疲劳标签的正确对应关系

## 1. 实验目标与结论边界

本实验建立 shuffled-label null distribution，对当前 Daily-affect 结果进行负对照检验。

实验分别回答三个层次的问题：

| 结论层次 | 指标 | 回答的问题 |
| --- | --- | --- |
| 序数预测有效 | QWK | 模型能否正确区分 1--5 级疲劳 |
| 总体相关性有效 | Expected raw Pearson r | 模型预测是否与总体疲劳等级相关 |
| 个体内状态相关性有效 | Within-subject centered r | 去除受试者平均差异后，模型能否跟踪 event-level 波动 |

标签置换结果支持 held-out 数据上的统计关联证据。生理因果关系需要独立的因果设计和外部验证。

## 2. 锁定实验条件

固定当前 cross-day 优先候选配置：

| 配置项 | 固定值 |
| --- | --- |
| Protocol | `cross_day` |
| Route | `A1_Wphysio_full` |
| Model | `dynamic_kernel_prior_uniform` |
| Normalization | `per_modality` |
| Adapter | `per_modality` |
| Objective | weighted CE + `0.5` ordinal + `0.1` rank |
| Model seeds | `240729,240730,240731` |
| Checkpoint selection | validation QWK |
| Final evaluation | untouched real-label test split |

当前七 seed 结果作为背景锚点：QWK `0.2515 +/- 0.0402`，Expected raw r `0.2833 +/- 0.0280`。正式 permutation comparison 使用相同三个 model seed 的 clean 结果，保持 seed、输入、训练预算和 test event 集合匹配。

实验开始后不再调整模型结构、损失权重、学习率、训练轮数、early-stopping patience 或评价指标。

## 3. 标签置换协议

### 3.1 置换单位

标签在完整 EMA event 层置换：

```text
tokens[event, 23, 4, 256] -> one permuted event label
```

每条 event 的 23 个窗口保持完整。窗口顺序、四模态 token、`modality_mask`、event 顺序和 split 均保持不变。

### 3.2 Split 边界

分别在 train 和 validation 内产生置换：

```text
train labels -> permute inside train only
val labels   -> permute inside val only
test labels  -> keep untouched
```

当前训练使用 validation QWK 选择 checkpoint。Null run 的 validation supervision 必须使用置换标签，使真实 validation 标签不进入 checkpoint selection。

以下训练分量统一读取置换后的监督标签：

- Class weight；
- Weighted cross-entropy；
- Cumulative ordinal loss；
- Within-subject ranking loss；
- Modality probe loss；
- Expected-score auxiliary loss（如当前锁定配置启用）；
- Validation metric 和 checkpoint selection。

最终 test QWK、Macro-F1、ordinal MAE、Expected RMSE、raw r 和 centered r 统一相对于原始真实 test 标签计算。

### 3.3 `global_shuffle`

在 train 和 validation split 内分别进行全局标签排列，严格保留每个 split 的类别计数。

该对照用于快速检查训练、checkpoint selection 和 test evaluation 流水线是否存在明显泄漏。

### 3.4 `within_subject_shuffle`

在每个 split 的每个 subject 内独立排列标签。该方法保留：

- 每名受试者的标签直方图；
- 受试者平均疲劳水平；
- split 的总体类别分布。

它破坏具体 EMA event 与标签的对应关系，主要检验模型是否具有超出 subject identity 和 subject mean 的 event-level 状态信息。

### 3.5 置换审计

每个 permutation 必须保存：

- `permutation_id`；
- permutation seed；
- mode：`global_shuffle` 或 `within_subject_shuffle`；
- train/validation 原始与置换后类别直方图；
- permutation index moved fraction；
- label value changed fraction；
- singleton subject 数量；
- 有效参与置换的 subject 和 event 数量；
- test labels unchanged 检查；
- split membership unchanged 检查；
- 每条 event 的 23-window membership unchanged 检查。

最低有效性门槛：

- `global_shuffle` 的 label value changed fraction 不低于 `0.50`；
- `within_subject_shuffle` 的 label value changed fraction 不低于 `0.30`。

低于门槛时，该 permutation 不进入 null distribution，并停止正式扩展以检查标签重复、组大小和置换实现。

## 4. 上游监督边界

运行前读取每个 bag 的 `supervision_boundary`：

### 4.1 `label_free_or_fixed`

该条件下，实验可以解释为当前 Daily-affect 训练与预测流水线的标签置换检查。

### 4.2 `fatigue_supervised_control`

该条件表示输入 token 已经通过真实 fatigue 标签训练。下游标签置换只能检验 Daily-affect head 对标签的依赖性。

此时增加以下至少一种检查：

1. 在匹配的 frozen/label-free token route 上重复 permutation test；
2. 使用相同 permutation mapping 重新训练上游监督 encoder，再生成 token 和训练 Daily-affect head。

报告必须明确区分 downstream-head null 与 full-pipeline null。

## 5. 分阶段服务器实验

### Phase 0：远端预检

在服务器上确认：

1. GPU、Python runtime 和 `PYTHONPATH` 可用；
2. 三个 seed 的 `ema_bags.npz` 完整；
3. train、validation、test event 集合符合预定义 split；
4. 各 split 的 event 和 subject-day 交叉情况符合 protocol；
5. `supervision_boundary` 与 route/token 配置一致；
6. 真实 clean metrics、predictions 和 checkpoint 可读取；
7. 输出使用独立目录，不覆盖现有正式结果。

任何数据契约或 split 审计失败时停止训练。

### Phase 1：Global shuffled-label smoke

运行规模：

```text
5 permutation IDs x 1 model seed = 5 runs
```

建议使用 model seed `240729`。五个 permutation 使用互不相同、可复现的 permutation seed。

继续门槛：

- `5/5` permutation audit 通过；
- 至少 `4/5` shuffled QWK 低于 matched clean QWK；
- 至少 `4/5` shuffled raw r 低于 matched clean raw r；
- shuffled centered r 未形成稳定正向结果；
- test 标签、test event 和test predictions的评价索引完全匹配。

任一门槛未通过时立即停止扩展，并按以下顺序检查：

1. Validation 是否仍读取真实标签；
2. Test event、test label 或 test normalization 是否进入训练；
3. 上游 token 是否带真实 fatigue 监督；
4. Subject/day identity 是否形成捷径；
5. 置换标签是否进入所有 loss；
6. 保存的 metrics 是否错误读取 clean prediction。

### Phase 2：Within-subject permutation test

Global smoke 通过后运行：

```text
30 permutation IDs x 3 matched model seeds = 90 runs
```

同一个 `permutation_id` 在三个 model seed 中使用完全相同的标签映射，使标签随机化和模型初始化随机性可以独立审计。

每个 permutation 先对三个 seed 的指标等权平均，产生一个 null statistic。最终得到 30 个独立 permutation-level null statistics。

经验显著性：

```text
p = (1 + count(null statistic >= real statistic)) / (B + 1)
```

其中 `B=30`。QWK、Macro-F1、raw r 和 centered r 使用上尾；RMSE、ordinal MAE 和 severe-error rate 使用下尾。30 次置换可达到的最小经验 p 值为：

```text
1 / 31 = 0.0323
```

若需要论文级 `p < 0.01` 分辨率，将 permutation IDs 扩展到 `99`，并保持相同三个 model seeds。

### Phase 3：时间结构敏感性检查

当 within-subject permutation 显著、且原始 EMA 标签表现出明显时间自相关时，增加非零循环平移对照：

```text
within-subject non-zero circular shift
```

每名受试者的 event 按真实时间排序，对标签序列执行非零循环平移。该对照保留标签边际分布和时间平滑结构，同时破坏当前 event 与输入的精确对齐。

循环平移属于补充稳健性证据，不替代 global 和 within-subject permutation。

## 6. 统计分析

### 6.1 Confirmatory metrics

按以下顺序解释：

1. QWK：Daily-affect 主指标；
2. Expected raw Pearson r：总体连续关联；
3. Within-subject centered r：个体内 event-level 关联；
4. Ordinal MAE 和 Expected RMSE：误差方向验证；
5. Macro-F1：类别均衡表现；
6. Prediction standard deviation：检查常数预测和动态范围塌缩。

当预测标准差低于预设数值容差、Pearson r 不可定义时，报告为 `NA_constant_prediction`。经验超越计数中将其视为没有正向相关信息，同时单独报告此类 run 数量。

### 6.2 Subject-day paired bootstrap

所有 test predictions 使用相同 event 顺序后，在每个 model seed 内按 subject-day block 重采样，计算 clean 与 shuffled prediction 的配对指标差。

报告内容：

- 每个 seed 的 delta；
- 每个 seed 的 bootstrap 95% CI；
- 三 seed delta 的 mean +/- std；
- 三 seed 的方向一致性；
- 各 permutation 中 clean 优于 shuffled 的比例。

Seed-averaged prediction 不进入一次性 pooled bootstrap。

## 7. 判定门槛

### 7.1 序数预测结论

当以下条件全部满足时，支持“序数预测显著超出标签置换零分布”：

- `p_QWK <= 0.05`；
- clean QWK 高于 null QWK 第 95 百分位；
- 至少 `2/3` model seeds 的 clean-minus-null delta 为正；
- ordinal MAE 方向与 QWK 结论一致。

### 7.2 总体相关性结论

当以下条件全部满足时，支持“总体预测与疲劳等级存在可泛化关联”：

- `p_raw_r <= 0.05`；
- clean raw r 高于 null raw r 第 95 百分位；
- 至少 `2/3` model seeds 方向一致；
- Expected RMSE 未出现与相关性结论冲突的系统性恶化。

### 7.3 个体内状态结论

当以下条件全部满足时，支持“模型能够跟踪个体内 event-level 疲劳波动”：

- `p_centered_r <= 0.05`；
- clean centered r 高于 within-subject null 第 95 百分位；
- 至少 `2/3` model seeds 方向一致；
- Seed-level subject-day bootstrap 提供一致的正向差异。

## 8. 结果解释矩阵

| 观察结果 | 允许形成的结论 |
| --- | --- |
| QWK、raw r、centered r 均通过 | 支持 event-level 多模态状态与 EMA 疲劳标签存在可泛化关联 |
| QWK 和 raw r 通过，centered r 未通过 | 关联主要由受试者或日期间差异贡献；个体内状态跟踪证据不足 |
| Global null 通过，within-subject null 未通过 | 模型利用了群体结构，event-level 对齐贡献有限 |
| Global shuffle 后仍有稳定高 QWK 或高 r | 触发泄漏、监督边界或评估实现审计，停止正式扩展 |
| 所有 shuffled 指标接近机会水平且真实值位于 null 极端尾部 | 支持真实结果依赖正确的输入--标签配对 |
| Frozen route 通过而 fatigue-supervised route 结果复杂 | 分别报告 label-free 与 supervised representation 的监督边界 |

## 9. 工程实现

建议新增入口：

```text
scripts/daily_affect/89_run_daily_affect_label_permutation.py
```

实现原则：

1. 原始 `ema_bags.npz` 保持只读；
2. 以内存 dataset 副本替换 train/validation 的 `label` 和 `label_zero_based`；
3. Test 位置的两个标签数组保持逐元素一致；
4. Null run 写入独立目录；
5. `config.json` 和 `metrics.json` 显式写入 permutation metadata；
6. 保存 test event-level probabilities、predicted class 和 expected score；
7. 所有已存在输出默认跳过，除非显式传入 overwrite 参数；
8. 服务器日志逐 run 刷新，便于中断后恢复。

建议增加测试：

- Global shuffle 保留 split 类别直方图；
- Within-subject shuffle 保留每 subject 类别直方图；
- Test 标签逐元素不变；
- 23-window event membership 不变；
- 相同 permutation seed 产生相同映射；
- 不同 model seed 复用相同 permutation mapping；
- Training loss 和 validation selection 使用置换标签；
- Final test metrics 使用真实标签；
- Singleton subject 和常数标签 subject 得到明确审计记录。

## 10. 输出目录与交付物

建议独立结果根：

```text
outputs/daily_affect_label_permutation_20260907/
|-- manifest.json
|-- permutation_audit.csv
|-- run_metrics.csv
|-- seed_level_deltas.csv
|-- null_distribution.csv
|-- empirical_p_values.json
|-- label_permutation_report.md
|-- logs/
`-- figures/
    |-- qwk_null_distribution.png
    |-- raw_r_null_distribution.png
    |-- centered_r_null_distribution.png
    `-- real_vs_null_seed_deltas.png
```

最终报告必须列出：

- 实际完成的 permutation 和 run 数量；
- 每种置换的有效标签变化比例；
- Clean 指标在 null distribution 中的百分位；
- QWK、raw r、centered r 的经验 p 值；
- 每 seed subject-day bootstrap delta 和 CI；
- Seed-level mean +/- std 与方向一致性；
- 上游 token 的 `supervision_boundary`；
- 最终状态：`pass`、`claim_limited` 或 `stop_and_audit`；
- 未完成的阶段和停止原因。

## 11. 执行顺序与停止规则

```text
Phase 0 remote preflight
  -> failure: stop and repair contract
  -> pass
Phase 1 global shuffle, 5 x 1
  -> gate failure: stop and audit leakage/supervision
  -> gate pass
Phase 2 within-subject shuffle, 30 x 3
  -> evaluate QWK/raw r/centered r separately
  -> optional Phase 3 temporal circular-shift control
  -> write final report and figures
```

实验遵循“直到有门槛未过再停下”：每一阶段只在上一阶段的数据、置换和指标门槛全部通过后扩展。
