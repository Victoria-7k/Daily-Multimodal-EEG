# 同被试融合结果指标选择报告

完整指标文件：

- `outputs/server_sync/within_subject_fusion_120s10s/within_subject_fusion_complete_metrics_v2_long.csv`
- `outputs/server_sync/within_subject_fusion_120s10s/within_subject_fusion_attention_top_metrics_v2.csv`
- `outputs/server_sync/within_subject_fusion_120s10s/within_subject_fusion_complete_metrics_v2.json`

## 指标含义

| 指标 | 含义 | 回答的问题 |
| --- | --- | --- |
| `pooled_raw_r` | 把所有 OOF event prediction/target 直接 pooled 后算 Pearson r | 全体样本混在一起看，预测高低是否和 fatigue 高低相关 |
| `within_subject_centered_r` | 每个 subject 内分别对 prediction/target 去均值，再 pooled 算 r | 去掉 subject baseline 后，模型能否追踪个体内部 fatigue 升降 |
| `per_subject_r_mean/std` | 每个 subject 单独算 r，再对 subject 取均值和标准差 | 平均到每个被试表现如何，以及被试间稳定性如何 |

## Event-Grouped 最佳处理方式
event-grouped 测的是：
没见过的 event

### Cross-Attention 内最佳

| 选择指标 | Model | 处理方式 | pooled raw r | centered r | per-subject r mean/std | RMSE | MAE | centered RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pooled raw r | `learnable_cross_attention` | `WdeepPre + FullSweepB3Lam005 + full` | 0.5076 | 0.2673 | 0.1641 / 0.2882 | 0.7717 | 0.6037 | 0.7672 |
| within-subject centered r | `learnable_cross_attention` | `WdeepPre + FullSweepB3Lam005 + full` | 0.5076 | 0.2673 | 0.1641 / 0.2882 | 0.7717 | 0.6037 | 0.7672 |
| per-subject r mean | `learnable_cross_attention` | `WdeepPre + FullSweepB3Lam005 + full` | 0.5076 | 0.2673 | 0.1641 / 0.2882 | 0.7717 | 0.6037 | 0.7672 |

### 推荐

目前 event-grouped 推荐：

`learnable_cross_attention + WdeepPre + FullSweepB3Lam005 + full`

理由：

- 它同时是 pooled raw r 和 within-subject centered r 的全模型第一。
- 在 cross-attention 范围内，它也是 per-subject r mean 第一。
- Ridge 的 `WphysioPre + B5A1Lam0001 + no_audio` 在 per-subject r mean 上略高，
  但 RMSE/centered RMSE 明显更差，且 Ridge 在其他 full-modality 设置里存在严重
  held-out feature shift 风险，因此更适合作为 diagnostic baseline。

## Session-Held-Out 最佳处理方式
session-held-out 测的是：
没见过的 session
所以 session-held-out 更难，也更能检验跨 session 泛化
session: 连续录制的一段时间，对于当前数据就是日期

### Cross-Attention 内最佳

| 选择指标 | Model | 处理方式 | pooled raw r | centered r | per-subject r mean/std | RMSE | MAE | centered RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pooled raw r | `learnable_cross_attention` | `WphysioPre + FullSweepB3Lam005 + full` | 0.4077 | 0.0625 | -0.0064 / 0.3306 | 0.8441 | 0.6617 | 0.8357 |
| within-subject centered r | `learnable_cross_attention` | `WphysioPre + FullSweepB3Lam005 + full` | 0.4077 | 0.0625 | -0.0064 / 0.3306 | 0.8441 | 0.6617 | 0.8357 |
| per-subject r mean | `learnable_cross_attention` | `WphysioPre + FullSweepB3Lam005 + full` | 0.4077 | 0.0625 | -0.0064 / 0.3306 | 0.8441 | 0.6617 | 0.8357 |

### 推荐

目前 session-held-out 推荐：

`learnable_cross_attention + WphysioPre + FullSweepB3Lam005 + full`

理由：

- 在 cross-attention 范围内，它同时是 pooled raw r、centered r、per-subject r mean
  第一。

- 即使推荐配置也是弱结果：per-subject r mean 为 `-0.0064`，
  说明跨 session 的个体内 fatigue tracking 目前基本没有稳定建立。

## 处理方式解释

| 缩写 | 含义 |
| --- | --- |
| `WdeepPre` | wear 使用 `wear_deep_sequence_preprocessed_v1` |
| `WphysioPre` | wear 使用 `wear_physio_features_preprocessed_v1` |
| `FullSweepB3Lam005` | video 使用 full-sweep B3，lambda `0.05` |
| `B5A1Lam0001` | video 使用 B5/A1，lambda `0.001` |
| `full` | EEG + wear + video + audio 四路输入 |
| `no_audio` | EEG + wear + video，去掉 audio |

## 当前训练是不是所有样本混在一起

当前训练是 per-subject within-subject 训练，不是跨被试混合训练。

实际执行方式：

1. runner 先遍历 experiment。
2. 对每个 experiment，再逐个 subject 建立 job。
3. 每个 subject/job 内按该 subject 的 folds 训练和测试。
4. 每个 fold 的训练只使用该 subject 的 `fold.train`，预测该 subject 的 `fold.test`。
5. 最后的总体指标把所有 subject 的 OOF predictions 合并后再计算。

也就是说：

- subject 之间没有混在一起训练同一个模型。
- 同一个 subject 的 train fold 内，所有训练 windows/events 会作为训练样本混合 batch 化。
- event-grouped 中 train/test 按 event 分开。
- session-held-out 中 train/test 按 session 分开。

对应代码位置：

- `src/daily_multimodal/training/within_subject_runner.py:376` 开始逐 subject 建 job。
- `src/daily_multimodal/training/within_subject_runner.py:194` 开始在该 subject 的 folds 内训练。
- `src/daily_multimodal/training/within_subject_runner.py:222` 到 `245` 使用 `fold.train`
  训练并对 `fold.test` 预测。
- `src/daily_multimodal/training/within_subject_runner.py:508` 到 `518` 检查 fold 不跨
  subject boundary。

## 总结

目前最稳的结论是：

- event-grouped：`WdeepPre + FullSweepB3Lam005 + full` 最值得作为当前主结果。
- session-held-out：`WphysioPre + FullSweepB3Lam005 + full` 是当前最好的
  cross-attention 结果，但跨 session 泛化仍然很弱。
- B3 视频路线目前最一致。
- audio 没有形成稳定必要性，后续仍应保留 no-audio 对照。
- 正式定稿前，应修正 runner 的 event aggregation key，让原生 summary 直接输出
  `614` event-level rows。
