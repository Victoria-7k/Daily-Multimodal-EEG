# Fusion Variant 消融结果（2026-08-20）

> 目的：回答三个问题 —— (a) 当前 modality-token attention 是否比朴素拼接更好；(b) 多头 + 多 latent query 是否有效；(c) EEG 当锚点的非对称 cross-attention 是否优于平权 self-attention。
> 实现：`scripts/32_run_eegpt_centered_loss.py` 的 `AttentionRegressor` 新增 `--fusion-variant`（`attention` / `concat` / `attention_multihead_pma` / `eeg_anchor`），所有变体 `encode()` 输出 `(B, hidden_dim)`，head/损失/评估链路不变。

## 1. 基线验证（attention 路径与 0814 逐位一致）

用归档 `eeg_encoder_256d_5route_fusion_video_only_seed240800_raw.json` 中三条代表行的**同 seed** 定点重跑（`--experiment-seed-fixed --seed <归档 seed>`、`--fusion-variant attention`）：

| protocol | experiment | seed | 复现 rmse / raw r / centered r | 归档 rmse / raw r / centered r | 一致 |
| --- | --- | ---: | --- | --- | --- |
| `cross_day` | `B0_Wphysio_full` | 240790 | 0.9481 / 0.3504 / 0.1176 | 0.9481 / 0.3504 / 0.1176 | ✅ 逐位 |
| `within_subject_day` | `B0_Wphysio_no_audio` | 240855 | 0.9243 / 0.4049 / 0.2062 | 0.9243 / 0.4049 / 0.2062 | ✅ 逐位 |
| `cross_subject` | `B0_Wphysio_no_audio` | 240735 | 0.9624 / 0.0841 / 0.0865 | 0.9624 / 0.0841 / 0.0865 | ✅ 逐位 |

结论：重构后的 `attention` 路径与 0814 基线完全等价，后续变体差异只来自融合结构。

## 2. 决策切片（32 runs，seed 240800 固定配对）

口径：`cross_day / within_subject_day` × `B0_Wphysio_full / B0_Wphysio_no_audio / A1_Wdeep_full / A1_Wdeep_no_audio` × `eeg_eegpt_partial_ft_v1`，`--experiment-seed-fixed --seed 240800`，loss=raw。

**按协议 × 变体的 test 均值（每格 4 runs）：**

| protocol | fusion variant | mean RMSE | mean MAE | mean raw r | mean centered r |
| --- | --- | ---: | ---: | ---: | ---: |
| `cross_day` | **concat** | **0.9387** | **0.7360** | **0.3112** | 0.0909 |
| `cross_day` | attention_multihead_pma | 0.9713 | 0.7605 | 0.2854 | 0.0694 |
| `cross_day` | eeg_anchor | 0.9529 | 0.7458 | 0.2843 | **0.1147** |
| `cross_day` | attention（现状） | 0.9695 | 0.7578 | 0.2830 | 0.0813 |
| `within_subject_day` | **concat** | 0.9280 | **0.7209** | **0.3967** | 0.1894 |
| `within_subject_day` | **eeg_anchor** | **0.9160** | **0.7143** | 0.3963 | 0.1806 |
| `within_subject_day` | attention（现状） | 0.9322 | 0.7244 | 0.3928 | 0.1769 |
| `within_subject_day` | attention_multihead_pma | 0.9415 | 0.7317 | 0.3893 | **0.1919** |

## 3. 中间结论（待全量确认）

- **实验 2 判定：concat 追平并反超 attention。** 两个协议下 `concat` 的 mean raw r 都 ≥ attention（`cross_day` +0.028、`within_subject_day` +0.004），mean RMSE 都更低（−0.031 / −0.004）。按决策树，**当前 attention 融合相比朴素拼接没有带来价值**，raw r 主线应优先研究"提升单模态表征质量"而非 attention 结构。
- **实验 3 判定：未通过。** `attention_multihead_pma`（4 头 + 8 latent query）在两个协议上的 mean raw r 均低于 concat，且 `cross_day` RMSE 最差；多查询跨模态注意力没有一致增益。
- **实验 4 判定：部分有信号。** `eeg_anchor` 在 `within_subject_day` 拿到最低 mean RMSE（0.9160）和次高 raw r（0.3963），在 `cross_day` 拿到最高 centered r（0.1147），但 raw r 未超过 concat。EEG 锚点方向值得保留观察。
- **口径提醒**：切片为单 seed（240800）× 每协议 4 runs，正式结论需全量矩阵方向一致确认（进行中，见下节）。

## 4. 全量矩阵确认（已完成）

`concat` 与 `eeg_anchor` 各 180 runs 全量矩阵（`--experiment-set custom` 36 项 × 5 条 EEG route × 3 协议，默认 seed 序列 240729+n 与归档 0814 矩阵**逐 run 同 seed 配对**，paired 180/180、unpaired 0）。

### 4.1 concat vs 归档 attention（同 seed 配对 Δ）

| protocol | N | Δ raw r 均值 | Δ centered r 均值 | Δ RMSE 均值 | raw r 胜/负/平 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cross_subject` | 60 | −0.0013 | +0.0102 | +0.0060 | 22/33/5 |
| `cross_day` | 60 | **+0.0137** | −0.0126 | **−0.0058** | 27/28/5 |
| `within_subject_day` | 60 | **+0.0394** | **+0.0311** | **−0.0143** | **42/12/6** |

按 EEG 分支（每分支 36 runs）：`eegpt_partial_ft` −0.0080（attention 略优，唯一负项）、`cbramod_frozen` +0.0230、`cbramod_partial_ft` +0.0330、`de_5band` +0.0159、`eegpt_frozen` +0.0225。

### 4.2 eeg_anchor vs 归档 attention（同 seed 配对 Δ）

| protocol | N | Δ raw r 均值 | Δ centered r 均值 | Δ RMSE 均值 | raw r 胜/负/平 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cross_subject` | 60 | +0.0122 | +0.0043 | **−0.0139** | 31/22/7 |
| `cross_day` | 60 | −0.0004 | −0.0080 | **−0.0082** | 29/28/3 |
| `within_subject_day` | 60 | −0.0124 | −0.0066 | −0.0010 | 21/26/13 |

## 5. 最终结论（全量矩阵方向一致确认）

1. **实验 2 结论成立：当前 attention 融合相比朴素拼接没有价值。** 全量 180-run 同 seed 配对下，`concat` 在 `within_subject_day` 明显更优（Δraw r +0.039、42 胜/12 负、ΔRMSE −0.014、Δcentered r +0.031），`cross_day` raw r/RMSE 也小幅更优（+0.014 / −0.006），`cross_subject` 持平。除最强 EEG 分支 `eegpt_partial_ft` 外，4/5 条 EEG 分支 concat 全部更优。→ **raw r 主线应转向"提升单模态表征质量"，而非继续打磨 modality-token attention；`concat + MLP` 可作为新融合基线。**
2. **实验 3 不通过。** `attention_multihead_pma`（4 头 + 8 latent query）在决策切片两个协议 raw r 均低于 concat，未进入全量。
3. **实验 4 部分信号、不构成主线。** `eeg_anchor` 在 `cross_subject` raw r（+0.012）与 `cross_day` RMSE（−0.008）有改善，但 `within_subject_day` raw r 下降（−0.012），无 concat 那种一致优势；可作为交叉验证中的次要候选，不推荐替换默认融合。
4. **观察**：attention 唯一"不亏"的支线恰是最强分支 `eegpt_partial_ft`——提示 attention 只在最强表征下勉强打平，弱表征下反而拖累；这也支持"先把单模态表征做强"的方向。

## 产物

- 决策切片：`outputs/server_sync/fusion_variant_20260820/`（四变体 json/md + `summary.{json,md}` + 日志）
- 全量矩阵与配对分析：`outputs/server_sync/fusion_variant_20260820/fusion_variant_{concat,eeg_anchor}_full_seed240800_raw.json`、`fusion_variant_{concat,eeg_anchor}_full_paired.{md,json}`、`fusion_variant_full.log`
- 定点复现：`outputs/server_sync/fusion_variant_20260820/cross_*_seed*.json`
- 汇总脚本：`scripts/54_summarize_fusion_variants.py`（切片汇总）、`scripts/55_summarize_fusion_variant_full_paired.py`（全量配对）
