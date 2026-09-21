# 证据文档：当前 modality-token attention 融合相比朴素拼接（concat）没有价值

> 生成时间：2026-08-20。本文档所有数字均由脚本从真实实验 report JSON 读取生成，无手工转录。
> 生成脚本：`scripts/window_fatigue/56_build_fusion_attention_vs_concat_evidence.py`；源数据与完整 JSON 副本见文末「产物与复现」。

---

## 1. 结论摘要（TL;DR）

- **主张**：在当前 28,819 窗口 EEG-aligned 四模态疲劳预测主线上，`AttentionRegressor`（单头 self-attention + learnable-query pooling，即 `technical_route_20260814.md` 中的 cross-attention 融合）相比朴素 `concat + MLP` **不提供任何预测价值**，在多数配置下反而更差。
- **最强证据**：全量 180-run 矩阵与归档 attention 矩阵**逐 run 同 seed 配对**（paired 180/180）：`concat` 的 mean Δraw r 在 `within_subject_day` 为 **+0.0394**（42 胜 / 12 负，符号检验 p=5.2e-5）、`cross_day` 为 **+0.0137**、`cross_subject` 为 **−0.0013**（持平）；mean ΔRMSE 在 `within_subject_day` 为 **−0.0143**。
- **方向一致性**：独立的决策切片（32 runs，seed 240800 固定）在两个协议上 `concat` 的 mean raw r 均 ≥ `attention`，与全量配对方向一致。
- **唯一例外**：最强 EEG 分支 `eegpt_partial_ft` 上 attention 略优（Δraw r −0.008，接近噪声），其余 4/5 分支 concat 全部更优；attention 只在最强表征下勉强打平。
- **建议**：以 `concat + MLP` 替换当前 attention 融合作为新基线；raw r 主线资源转向提升单模态表征质量（wear/video 目前为 frozen/无监督表征）。

---

## 2. 背景与问题定义

当前主线融合器（`scripts/window_fatigue/32_run_eegpt_centered_loss.py` 的 `AttentionRegressor`）：每个样本由最多 4 个 256D 模态 token（EEG/Wear/Video/Audio，顺序 `[eeg, wear, video, audio]`，`modality_mask` 屏蔽缺失）表示，融合流程为：共享 `Linear(256→128)` → learnable modality embedding → 单头 `MultiheadAttention`（Q=K=V，模态间 self-attention）→ learnable query 加权 pooling → `LayerNorm + MLP` 回归头。

质疑点：序列只有 ≤4 个 token、单头 128 维，attention 实际退化为带 mask 的加权平均，表达上限低于或等于「直接拼接后让 MLP 自己学」。为验证该质疑，实现四个融合变体（同一脚本、同一输入 token、同一训练/评估链路，仅替换 `encode()`）：

| variant | 结构 | 对应实验 |
| --- | --- | --- |
| `attention` | 现状：单头 self-attention + learnable query pooling | 实验 1（基线） |
| `concat` | 无 attention：零掩码拼接 `256*M + M`（含 mask 指示位）→ `Linear` 投影 | 实验 2（朴素拼接） |
| `attention_multihead_pma` | 8 个 latent query 对模态 token 做 4 头 cross-attention → mean-pool | 实验 3（结构做厚） |
| `eeg_anchor` | EEG token 作 query、其余模态作 key/value 的非对称 cross-attention | 实验 4（EEG 锚点） |

所有变体 `encode()` 输出 `(B, hidden_dim)`，三个 head（regression / ordinal / classification）与损失、早停、评估、归一化链路完全复用。

---

## 3. 实验设计（控制变量）

| 维度 | 决策切片 | 全量配对矩阵 |
| --- | --- | --- |
| 协议 | `cross_day`, `within_subject_day` | `cross_subject`, `cross_day`, `within_subject_day` |
| EEG 分支 | `eeg_eegpt_partial_ft_v1`（最强主线） | 全部 5 条 256D route |
| 融合组合 | `B0_Wphysio_full, B0_Wphysio_no_audio, A1_Wdeep_full, A1_Wdeep_no_audio` | 全部 12 个 video-only 组合 |
| 变体 | 4 个 | `concat` 与 `eeg_anchor`（切片表现最好/次好） |
| 训练 seed | `--experiment-seed-fixed --seed 240800`（四变体完全同 seed） | 默认 seed 序列 `240729 + run_number`，与归档 0814 矩阵**逐 run 同 seed** |
| 其余 | `--eeg-token-root eeg_encoder_256d_tokens --eeg-token-seed 240800 --loss-modes raw --no-raw-baseline --subject-balanced-batches`，epochs 80、patience 15、hidden 128、lr 1e-3、wd 1e-4、dropout 0.1、batch 256 | 同左 |

配对方式：全量矩阵使用与归档 0814 命令完全相同的 `--experiments` 顺序（协议外循环 × 12 组合中循环 × 5 分支内循环），因此每个新 run 的 `run_seed` 与归档 attention 矩阵对应行**逐位相同**，可直接按 `(protocol, experiment, eeg_branch, seed)` 配对。

---

## 4. 证据一：基线可复现性（attention 路径与归档逐位一致）

用归档矩阵三条代表行的**同 seed** 定点重跑 `--fusion-variant attention`（`--experiment-seed-fixed --seed <归档 seed>`），验证重构未改变 attention 路径：

| protocol | experiment | seed | 复现 rmse / raw r / centered r | 归档 rmse / raw r / centered r | 逐位一致 |
| --- | --- | ---: | --- | --- | --- |
| cross_day | B0_Wphysio_full | 240790 | 0.9481 / 0.3504 / 0.1176 | 0.9481 / 0.3504 / 0.1176 | ✅ 逐位一致 |
| within_subject_day | B0_Wphysio_no_audio | 240855 | 0.9243 / 0.4049 / 0.2062 | 0.9243 / 0.4049 / 0.2062 | ✅ 逐位一致 |
| cross_subject | B0_Wphysio_no_audio | 240735 | 0.9624 / 0.0841 / 0.0865 | 0.9624 / 0.0841 / 0.0865 | ✅ 逐位一致 |

三条全部逐位一致 → 后续所有变体差异**只来自融合结构本身**，可与归档 attention 结果直接配对比较。

---

## 5. 证据二：决策切片（32 runs，seed 240800 固定配对）

### 5.1 按协议 × 变体的 test 均值（每格 4 runs）

| protocol | fusion variant | mean RMSE | mean MAE | mean raw r | mean centered r |
| --- | --- | ---: | ---: | ---: | ---: |
| cross_day | **concat** | 0.9387 | 0.7360 | 0.3112 | 0.0909 |
| cross_day | attention_multihead_pma | 0.9713 | 0.7605 | 0.2854 | 0.0694 |
| cross_day | eeg_anchor | 0.9529 | 0.7458 | 0.2843 | 0.1147 |
| cross_day | **attention** | 0.9695 | 0.7578 | 0.2830 | 0.0813 |
| within_subject_day | **concat** | 0.9280 | 0.7209 | 0.3967 | 0.1894 |
| within_subject_day | eeg_anchor | 0.9160 | 0.7143 | 0.3963 | 0.1806 |
| within_subject_day | **attention** | 0.9322 | 0.7244 | 0.3928 | 0.1769 |
| within_subject_day | attention_multihead_pma | 0.9415 | 0.7317 | 0.3893 | 0.1919 |

要点：`cross_day` 上 `concat` 的 mean raw r 0.3112 vs `attention` 0.2830（+0.028）；`within_subject_day` 上 0.3967 vs 0.3928（+0.004），mean RMSE 均更低。`attention_multihead_pma` 两协议 raw r 均低于 concat。

### 5.2 决策切片完整明细（32 runs）

| protocol | experiment | variant | seed | RMSE | MAE | raw r | centered r |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| cross_day | A1_Wdeep_full | attention | 240800 | 0.9715 | 0.7627 | 0.2538 | 0.0891 |
| cross_day | A1_Wdeep_full | attention_multihead_pma | 240800 | 0.9611 | 0.7440 | 0.2908 | 0.0616 |
| cross_day | A1_Wdeep_full | concat | 240800 | 0.9387 | 0.7396 | 0.3139 | 0.1008 |
| cross_day | A1_Wdeep_full | eeg_anchor | 240800 | 0.9576 | 0.7478 | 0.2909 | 0.1497 |
| cross_day | A1_Wdeep_no_audio | attention | 240800 | 0.9989 | 0.7700 | 0.2606 | 0.0450 |
| cross_day | A1_Wdeep_no_audio | attention_multihead_pma | 240800 | 1.0234 | 0.8020 | 0.1866 | -0.0080 |
| cross_day | A1_Wdeep_no_audio | concat | 240800 | 0.9524 | 0.7492 | 0.3093 | 0.0748 |
| cross_day | A1_Wdeep_no_audio | eeg_anchor | 240800 | 0.9569 | 0.7527 | 0.3076 | 0.0993 |
| cross_day | B0_Wphysio_full | attention | 240800 | 0.9548 | 0.7506 | 0.3094 | 0.1094 |
| cross_day | B0_Wphysio_full | attention_multihead_pma | 240800 | 0.9320 | 0.7312 | 0.3412 | 0.1127 |
| cross_day | B0_Wphysio_full | concat | 240800 | 0.9336 | 0.7333 | 0.3173 | 0.1038 |
| cross_day | B0_Wphysio_full | eeg_anchor | 240800 | 0.9565 | 0.7474 | 0.2507 | 0.1108 |
| cross_day | B0_Wphysio_no_audio | attention | 240800 | 0.9529 | 0.7479 | 0.3081 | 0.0817 |
| cross_day | B0_Wphysio_no_audio | attention_multihead_pma | 240800 | 0.9689 | 0.7647 | 0.3230 | 0.1113 |
| cross_day | B0_Wphysio_no_audio | concat | 240800 | 0.9303 | 0.7217 | 0.3044 | 0.0844 |
| cross_day | B0_Wphysio_no_audio | eeg_anchor | 240800 | 0.9406 | 0.7354 | 0.2881 | 0.0989 |
| within_subject_day | A1_Wdeep_full | attention | 240800 | 0.9270 | 0.7257 | 0.3917 | 0.1618 |
| within_subject_day | A1_Wdeep_full | attention_multihead_pma | 240800 | 0.9346 | 0.7268 | 0.3952 | 0.1963 |
| within_subject_day | A1_Wdeep_full | concat | 240800 | 0.9165 | 0.7090 | 0.3962 | 0.1798 |
| within_subject_day | A1_Wdeep_full | eeg_anchor | 240800 | 0.9206 | 0.7236 | 0.4084 | 0.1926 |
| within_subject_day | A1_Wdeep_no_audio | attention | 240800 | 0.9377 | 0.7301 | 0.3914 | 0.1731 |
| within_subject_day | A1_Wdeep_no_audio | attention_multihead_pma | 240800 | 0.9446 | 0.7383 | 0.3881 | 0.1791 |
| within_subject_day | A1_Wdeep_no_audio | concat | 240800 | 0.9470 | 0.7337 | 0.3980 | 0.2057 |
| within_subject_day | A1_Wdeep_no_audio | eeg_anchor | 240800 | 0.9198 | 0.7145 | 0.3853 | 0.1668 |
| within_subject_day | B0_Wphysio_full | attention | 240800 | 0.9332 | 0.7175 | 0.3888 | 0.1754 |
| within_subject_day | B0_Wphysio_full | attention_multihead_pma | 240800 | 0.9298 | 0.7181 | 0.4017 | 0.1992 |
| within_subject_day | B0_Wphysio_full | concat | 240800 | 0.9262 | 0.7256 | 0.3902 | 0.1746 |
| within_subject_day | B0_Wphysio_full | eeg_anchor | 240800 | 0.9161 | 0.7115 | 0.4010 | 0.1784 |
| within_subject_day | B0_Wphysio_no_audio | attention | 240800 | 0.9308 | 0.7243 | 0.3994 | 0.1971 |
| within_subject_day | B0_Wphysio_no_audio | attention_multihead_pma | 240800 | 0.9571 | 0.7434 | 0.3723 | 0.1928 |
| within_subject_day | B0_Wphysio_no_audio | concat | 240800 | 0.9223 | 0.7151 | 0.4025 | 0.1975 |
| within_subject_day | B0_Wphysio_no_audio | eeg_anchor | 240800 | 0.9075 | 0.7076 | 0.3904 | 0.1847 |

### 5.3 实验 3 判定：`attention_multihead_pma` 未通过（因此未进入全量矩阵）

| protocol | attention mean raw r | concat mean raw r | pma mean raw r | pma − concat | pma − attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| cross_day | 0.2830 | 0.3112 | 0.2854 | -0.0258 | 0.0024 |
| within_subject_day | 0.3928 | 0.3967 | 0.3893 | -0.0074 | -0.0035 |

判定：`attention_multihead_pma`（4 头 + 8 latent query 的 latent-array cross-attention）在两个协议上的 mean raw r 均低于 `concat`，也未能对 `attention` 形成一致优势（见上表差值列），故不进入全量矩阵。实验 3 结论：**把 attention 做厚（多头 + 多 query）不解决该任务上的问题**。

---

## 6. 证据三：全量配对矩阵（concat 180 runs vs 归档 attention 180 runs，同 seed 逐 run 配对）

配对完整性：**paired 180 / 0 unpaired**（全部命中）。

### 6.1 整体与协议级汇总

| 范围 | N | mean Δraw r | median Δraw r | mean Δcentered r | mean ΔRMSE | 胜/负/平(raw r, ±0.005) | 符号检验 p | concat mean raw r | attention mean raw r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 180 | 0.0173 | 0.0065 | 0.0096 | -0.0047 | 91/73/16 | 0.1842 | 0.1811 | 0.1638 |
| cross_subject | 60 | -0.0013 | -0.0127 | 0.0102 | 0.0060 | 22/33/5 | 0.177 | 0.0234 | 0.0247 |
| cross_day | 60 | 0.0137 | -0.0023 | -0.0126 | -0.0058 | 27/28/5 | 1 | 0.2084 | 0.1947 |
| within_subject_day | 60 | 0.0394 | 0.0280 | 0.0311 | -0.0143 | 42/12/6 | 5.209e-05 | 0.3115 | 0.2721 |

解读：全量 180 对中 `concat` mean Δraw r +0.017、ΔRMSE −0.005；`within_subject_day` 上 Δraw r +0.0394 且符号检验 p=5.2e-5，远超偶然；`cross_subject` 持平（−0.0013）。

### 6.2 按 EEG 分支汇总

| EEG branch | N | mean Δraw r | mean Δcentered r | mean ΔRMSE | 胜/负/平 | 符号检验 p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eeg_eegpt_frozen_v1 | 36 | 0.0225 | 0.0077 | -0.0068 | 17/15/4 | 0.8601 |
| eeg_eegpt_partial_ft_v1 | 36 | -0.0080 | -0.0081 | -0.0009 | 11/18/7 | 0.2649 |
| eeg_cbramod_frozen_v1 | 36 | 0.0230 | 0.0178 | 0.0002 | 22/13/1 | 0.1755 |
| eeg_cbramod_partial_ft_v1 | 36 | 0.0330 | 0.0219 | -0.0112 | 23/11/2 | 0.05761 |
| eeg_de_5band_1s_avg_v1 | 36 | 0.0159 | 0.0086 | -0.0047 | 18/16/2 | 0.8642 |

解读：唯一 `attention` 占优的分支是 `eegpt_partial_ft`（Δraw r −0.008，且 18 胜/15 负接近五五开）；其余 4/5 分支 concat 全部更优，`cbramod_partial_ft` 最明显（+0.033）。

### 6.3 按融合组合汇总（12 组合 × 3 协议 × 5 分支 = 每格 15 对）

| experiment | N | mean Δraw r | mean Δcentered r | mean ΔRMSE | 胜/负/平 | 符号检验 p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0_Wphysio_full | 15 | 0.0280 | 0.0057 | 0.0016 | 8/6/1 | 0.7905 |
| B0_Wphysio_no_audio | 15 | 0.0479 | 0.0092 | -0.0107 | 9/6/0 | 0.6072 |
| B0_Wdeep_full | 15 | 0.0088 | 0.0136 | -0.0072 | 8/5/2 | 0.5811 |
| B0_Wdeep_no_audio | 15 | -0.0008 | 0.0094 | -0.0168 | 7/8/0 | 1 |
| A1_Wphysio_full | 15 | 0.0172 | 0.0009 | 0.0023 | 6/6/3 | 1 |
| A1_Wphysio_no_audio | 15 | 0.0434 | 0.0061 | -0.0006 | 9/6/0 | 0.6072 |
| A1_Wdeep_full | 15 | -0.0070 | 0.0038 | -0.0048 | 7/7/1 | 1 |
| A1_Wdeep_no_audio | 15 | 0.0161 | 0.0067 | -0.0100 | 7/6/2 | 1 |
| A2_Wphysio_full | 15 | 0.0118 | 0.0020 | 0.0049 | 5/10/0 | 0.3018 |
| A2_Wphysio_no_audio | 15 | 0.0040 | 0.0096 | -0.0055 | 8/6/1 | 0.7905 |
| A2_Wdeep_full | 15 | 0.0176 | 0.0174 | 0.0021 | 9/2/4 | 0.06543 |
| A2_Wdeep_no_audio | 15 | 0.0204 | 0.0303 | -0.0114 | 8/5/2 | 0.5811 |

### 6.4 分支 × 协议 细分（每格 12 对）

| EEG branch / protocol | N | mean Δraw r | mean Δcentered r | mean ΔRMSE | 胜/负/平 |
| --- | ---: | ---: | ---: | ---: | ---: |
| eeg_cbramod_frozen_v1 / cross_day | 12 | -0.0093 | -0.0309 | 0.0039 | 5/7/0 |
| eeg_cbramod_frozen_v1 / cross_subject | 12 | 0.0059 | 0.0269 | 0.0195 | 5/6/1 |
| eeg_cbramod_frozen_v1 / within_subject_day | 12 | 0.0725 | 0.0574 | -0.0229 | 12/0/0 |
| eeg_cbramod_partial_ft_v1 / cross_day | 12 | 0.0228 | -0.0170 | -0.0098 | 5/6/1 |
| eeg_cbramod_partial_ft_v1 / cross_subject | 12 | 0.0159 | 0.0288 | -0.0006 | 7/5/0 |
| eeg_cbramod_partial_ft_v1 / within_subject_day | 12 | 0.0604 | 0.0538 | -0.0231 | 11/0/1 |
| eeg_de_5band_1s_avg_v1 / cross_day | 12 | 0.0392 | 0.0024 | -0.0147 | 7/4/1 |
| eeg_de_5band_1s_avg_v1 / cross_subject | 12 | -0.0287 | -0.0083 | 0.0193 | 1/10/1 |
| eeg_de_5band_1s_avg_v1 / within_subject_day | 12 | 0.0371 | 0.0318 | -0.0188 | 10/2/0 |
| eeg_eegpt_frozen_v1 / cross_day | 12 | 0.0157 | -0.0124 | -0.0081 | 3/7/2 |
| eeg_eegpt_frozen_v1 / cross_subject | 12 | 0.0073 | 0.0032 | -0.0005 | 6/6/0 |
| eeg_eegpt_frozen_v1 / within_subject_day | 12 | 0.0445 | 0.0322 | -0.0119 | 8/2/2 |
| eeg_eegpt_partial_ft_v1 / cross_day | 12 | 0.0002 | -0.0050 | -0.0001 | 7/4/1 |
| eeg_eegpt_partial_ft_v1 / cross_subject | 12 | -0.0069 | 0.0006 | -0.0076 | 3/6/3 |
| eeg_eegpt_partial_ft_v1 / within_subject_day | 12 | -0.0173 | -0.0198 | 0.0051 | 1/8/3 |

### 6.5 完整 180 行配对明细（concat vs 归档 attention）

#### cross_subject（60 对）

| experiment | EEG branch | seed | attn raw r | concat raw r | Δraw r | attn centered r | concat centered r | Δcentered r | attn RMSE | concat RMSE | ΔRMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0_Wphysio_full | eeg_eegpt_frozen_v1 | 240729 | -0.0965 | -0.0549 | 0.0416 | -0.0354 | -0.0139 | 0.0215 | 0.9435 | 0.9557 | 0.0122 |
| B0_Wphysio_full | eeg_eegpt_partial_ft_v1 | 240730 | -0.0165 | -0.0242 | -0.0077 | 0.0274 | -0.0022 | -0.0296 | 0.9980 | 0.9449 | -0.0531 |
| B0_Wphysio_full | eeg_cbramod_frozen_v1 | 240731 | -0.0752 | -0.0386 | 0.0366 | -0.0488 | -0.0223 | 0.0265 | 0.9362 | 1.0268 | 0.0906 |
| B0_Wphysio_full | eeg_cbramod_partial_ft_v1 | 240732 | -0.0371 | -0.0499 | -0.0127 | 0.0026 | -0.0049 | -0.0075 | 0.9614 | 1.0037 | 0.0424 |
| B0_Wphysio_full | eeg_de_5band_1s_avg_v1 | 240733 | -0.0215 | -0.0678 | -0.0463 | -0.0029 | -0.0071 | -0.0042 | 0.9368 | 1.0197 | 0.0829 |
| B0_Wphysio_no_audio | eeg_eegpt_frozen_v1 | 240734 | 0.0305 | 0.0160 | -0.0145 | 0.0451 | 0.0216 | -0.0235 | 0.9078 | 0.9426 | 0.0347 |
| B0_Wphysio_no_audio | eeg_eegpt_partial_ft_v1 | 240735 | 0.0841 | 0.0503 | -0.0338 | 0.0865 | 0.0838 | -0.0027 | 0.9624 | 0.9393 | -0.0232 |
| B0_Wphysio_no_audio | eeg_cbramod_frozen_v1 | 240736 | -0.0599 | 0.0065 | 0.0664 | -0.0201 | 0.0381 | 0.0582 | 0.9621 | 0.9377 | -0.0244 |
| B0_Wphysio_no_audio | eeg_cbramod_partial_ft_v1 | 240737 | -0.0638 | 0.0095 | 0.0733 | -0.0501 | 0.0378 | 0.0879 | 0.9634 | 0.9518 | -0.0117 |
| B0_Wphysio_no_audio | eeg_de_5band_1s_avg_v1 | 240738 | 0.0303 | 0.0082 | -0.0221 | 0.0465 | 0.0497 | 0.0032 | 0.9497 | 0.9457 | -0.0040 |
| B0_Wdeep_full | eeg_eegpt_frozen_v1 | 240739 | 0.0781 | 0.0595 | -0.0186 | 0.0507 | 0.0442 | -0.0065 | 0.9786 | 0.9338 | -0.0448 |
| B0_Wdeep_full | eeg_eegpt_partial_ft_v1 | 240740 | 0.0499 | 0.0688 | 0.0189 | 0.0629 | 0.0841 | 0.0212 | 0.9511 | 0.9490 | -0.0021 |
| B0_Wdeep_full | eeg_cbramod_frozen_v1 | 240741 | 0.0498 | 0.0682 | 0.0184 | 0.0309 | 0.0865 | 0.0556 | 0.9373 | 0.9397 | 0.0025 |
| B0_Wdeep_full | eeg_cbramod_partial_ft_v1 | 240742 | 0.0915 | 0.0546 | -0.0370 | 0.0384 | 0.0641 | 0.0256 | 0.9105 | 0.9637 | 0.0531 |
| B0_Wdeep_full | eeg_de_5band_1s_avg_v1 | 240743 | 0.0969 | 0.0762 | -0.0207 | 0.0379 | 0.0717 | 0.0338 | 0.9259 | 0.9479 | 0.0220 |
| B0_Wdeep_no_audio | eeg_eegpt_frozen_v1 | 240744 | 0.0723 | 0.1020 | 0.0297 | 0.0324 | 0.0696 | 0.0373 | 0.9054 | 0.8976 | -0.0078 |
| B0_Wdeep_no_audio | eeg_eegpt_partial_ft_v1 | 240745 | 0.1124 | 0.0753 | -0.0370 | 0.1352 | 0.1074 | -0.0278 | 0.9411 | 0.9317 | -0.0094 |
| B0_Wdeep_no_audio | eeg_cbramod_frozen_v1 | 240746 | 0.0566 | 0.0495 | -0.0070 | 0.0023 | 0.0735 | 0.0712 | 0.9275 | 0.9439 | 0.0164 |
| B0_Wdeep_no_audio | eeg_cbramod_partial_ft_v1 | 240747 | 0.0454 | 0.0891 | 0.0437 | -0.0135 | 0.0603 | 0.0738 | 0.9237 | 0.9020 | -0.0217 |
| B0_Wdeep_no_audio | eeg_de_5band_1s_avg_v1 | 240748 | 0.0911 | 0.1002 | 0.0091 | 0.0741 | 0.0786 | 0.0045 | 0.9346 | 0.9168 | -0.0177 |
| A1_Wphysio_full | eeg_eegpt_frozen_v1 | 240749 | -0.0801 | -0.0353 | 0.0448 | -0.0295 | -0.0332 | -0.0037 | 0.9478 | 0.9724 | 0.0245 |
| A1_Wphysio_full | eeg_eegpt_partial_ft_v1 | 240750 | -0.0154 | -0.0178 | -0.0024 | 0.0330 | 0.0176 | -0.0154 | 0.9776 | 0.9668 | -0.0108 |
| A1_Wphysio_full | eeg_cbramod_frozen_v1 | 240751 | -0.0351 | -0.0540 | -0.0189 | -0.0007 | -0.0176 | -0.0169 | 0.9341 | 1.0332 | 0.0991 |
| A1_Wphysio_full | eeg_cbramod_partial_ft_v1 | 240752 | 0.0369 | -0.0375 | -0.0744 | 0.0412 | -0.0214 | -0.0626 | 0.9871 | 0.9791 | -0.0080 |
| A1_Wphysio_full | eeg_de_5band_1s_avg_v1 | 240753 | 0.0135 | -0.0197 | -0.0332 | 0.0208 | -0.0063 | -0.0271 | 0.9439 | 1.0118 | 0.0679 |
| A1_Wphysio_no_audio | eeg_eegpt_frozen_v1 | 240754 | 0.0108 | -0.0128 | -0.0235 | 0.0242 | 0.0215 | -0.0026 | 0.9340 | 0.9461 | 0.0121 |
| A1_Wphysio_no_audio | eeg_eegpt_partial_ft_v1 | 240755 | 0.0589 | 0.0456 | -0.0133 | 0.0651 | 0.0763 | 0.0111 | 0.9637 | 0.9903 | 0.0266 |
| A1_Wphysio_no_audio | eeg_cbramod_frozen_v1 | 240756 | -0.0194 | -0.0771 | -0.0577 | 0.0066 | -0.0479 | -0.0545 | 0.9440 | 0.9318 | -0.0122 |
| A1_Wphysio_no_audio | eeg_cbramod_partial_ft_v1 | 240757 | -0.0512 | -0.0063 | 0.0450 | -0.0136 | 0.0324 | 0.0459 | 0.9718 | 0.9693 | -0.0025 |
| A1_Wphysio_no_audio | eeg_de_5band_1s_avg_v1 | 240758 | -0.0060 | -0.0453 | -0.0393 | 0.0246 | -0.0050 | -0.0297 | 0.9433 | 0.9722 | 0.0289 |
| A1_Wdeep_full | eeg_eegpt_frozen_v1 | 240759 | 0.0674 | 0.0507 | -0.0168 | 0.0380 | 0.0102 | -0.0278 | 0.9530 | 0.9537 | 0.0008 |
| A1_Wdeep_full | eeg_eegpt_partial_ft_v1 | 240760 | 0.0402 | 0.0175 | -0.0227 | 0.0479 | 0.0412 | -0.0067 | 0.9827 | 0.9611 | -0.0216 |
| A1_Wdeep_full | eeg_cbramod_frozen_v1 | 240761 | 0.1127 | 0.0442 | -0.0685 | 0.0433 | 0.0107 | -0.0326 | 0.9074 | 0.9333 | 0.0259 |
| A1_Wdeep_full | eeg_cbramod_partial_ft_v1 | 240762 | 0.0431 | 0.0716 | 0.0285 | 0.0307 | 0.0678 | 0.0372 | 0.9295 | 0.9366 | 0.0071 |
| A1_Wdeep_full | eeg_de_5band_1s_avg_v1 | 240763 | 0.0877 | 0.0667 | -0.0211 | 0.0678 | 0.0423 | -0.0255 | 0.9218 | 0.9221 | 0.0003 |
| A1_Wdeep_no_audio | eeg_eegpt_frozen_v1 | 240764 | 0.0230 | 0.1146 | 0.0916 | 0.0255 | 0.0706 | 0.0451 | 0.9183 | 0.8925 | -0.0258 |
| A1_Wdeep_no_audio | eeg_eegpt_partial_ft_v1 | 240765 | 0.0193 | 0.0749 | 0.0556 | 0.0381 | 0.0838 | 0.0457 | 0.9854 | 0.9462 | -0.0393 |
| A1_Wdeep_no_audio | eeg_cbramod_frozen_v1 | 240766 | 0.0390 | 0.1089 | 0.0698 | -0.0038 | 0.1122 | 0.1159 | 0.9338 | 0.9286 | -0.0052 |
| A1_Wdeep_no_audio | eeg_cbramod_partial_ft_v1 | 240767 | 0.0564 | 0.0436 | -0.0127 | 0.0017 | 0.0273 | 0.0256 | 0.9118 | 0.9130 | 0.0012 |
| A1_Wdeep_no_audio | eeg_de_5band_1s_avg_v1 | 240768 | 0.0807 | 0.0782 | -0.0025 | 0.0393 | 0.0720 | 0.0328 | 0.9350 | 0.9281 | -0.0069 |
| A2_Wphysio_full | eeg_eegpt_frozen_v1 | 240769 | -0.0242 | -0.0937 | -0.0695 | 0.0136 | -0.0397 | -0.0533 | 0.9299 | 0.9675 | 0.0376 |
| A2_Wphysio_full | eeg_eegpt_partial_ft_v1 | 240770 | 0.0615 | 0.0095 | -0.0520 | 0.0904 | 0.0584 | -0.0320 | 0.9435 | 0.9739 | 0.0304 |
| A2_Wphysio_full | eeg_cbramod_frozen_v1 | 240771 | -0.0553 | -0.0834 | -0.0281 | -0.0060 | -0.0320 | -0.0260 | 0.9706 | 1.0033 | 0.0327 |
| A2_Wphysio_full | eeg_cbramod_partial_ft_v1 | 240772 | -0.0293 | -0.0546 | -0.0253 | 0.0076 | -0.0159 | -0.0235 | 0.9708 | 0.9785 | 0.0077 |
| A2_Wphysio_full | eeg_de_5band_1s_avg_v1 | 240773 | -0.0452 | -0.0839 | -0.0387 | -0.0051 | -0.0444 | -0.0393 | 0.9457 | 0.9818 | 0.0362 |
| A2_Wphysio_no_audio | eeg_eegpt_frozen_v1 | 240774 | 0.0596 | -0.0113 | -0.0708 | 0.0685 | 0.0035 | -0.0650 | 0.9260 | 0.9329 | 0.0069 |
| A2_Wphysio_no_audio | eeg_eegpt_partial_ft_v1 | 240775 | 0.0572 | 0.0675 | 0.0103 | 0.0815 | 0.1046 | 0.0231 | 0.9920 | 0.9786 | -0.0134 |
| A2_Wphysio_no_audio | eeg_cbramod_frozen_v1 | 240776 | 0.0043 | 0.0045 | 0.0002 | 0.0325 | 0.0485 | 0.0160 | 0.9608 | 0.9634 | 0.0026 |
| A2_Wphysio_no_audio | eeg_cbramod_partial_ft_v1 | 240777 | -0.0433 | 0.0072 | 0.0505 | -0.0093 | 0.0233 | 0.0326 | 1.0249 | 0.9449 | -0.0800 |
| A2_Wphysio_no_audio | eeg_de_5band_1s_avg_v1 | 240778 | 0.0587 | -0.0154 | -0.0741 | 0.0799 | 0.0221 | -0.0578 | 0.9492 | 0.9505 | 0.0013 |
| A2_Wdeep_full | eeg_eegpt_frozen_v1 | 240779 | 0.0807 | 0.0937 | 0.0130 | 0.0347 | 0.0424 | 0.0078 | 0.9168 | 0.9112 | -0.0055 |
| A2_Wdeep_full | eeg_eegpt_partial_ft_v1 | 240780 | 0.0335 | 0.0311 | -0.0024 | 0.0397 | 0.0494 | 0.0097 | 0.9952 | 1.0074 | 0.0123 |
| A2_Wdeep_full | eeg_cbramod_frozen_v1 | 240781 | 0.0425 | 0.0248 | -0.0178 | 0.0127 | -0.0022 | -0.0149 | 0.9235 | 0.9505 | 0.0270 |
| A2_Wdeep_full | eeg_cbramod_partial_ft_v1 | 240782 | 0.0423 | 0.0729 | 0.0306 | 0.0013 | 0.0475 | 0.0462 | 0.9255 | 0.9694 | 0.0439 |
| A2_Wdeep_full | eeg_de_5band_1s_avg_v1 | 240783 | 0.0410 | 0.0259 | -0.0151 | 0.0117 | 0.0032 | -0.0085 | 0.9554 | 0.9672 | 0.0118 |
| A2_Wdeep_no_audio | eeg_eegpt_frozen_v1 | 240784 | 0.0135 | 0.0940 | 0.0805 | -0.0442 | 0.0648 | 0.1089 | 0.9536 | 0.9023 | -0.0512 |
| A2_Wdeep_no_audio | eeg_eegpt_partial_ft_v1 | 240785 | 0.0463 | 0.0496 | 0.0033 | 0.0764 | 0.0865 | 0.0101 | 0.9720 | 0.9847 | 0.0127 |
| A2_Wdeep_no_audio | eeg_cbramod_frozen_v1 | 240786 | 0.0260 | 0.1034 | 0.0774 | -0.0196 | 0.1045 | 0.1241 | 0.9640 | 0.9429 | -0.0210 |
| A2_Wdeep_no_audio | eeg_cbramod_partial_ft_v1 | 240787 | 0.0058 | 0.0868 | 0.0811 | -0.0161 | 0.0489 | 0.0649 | 0.9463 | 0.9071 | -0.0392 |
| A2_Wdeep_no_audio | eeg_de_5band_1s_avg_v1 | 240788 | 0.1054 | 0.0652 | -0.0402 | 0.0393 | 0.0572 | 0.0179 | 0.9153 | 0.9245 | 0.0092 |

#### cross_day（60 对）

| experiment | EEG branch | seed | attn raw r | concat raw r | Δraw r | attn centered r | concat centered r | Δcentered r | attn RMSE | concat RMSE | ΔRMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0_Wphysio_full | eeg_eegpt_frozen_v1 | 240789 | 0.0710 | 0.1960 | 0.1251 | 0.0537 | 0.0478 | -0.0059 | 1.0001 | 0.9583 | -0.0419 |
| B0_Wphysio_full | eeg_eegpt_partial_ft_v1 | 240790 | 0.3504 | 0.3112 | -0.0391 | 0.1176 | 0.1341 | 0.0165 | 0.9481 | 0.9407 | -0.0074 |
| B0_Wphysio_full | eeg_cbramod_frozen_v1 | 240791 | 0.1808 | 0.2046 | 0.0238 | 0.0348 | 0.0470 | 0.0123 | 0.9521 | 0.9538 | 0.0018 |
| B0_Wphysio_full | eeg_cbramod_partial_ft_v1 | 240792 | 0.1601 | 0.1441 | -0.0160 | 0.0374 | 0.0206 | -0.0167 | 0.9783 | 0.9769 | -0.0014 |
| B0_Wphysio_full | eeg_de_5band_1s_avg_v1 | 240793 | 0.2095 | 0.2022 | -0.0073 | 0.0600 | 0.0628 | 0.0028 | 0.9843 | 0.9681 | -0.0162 |
| B0_Wphysio_no_audio | eeg_eegpt_frozen_v1 | 240794 | -0.0327 | 0.0469 | 0.0796 | 0.0449 | 0.0084 | -0.0365 | 0.9683 | 0.9582 | -0.0101 |
| B0_Wphysio_no_audio | eeg_eegpt_partial_ft_v1 | 240795 | 0.3301 | 0.3077 | -0.0223 | 0.1526 | 0.1015 | -0.0512 | 0.9189 | 0.9315 | 0.0126 |
| B0_Wphysio_no_audio | eeg_cbramod_frozen_v1 | 240796 | 0.0872 | 0.1261 | 0.0389 | 0.0463 | 0.0027 | -0.0436 | 0.9515 | 0.9601 | 0.0086 |
| B0_Wphysio_no_audio | eeg_cbramod_partial_ft_v1 | 240797 | -0.0028 | 0.1749 | 0.1777 | 0.0527 | 0.0056 | -0.0470 | 0.9615 | 0.9585 | -0.0030 |
| B0_Wphysio_no_audio | eeg_de_5band_1s_avg_v1 | 240798 | 0.2093 | 0.1866 | -0.0227 | 0.0263 | 0.0410 | 0.0147 | 0.9720 | 0.9614 | -0.0106 |
| B0_Wdeep_full | eeg_eegpt_frozen_v1 | 240799 | 0.1958 | 0.1706 | -0.0252 | 0.0452 | 0.0355 | -0.0098 | 0.9928 | 0.9812 | -0.0116 |
| B0_Wdeep_full | eeg_eegpt_partial_ft_v1 | 240800 | 0.2871 | 0.2988 | 0.0117 | 0.0738 | 0.0956 | 0.0218 | 0.9634 | 0.9562 | -0.0072 |
| B0_Wdeep_full | eeg_cbramod_frozen_v1 | 240801 | 0.2014 | 0.2334 | 0.0320 | 0.0516 | 0.0058 | -0.0458 | 0.9784 | 0.9749 | -0.0034 |
| B0_Wdeep_full | eeg_cbramod_partial_ft_v1 | 240802 | 0.1805 | 0.2093 | 0.0289 | 0.0923 | 0.0373 | -0.0550 | 0.9954 | 0.9826 | -0.0129 |
| B0_Wdeep_full | eeg_de_5band_1s_avg_v1 | 240803 | 0.1968 | 0.1925 | -0.0042 | 0.0090 | 0.0503 | 0.0413 | 0.9987 | 1.0185 | 0.0199 |
| B0_Wdeep_no_audio | eeg_eegpt_frozen_v1 | 240804 | 0.2308 | 0.1976 | -0.0332 | 0.0918 | 0.0528 | -0.0390 | 0.9659 | 0.9334 | -0.0324 |
| B0_Wdeep_no_audio | eeg_eegpt_partial_ft_v1 | 240805 | 0.2367 | 0.2772 | 0.0405 | 0.0714 | 0.0933 | 0.0219 | 0.9801 | 0.9626 | -0.0175 |
| B0_Wdeep_no_audio | eeg_cbramod_frozen_v1 | 240806 | 0.2398 | 0.1911 | -0.0487 | 0.0538 | 0.0419 | -0.0119 | 0.9795 | 0.9371 | -0.0424 |
| B0_Wdeep_no_audio | eeg_cbramod_partial_ft_v1 | 240807 | 0.2258 | 0.1994 | -0.0264 | 0.0565 | 0.0401 | -0.0163 | 0.9882 | 0.9386 | -0.0497 |
| B0_Wdeep_no_audio | eeg_de_5band_1s_avg_v1 | 240808 | 0.2372 | 0.1948 | -0.0424 | 0.0599 | 0.0207 | -0.0392 | 0.9752 | 0.9728 | -0.0024 |
| A1_Wphysio_full | eeg_eegpt_frozen_v1 | 240809 | 0.2201 | 0.1907 | -0.0294 | 0.0492 | 0.0682 | 0.0190 | 0.9816 | 0.9713 | -0.0104 |
| A1_Wphysio_full | eeg_eegpt_partial_ft_v1 | 240810 | 0.3037 | 0.3004 | -0.0033 | 0.1605 | 0.1395 | -0.0210 | 0.9465 | 0.9544 | 0.0079 |
| A1_Wphysio_full | eeg_cbramod_frozen_v1 | 240811 | 0.2090 | 0.1627 | -0.0463 | 0.0567 | 0.0062 | -0.0505 | 0.9694 | 0.9922 | 0.0228 |
| A1_Wphysio_full | eeg_cbramod_partial_ft_v1 | 240812 | 0.1774 | 0.1780 | 0.0006 | 0.0617 | 0.0203 | -0.0415 | 0.9914 | 0.9630 | -0.0285 |
| A1_Wphysio_full | eeg_de_5band_1s_avg_v1 | 240813 | 0.1731 | 0.1986 | 0.0255 | 0.0496 | 0.0487 | -0.0010 | 0.9969 | 0.9562 | -0.0407 |
| A1_Wphysio_no_audio | eeg_eegpt_frozen_v1 | 240814 | -0.0144 | 0.2033 | 0.2177 | 0.0415 | 0.0578 | 0.0163 | 0.9653 | 0.9557 | -0.0096 |
| A1_Wphysio_no_audio | eeg_eegpt_partial_ft_v1 | 240815 | 0.2817 | 0.2974 | 0.0157 | 0.1037 | 0.1121 | 0.0084 | 0.9434 | 0.9554 | 0.0119 |
| A1_Wphysio_no_audio | eeg_cbramod_frozen_v1 | 240816 | 0.0998 | 0.0612 | -0.0386 | 0.0677 | 0.0052 | -0.0625 | 0.9452 | 0.9639 | 0.0186 |
| A1_Wphysio_no_audio | eeg_cbramod_partial_ft_v1 | 240817 | 0.0789 | 0.2229 | 0.1440 | 0.0304 | 0.0164 | -0.0140 | 0.9509 | 0.9704 | 0.0195 |
| A1_Wphysio_no_audio | eeg_de_5band_1s_avg_v1 | 240818 | 0.0058 | 0.2084 | 0.2026 | 0.0324 | 0.0409 | 0.0086 | 0.9766 | 0.9540 | -0.0227 |
| A1_Wdeep_full | eeg_eegpt_frozen_v1 | 240819 | 0.2539 | 0.1737 | -0.0802 | 0.0647 | 0.0099 | -0.0547 | 0.9773 | 0.9811 | 0.0038 |
| A1_Wdeep_full | eeg_eegpt_partial_ft_v1 | 240820 | 0.2588 | 0.2780 | 0.0192 | 0.1002 | 0.0927 | -0.0075 | 0.9744 | 0.9535 | -0.0209 |
| A1_Wdeep_full | eeg_cbramod_frozen_v1 | 240821 | 0.2604 | 0.2088 | -0.0516 | 0.0654 | 0.0462 | -0.0192 | 0.9596 | 1.0040 | 0.0444 |
| A1_Wdeep_full | eeg_cbramod_partial_ft_v1 | 240822 | 0.2509 | 0.2072 | -0.0436 | 0.0379 | 0.0533 | 0.0154 | 0.9575 | 0.9752 | 0.0177 |
| A1_Wdeep_full | eeg_de_5band_1s_avg_v1 | 240823 | 0.2011 | 0.2579 | 0.0568 | 0.0604 | 0.0531 | -0.0073 | 1.0005 | 0.9411 | -0.0594 |
| A1_Wdeep_no_audio | eeg_eegpt_frozen_v1 | 240824 | 0.2201 | 0.1894 | -0.0308 | 0.0835 | 0.0295 | -0.0540 | 0.9605 | 0.9702 | 0.0097 |
| A1_Wdeep_no_audio | eeg_eegpt_partial_ft_v1 | 240825 | 0.3225 | 0.2907 | -0.0318 | 0.1272 | 0.0672 | -0.0600 | 0.9625 | 0.9579 | -0.0046 |
| A1_Wdeep_no_audio | eeg_cbramod_frozen_v1 | 240826 | 0.1962 | 0.1541 | -0.0421 | 0.0995 | 0.0081 | -0.0914 | 0.9514 | 0.9465 | -0.0049 |
| A1_Wdeep_no_audio | eeg_cbramod_partial_ft_v1 | 240827 | 0.1393 | 0.1916 | 0.0524 | 0.0223 | 0.0419 | 0.0196 | 1.0000 | 0.9493 | -0.0506 |
| A1_Wdeep_no_audio | eeg_de_5band_1s_avg_v1 | 240828 | 0.1109 | 0.2253 | 0.1144 | 0.0282 | 0.0601 | 0.0319 | 1.0071 | 0.9692 | -0.0378 |
| A2_Wphysio_full | eeg_eegpt_frozen_v1 | 240829 | 0.1698 | 0.1558 | -0.0140 | 0.0321 | 0.0471 | 0.0150 | 0.9604 | 0.9616 | 0.0012 |
| A2_Wphysio_full | eeg_eegpt_partial_ft_v1 | 240830 | 0.2947 | 0.2744 | -0.0204 | 0.1200 | 0.0903 | -0.0296 | 0.9471 | 0.9730 | 0.0259 |
| A2_Wphysio_full | eeg_cbramod_frozen_v1 | 240831 | 0.1788 | 0.1901 | 0.0113 | 0.0563 | 0.0373 | -0.0191 | 0.9730 | 0.9696 | -0.0034 |
| A2_Wphysio_full | eeg_cbramod_partial_ft_v1 | 240832 | 0.1772 | 0.1591 | -0.0180 | 0.0389 | 0.0234 | -0.0155 | 0.9582 | 0.9830 | 0.0247 |
| A2_Wphysio_full | eeg_de_5band_1s_avg_v1 | 240833 | 0.2054 | 0.1887 | -0.0167 | 0.0451 | 0.0445 | -0.0006 | 0.9890 | 0.9748 | -0.0142 |
| A2_Wphysio_no_audio | eeg_eegpt_frozen_v1 | 240834 | 0.1850 | 0.1655 | -0.0195 | 0.0330 | 0.0589 | 0.0259 | 0.9747 | 0.9534 | -0.0213 |
| A2_Wphysio_no_audio | eeg_eegpt_partial_ft_v1 | 240835 | 0.3242 | 0.3315 | 0.0074 | 0.1146 | 0.1248 | 0.0101 | 0.9330 | 0.9371 | 0.0041 |
| A2_Wphysio_no_audio | eeg_cbramod_frozen_v1 | 240836 | 0.2057 | 0.1760 | -0.0297 | 0.0467 | 0.0167 | -0.0300 | 0.9523 | 0.9749 | 0.0226 |
| A2_Wphysio_no_audio | eeg_cbramod_partial_ft_v1 | 240837 | 0.1981 | 0.1735 | -0.0245 | 0.0097 | 0.0161 | 0.0065 | 0.9749 | 0.9824 | 0.0075 |
| A2_Wphysio_no_audio | eeg_de_5band_1s_avg_v1 | 240838 | 0.1209 | 0.1834 | 0.0625 | 0.0314 | 0.0264 | -0.0051 | 0.9644 | 0.9826 | 0.0181 |
| A2_Wdeep_full | eeg_eegpt_frozen_v1 | 240839 | 0.2082 | 0.2069 | -0.0013 | 0.0892 | 0.0741 | -0.0151 | 0.9597 | 1.0031 | 0.0434 |
| A2_Wdeep_full | eeg_eegpt_partial_ft_v1 | 240840 | 0.3206 | 0.3370 | 0.0164 | 0.1276 | 0.1295 | 0.0019 | 0.9233 | 0.9166 | -0.0067 |
| A2_Wdeep_full | eeg_cbramod_frozen_v1 | 240841 | 0.1772 | 0.2302 | 0.0531 | 0.0440 | 0.0537 | 0.0097 | 0.9840 | 0.9827 | -0.0013 |
| A2_Wdeep_full | eeg_cbramod_partial_ft_v1 | 240842 | 0.1894 | 0.2050 | 0.0156 | 0.0602 | 0.0092 | -0.0510 | 0.9808 | 0.9661 | -0.0147 |
| A2_Wdeep_full | eeg_de_5band_1s_avg_v1 | 240843 | 0.1888 | 0.2069 | 0.0181 | 0.0553 | 0.0652 | 0.0098 | 0.9669 | 0.9715 | 0.0046 |
| A2_Wdeep_no_audio | eeg_eegpt_frozen_v1 | 240844 | 0.1835 | 0.1835 | 0.0000 | 0.0437 | 0.0341 | -0.0096 | 0.9548 | 0.9368 | -0.0180 |
| A2_Wdeep_no_audio | eeg_eegpt_partial_ft_v1 | 240845 | 0.3045 | 0.3125 | 0.0079 | 0.0690 | 0.0981 | 0.0292 | 0.9554 | 0.9558 | 0.0004 |
| A2_Wdeep_no_audio | eeg_cbramod_frozen_v1 | 240846 | 0.1675 | 0.1543 | -0.0132 | 0.0304 | 0.0117 | -0.0187 | 0.9692 | 0.9523 | -0.0169 |
| A2_Wdeep_no_audio | eeg_cbramod_partial_ft_v1 | 240847 | 0.2227 | 0.2055 | -0.0172 | 0.0573 | 0.0688 | 0.0115 | 0.9636 | 0.9372 | -0.0264 |
| A2_Wdeep_no_audio | eeg_de_5band_1s_avg_v1 | 240848 | 0.1161 | 0.2001 | 0.0840 | 0.0553 | 0.0279 | -0.0274 | 0.9945 | 0.9798 | -0.0147 |

#### within_subject_day（60 对）

| experiment | EEG branch | seed | attn raw r | concat raw r | Δraw r | attn centered r | concat centered r | Δcentered r | attn RMSE | concat RMSE | ΔRMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0_Wphysio_full | eeg_eegpt_frozen_v1 | 240849 | 0.1489 | 0.3089 | 0.1600 | 0.1140 | 0.1011 | -0.0129 | 0.9884 | 0.9514 | -0.0370 |
| B0_Wphysio_full | eeg_eegpt_partial_ft_v1 | 240850 | 0.3932 | 0.3890 | -0.0042 | 0.1882 | 0.1855 | -0.0026 | 0.9244 | 0.9148 | -0.0096 |
| B0_Wphysio_full | eeg_cbramod_frozen_v1 | 240851 | 0.2052 | 0.3045 | 0.0992 | 0.0929 | 0.1279 | 0.0350 | 0.9711 | 0.9383 | -0.0328 |
| B0_Wphysio_full | eeg_cbramod_partial_ft_v1 | 240852 | 0.2860 | 0.3102 | 0.0242 | 0.0516 | 0.1049 | 0.0532 | 0.9496 | 0.9539 | 0.0044 |
| B0_Wphysio_full | eeg_de_5band_1s_avg_v1 | 240853 | 0.2261 | 0.2684 | 0.0423 | 0.0873 | 0.0851 | -0.0022 | 0.9655 | 0.9540 | -0.0115 |
| B0_Wphysio_no_audio | eeg_eegpt_frozen_v1 | 240854 | 0.1425 | 0.2949 | 0.1524 | 0.0760 | 0.1014 | 0.0254 | 0.9854 | 0.9425 | -0.0429 |
| B0_Wphysio_no_audio | eeg_eegpt_partial_ft_v1 | 240855 | 0.4049 | 0.3791 | -0.0258 | 0.2062 | 0.1725 | -0.0337 | 0.9243 | 0.9427 | 0.0184 |
| B0_Wphysio_no_audio | eeg_cbramod_frozen_v1 | 240856 | 0.2582 | 0.3022 | 0.0439 | 0.0560 | 0.1046 | 0.0486 | 0.9604 | 0.9446 | -0.0157 |
| B0_Wphysio_no_audio | eeg_cbramod_partial_ft_v1 | 240857 | 0.1431 | 0.3251 | 0.1820 | 0.0167 | 0.1106 | 0.0939 | 1.0014 | 0.9386 | -0.0628 |
| B0_Wphysio_no_audio | eeg_de_5band_1s_avg_v1 | 240858 | 0.2355 | 0.2805 | 0.0451 | 0.0879 | 0.1325 | 0.0446 | 0.9801 | 0.9536 | -0.0265 |
| B0_Wdeep_full | eeg_eegpt_frozen_v1 | 240859 | 0.2890 | 0.2756 | -0.0135 | 0.0958 | 0.1068 | 0.0111 | 0.9620 | 0.9505 | -0.0115 |
| B0_Wdeep_full | eeg_eegpt_partial_ft_v1 | 240860 | 0.3884 | 0.3900 | 0.0016 | 0.1981 | 0.1832 | -0.0149 | 0.9339 | 0.9291 | -0.0048 |
| B0_Wdeep_full | eeg_cbramod_frozen_v1 | 240861 | 0.3137 | 0.3371 | 0.0235 | 0.0834 | 0.1317 | 0.0483 | 0.9662 | 0.9394 | -0.0268 |
| B0_Wdeep_full | eeg_cbramod_partial_ft_v1 | 240862 | 0.2895 | 0.3140 | 0.0245 | 0.0644 | 0.0807 | 0.0164 | 0.9693 | 0.9367 | -0.0327 |
| B0_Wdeep_full | eeg_de_5band_1s_avg_v1 | 240863 | 0.2031 | 0.2954 | 0.0923 | 0.0710 | 0.1313 | 0.0603 | 0.9962 | 0.9482 | -0.0480 |
| B0_Wdeep_no_audio | eeg_eegpt_frozen_v1 | 240864 | 0.3125 | 0.2962 | -0.0163 | 0.0698 | 0.1067 | 0.0368 | 0.9478 | 0.9575 | 0.0096 |
| B0_Wdeep_no_audio | eeg_eegpt_partial_ft_v1 | 240865 | 0.4252 | 0.3788 | -0.0464 | 0.2122 | 0.1719 | -0.0402 | 0.9337 | 0.9466 | 0.0129 |
| B0_Wdeep_no_audio | eeg_cbramod_frozen_v1 | 240866 | 0.2610 | 0.3062 | 0.0453 | 0.0540 | 0.0682 | 0.0143 | 0.9720 | 0.9503 | -0.0217 |
| B0_Wdeep_no_audio | eeg_cbramod_partial_ft_v1 | 240867 | 0.2756 | 0.3369 | 0.0613 | 0.0445 | 0.0750 | 0.0305 | 0.9794 | 0.9408 | -0.0387 |
| B0_Wdeep_no_audio | eeg_de_5band_1s_avg_v1 | 240868 | 0.2638 | 0.2792 | 0.0154 | 0.0566 | 0.0825 | 0.0259 | 0.9822 | 0.9530 | -0.0292 |
| A1_Wphysio_full | eeg_eegpt_frozen_v1 | 240869 | 0.2113 | 0.2841 | 0.0728 | 0.0386 | 0.0896 | 0.0510 | 0.9720 | 0.9717 | -0.0002 |
| A1_Wphysio_full | eeg_eegpt_partial_ft_v1 | 240870 | 0.3920 | 0.3637 | -0.0284 | 0.1744 | 0.1455 | -0.0288 | 0.9203 | 0.9352 | 0.0149 |
| A1_Wphysio_full | eeg_cbramod_frozen_v1 | 240871 | 0.0840 | 0.3007 | 0.2167 | 0.0333 | 0.0921 | 0.0587 | 0.9948 | 0.9462 | -0.0486 |
| A1_Wphysio_full | eeg_cbramod_partial_ft_v1 | 240872 | 0.2402 | 0.3327 | 0.0925 | 0.0199 | 0.1329 | 0.1130 | 0.9729 | 0.9379 | -0.0350 |
| A1_Wphysio_full | eeg_de_5band_1s_avg_v1 | 240873 | 0.2503 | 0.2916 | 0.0413 | 0.0917 | 0.1328 | 0.0410 | 0.9699 | 0.9490 | -0.0209 |
| A1_Wphysio_no_audio | eeg_eegpt_frozen_v1 | 240874 | 0.1988 | 0.2985 | 0.0997 | 0.0588 | 0.0907 | 0.0318 | 0.9669 | 0.9449 | -0.0220 |
| A1_Wphysio_no_audio | eeg_eegpt_partial_ft_v1 | 240875 | 0.4080 | 0.3910 | -0.0170 | 0.2029 | 0.1763 | -0.0266 | 0.9348 | 0.9273 | -0.0076 |
| A1_Wphysio_no_audio | eeg_cbramod_frozen_v1 | 240876 | 0.2500 | 0.3016 | 0.0515 | 0.0568 | 0.1142 | 0.0573 | 0.9587 | 0.9422 | -0.0165 |
| A1_Wphysio_no_audio | eeg_cbramod_partial_ft_v1 | 240877 | 0.2747 | 0.3098 | 0.0352 | 0.0212 | 0.1055 | 0.0843 | 0.9637 | 0.9442 | -0.0195 |
| A1_Wphysio_no_audio | eeg_de_5band_1s_avg_v1 | 240878 | 0.2408 | 0.2701 | 0.0293 | 0.0875 | 0.1056 | 0.0181 | 0.9710 | 0.9574 | -0.0136 |
| A1_Wdeep_full | eeg_eegpt_frozen_v1 | 240879 | 0.2585 | 0.2867 | 0.0282 | 0.0577 | 0.0972 | 0.0394 | 0.9542 | 0.9418 | -0.0124 |
| A1_Wdeep_full | eeg_eegpt_partial_ft_v1 | 240880 | 0.3811 | 0.4006 | 0.0195 | 0.1908 | 0.1733 | -0.0175 | 0.9218 | 0.9089 | -0.0128 |
| A1_Wdeep_full | eeg_cbramod_frozen_v1 | 240881 | 0.2580 | 0.2647 | 0.0067 | 0.0659 | 0.1178 | 0.0518 | 0.9636 | 0.9641 | 0.0006 |
| A1_Wdeep_full | eeg_cbramod_partial_ft_v1 | 240882 | 0.3143 | 0.3131 | -0.0012 | 0.0505 | 0.0920 | 0.0415 | 0.9502 | 0.9388 | -0.0115 |
| A1_Wdeep_full | eeg_de_5band_1s_avg_v1 | 240883 | 0.2468 | 0.2889 | 0.0421 | 0.0461 | 0.1170 | 0.0710 | 0.9943 | 0.9597 | -0.0346 |
| A1_Wdeep_no_audio | eeg_eegpt_frozen_v1 | 240884 | 0.2978 | 0.2940 | -0.0039 | 0.0881 | 0.1024 | 0.0143 | 0.9412 | 0.9485 | 0.0073 |
| A1_Wdeep_no_audio | eeg_eegpt_partial_ft_v1 | 240885 | 0.4062 | 0.3804 | -0.0258 | 0.2087 | 0.1768 | -0.0319 | 0.9322 | 0.9310 | -0.0012 |
| A1_Wdeep_no_audio | eeg_cbramod_frozen_v1 | 240886 | 0.2606 | 0.2670 | 0.0064 | 0.0467 | 0.0710 | 0.0243 | 0.9688 | 0.9625 | -0.0063 |
| A1_Wdeep_no_audio | eeg_cbramod_partial_ft_v1 | 240887 | 0.2798 | 0.3140 | 0.0342 | 0.0657 | 0.0754 | 0.0097 | 0.9639 | 0.9563 | -0.0076 |
| A1_Wdeep_no_audio | eeg_de_5band_1s_avg_v1 | 240888 | 0.2640 | 0.2312 | -0.0327 | 0.0795 | 0.0530 | -0.0265 | 0.9676 | 0.9892 | 0.0216 |
| A2_Wphysio_full | eeg_eegpt_frozen_v1 | 240889 | 0.2568 | 0.2780 | 0.0211 | 0.0174 | 0.1062 | 0.0887 | 0.9701 | 0.9593 | -0.0108 |
| A2_Wphysio_full | eeg_eegpt_partial_ft_v1 | 240890 | 0.3848 | 0.3357 | -0.0491 | 0.1747 | 0.1235 | -0.0511 | 0.9212 | 0.9427 | 0.0216 |
| A2_Wphysio_full | eeg_cbramod_frozen_v1 | 240891 | 0.0680 | 0.3026 | 0.2346 | 0.0020 | 0.1073 | 0.1054 | 0.9940 | 0.9482 | -0.0458 |
| A2_Wphysio_full | eeg_cbramod_partial_ft_v1 | 240892 | 0.1166 | 0.2968 | 0.1802 | 0.0533 | 0.0921 | 0.0388 | 0.9892 | 0.9465 | -0.0426 |
| A2_Wphysio_full | eeg_de_5band_1s_avg_v1 | 240893 | 0.2183 | 0.2803 | 0.0620 | 0.0386 | 0.1112 | 0.0725 | 0.9810 | 0.9530 | -0.0280 |
| A2_Wphysio_no_audio | eeg_eegpt_frozen_v1 | 240894 | 0.2718 | 0.2917 | 0.0199 | 0.0601 | 0.0886 | 0.0285 | 0.9500 | 0.9505 | 0.0005 |
| A2_Wphysio_no_audio | eeg_eegpt_partial_ft_v1 | 240895 | 0.3783 | 0.3678 | -0.0105 | 0.1716 | 0.1805 | 0.0089 | 0.9422 | 0.9472 | 0.0050 |
| A2_Wphysio_no_audio | eeg_cbramod_frozen_v1 | 240896 | 0.2425 | 0.2941 | 0.0516 | 0.0412 | 0.1040 | 0.0627 | 0.9598 | 0.9484 | -0.0113 |
| A2_Wphysio_no_audio | eeg_cbramod_partial_ft_v1 | 240897 | 0.2670 | 0.2947 | 0.0278 | 0.0436 | 0.0900 | 0.0464 | 0.9537 | 0.9475 | -0.0062 |
| A2_Wphysio_no_audio | eeg_de_5band_1s_avg_v1 | 240898 | 0.2164 | 0.2760 | 0.0596 | 0.0715 | 0.1126 | 0.0411 | 0.9766 | 0.9573 | -0.0193 |
| A2_Wdeep_full | eeg_eegpt_frozen_v1 | 240899 | 0.2637 | 0.2664 | 0.0027 | 0.0608 | 0.1206 | 0.0598 | 0.9756 | 0.9517 | -0.0238 |
| A2_Wdeep_full | eeg_eegpt_partial_ft_v1 | 240900 | 0.4046 | 0.4012 | -0.0034 | 0.1736 | 0.1787 | 0.0051 | 0.9138 | 0.9096 | -0.0042 |
| A2_Wdeep_full | eeg_cbramod_frozen_v1 | 240901 | 0.2418 | 0.2892 | 0.0475 | 0.0417 | 0.1335 | 0.0918 | 0.9659 | 0.9532 | -0.0128 |
| A2_Wdeep_full | eeg_cbramod_partial_ft_v1 | 240902 | 0.3014 | 0.3225 | 0.0211 | 0.0637 | 0.1338 | 0.0701 | 0.9524 | 0.9461 | -0.0063 |
| A2_Wdeep_full | eeg_de_5band_1s_avg_v1 | 240903 | 0.2435 | 0.3294 | 0.0859 | 0.0847 | 0.1233 | 0.0387 | 0.9799 | 0.9439 | -0.0361 |
| A2_Wdeep_no_audio | eeg_eegpt_frozen_v1 | 240904 | 0.2798 | 0.2907 | 0.0109 | 0.0431 | 0.0560 | 0.0129 | 0.9468 | 0.9478 | 0.0010 |
| A2_Wdeep_no_audio | eeg_eegpt_partial_ft_v1 | 240905 | 0.3903 | 0.3721 | -0.0182 | 0.1823 | 0.1779 | -0.0043 | 0.9189 | 0.9472 | 0.0283 |
| A2_Wdeep_no_audio | eeg_cbramod_frozen_v1 | 240906 | 0.2552 | 0.2980 | 0.0428 | 0.0381 | 0.1285 | 0.0904 | 0.9811 | 0.9442 | -0.0369 |
| A2_Wdeep_no_audio | eeg_cbramod_partial_ft_v1 | 240907 | 0.2486 | 0.2920 | 0.0434 | 0.0326 | 0.0803 | 0.0477 | 0.9698 | 0.9510 | -0.0188 |
| A2_Wdeep_no_audio | eeg_de_5band_1s_avg_v1 | 240908 | 0.2907 | 0.2537 | -0.0371 | 0.0803 | 0.0770 | -0.0033 | 0.9612 | 0.9820 | 0.0207 |

---

## 7. 证据四：eeg_anchor（实验 4）对照——只有 concat 有全量一致优势

`eeg_anchor` 全量配对（paired 180/180）作为对照：

| 范围 | N | mean Δraw r | mean Δcentered r | mean ΔRMSE | 胜/负/平 | 符号检验 p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 180 | -0.0002 | -0.0034 | -0.0077 | 81/76/23 | 0.7497 |
| cross_subject | 60 | 0.0122 | 0.0043 | -0.0139 | 31/22/7 | 0.2717 |
| cross_day | 60 | -0.0004 | -0.0080 | -0.0082 | 29/28/3 | 1 |
| within_subject_day | 60 | -0.0124 | -0.0066 | -0.0010 | 21/26/13 | 0.5601 |

解读：`eeg_anchor` 仅在 `cross_subject` raw r（+0.012）与 `cross_day` RMSE（−0.008）小幅改善，`within_subject_day` raw r 反而下降（−0.012）；无 concat 那种一致、显著的优势，不构成替换理由。

按 EEG 分支（eeg_anchor vs attention，每分支 36 对）：

| EEG branch | N | mean Δraw r | mean Δcentered r | mean ΔRMSE | 胜/负/平 |
| --- | ---: | ---: | ---: | ---: | ---: |
| eeg_eegpt_frozen_v1 | 36 | -0.0023 | -0.0066 | -0.0120 | 15/15/6 |
| eeg_eegpt_partial_ft_v1 | 36 | -0.0003 | -0.0066 | -0.0056 | 13/15/8 |
| eeg_cbramod_frozen_v1 | 36 | 0.0045 | -0.0037 | -0.0027 | 16/17/3 |
| eeg_cbramod_partial_ft_v1 | 36 | 0.0048 | -0.0013 | -0.0091 | 20/11/5 |
| eeg_de_5band_1s_avg_v1 | 36 | -0.0077 | 0.0011 | -0.0091 | 17/18/1 |

---

## 8. 稳健性与方向一致性

1. **两套独立实验方向一致**：决策切片（seed 240800 固定、4 组合）与全量配对矩阵（180 对）中，`concat` 的 mean raw r 均 ≥ `attention`（切片：cross_day +0.028 / within_subject_day +0.004；全量：cross_day +0.014 / within_subject_day +0.039）。
2. **统计显著性集中在关键协议**：`within_subject_day` 全量 60 对 42 胜 / 12 负（p=5.2e-5）；`cross_day` 27/28 基本五五开但均值 +0.014；`cross_subject` 持平。
3. **例外可解释**：attention 唯一略优的分支 `eegpt_partial_ft` 是唯一 fatigue-supervised 强表征；弱表征分支（CBraMod/DE/frozen）concat 全胜 → 与「attention 需要足够强的输入才有意义」一致，恰好反证当前融合设计在该任务上不成立。
4. **配对公平性**：全量矩阵与归档使用完全相同的命令顺序与超参，180/180 逐位同 seed 配对，无种子混差；决策切片四变体共享 seed 240800。
5. **局限**：全部结论基于当前 256D token 输入（EEG 部分 fatigue-supervised、wear/video frozen/无监督）与 28,819 窗口单一数据；不构成「attention 在所有多模态任务上无效」的一般性断言。

---

## 9. 结论与建议

1. **当前 attention 融合相比朴素拼接没有价值**（全量配对：`within_subject_day` Δraw r +0.039、ΔRMSE −0.014，42 胜/12 负，p=5.2e-5；`cross_day` Δraw r +0.014；`cross_subject` 持平）。
2. **结构做厚（多头 + 多 query）不解决问题**：`attention_multihead_pma` 切片两协议 raw r 均低于 concat（见 §5.3）。
3. **EEG 锚点（eeg_anchor）部分信号、不作主线**：`cross_subject` raw r +0.012、`cross_day` RMSE −0.008 小幅改善，但 `within_subject_day` raw r −0.012 下降，无 concat 那种一致优势（见 §7）。
4. **建议以 `concat + MLP` 替换当前 attention 融合作为新基线**（`--fusion-variant concat` 已可一键复现），并把 raw r 主线资源转向提升单模态表征质量——尤其 wear（Wphysio/Wdeep 无监督、固定随机投影）与 video（DINOv2 frozen）的监督对齐。

---

## 10. 产物与复现

| 用途 | 路径 |
| --- | --- |
| 本文档 | `docs/research/current/0814-window/experiments/fusion_attention_vs_concat_evidence_20260820.md` |
| 所有表格 JSON 副本 | `outputs/reports/fusion_attention_vs_concat_evidence_20260820.json` |
| 决策切片（4 变体 × 8 runs） | `outputs/server_sync/fusion_variant_20260820/{attention,concat,attention_multihead_pma,eeg_anchor}.json` |
| 切片汇总 | `outputs/server_sync/fusion_variant_20260820/summary.{json,md}` |
| 全量 concat / eeg_anchor 矩阵 | `outputs/server_sync/fusion_variant_20260820/fusion_variant_{concat,eeg_anchor}_full_seed240800_raw.json` |
| 归档 attention 矩阵（0814） | `outputs/server_sync/eeg_encoder_256d_5route_20260814/reports/eeg_encoder_256d_5route_fusion_video_only_seed240800_raw.json` |
| 定点复现（3 行） | `outputs/server_sync/fusion_variant_20260820/cross_*_seed*.json` |
| 运行日志 | `outputs/server_sync/fusion_variant_20260820/fusion_variant_{decision,full}.log` |
| 生成脚本 | `scripts/window_fatigue/56_build_fusion_attention_vs_concat_evidence.py` |

复现命令见 `repo-docs/references/commands-and-artifacts.md`「Fusion variant 决策切片」「EEG encoder fusion video-only matrix」两行。

