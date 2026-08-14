# 当前多模态疲劳预测技术路线总结

## 任务与输入

当前任务是基于 EEG 对齐后的多模态 10 秒窗口预测 `fatigue`。主数据口径为：

- 样本数：`28819` 个 10 秒窗口。
- EEG 原始输入：`X.npy`，shape 为 `(28819, 2000, 59)`，即每个窗口 10 秒、200 Hz、59 通道。
- 标签：`fatigue`，来自对齐后的窗口索引和 `y.npy`。
- 评估协议：`cross_day` 和 `within_subject_day` 作为主协议；`cross_subject` 保留为诊断协议。
- 划分规则：训练集使用 `pretrain + finetune`，验证集使用 `val` 做早停和模型选择，最终指标只在 `test` split 上计算。

每个样本在融合阶段被表示为最多四个 256D modality token：

| 模态 | 输入数据类型 | 融合 token |
| --- | --- | --- |
| EEG | 59 通道 EEG，200 Hz，10 秒窗口 | `eeg_emb (N,256)` |
| Wear | PPG、GSR、三轴 ACC | `wear_emb (N,256)` |
| Video | 2 倍主脸 ROI 视频窗口 | `video_emb (N,256)` |
| Audio | 视频音轨切出的音频窗口 / openSMILE 特征 | `audio_emb (N,256)` |

`modality_mask` 的顺序为 `[eeg, wear, video, audio]`，用于在融合时屏蔽缺失模态。

## 单模态 Embedding

### EEG

当前 EEG 接入口径已经改为完整 256D EEG embedding，而不是 EEG-only 预测分数。最新矩阵包含五条 EEG 表征路线：

| EEG route | 方法 | 监督边界 |
| --- | --- | --- |
| `eegpt_frozen_v1` | 复用现有 EEGPT frozen 256D embedding | frozen baseline |
| `eegpt_partial_ft_v1` | EEGPT 端到端回归训练，解冻最后若干 transformer block、norm、projection/head；取 encoder pooled hidden state，经 256D projection 输出 | fatigue-supervised，仅用对应 protocol 的 train/val |
| `cbramod_frozen_v1` | CBraMod encoder frozen，encoder hidden state pooling 后接 256D projection/head | fatigue-supervised projection/head，仅用 train/val |
| `cbramod_partial_ft_v1` | CBraMod partial fine-tune，解冻后部 block、norm、projection/head；输出 256D pooled embedding | fatigue-supervised，仅用 train/val |
| `eeg_de_5band_1s_avg_v1` | 五频带 DE 特征 `[1,4), [4,8), [8,13), [13,30), [30,45)`；每通道每秒提取后 10 秒平均，得到 295D，再由 MLP 倒数 256D projection 输出 | fatigue-supervised MLP，仅用 train/val |

当前结果中，`eegpt_partial_ft_v1` 是主线最强 EEG 表征。它在 EEG-only 上达到：

| protocol | RMSE | raw r | centered r |
| --- | ---: | ---: | ---: |
| `cross_day` | 0.9272 | 0.2741 | 0.1005 |
| `within_subject_day` | 0.9270 | 0.3749 | 0.1489 |

### Wear

Wear 输入包含 PPG、GSR 和三轴 ACC。当前融合矩阵使用两类 256D wearable embedding：

| Wear route | 方法 |
| --- | --- |
| `Wphysio` | 从 PPG 提取 HR/HRV，从 GSR 提取 slope/SCR，从 ACC 提取 motion/stationary 等可解释生理与运动特征，再投影为 256D |
| `Wdeep` | 将 PPG/GSR/ACC 重采样并组成序列，经固定随机 1D convolution / TCN-like 特征提取、池化统计和固定 256D projection 输出 |

`Wdeep` 本身不是监督训练的 wearable encoder；监督训练发生在最终融合回归器中。

### Video

Video 使用对齐到 10 秒窗口的 2 倍主脸 ROI clip，并由 DINOv2-Base frozen encoder 提取 256D 视频表示。当前视频候选为：

| Video route | 方法 |
| --- | --- |
| `B0` | 2 倍主脸 ROI DINOv2 embedding |
| `A1` | 2 倍主脸 ROI DINOv2 embedding，加入轻量颜色/亮度增强 |
| `A2` | 2 倍主脸 ROI DINOv2 embedding，在 A1 基础上加入 grayscale 增强 |

这些 video embedding 是内容表征，用于后续多模态疲劳预测。

### Audio

Audio 当前使用 openSMILE eGeMAPS Functionals。流程是先从视频音轨按窗口切出音频，再计算 eGeMAPS functionals，并投影为 256D `audio_emb`。在完整四模态配置中，audio token 与 EEG、Wear、Video token 一起进入融合器。

## Cross-Attention 融合

当前融合器是轻量 modality-token attention regression：

1. 将每个可用模态表示为 256D token：EEG、Wear、Video、Audio。
2. 每个 token 先经过 `Linear(256 -> hidden_dim)`，当前 `hidden_dim=128`。
3. 加入 learnable modality embedding。
4. 使用单头 `MultiheadAttention` 在模态 token 之间建模互补关系。
5. 使用 learnable query 对 attention 后的 token 做加权 pooling。
6. pooling 后进入 `LayerNorm + MLP` 回归头，输出 fatigue 预测。

训练配置为 AdamW，学习率 `1e-3`，weight decay `1e-4`，batch size `256`，最多 `80` epoch，patience `15`，dropout `0.1`。所有 normalization 都只在 train split 上拟合。

## 最新实验表现

最新结果来自 `eeg_encoder_256d_5route_fusion_video_only_seed240800_raw`：

- EEG 256D matrix：`15` runs。
- 四模态融合 matrix：`180` runs。
- EEG routes：5 条。
- Video routes：`B0/A1/A2`。
- Wear routes：`Wphysio/Wdeep`。
- Fusion 设置：含视频路线，包含 full 四模态和无音频可用时的 EEG+Wear+Video 配置。

### 主协议最佳结果

| protocol | 选择标准 | EEG route | fusion route | RMSE | raw r | centered r |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `cross_day` | 最低 RMSE | EEGPT partial FT | `B0_Wphysio_no_audio` | 0.9189 | 0.3301 | 0.1526 |
| `cross_day` | 最高 raw r | EEGPT partial FT | `B0_Wphysio_full` | 0.9481 | 0.3504 | 0.1176 |
| `within_subject_day` | 最低 RMSE | EEGPT partial FT | `A2_Wdeep_full` | 0.9138 | 0.4046 | 0.1736 |
| `within_subject_day` | 最高 raw r | EEGPT partial FT | `B0_Wdeep_no_audio` | 0.9337 | 0.4252 | 0.2122 |

### 按 EEG route 的融合平均表现

| protocol | EEG route | mean RMSE | mean raw r | mean centered r |
| --- | --- | ---: | ---: | ---: |
| `cross_day` | EEGPT frozen | 0.9718 | 0.1576 | 0.0561 |
| `cross_day` | EEGPT partial FT | 0.9497 | 0.3012 | 0.1115 |
| `cross_day` | CBraMod frozen | 0.9638 | 0.1836 | 0.0544 |
| `cross_day` | CBraMod partial FT | 0.9751 | 0.1664 | 0.0464 |
| `cross_day` | DE+MLP | 0.9855 | 0.1646 | 0.0428 |
| `within_subject_day` | EEGPT frozen | 0.9634 | 0.2443 | 0.0650 |
| `within_subject_day` | EEGPT partial FT | 0.9268 | 0.3964 | 0.1903 |
| `within_subject_day` | CBraMod frozen | 0.9714 | 0.2249 | 0.0510 |
| `within_subject_day` | CBraMod partial FT | 0.9680 | 0.2531 | 0.0440 |
| `within_subject_day` | DE+MLP | 0.9771 | 0.2416 | 0.0736 |

## 当前结论

当前整条技术路线可以概括为：

`10s EEG/Wear/Video/Audio aligned windows -> 每模态 256D embedding -> modality-token cross-attention -> fatigue regression`

其中 EEG 分支是当前性能提升的主要来源。完整 256D EEG embedding 接入口径下，`EEGPT partial FT` 在 EEG-only 和多模态融合中都保持领先；CBraMod 和 DE+MLP 目前作为对照与补充分析路线保留。融合层面，轻量 cross-attention 能在不同 Wear 和 Video 表征之间选择互补信息，当前主协议最强结果集中在 `EEGPT partial FT + Wphysio/Wdeep + B0/A2 video` 的组合上。

主要结果文件：

- `G:\Daily Multimodal\outputs\server_sync\eeg_encoder_256d_5route_20260814\reports\eeg_encoder_256d_5route_matrix_seed240800_20260814.json`
- `G:\Daily Multimodal\outputs\server_sync\eeg_encoder_256d_5route_20260814\reports\eeg_encoder_256d_5route_fusion_video_only_seed240800_raw.json`
- `G:\Daily Multimodal\outputs\server_sync\eeg_encoder_256d_5route_20260814\eeg_encoder_256d_5route_results_summary.md`
