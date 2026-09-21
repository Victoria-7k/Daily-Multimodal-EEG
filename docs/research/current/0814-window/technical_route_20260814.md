# 当前多模态疲劳预测技术路线总结

> Status: Current parallel route
> Route role: 0814 window；与 0906 EMA-bag 并列使用
> Research index: [研究文档索引](../../README.md)

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
| `Wmoment_frozen` | MOMENT-1-small 冻结 + 可学习 256D 投影头（fatigue 监督，train/val；2026-08-20 新增） |
| `Wmoment_partial_ft` | MOMENT-1-small 解冻最后 2 个 transformer block + final norm + 投影头（fatigue 监督；2026-08-20 新增） |

`Wdeep` 本身不是监督训练的 wearable encoder；监督训练发生在最终融合回归器中。`Wmoment_*` 两档是 fatigue-supervised representation，监督边界与 EEG 侧 partial FT 一致。

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

最新结果来自 `eeg_encoder_256d_5route_fusion_video_only_seed240800_raw`。当前报告使用真正的 256D EEG embedding 接入口径：每条 EEG route 都输出 `eeg_emb (28819,256)`，再与 Wear、Video、Audio 的 256D embedding 一起进入 modality-token attention fusion。

- EEG 256D matrix：`15` runs。
- 四模态融合 matrix：`180` runs。
- 划分协议：`cross_subject`、`cross_day`、`within_subject_day`。
- EEG routes：`EEGPT frozen`、`EEGPT partial FT`、`CBraMod frozen`、`CBraMod partial FT`、`DE+MLP`。
- Fusion routes：`B0/A1/A2` × `Wphysio/Wdeep` × `full/no_audio`，即保留含视频路线，去掉 `no_video` 和 `bio_only`。

### 各划分协议 Raw R Top 3（2026-08-20 更新：含 Wear × MOMENT）

> 口径：方案 C 矩阵（2 EEG × 24 fusion routes × 3 protocols，seed 240729 固定，`--experiment-seed-fixed`）；EEG routes 为 `eegpt_partial_ft_v1` / `eegpt_frozen_v1`，Wear 含 `Wphysio/Wdeep/Wmoment_frozen/Wmoment_ft` 四档。旧 5-EEG × 12-route 的 180-run 矩阵结果仍见 `outputs/server_sync/eeg_encoder_256d_5route_20260814/reports/`（seed 为 run_number 递增口径，与新表不可直接混比）。

| protocol | rank | EEG route | fusion route | raw r | centered r | RMSE | MAE |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `cross_subject` | 1 | eegpt_partial_ft_v1 | `B0_Wphysio_no_audio` | 0.1118 | 0.1256 | 0.9187 | 0.7277 |
| `cross_subject` | 2 | eegpt_frozen_v1 | `A2_Wdeep_full` | 0.0950 | 0.0994 | 0.9211 | 0.7554 |
| `cross_subject` | 3 | eegpt_partial_ft_v1 | `A2_Wmoment_frozen_full` | 0.0899 | 0.1128 | 0.9334 | 0.7387 |
| `cross_day` | 1 | eegpt_partial_ft_v1 | `A1_Wphysio_no_audio` | 0.3554 | 0.1289 | 0.9208 | 0.7190 |
| `cross_day` | 2 | eegpt_partial_ft_v1 | `A1_Wmoment_frozen_full` | 0.3319 | 0.1417 | 0.9159 | 0.7139 |
| `cross_day` | 3 | eegpt_partial_ft_v1 | `A2_Wmoment_ft_full` | 0.3231 | 0.1097 | 0.9122 | 0.7153 |
| `within_subject_day` | 1 | eegpt_partial_ft_v1 | `A1_Wphysio_no_audio` | 0.3985 | 0.1859 | 0.9141 | 0.7054 |
| `within_subject_day` | 2 | eegpt_partial_ft_v1 | `B0_Wphysio_no_audio` | 0.3976 | 0.1981 | 0.9316 | 0.7111 |
| `within_subject_day` | 3 | eegpt_partial_ft_v1 | `B0_Wmoment_frozen_no_audio` | 0.3957 | 0.1796 | 0.9209 | 0.7182 |

### 按 EEG Route 分类的四模态平均表现（2026-08-20 更新）

> 口径同 Top 3：每个 EEG route 在 24 条 fusion route（3 video × 4 wear × 2 audio）上对 test 指标取平均。

| protocol | EEG route | mean RMSE | mean MAE | mean raw r | mean centered r |
| --- | --- | ---: | ---: | ---: | ---: |
| `cross_subject` | `eegpt_partial_ft_v1` | 0.9686 | 0.7726 | 0.0435 | 0.0670 |
| `cross_subject` | `eegpt_frozen_v1` | 0.9207 | 0.7383 | 0.0344 | 0.0396 |
| `cross_day` | `eegpt_partial_ft_v1` | 0.9432 | 0.7393 | 0.2872 | 0.0978 |
| `cross_day` | `eegpt_frozen_v1` | 0.9574 | 0.7423 | 0.2036 | 0.0459 |
| `within_subject_day` | `eegpt_partial_ft_v1` | 0.9264 | 0.7218 | 0.3846 | 0.1776 |
| `within_subject_day` | `eegpt_frozen_v1` | 0.9569 | 0.7396 | 0.2567 | 0.0692 |

### EEG-Only 全部结果

| protocol | EEG route | RMSE | MAE | raw r | centered r | best epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `cross_subject` | CBraMod frozen | 0.9430 | 0.7746 | 0.0581 | 0.0965 | 6 |
| `cross_subject` | CBraMod partial FT | 0.9605 | 0.8046 | 0.0680 | 0.0778 | 1 |
| `cross_subject` | DE+MLP | 0.9397 | 0.7040 | -0.0002 | -0.0039 | 15 |
| `cross_subject` | EEGPT frozen | 0.9108 | 0.7080 | 0.0300 | 0.0318 | 13 |
| `cross_subject` | EEGPT partial FT | 0.9861 | 0.7817 | 0.0517 | 0.0819 | 7 |
| `cross_day` | CBraMod frozen | 0.9559 | 0.7071 | 0.0401 | 0.0479 | 2 |
| `cross_day` | CBraMod partial FT | 0.9517 | 0.7346 | 0.0431 | 0.0421 | 1 |
| `cross_day` | DE+MLP | 0.9518 | 0.6994 | 0.0702 | 0.0392 | 12 |
| `cross_day` | EEGPT frozen | 0.9566 | 0.6897 | 0.0526 | 0.0244 | 3 |
| `cross_day` | EEGPT partial FT | 0.9272 | 0.7139 | 0.2741 | 0.1005 | 2 |
| `within_subject_day` | CBraMod frozen | 0.9880 | 0.7710 | 0.0829 | 0.0508 | 11 |
| `within_subject_day` | CBraMod partial FT | 1.0120 | 0.8164 | 0.1487 | 0.0142 | 3 |
| `within_subject_day` | DE+MLP | 1.3517 | 0.8062 | -0.0411 | -0.0374 | 3 |
| `within_subject_day` | EEGPT frozen | 0.9910 | 0.7300 | 0.0905 | 0.0577 | 15 |
| `within_subject_day` | EEGPT partial FT | 0.9270 | 0.7360 | 0.3749 | 0.1489 | 19 |

## 当前结论

当前整条技术路线可以概括为：

`10s EEG/Wear/Video/Audio aligned windows -> 每模态 256D embedding -> modality-token cross-attention -> fatigue regression`

三个划分协议下，raw r 排名前列的组合都集中在 `EEGPT partial FT` 分支，说明当前 256D EEG embedding 的主要收益来自 EEGPT 局部微调。融合层面，强组合主要分布在 `EEGPT partial FT + Wphysio/Wdeep + B0/A1/A2 video`，其中 `full` 与 `no_audio` 都能进入前三，表明视频与 wearable 分支是当前稳定贡献来源，audio 是否加入需要按具体协议和路线配对判断。

主要结果文件：

- `G:\Daily Multimodal\outputs\server_sync\eeg_encoder_256d_5route_20260814\reports\eeg_encoder_256d_5route_matrix_seed240800_20260814.json`
- `G:\Daily Multimodal\outputs\server_sync\eeg_encoder_256d_5route_20260814\reports\eeg_encoder_256d_5route_fusion_video_only_seed240800_raw.json`
- `G:\Daily Multimodal\outputs\server_sync\eeg_encoder_256d_5route_20260814\eeg_encoder_256d_5route_results_summary.md`

## 2026-08-20 更新：Wear × MOMENT

Wear 侧新增两条 MOMENT-1-small 预训练路线（见上表），在 `EEGPT partial FT + A1 + full` 固定配置下做 3-seed paired 实验（融合 seed 240729/240730/240731 × wear token seed 240800/240801/240802，`--experiment-seed-fixed`），结果详见 [wear_moment_results_20260820.md](../../supporting/wear/wear_moment_results_20260820.md)：

- **`Wmoment_frozen` 建议纳入主线候选**：`cross_day` 上 3/3 seeds 一致优于 `Wdeep`（raw r Δ +0.063、RMSE Δ -0.039），`cross_subject` 稳定更优（raw r Δ +0.035、RMSE Δ -0.021），`within_subject_day` 持平（raw r Δ -0.008、RMSE Δ 0.000）；相对 `Wphysio` 三协议均更优。典型数值（`A1_Wmoment_frozen_full` vs `A1_Wdeep_full`，seed 240729）：`cross_day` raw r 0.3319 / RMSE 0.9159（同配置 `Wdeep` 0.2579 / 0.9479）。
  - 方案 C 全维度扩展（144 runs，2 EEG × 24 routes × 3 protocols，seed 240729 固定）进一步确认：主线 EEG（`eegpt_partial_ft_v1`）× `cross_day` 下 frozen 相对 Wdeep 在**全部 6 个 video/audio 配置上都更优**（raw r Δ +0.033~+0.087、RMSE 全部更低），`within_subject_day` 上 RMSE 6/6 更低；低监督 EEG（`eegpt_frozen_v1`）下增益不稳定；推荐主线组合 `eeg_eegpt_partial_ft_v1 + Wmoment_frozen`（cross_day 最佳 `A2_Wmoment_frozen_full`）。
- **`Wmoment_partial_ft` 保留为诊断/可选**：仅在 `cross_subject` 上 3/3 seeds 优于 frozen，两个主协议上不稳定（方案 C 36 配置组方向混杂），不作为默认路线。
- 监督边界：两条 route 的投影头/微调均为 fatigue-supervised（train/val），mask 复用现有 `wear_mask`（24,127/28,819）；MOMENT-1-small 权重经 hf-mirror 获取（repo sha `411e2882`）。

## 附表：全部四模态融合实验结果（2026-08-20 更新）

> 口径同 Top 3：方案 C 矩阵全部 144 runs（2 EEG × 24 fusion routes × 3 protocols，seed 240729 固定），按 protocol 分组、组内按 raw r 从高到低排序。

| protocol | rank in protocol | EEG route | fusion route | RMSE | MAE | raw r | centered r | per-subject r mean | best epoch |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cross_subject` | 1 | eegpt_partial_ft_v1 | `B0_Wphysio_no_audio` | 0.9187 | 0.7277 | 0.1118 | 0.1256 | 0.1435 | 1 |
| `cross_subject` | 2 | eegpt_frozen_v1 | `A2_Wdeep_full` | 0.9211 | 0.7554 | 0.0950 | 0.0994 | 0.0755 | 12 |
| `cross_subject` | 3 | eegpt_partial_ft_v1 | `A2_Wmoment_frozen_full` | 0.9334 | 0.7387 | 0.0899 | 0.1128 | 0.1189 | 8 |
| `cross_subject` | 4 | eegpt_frozen_v1 | `A1_Wmoment_ft_no_audio` | 0.9085 | 0.7298 | 0.0806 | 0.0713 | 0.0737 | 3 |
| `cross_subject` | 5 | eegpt_frozen_v1 | `A1_Wmoment_frozen_no_audio` | 0.9042 | 0.7195 | 0.0797 | 0.0662 | 0.0731 | 1 |
| `cross_subject` | 6 | eegpt_frozen_v1 | `A1_Wdeep_no_audio` | 0.9186 | 0.7365 | 0.0793 | 0.0440 | 0.0476 | 12 |
| `cross_subject` | 7 | eegpt_frozen_v1 | `B0_Wmoment_frozen_no_audio` | 0.9091 | 0.7393 | 0.0777 | 0.0565 | 0.0624 | 1 |
| `cross_subject` | 8 | eegpt_partial_ft_v1 | `B0_Wphysio_full` | 0.9321 | 0.7392 | 0.0769 | 0.1010 | 0.1113 | 8 |
| `cross_subject` | 9 | eegpt_partial_ft_v1 | `A1_Wmoment_ft_full` | 0.9303 | 0.7394 | 0.0761 | 0.0722 | 0.0808 | 12 |
| `cross_subject` | 10 | eegpt_partial_ft_v1 | `B0_Wdeep_no_audio` | 0.9823 | 0.7902 | 0.0731 | 0.1090 | 0.1297 | 4 |
| `cross_subject` | 11 | eegpt_partial_ft_v1 | `A1_Wmoment_frozen_full` | 0.9323 | 0.7355 | 0.0695 | 0.0922 | 0.0999 | 8 |
| `cross_subject` | 12 | eegpt_partial_ft_v1 | `B0_Wmoment_frozen_full` | 0.9468 | 0.7513 | 0.0676 | 0.1005 | 0.1101 | 8 |
| `cross_subject` | 13 | eegpt_frozen_v1 | `A1_Wdeep_full` | 0.9378 | 0.7581 | 0.0656 | 0.0386 | 0.0288 | 16 |
| `cross_subject` | 14 | eegpt_frozen_v1 | `A2_Wmoment_frozen_no_audio` | 0.9092 | 0.7357 | 0.0644 | 0.0343 | 0.0369 | 1 |
| `cross_subject` | 15 | eegpt_frozen_v1 | `A2_Wmoment_frozen_full` | 0.9042 | 0.7191 | 0.0638 | 0.0545 | 0.0499 | 8 |
| `cross_subject` | 16 | eegpt_frozen_v1 | `A2_Wmoment_ft_no_audio` | 0.9269 | 0.7499 | 0.0614 | 0.0630 | 0.0593 | 3 |
| `cross_subject` | 17 | eegpt_partial_ft_v1 | `A1_Wphysio_no_audio` | 0.9801 | 0.7802 | 0.0601 | 0.0706 | 0.0772 | 14 |
| `cross_subject` | 18 | eegpt_partial_ft_v1 | `A1_Wdeep_no_audio` | 0.9880 | 0.7954 | 0.0558 | 0.0966 | 0.1217 | 4 |
| `cross_subject` | 19 | eegpt_frozen_v1 | `B0_Wdeep_full` | 0.9256 | 0.7446 | 0.0554 | 0.0161 | -0.0009 | 11 |
| `cross_subject` | 20 | eegpt_partial_ft_v1 | `A1_Wmoment_frozen_no_audio` | 0.9656 | 0.7781 | 0.0548 | 0.0644 | 0.0710 | 8 |
| `cross_subject` | 21 | eegpt_frozen_v1 | `B0_Wmoment_frozen_full` | 0.9076 | 0.7246 | 0.0545 | 0.0739 | 0.0762 | 14 |
| `cross_subject` | 22 | eegpt_frozen_v1 | `A2_Wphysio_no_audio` | 0.9071 | 0.7194 | 0.0536 | 0.0561 | 0.0587 | 8 |
| `cross_subject` | 23 | eegpt_frozen_v1 | `B0_Wmoment_ft_no_audio` | 0.9149 | 0.7367 | 0.0475 | 0.0423 | 0.0405 | 3 |
| `cross_subject` | 24 | eegpt_partial_ft_v1 | `B0_Wmoment_frozen_no_audio` | 0.9757 | 0.7840 | 0.0460 | 0.0477 | 0.0534 | 8 |
| `cross_subject` | 25 | eegpt_frozen_v1 | `B0_Wdeep_no_audio` | 0.9157 | 0.7348 | 0.0448 | 0.0552 | 0.0396 | 10 |
| `cross_subject` | 26 | eegpt_partial_ft_v1 | `A2_Wphysio_full` | 0.9779 | 0.7742 | 0.0418 | 0.0621 | 0.0682 | 9 |
| `cross_subject` | 27 | eegpt_partial_ft_v1 | `A1_Wphysio_full` | 0.9528 | 0.7528 | 0.0409 | 0.0800 | 0.0861 | 8 |
| `cross_subject` | 28 | eegpt_partial_ft_v1 | `A2_Wdeep_no_audio` | 1.0037 | 0.8052 | 0.0383 | 0.0538 | 0.0649 | 14 |
| `cross_subject` | 29 | eegpt_frozen_v1 | `A2_Wdeep_no_audio` | 0.9193 | 0.7409 | 0.0380 | 0.0376 | 0.0305 | 32 |
| `cross_subject` | 30 | eegpt_frozen_v1 | `A1_Wphysio_full` | 0.9311 | 0.7485 | 0.0369 | 0.0665 | 0.0662 | 14 |
| `cross_subject` | 31 | eegpt_frozen_v1 | `A1_Wphysio_no_audio` | 0.9408 | 0.7566 | 0.0367 | 0.0564 | 0.0638 | 6 |
| `cross_subject` | 32 | eegpt_partial_ft_v1 | `A2_Wdeep_full` | 0.9660 | 0.7769 | 0.0272 | 0.0220 | 0.0387 | 14 |
| `cross_subject` | 33 | eegpt_partial_ft_v1 | `B0_Wmoment_ft_no_audio` | 0.9832 | 0.7830 | 0.0270 | 0.0585 | 0.0642 | 6 |
| `cross_subject` | 34 | eegpt_frozen_v1 | `B0_Wmoment_ft_full` | 0.9120 | 0.7194 | 0.0257 | 0.0224 | 0.0251 | 10 |
| `cross_subject` | 35 | eegpt_partial_ft_v1 | `A2_Wmoment_frozen_no_audio` | 1.0040 | 0.8062 | 0.0252 | 0.0493 | 0.0552 | 10 |
| `cross_subject` | 36 | eegpt_partial_ft_v1 | `A2_Wmoment_ft_full` | 0.9773 | 0.7769 | 0.0247 | 0.0659 | 0.0751 | 7 |
| `cross_subject` | 37 | eegpt_frozen_v1 | `A2_Wmoment_ft_full` | 0.9318 | 0.7564 | 0.0226 | 0.0375 | 0.0386 | 12 |
| `cross_subject` | 38 | eegpt_partial_ft_v1 | `B0_Wmoment_ft_full` | 0.9582 | 0.7611 | 0.0208 | 0.0699 | 0.0756 | 6 |
| `cross_subject` | 39 | eegpt_partial_ft_v1 | `A2_Wmoment_ft_no_audio` | 0.9897 | 0.7922 | 0.0207 | 0.0395 | 0.0457 | 8 |
| `cross_subject` | 40 | eegpt_partial_ft_v1 | `A2_Wphysio_no_audio` | 0.9783 | 0.7800 | 0.0189 | 0.0387 | 0.0437 | 10 |
| `cross_subject` | 41 | eegpt_partial_ft_v1 | `B0_Wdeep_full` | 0.9774 | 0.7793 | 0.0054 | 0.0331 | 0.0381 | 10 |
| `cross_subject` | 42 | eegpt_partial_ft_v1 | `A1_Wmoment_ft_no_audio` | 1.0092 | 0.8080 | -0.0103 | 0.0195 | 0.0238 | 10 |
| `cross_subject` | 43 | eegpt_partial_ft_v1 | `A1_Wdeep_full` | 0.9829 | 0.7879 | -0.0182 | 0.0229 | 0.0325 | 10 |
| `cross_subject` | 44 | eegpt_frozen_v1 | `B0_Wphysio_no_audio` | 0.9256 | 0.7321 | -0.0196 | 0.0033 | 0.0114 | 11 |
| `cross_subject` | 45 | eegpt_frozen_v1 | `A1_Wmoment_frozen_full` | 0.9283 | 0.7421 | -0.0473 | 0.0221 | 0.0339 | 10 |
| `cross_subject` | 46 | eegpt_frozen_v1 | `A2_Wphysio_full` | 0.9429 | 0.7459 | -0.0545 | -0.0219 | -0.0216 | 21 |
| `cross_subject` | 47 | eegpt_frozen_v1 | `A1_Wmoment_ft_full` | 0.9172 | 0.7275 | -0.0640 | -0.0400 | -0.0314 | 3 |
| `cross_subject` | 48 | eegpt_frozen_v1 | `B0_Wphysio_full` | 0.9385 | 0.7463 | -0.0722 | -0.0056 | -0.0092 | 13 |
| `cross_day` | 1 | eegpt_partial_ft_v1 | `A1_Wphysio_no_audio` | 0.9208 | 0.7190 | 0.3554 | 0.1289 | 0.1246 | 11 |
| `cross_day` | 2 | eegpt_partial_ft_v1 | `A1_Wmoment_frozen_full` | 0.9159 | 0.7139 | 0.3319 | 0.1417 | 0.1064 | 7 |
| `cross_day` | 3 | eegpt_partial_ft_v1 | `A2_Wmoment_ft_full` | 0.9122 | 0.7153 | 0.3231 | 0.1097 | 0.1013 | 1 |
| `cross_day` | 4 | eegpt_partial_ft_v1 | `A1_Wmoment_ft_no_audio` | 0.9296 | 0.7385 | 0.3170 | 0.0849 | 0.0543 | 1 |
| `cross_day` | 5 | eegpt_partial_ft_v1 | `B0_Wmoment_ft_full` | 0.9297 | 0.7285 | 0.3135 | 0.1286 | 0.1067 | 18 |
| `cross_day` | 6 | eegpt_partial_ft_v1 | `B0_Wphysio_no_audio` | 0.9446 | 0.7384 | 0.3092 | 0.1070 | 0.1027 | 7 |
| `cross_day` | 7 | eegpt_partial_ft_v1 | `A2_Wphysio_no_audio` | 0.9347 | 0.7232 | 0.3019 | 0.1349 | 0.1125 | 5 |
| `cross_day` | 8 | eegpt_partial_ft_v1 | `A1_Wphysio_full` | 0.9579 | 0.7472 | 0.2995 | 0.1051 | 0.0744 | 12 |
| `cross_day` | 9 | eegpt_partial_ft_v1 | `B0_Wmoment_frozen_no_audio` | 0.9467 | 0.7525 | 0.2993 | 0.0958 | 0.0729 | 2 |
| `cross_day` | 10 | eegpt_partial_ft_v1 | `B0_Wmoment_ft_no_audio` | 0.9348 | 0.7448 | 0.2984 | 0.0756 | 0.0470 | 1 |
| `cross_day` | 11 | eegpt_partial_ft_v1 | `A2_Wmoment_frozen_no_audio` | 0.9443 | 0.7460 | 0.2944 | 0.0762 | 0.0654 | 2 |
| `cross_day` | 12 | eegpt_partial_ft_v1 | `A1_Wmoment_frozen_no_audio` | 0.9510 | 0.7535 | 0.2915 | 0.0807 | 0.0610 | 2 |
| `cross_day` | 13 | eegpt_partial_ft_v1 | `B0_Wmoment_frozen_full` | 0.9314 | 0.7227 | 0.2891 | 0.1121 | 0.1049 | 6 |
| `cross_day` | 14 | eegpt_partial_ft_v1 | `B0_Wphysio_full` | 0.9580 | 0.7483 | 0.2870 | 0.1202 | 0.0706 | 15 |
| `cross_day` | 15 | eegpt_partial_ft_v1 | `A1_Wmoment_ft_full` | 0.9350 | 0.7274 | 0.2841 | 0.0981 | 0.0734 | 8 |
| `cross_day` | 16 | eegpt_partial_ft_v1 | `A2_Wmoment_frozen_full` | 0.9462 | 0.7451 | 0.2837 | 0.1119 | 0.0957 | 8 |
| `cross_day` | 17 | eegpt_partial_ft_v1 | `A2_Wmoment_ft_no_audio` | 0.9481 | 0.7517 | 0.2771 | 0.0608 | 0.0437 | 1 |
| `cross_day` | 18 | eegpt_partial_ft_v1 | `A2_Wphysio_full` | 0.9188 | 0.7147 | 0.2731 | 0.1209 | 0.1154 | 3 |
| `cross_day` | 19 | eegpt_partial_ft_v1 | `B0_Wdeep_no_audio` | 0.9635 | 0.7561 | 0.2660 | 0.0926 | 0.0585 | 3 |
| `cross_day` | 20 | eegpt_partial_ft_v1 | `A2_Wdeep_no_audio` | 0.9662 | 0.7543 | 0.2580 | 0.0738 | 0.0454 | 3 |
| `cross_day` | 21 | eegpt_partial_ft_v1 | `A1_Wdeep_full` | 0.9479 | 0.7388 | 0.2579 | 0.0999 | 0.0867 | 8 |
| `cross_day` | 22 | eegpt_frozen_v1 | `A2_Wmoment_frozen_no_audio` | 0.9534 | 0.7481 | 0.2501 | 0.0740 | 0.0597 | 2 |
| `cross_day` | 23 | eegpt_partial_ft_v1 | `A1_Wdeep_no_audio` | 0.9633 | 0.7539 | 0.2491 | 0.0614 | 0.0380 | 3 |
| `cross_day` | 24 | eegpt_frozen_v1 | `B0_Wdeep_full` | 0.9552 | 0.7544 | 0.2473 | 0.0915 | 0.0263 | 10 |
| `cross_day` | 25 | eegpt_frozen_v1 | `A1_Wdeep_no_audio` | 0.9419 | 0.7391 | 0.2404 | 0.0501 | 0.0361 | 4 |
| `cross_day` | 26 | eegpt_frozen_v1 | `A2_Wdeep_full` | 0.9614 | 0.7532 | 0.2362 | 0.0908 | 0.0660 | 10 |
| `cross_day` | 27 | eegpt_partial_ft_v1 | `B0_Wdeep_full` | 0.9698 | 0.7547 | 0.2354 | 0.0592 | 0.0481 | 8 |
| `cross_day` | 28 | eegpt_frozen_v1 | `A1_Wphysio_full` | 0.9577 | 0.7455 | 0.2332 | 0.0492 | 0.0262 | 23 |
| `cross_day` | 29 | eegpt_frozen_v1 | `A1_Wphysio_no_audio` | 0.9445 | 0.7212 | 0.2295 | 0.0932 | 0.0579 | 15 |
| `cross_day` | 30 | eegpt_frozen_v1 | `B0_Wmoment_frozen_full` | 0.9487 | 0.7241 | 0.2173 | 0.0693 | 0.0126 | 11 |
| `cross_day` | 31 | eegpt_frozen_v1 | `A2_Wmoment_ft_full` | 0.9470 | 0.7207 | 0.2157 | 0.0525 | -0.0046 | 13 |
| `cross_day` | 32 | eegpt_frozen_v1 | `A1_Wmoment_ft_full` | 0.9526 | 0.7368 | 0.2142 | 0.0324 | -0.0031 | 17 |
| `cross_day` | 33 | eegpt_frozen_v1 | `A2_Wdeep_no_audio` | 0.9609 | 0.7546 | 0.2137 | 0.0411 | 0.0325 | 4 |
| `cross_day` | 34 | eegpt_frozen_v1 | `A2_Wmoment_frozen_full` | 0.9562 | 0.7419 | 0.2129 | 0.0502 | 0.0169 | 17 |
| `cross_day` | 35 | eegpt_frozen_v1 | `A1_Wmoment_frozen_full` | 0.9441 | 0.7278 | 0.2120 | 0.0440 | 0.0133 | 17 |
| `cross_day` | 36 | eegpt_frozen_v1 | `A2_Wmoment_ft_no_audio` | 0.9556 | 0.7492 | 0.2095 | 0.0079 | -0.0059 | 4 |
| `cross_day` | 37 | eegpt_frozen_v1 | `A1_Wmoment_ft_no_audio` | 0.9535 | 0.7423 | 0.2088 | 0.0307 | 0.0175 | 4 |
| `cross_day` | 38 | eegpt_frozen_v1 | `B0_Wmoment_ft_full` | 0.9585 | 0.7353 | 0.2034 | 0.0506 | 0.0158 | 13 |
| `cross_day` | 39 | eegpt_frozen_v1 | `A1_Wmoment_frozen_no_audio` | 0.9652 | 0.7555 | 0.2032 | 0.0261 | 0.0100 | 4 |
| `cross_day` | 40 | eegpt_frozen_v1 | `B0_Wphysio_full` | 0.9687 | 0.7536 | 0.1975 | 0.0445 | 0.0287 | 17 |
| `cross_day` | 41 | eegpt_partial_ft_v1 | `A2_Wdeep_full` | 0.9659 | 0.7548 | 0.1971 | 0.0669 | 0.0401 | 8 |
| `cross_day` | 42 | eegpt_frozen_v1 | `A1_Wdeep_full` | 0.9765 | 0.7627 | 0.1963 | 0.0844 | 0.0583 | 8 |
| `cross_day` | 43 | eegpt_frozen_v1 | `A2_Wphysio_full` | 0.9568 | 0.7388 | 0.1961 | 0.0243 | 0.0026 | 18 |
| `cross_day` | 44 | eegpt_frozen_v1 | `B0_Wmoment_frozen_no_audio` | 0.9859 | 0.7708 | 0.1899 | -0.0076 | 0.0056 | 13 |
| `cross_day` | 45 | eegpt_frozen_v1 | `A2_Wphysio_no_audio` | 0.9406 | 0.7202 | 0.1858 | 0.0498 | 0.0266 | 18 |
| `cross_day` | 46 | eegpt_frozen_v1 | `B0_Wmoment_ft_no_audio` | 0.9678 | 0.7540 | 0.1837 | -0.0147 | -0.0233 | 4 |
| `cross_day` | 47 | eegpt_frozen_v1 | `B0_Wdeep_no_audio` | 0.9570 | 0.7448 | 0.1796 | 0.0161 | 0.0085 | 10 |
| `cross_day` | 48 | eegpt_frozen_v1 | `B0_Wphysio_no_audio` | 0.9675 | 0.7197 | 0.0100 | 0.0512 | 0.0840 | 16 |
| `within_subject_day` | 1 | eegpt_partial_ft_v1 | `A1_Wphysio_no_audio` | 0.9141 | 0.7054 | 0.3985 | 0.1859 | 0.1774 | 18 |
| `within_subject_day` | 2 | eegpt_partial_ft_v1 | `B0_Wphysio_no_audio` | 0.9316 | 0.7111 | 0.3976 | 0.1981 | 0.1881 | 22 |
| `within_subject_day` | 3 | eegpt_partial_ft_v1 | `B0_Wmoment_frozen_no_audio` | 0.9209 | 0.7182 | 0.3957 | 0.1796 | 0.1829 | 2 |
| `within_subject_day` | 4 | eegpt_partial_ft_v1 | `A2_Wmoment_frozen_full` | 0.9098 | 0.7168 | 0.3950 | 0.1899 | 0.1863 | 6 |
| `within_subject_day` | 5 | eegpt_partial_ft_v1 | `B0_Wmoment_frozen_full` | 0.9139 | 0.7205 | 0.3933 | 0.1765 | 0.1729 | 6 |
| `within_subject_day` | 6 | eegpt_partial_ft_v1 | `B0_Wdeep_full` | 0.9274 | 0.7272 | 0.3926 | 0.2008 | 0.2043 | 16 |
| `within_subject_day` | 7 | eegpt_partial_ft_v1 | `A1_Wmoment_frozen_no_audio` | 0.9232 | 0.7202 | 0.3901 | 0.1718 | 0.1745 | 2 |
| `within_subject_day` | 8 | eegpt_partial_ft_v1 | `A2_Wmoment_frozen_no_audio` | 0.9283 | 0.7232 | 0.3889 | 0.1745 | 0.1783 | 2 |
| `within_subject_day` | 9 | eegpt_partial_ft_v1 | `A1_Wmoment_ft_full` | 0.9213 | 0.7193 | 0.3885 | 0.1764 | 0.1702 | 6 |
| `within_subject_day` | 10 | eegpt_partial_ft_v1 | `B0_Wmoment_ft_no_audio` | 0.9237 | 0.7198 | 0.3883 | 0.1729 | 0.1749 | 2 |
| `within_subject_day` | 11 | eegpt_partial_ft_v1 | `A2_Wmoment_ft_no_audio` | 0.9239 | 0.7215 | 0.3874 | 0.1688 | 0.1698 | 2 |
| `within_subject_day` | 12 | eegpt_partial_ft_v1 | `A2_Wphysio_no_audio` | 0.9152 | 0.7154 | 0.3872 | 0.1719 | 0.1656 | 13 |
| `within_subject_day` | 13 | eegpt_partial_ft_v1 | `A1_Wdeep_full` | 0.9270 | 0.7228 | 0.3871 | 0.1882 | 0.1823 | 6 |
| `within_subject_day` | 14 | eegpt_partial_ft_v1 | `A2_Wdeep_full` | 0.9324 | 0.7305 | 0.3871 | 0.1954 | 0.1949 | 16 |
| `within_subject_day` | 15 | eegpt_partial_ft_v1 | `B0_Wdeep_no_audio` | 0.9513 | 0.7334 | 0.3837 | 0.1712 | 0.1882 | 21 |
| `within_subject_day` | 16 | eegpt_partial_ft_v1 | `A2_Wphysio_full` | 0.9257 | 0.7240 | 0.3822 | 0.1729 | 0.1645 | 13 |
| `within_subject_day` | 17 | eegpt_partial_ft_v1 | `A2_Wmoment_ft_full` | 0.9136 | 0.7180 | 0.3816 | 0.1729 | 0.1689 | 6 |
| `within_subject_day` | 18 | eegpt_partial_ft_v1 | `A1_Wmoment_ft_no_audio` | 0.9283 | 0.7274 | 0.3814 | 0.1652 | 0.1648 | 2 |
| `within_subject_day` | 19 | eegpt_partial_ft_v1 | `A1_Wmoment_frozen_full` | 0.9255 | 0.7153 | 0.3774 | 0.1712 | 0.1610 | 13 |
| `within_subject_day` | 20 | eegpt_partial_ft_v1 | `B0_Wmoment_ft_full` | 0.9171 | 0.7198 | 0.3758 | 0.1741 | 0.1704 | 6 |
| `within_subject_day` | 21 | eegpt_partial_ft_v1 | `B0_Wphysio_full` | 0.9297 | 0.7182 | 0.3727 | 0.1606 | 0.1379 | 20 |
| `within_subject_day` | 22 | eegpt_partial_ft_v1 | `A2_Wdeep_no_audio` | 0.9423 | 0.7296 | 0.3709 | 0.1778 | 0.1748 | 6 |
| `within_subject_day` | 23 | eegpt_partial_ft_v1 | `A1_Wdeep_no_audio` | 0.9431 | 0.7333 | 0.3660 | 0.1806 | 0.1827 | 13 |
| `within_subject_day` | 24 | eegpt_partial_ft_v1 | `A1_Wphysio_full` | 0.9452 | 0.7320 | 0.3611 | 0.1641 | 0.1547 | 30 |
| `within_subject_day` | 25 | eegpt_frozen_v1 | `B0_Wphysio_no_audio` | 0.9429 | 0.7254 | 0.2948 | 0.0739 | 0.0619 | 19 |
| `within_subject_day` | 26 | eegpt_frozen_v1 | `A2_Wdeep_no_audio` | 0.9403 | 0.7378 | 0.2943 | 0.0682 | 0.0625 | 30 |
| `within_subject_day` | 27 | eegpt_frozen_v1 | `A1_Wmoment_frozen_full` | 0.9435 | 0.7380 | 0.2926 | 0.1097 | 0.0598 | 28 |
| `within_subject_day` | 28 | eegpt_frozen_v1 | `B0_Wmoment_ft_no_audio` | 0.9562 | 0.7283 | 0.2877 | 0.0789 | 0.0824 | 13 |
| `within_subject_day` | 29 | eegpt_frozen_v1 | `B0_Wdeep_no_audio` | 0.9419 | 0.7332 | 0.2849 | 0.0663 | 0.0640 | 30 |
| `within_subject_day` | 30 | eegpt_frozen_v1 | `B0_Wmoment_frozen_full` | 0.9449 | 0.7360 | 0.2829 | 0.0928 | 0.0611 | 24 |
| `within_subject_day` | 31 | eegpt_frozen_v1 | `A1_Wphysio_no_audio` | 0.9505 | 0.7291 | 0.2781 | 0.0702 | 0.0548 | 27 |
| `within_subject_day` | 32 | eegpt_frozen_v1 | `A1_Wmoment_ft_no_audio` | 0.9601 | 0.7415 | 0.2706 | 0.0797 | 0.0721 | 16 |
| `within_subject_day` | 33 | eegpt_frozen_v1 | `A1_Wdeep_no_audio` | 0.9535 | 0.7234 | 0.2689 | 0.0937 | 0.1206 | 24 |
| `within_subject_day` | 34 | eegpt_frozen_v1 | `A2_Wphysio_full` | 0.9487 | 0.7301 | 0.2688 | 0.0999 | 0.0730 | 51 |
| `within_subject_day` | 35 | eegpt_frozen_v1 | `A1_Wdeep_full` | 0.9592 | 0.7458 | 0.2632 | 0.0874 | 0.0835 | 27 |
| `within_subject_day` | 36 | eegpt_frozen_v1 | `A2_Wdeep_full` | 0.9736 | 0.7529 | 0.2548 | 0.0768 | 0.0816 | 63 |
| `within_subject_day` | 37 | eegpt_frozen_v1 | `B0_Wmoment_ft_full` | 0.9531 | 0.7284 | 0.2519 | 0.0547 | 0.0193 | 24 |
| `within_subject_day` | 38 | eegpt_frozen_v1 | `A2_Wmoment_ft_no_audio` | 0.9619 | 0.7417 | 0.2514 | 0.0284 | 0.0291 | 27 |
| `within_subject_day` | 39 | eegpt_frozen_v1 | `B0_Wdeep_full` | 0.9562 | 0.7479 | 0.2487 | 0.0571 | 0.0536 | 16 |
| `within_subject_day` | 40 | eegpt_frozen_v1 | `B0_Wphysio_full` | 0.9568 | 0.7337 | 0.2453 | 0.0704 | 0.0575 | 28 |
| `within_subject_day` | 41 | eegpt_frozen_v1 | `A2_Wphysio_no_audio` | 0.9613 | 0.7404 | 0.2419 | 0.0760 | 0.0660 | 25 |
| `within_subject_day` | 42 | eegpt_frozen_v1 | `A2_Wmoment_frozen_no_audio` | 0.9614 | 0.7384 | 0.2393 | 0.0742 | 0.0763 | 7 |
| `within_subject_day` | 43 | eegpt_frozen_v1 | `B0_Wmoment_frozen_no_audio` | 0.9638 | 0.7542 | 0.2318 | 0.0859 | 0.0932 | 9 |
| `within_subject_day` | 44 | eegpt_frozen_v1 | `A2_Wmoment_frozen_full` | 0.9653 | 0.7641 | 0.2273 | 0.0443 | 0.0461 | 24 |
| `within_subject_day` | 45 | eegpt_frozen_v1 | `A1_Wphysio_full` | 0.9678 | 0.7438 | 0.2260 | 0.0419 | 0.0286 | 54 |
| `within_subject_day` | 46 | eegpt_frozen_v1 | `A2_Wmoment_ft_full` | 0.9629 | 0.7358 | 0.2211 | 0.0487 | 0.0387 | 8 |
| `within_subject_day` | 47 | eegpt_frozen_v1 | `A1_Wmoment_ft_full` | 0.9734 | 0.7447 | 0.2176 | 0.0224 | 0.0248 | 30 |
| `within_subject_day` | 48 | eegpt_frozen_v1 | `A1_Wmoment_frozen_no_audio` | 0.9656 | 0.7568 | 0.2161 | 0.0599 | 0.0611 | 26 |
