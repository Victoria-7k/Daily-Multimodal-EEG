# Daily Multimodal EEG-Aligned Fatigue Prediction

本仓库当前主线是基于 EEG 对齐后的多模态 10 秒窗口预测 `fatigue`：

```text
10s EEG/Wear/Video/Audio aligned windows
-> 每模态 256D embedding
-> modality-token cross-attention
-> fatigue regression
```

更完整的技术路线见 [technical_route_20260814.md](technical_route_20260814.md)，仓库行为导览见 [repo-docs/README.md](repo-docs/README.md)。当前脚本入口说明见 [scripts/README.md](scripts/README.md)。

## 当前任务口径

| 项目 | 当前口径 |
| --- | --- |
| 样本 | `28819` 个 EEG-aligned 10 秒窗口 |
| EEG 原始输入 | `X.npy`，shape 为 `(28819, 2000, 59)`；每窗 10 秒、200 Hz、59 通道 |
| 标签 | `fatigue`，来自对齐后的窗口索引和 `y.npy` |
| 主评估协议 | `cross_day`、`within_subject_day` |
| 诊断协议 | `cross_subject` |
| 划分规则 | train 使用 `pretrain + finetune`，val 用于早停和模型选择，最终指标只在 test split 计算 |

每个窗口在融合阶段最多包含四个 256D modality token：

| 模态 | 输入 | Token |
| --- | --- | --- |
| EEG | 59 通道 EEG 10 秒窗口 | `eeg_emb (N,256)` |
| Wear | PPG、GSR、三轴 ACC | `wear_emb (N,256)` |
| Video | 2 倍主脸 ROI 视频窗口 | `video_emb (N,256)` |
| Audio | 视频音轨切窗后的 openSMILE 特征 | `audio_emb (N,256)` |

`modality_mask` 顺序为 `[eeg, wear, video, audio]`。

## 模态路线

### EEG

当前 EEG 接入口径是完整 256D EEG embedding。最新矩阵包含五条路线：

| EEG route | 方法 | 监督边界 |
| --- | --- | --- |
| `eegpt_frozen_v1` | 复用现有 EEGPT frozen 256D embedding | frozen baseline |
| `eegpt_partial_ft_v1` | EEGPT 端到端回归训练，解冻最后若干 transformer block、norm、projection/head；取 encoder pooled hidden state，经 256D projection 输出 | fatigue-supervised，仅用对应 protocol 的 train/val |
| `cbramod_frozen_v1` | CBraMod encoder frozen，encoder hidden state pooling 后接 256D projection/head | fatigue-supervised projection/head，仅用 train/val |
| `cbramod_partial_ft_v1` | CBraMod partial fine-tune，解冻后部 block、norm、projection/head；输出 256D pooled embedding | fatigue-supervised，仅用 train/val |
| `eeg_de_5band_1s_avg_v1` | 五频带 DE 特征按 1 秒提取并做 10 秒平均，得到 295D，再由 MLP 倒数 256D projection 输出 | fatigue-supervised MLP，仅用 train/val |

当前主线最强 EEG 表征是 `eegpt_partial_ft_v1`。EEG-only 结果：

| protocol | RMSE | raw r | centered r |
| --- | ---: | ---: | ---: |
| `cross_day` | 0.9272 | 0.2741 | 0.1005 |
| `within_subject_day` | 0.9270 | 0.3749 | 0.1489 |

### Wear / Video / Audio

| 模态 | 当前路线 |
| --- | --- |
| Wear | `Wphysio` 从 PPG/GSR/ACC 提取 HR/HRV、SCR/slope、motion/stationary 等可解释特征并投影到 256D；`Wdeep` 使用固定随机 1D convolution / TCN-like 序列特征、池化统计和固定 256D projection。`Wdeep` 本身是固定提取器，监督训练发生在最终融合回归器。 |
| Video | `B0`、`A1`、`A2` 均为 2 倍主脸 ROI DINOv2-Base frozen 256D embedding；`A1` 加入轻量颜色/亮度增强，`A2` 在 A1 基础上加入 grayscale 增强。 |
| Audio | 从视频音轨按窗口切出音频，计算 openSMILE eGeMAPS Functionals，再投影为 256D `audio_emb`。 |

## 融合模型

当前融合器是轻量 modality-token attention regression：

1. 每个可用模态作为一个 256D token。
2. token 经过 `Linear(256 -> hidden_dim)`，当前 `hidden_dim=128`。
3. 加入 learnable modality embedding。
4. 使用单头 `MultiheadAttention` 建模模态互补关系。
5. 使用 learnable query 对 attention 后的 token 加权 pooling。
6. 经过 `LayerNorm + MLP` 回归头输出 fatigue 预测。

训练配置为 AdamW、学习率 `1e-3`、weight decay `1e-4`、batch size `256`、最多 `80` epoch、patience `15`、dropout `0.1`。所有 normalization 只在 train split 拟合。

## 最新结果

最新结果来自 `eeg_encoder_256d_5route_fusion_video_only_seed240800_raw`：

- EEG 256D matrix：`15` runs。
- 四模态融合 matrix：`180` runs。
- EEG routes：5 条。
- Video routes：`B0/A1/A2`。
- Wear routes：`Wphysio/Wdeep`。
- Fusion 设置：`full` 四模态与 `no_audio` EEG+Wear+Video。

主协议最佳结果：

| protocol | 选择标准 | EEG route | fusion route | RMSE | raw r | centered r |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `cross_day` | 最低 RMSE | EEGPT partial FT | `B0_Wphysio_no_audio` | 0.9189 | 0.3301 | 0.1526 |
| `cross_day` | 最高 raw r | EEGPT partial FT | `B0_Wphysio_full` | 0.9481 | 0.3504 | 0.1176 |
| `within_subject_day` | 最低 RMSE | EEGPT partial FT | `A2_Wdeep_full` | 0.9138 | 0.4046 | 0.1736 |
| `within_subject_day` | 最高 raw r | EEGPT partial FT | `B0_Wdeep_no_audio` | 0.9337 | 0.4252 | 0.2122 |

按 EEG route 聚合时，`EEGPT partial FT` 在两个主协议中也保持领先：

| protocol | EEG route | mean RMSE | mean raw r | mean centered r |
| --- | --- | ---: | ---: | ---: |
| `cross_day` | EEGPT partial FT | 0.9497 | 0.3012 | 0.1115 |
| `within_subject_day` | EEGPT partial FT | 0.9268 | 0.3964 | 0.1903 |

主要结果文件：

- `outputs/server_sync/eeg_encoder_256d_5route_20260814/reports/eeg_encoder_256d_5route_matrix_seed240800_20260814.json`
- `outputs/server_sync/eeg_encoder_256d_5route_20260814/reports/eeg_encoder_256d_5route_fusion_video_only_seed240800_raw.json`
- `outputs/server_sync/eeg_encoder_256d_5route_20260814/eeg_encoder_256d_5route_results_summary.md`

## 当前复现实验顺序

1. 如需重新生成非 EEG 模态 embedding，运行：

```bash
python scripts/12_extract_audio_embeddings.py
python scripts/15_extract_wear_embeddings.py
python scripts/27_extract_dinov2_roi_embeddings.py
```

2. 生成五条 EEG 256D token：

```bash
python scripts/34_run_eeg_encoder_matrix.py \
  --profiles eegpt_frozen_v1,eegpt_partial_ft_v1,cbramod_frozen_v1,cbramod_partial_ft_v1,eeg_de_5band_1s_avg_v1 \
  --protocols cross_subject,cross_day,within_subject_day \
  --seeds 240800 \
  --cbramod-checkpoint outputs/checkpoints/cbramod-pretrained \
  --predictions-dir outputs/predictions/eeg_encoder_256d_5route_matrix_seed240800_20260814 \
  --embeddings-dir /vePFS-0x0d/DailyEEG_multimodal/embeddings/eeg_encoder_256d_tokens \
  --out-json outputs/reports/eeg_encoder_256d_5route_matrix_seed240800_20260814.json \
  --out-md outputs/reports/eeg_encoder_256d_5route_matrix_seed240800_20260814.md
```

3. 运行当前 full/no_audio video-only fusion matrix：

```bash
python scripts/32_run_eegpt_centered_loss.py \
  --root /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned \
  --embeddings-root /vePFS-0x0d/DailyEEG_multimodal/embeddings \
  --splits-root /vePFS-0x0d/DailyEEG/splits_new \
  --experiment-set video_only \
  --protocols cross_subject,cross_day,within_subject_day \
  --eeg-branches eeg_eegpt_frozen_v1,eeg_eegpt_partial_ft_v1,eeg_cbramod_frozen_v1,eeg_cbramod_partial_ft_v1,eeg_de_5band_1s_avg_v1 \
  --eeg-token-root eeg_encoder_256d_tokens \
  --eeg-token-seed 240800 \
  --loss-modes raw \
  --no-raw-baseline \
  --subject-balanced-batches \
  --out-json outputs/reports/eeg_encoder_256d_5route_fusion_video_only_seed240800_raw.json \
  --out-md outputs/reports/eeg_encoder_256d_5route_fusion_video_only_seed240800_raw.md \
  --predictions-dir outputs/predictions/eeg_encoder_256d_5route_fusion_video_only_seed240800_raw
```

## 本地验证

```bash
python -m compileall -q src scripts tests
$env:PYTHONPATH='src'; python -m unittest tests.test_eeg_encoder_matrix tests.test_eeg_encoder_fusion_tokens tests.test_centered_metrics -v
```

repo-docs 结构检查：

```bash
python C:\Users\28303\.codex\skills\repo-docs\scripts\validate_repo_docs.py repo-docs --repo-root .
```
