# Current Script Entry Points

当前 `scripts/` 顶层只保留最新 EEG-aligned 四模态路线从 embedding 生成开始需要直接调用的入口。

## 保留入口

| 脚本 | 用途 |
| --- | --- |
| `12_extract_audio_embeddings.py` | 生成 openSMILE eGeMAPS audio 256D embedding。 |
| `15_extract_wear_embeddings.py` | 生成 Wphysio / Wdeep wearable 256D embedding。 |
| `27_extract_dinov2_roi_embeddings.py` | 生成 B0/A1/A2 2x ROI DINOv2 video 256D embedding。 |
| `34_run_eeg_encoder_matrix.py` | 生成并评估五条 EEG 256D embedding 路线。 |
| `32_run_eegpt_centered_loss.py` | 将 EEG/Wear/Video/Audio tokens 接入 modality-token attention fusion。 |

## 当前复现实验顺序

1. 如需重新生成非 EEG 模态 embedding，先运行 `12`、`15`、`27`。
2. 运行 `34_run_eeg_encoder_matrix.py`，输出 protocol/profile/seed 对应的 `eeg_emb (28819,256)`。
3. 运行 `32_run_eegpt_centered_loss.py --experiment-set video_only --eeg-token-root eeg_encoder_256d_tokens`，完成当前 full/no_audio video-only fusion matrix。

历史实验、诊断、旧口径和调参脚本已移动到 `scripts/archive_legacy/`。
