# Daily Multimodal Embedding 读者指南

这个仓库当前主线是 EEG-aligned 多模态疲劳预测：`28819` 个 10 秒窗口，每个窗口最多由 EEG、Wear、Video、Audio 四个 256D modality token 表示，再接入轻量 modality-token cross-attention 回归 `fatigue`。根目录 [当前项目总览](../README.md) 和 [2026-08-14 技术路线总结](../technical_route_20260814.md) 给出当前实验口径、路线矩阵和最新结果。

如果需要理解仓库如何从原始 EEG、PPG、GSR、ACC、面部录像和录音素材生成可复现窗口级 embedding，先读 [一条事件如何变成 smoke embedding](walkthroughs/one-real-run.md)。如果已经知道数据背景，直接查 [运行命令和产物](references/commands-and-artifacts.md)、[字段契约](references/data-contracts.md) 或 [统一 embedding 契约](modules/embedding-contract.md)。

## 阅读路径

| 你想做什么 | 从哪里开始 |
| --- | --- |
| 快速理解当前主线 | 读根目录 [当前项目总览](../README.md) 和 [2026-08-14 技术路线总结](../technical_route_20260814.md)，先确认 28,819 行 EEG-aligned 技术路线。 |
| 理解历史工程闭环 | 读 [一条事件如何变成 smoke embedding](walkthroughs/one-real-run.md)，建立 manifest、窗口索引和 embedding 打包运行模型。 |
| 改窗口逻辑 | 读 [事件窗口的白话模型](modules/event-window.md)，再查 `build_window_index` 的字段输出。 |
| 改 embedding 输出 | 读 [统一 embedding 契约](modules/embedding-contract.md)，再查 `.npz` 和报告字段。 |
| 在服务器复现实验 | 查 [运行命令和产物](references/commands-and-artifacts.md)，按阶段顺序运行脚本。 |
| 查真实 embedding 结论 | 读 [主 walkthrough 的真实模态和 ablation 步骤](walkthroughs/one-real-run.md#step-12-all-real-打包形成训练入口兼容产物)，再查阶段 17/18 命令。 |
| 排查 EEG 时间窗口 | 查 [EEG coverage audit 命令](references/commands-and-artifacts.md) 和 [EEG 覆盖字段](references/data-contracts.md#eeg-coverage-audit-字段)。 |
| 查字段名 | 查 [字段契约](references/data-contracts.md)，避免从叙述页里翻源码。 |

当前本地 `outputs/` 副本显示：事件总数为 `1272`，完整 wear 事件为 `1127`，有视频日期候选的事件为 `1103`，完整多模态候选为 `995`。这些数字来自 [manifest 汇总报告](../outputs/reports/manifest_summary.json) 和服务器验证记录；新的数据同步后应重新核对。

本指南同时保留 historical basic 到 real v2 工程线和当前 EEG-aligned 主线。阶段 8 到阶段 18 记录了从完整候选集 embedding、轻量 baseline、真实 embedding 契约、失败清单、四模态缓存、Audio/Face/EEG/Wear 单模态真实 embedding、all-real 打包到 fair leakage controls 的演进；这些历史入口已集中归档在 `scripts/archive_legacy/`。2026-08-14 起，顶层 `scripts/` 只保留当前路线需要直接调用的 embedding 与融合入口：`12_extract_audio_embeddings.py`、`15_extract_wear_embeddings.py`、`27_extract_dinov2_roi_embeddings.py`、`34_run_eeg_encoder_matrix.py` 和 `32_run_eegpt_centered_loss.py`。

当前正式融合口径使用 `/vePFS-0x0d/DailyEEG_multimodal/embeddings/eeg_encoder_256d_tokens`，将五条 EEG 256D route 接入 B0/A1/A2 video、Wphysio/Wdeep wear 和 full/no_audio 融合矩阵。最新主协议结果集中在 `EEGPT partial FT + Wphysio/Wdeep + B0/A2 video`：`cross_day` 最低 RMSE 为 `B0_Wphysio_no_audio` 的 `0.9189`，`within_subject_day` 最低 RMSE 为 `A2_Wdeep_full` 的 `0.9138`，最高 raw r 为 `B0_Wdeep_no_audio` 的 `0.4252`。除 `eegpt_frozen_v1` baseline 外，当前 EEG 256D tokens 属于 fatigue-supervised representation，解读和复用时需要保留这个监督边界。

后续接入新的 encoder、修订 split 或新增结果矩阵时，应同步更新根目录 README、[统一 embedding 契约](modules/embedding-contract.md)、[运行命令和产物](references/commands-and-artifacts.md) 和 [变更记录](change-log.md)。

证据状态：除特别标注外，本页基于当前源码、测试、配置和本地同步产物已确认。
