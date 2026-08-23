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
| 了解 Wear 预训练模型接入方案 | 读根目录 [Wear 模型选择方案](../wear_model_selection_20260820.md)，含候选模型开放获取核验、推荐路线、工程落点和验收标准。 |
| 执行 Wear × MOMENT 实验 | 读根目录 [Wear × MOMENT 接入实验方案](../wear_moment_experiment_plan_20260820.md)，含两阶段实验矩阵、工程改动清单与验收门槛。 |
| 查 Wear × MOMENT 结果 | 读根目录 [Wear × MOMENT 实验结果](../wear_moment_results_20260820.md)，含 3-seed paired 门槛判定与完整结果表；产物在 `outputs/server_sync/wear_moment_20260820/`。 |
| 接手 Wear × MOMENT 工作 | 读根目录 [Wear × MOMENT Handoff](../wear_moment_handoff_20260820.md)，含脚本清单、服务器环境/数据事实、复现命令与注意事项。 |

当前本地 `outputs/` 副本显示：事件总数为 `1272`，完整 wear 事件为 `1127`，有视频日期候选的事件为 `1103`，完整多模态候选为 `995`。这些数字来自 [manifest 汇总报告](../outputs/reports/manifest_summary.json) 和服务器验证记录；新的数据同步后应重新核对。

本指南同时保留 historical basic 到 real v2 工程线和当前 EEG-aligned 主线。阶段 8 到阶段 18 记录了从完整候选集 embedding、轻量 baseline、真实 embedding 契约、失败清单、四模态缓存、Audio/Face/EEG/Wear 单模态真实 embedding、all-real 打包到 fair leakage controls 的演进；这些历史入口已集中归档在 `scripts/archive_legacy/`。2026-08-14 起，顶层 `scripts/` 只保留当前路线需要直接调用的 embedding 与融合入口：`12_extract_audio_embeddings.py`、`15_extract_wear_embeddings.py`、`16_run_wear_moment_matrix.py`（2026-08-20 新增，Wear × MOMENT token）、`27_extract_dinov2_roi_embeddings.py`、`34_run_eeg_encoder_matrix.py` 和 `32_run_eegpt_centered_loss.py`。

当前正式融合口径使用 `/vePFS-0x0d/DailyEEG_multimodal/embeddings/eeg_encoder_256d_tokens`，将五条 EEG 256D route 接入 B0/A1/A2 video、Wphysio/Wdeep wear 和 full/no_audio 融合矩阵。最新主协议结果集中在 `EEGPT partial FT + Wphysio/Wdeep + B0/A2 video`：`cross_day` 最低 RMSE 为 `B0_Wphysio_no_audio` 的 `0.9189`，`within_subject_day` 最低 RMSE 为 `A2_Wdeep_full` 的 `0.9138`，最高 raw r 为 `B0_Wdeep_no_audio` 的 `0.4252`。除 `eegpt_frozen_v1` baseline 外，当前 EEG 256D tokens 属于 fatigue-supervised representation，解读和复用时需要保留这个监督边界。

2026-08-17 新增 `eeg_cnn_dual_branch_v1` EEG-only 对照 profile：它来自用户提供的 `networks.py` dual-branch CNN，并适配 canonical 200 Hz、10 秒、59 通道 EEG 输入。该 profile 使用对应 protocol 的 train split 监督训练、val 选 best epoch、test 冻结评估；seed 240800 三协议结果和可视化位于 `outputs/server_sync/eeg_cnn_dual_branch_20260817_gpu_cached/summary/`。它用于 EEG-only encoder 对照，当前五条 256D 四模态 fusion 主线仍按上一段口径解读。

后续接入新的 encoder、修订 split 或新增结果矩阵时，应同步更新根目录 README、[统一 embedding 契约](modules/embedding-contract.md)、[运行命令和产物](references/commands-and-artifacts.md) 和 [变更记录](change-log.md)。

证据状态：除特别标注外，本页基于当前源码、测试、配置和本地同步产物已确认。
