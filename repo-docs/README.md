# Daily Multimodal Embedding 读者指南

这个仓库把 Daily Multimodal 的原始 EEG、PPG、GSR、ACC、面部录像和录音素材，整理成可复现、可替换、可训练的窗口级 embedding 工程闭环。当前代码已经从事件级 `manifest`、窗口索引和 `basic` smoke embedding，推进到四模态真实 embedding、all-real 打包、阶段 18 ablation、公平对照（fair embedding leakage controls）和 EEG 时间覆盖审计。

先读 [一条事件如何变成 smoke embedding](walkthroughs/one-real-run.md)。这条路径会跟着一个评分事件走完 manifest、视频音频精确对齐、窗口索引、单事件探针、embedding 打包这些行为。已经知道数据背景的读者，可以直接查 [运行命令和产物](references/commands-and-artifacts.md) 或 [字段契约](references/data-contracts.md)。

## 阅读路径

| 你想做什么 | 从哪里开始 |
| --- | --- |
| 快速理解项目 | 读 [一条事件如何变成 smoke embedding](walkthroughs/one-real-run.md)，先建立运行模型。 |
| 改窗口逻辑 | 读 [事件窗口的白话模型](modules/event-window.md)，再查 `build_window_index` 的字段输出。 |
| 改 embedding 输出 | 读 [统一 embedding 契约](modules/embedding-contract.md)，再查 `.npz` 和报告字段。 |
| 在服务器复现实验 | 查 [运行命令和产物](references/commands-and-artifacts.md)，按阶段顺序运行脚本。 |
| 查真实 embedding 结论 | 读 [主 walkthrough 的真实模态和 ablation 步骤](walkthroughs/one-real-run.md#step-12-all-real-打包形成训练入口兼容产物)，再查阶段 17/18 命令。 |
| 排查 EEG 时间窗口 | 查 [EEG coverage audit 命令](references/commands-and-artifacts.md) 和 [EEG 覆盖字段](references/data-contracts.md#eeg-coverage-audit-字段)。 |
| 查字段名 | 查 [字段契约](references/data-contracts.md)，避免从叙述页里翻源码。 |

当前本地 `outputs/` 副本显示：事件总数为 `1272`，完整 wear 事件为 `1127`，有视频日期候选的事件为 `1103`，完整多模态候选为 `995`。这些数字来自 [manifest 汇总报告](../outputs/reports/manifest_summary.json) 和服务器验证记录；新的数据同步后应重新核对。

本指南覆盖的是当前 basic 到 real 的完整工程线。阶段 8 到阶段 10 已经形成从完整候选集 embedding、轻量 baseline，到第一版融合升级对照的闭环；阶段 11 到阶段 18 已完成真实 embedding 契约、失败清单、四模态缓存、Audio/Face/EEG/Wear 单模态真实 embedding、all-real 打包和全量 ablation。`scripts/18_run_fair_embedding_ablation.py` 继续提供泄漏控制：比较 aligned basic、去路径 basic、path-only 和 real embedding，避免把路径/样本元数据误判为真实信号。当前工作区还新增了 EEG coverage audit，用于把 EEG 窗口 offset 分类成 in-range、负 offset、超出记录尾部、部分重叠或整天偏移候选。具体入口集中在 [运行命令和产物](references/commands-and-artifacts.md)。后续接入真正 OpenFace Apptainer wrapper、新的真实 encoder 或 EEG 时间修正时，应同步更新 [统一 embedding 契约](modules/embedding-contract.md) 和 [变更记录](change-log.md)。

证据状态：除特别标注外，本页基于当前源码、测试、配置和本地同步产物已确认。
