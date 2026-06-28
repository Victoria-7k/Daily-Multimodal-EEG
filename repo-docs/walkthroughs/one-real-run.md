# 一条事件如何变成 smoke embedding

这条导览跟着一条评分事件走到最终输出。读完以后，你应该能回答三件事：事件从哪里来，窗口如何定界，后续模型为什么能稳定读取结果。精确命令集中放在 [运行命令和产物](../references/commands-and-artifacts.md)，字段名集中放在 [字段契约](../references/data-contracts.md)。

## Step 1: 只读元数据生成事件记录

一次真实运行从 `scripts/01_build_manifest.py` 开始。入口只负责把 `src/` 放进 `sys.path`，实际工作交给 [manifest 构建器](../../src/daily_multimodal/manifest/build_manifest.py)。构建器遍历 EEG 数据集里的 `sub-*/ses-*/beh/*.tsv`，用每一行评分记录里的 `absolute_onset_time` 作为事件时间，再去匹配 EEG sidecar、wear CSV 时间段和视频日期目录。

这个阶段不读取大型 BDF、MP4 或原始音频内容。它只收集路径、时间、标签和可用性布尔值，所以适合先在服务器上做只读覆盖率检查。当前本地同步的 [汇总报告](../../outputs/reports/manifest_summary.json) 确认有 `1272` 个事件，其中 `995` 个满足日期级完整多模态候选。

## Step 2: 视频音频候选变成精确片段

日期级视频候选只能说明某一天目录里有 MP4，不能保证某个评分窗口被哪段视频覆盖。`scripts/02_align_video_audio.py` 会先从 manifest 收集候选 MP4，再让 [视频音频对齐模块](../../src/daily_multimodal/alignment/video_audio_alignment.py) 调用 `ffprobe`，读取 `creation_time`、`duration` 和音频流元数据。

对齐模块把 MP4 的 UTC `creation_time` 转到 `Asia/Shanghai`，再和事件窗口相交。相交结果写入 `video_candidates`，包含 `clip_start_seconds`、`clip_end_seconds`、`overlap_seconds`、`covers_window` 和音频流信息。这个步骤让后面的 `basic` embedding 优先使用精确 `video_candidates`，而不是只拿日期目录里的第一个候选。

## Step 3: 事件切成可复用窗口

窗口索引把事件记录变成训练或探针可以复用的样本记录。`scripts/03_build_window_index.py` 调用 [窗口构建函数](../../src/daily_multimodal/alignment/event_windows.py)，默认取事件发生前 `-10` 到 `0` 秒，窗口长度 `10` 秒，步长 `5` 秒。这个默认配置会为一个事件生成一个基础窗口；如果范围更长，则按步长滑动生成多个窗口。

窗口记录的核心变化是 `event_id` 变成稳定的 `sample_id`，例如测试里确认的 `sub-02_ses-03_00_row-0012_win-0000`。同一条记录还携带 EEG 起点、wear 路径、视频候选、标签和 `has_eeg`、`has_wear`、`has_face`、`has_audio` 等可用性标记。想理解这层概念，可以继续读 [事件窗口的白话模型](../modules/event-window.md)。

## Step 4: 单事件探针先检查形状和非空数据

在真正写 embedding 前，`scripts/03_probe_one_event.py` 会选一个窗口并生成探针报告。探针不训练模型，也不做复杂特征；它检查 EEG 预期重采样形状、PPG/GSR/ACC 在窗口内的行数，以及视频和音频候选数量。当前本地 [探针形状报告](../../outputs/reports/probe_one_event_shapes.txt) 显示一个同步样本有 `2500` 个 EEG 目标采样点、`200` 行 PPG、`400` 行 GSR、`200` 行 ACC 和 `5` 个视频候选。

这个探针的价值是把时间对齐错误尽早暴露出来。窗口开始结束时间一旦错了，wear 行数、EEG offset 或视频候选数量会先变得异常，而不是等到训练脚本里才发现样本不可用。

## Step 5: basic encoder 写出统一 embedding 契约

`scripts/04_extract_one_event_embeddings.py` 和 `scripts/05_extract_smoke_embeddings.py` 最终都会进入 [embedding 批处理](../../src/daily_multimodal/embeddings/pipeline.py)。批处理逐个窗口调用 [basic encoder](../../src/daily_multimodal/embeddings/basic.py)，再把结果保存成压缩 `.npz` 和 JSON 报告。

当前 `basic` encoder 的定位是 smoke 流水线。`wear_emb` 会读取窗口内 PPG/GSR/ACC 数值并计算统计特征；`eeg_emb`、`face_emb`、`audio_emb` 目前用 metadata-derived placeholder 检查文件存在性、路径、统一维度和 mask。输出固定包含 `eeg_emb`、`wear_emb`、`face_emb`、`audio_emb` 四组 `(N, 256)` 数组，以及顺序为 `[eeg, wear, face, audio]` 的 `modality_mask`。更细的输出约束在 [统一 embedding 契约](../modules/embedding-contract.md)。

## Step 6: subject 级入口扩大同一条路径

`scripts/06_extract_subject_embeddings.py` 没有引入新的 embedding 形状，它只是先用 [subject 选择器](../../src/daily_multimodal/embeddings/subject.py) 过滤某个 `subject_id` 的窗口，再复用同一个批处理和保存逻辑。`--require-all-modalities` 打开后，窗口必须同时具备 `has_eeg`、`has_ppg`、`has_gsr`、`has_acc`、`has_face` 和 `has_audio`。

这说明阶段 7 的风险主要不在输出契约，而在样本选择、长时间 CSV 读取、精确视频候选和服务器资源。阶段 4-6 的服务器验证报告已经记录过一次性能问题：重复读取大型 wear CSV 会拖慢 smoke test；当前实现通过缓存 CSV 解析和二分定位窗口范围解决。

## 验证

本地最小验证命令是：

```powershell
python -m pytest tests -q
```

理解主线后，可以按 [运行命令和产物](../references/commands-and-artifacts.md) 在服务器或同步副本上复现阶段命令。没有真实数据路径时，单元测试仍能确认 manifest 匹配、窗口切分、视频音频对齐、embedding 输出和 subject 选择这些核心行为。

证据状态：除特别标注外，本页基于当前源码、测试、配置、本地输出副本和阶段验证记录已确认。
