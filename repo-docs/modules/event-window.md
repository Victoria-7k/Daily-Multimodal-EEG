# 事件窗口

## 白话模型

事件窗口是把“某一刻的评分事件”变成“可以被模型读取的一段固定时间样本”。原始评分行只告诉项目某个事件发生在 `absolute_onset_time`，但 EEG、wear、视频和音频都需要一个开始时间、结束时间和稳定样本编号。窗口层补上这些内容。

当前默认事件范围是情绪评分事件前 120 秒到事件发生时刻，并按 10 秒窗口、10 秒步长切成 12 个不重叠样本。事件本身如果没有足够的前置两分钟历史，或者精确 `video_candidates` 不能覆盖完整两分钟事件范围，会在窗口展开前被跳过；跳过的事件和原因写入窗口索引 summary。这个选择让后续模型只读评分前的历史信号，不把评分之后的数据混进基础闭环。

## 代码模型

[窗口构建函数](../../src/daily_multimodal/alignment/event_windows.py) 接收 manifest rows 和几个时间参数。它先验证窗口大小、步长和范围，再对每个事件确认完整两分钟历史和精确视频覆盖，然后计算每个 10 秒样本的 `window_start_time`、`window_end_time` 和 offset 字段。输出记录保留原始路径和标签，同时把精确 `video_candidates` 重新换算成当前 10 秒样本的 `clip_start_seconds`、`clip_end_seconds`、`overlap_seconds` 和 `covers_window`，再转成 `has_face`、`has_audio`。

最小调用形状是：

```python
windows = build_window_index(
    rows,
    start_seconds=-120,
    end_seconds=0,
    window_size_seconds=10,
    stride_seconds=10,
)
```

[窗口测试](../../tests/test_event_window.py) 确认默认参数会把 `2025-02-28 14:13:10` 的事件切成 12 个样本，第一个样本是 `2025-02-28 14:11:10` 到 `2025-02-28 14:11:20`，最后一个样本结束在事件时刻。测试还覆盖不足两分钟历史、精确视频不足两分钟覆盖、以及 per-window `video_candidates` 重新换算。

当前本地同步副本用 `outputs/server_sync/events_manifest_with_video_audio.jsonl` 重建后，`outputs/reports/window_index_summary.json` 记录 `1272` 个事件中 `720` 个保留、`552` 个跳过，保留事件生成 `8640` 个 10 秒窗口。跳过原因是 `insufficient_video_coverage=502` 和 `insufficient_pre_event_history=50`。

## 接下去阅读

在主路径里，[Step 3: 事件切成可复用窗口](../walkthroughs/one-real-run.md#step-3-事件切成可复用窗口) 解释窗口为什么出现在 embedding 之前。需要精确字段时查 [字段契约](../references/data-contracts.md)；需要运行脚本时查 [运行命令和产物](../references/commands-and-artifacts.md)。

证据状态：除特别标注外，本页基于当前源码和测试已确认。
