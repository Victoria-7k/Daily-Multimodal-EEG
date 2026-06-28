# 事件窗口

## 白话模型

事件窗口是把“某一刻的评分事件”变成“可以被模型读取的一段固定时间样本”。原始评分行只告诉项目某个事件发生在 `absolute_onset_time`，但 EEG、wear、视频和音频都需要一个开始时间、结束时间和稳定样本编号。窗口层补上这些内容。

当前默认窗口是事件前 10 秒到事件发生时刻。这个选择让探针和 smoke embedding 先对齐同一段历史信号，而不是把评分之后的数据混进基础闭环。若扩大窗口范围，构建器会按 `stride_seconds` 滑动生成多个 `sample_id`。

## 代码模型

[窗口构建函数](../../src/daily_multimodal/alignment/event_windows.py) 接收 manifest rows 和几个时间参数。它先验证窗口大小、步长和范围，再对每个事件计算 `window_start_time`、`window_end_time` 和 offset 字段。输出记录保留原始路径和标签，同时把日期级视频候选或精确 `video_candidates` 转成 `has_face`、`has_audio`。

最小调用形状是：

```python
windows = build_window_index(
    rows,
    start_seconds=-10,
    end_seconds=0,
    window_size_seconds=10,
    stride_seconds=5,
)
```

[窗口测试](../../tests/test_event_window.py) 确认默认参数会把 `2025-02-28 14:13:10` 的事件切成 `2025-02-28 14:13:00` 到 `2025-02-28 14:13:10` 的样本，并生成 `sub-02_ses-03_00_row-0012_win-0000`。同一测试还确认当范围是 `-20` 到 `0` 秒时，窗口开始 offset 会是 `-20`、`-15`、`-10`。

## 接下去阅读

在主路径里，[Step 3: 事件切成可复用窗口](../walkthroughs/one-real-run.md#step-3-事件切成可复用窗口) 解释窗口为什么出现在 embedding 之前。需要精确字段时查 [字段契约](../references/data-contracts.md)；需要运行脚本时查 [运行命令和产物](../references/commands-and-artifacts.md)。

证据状态：除特别标注外，本页基于当前源码和测试已确认。

