# 术语表

| 术语 | 项目里的意思 | 延伸阅读 |
| --- | --- | --- |
| manifest | 事件级清单；把评分行、跨模态路径、标签和可用性标记放在同一行 | [一条事件如何变成 smoke embedding](walkthroughs/one-real-run.md) |
| event | 一条评分记录对应的绝对时间点，是跨模态对齐的中心 | [字段契约](references/data-contracts.md) |
| window index | 窗口级样本清单；把事件切成固定时长样本并生成 `sample_id` | [事件窗口](modules/event-window.md) |
| `sample_id` | 训练和探针使用的稳定窗口编号 | [字段契约](references/data-contracts.md) |
| precise video candidate | 由 `ffprobe` 确认和事件窗口相交的 MP4 片段 | [字段契约](references/data-contracts.md) |
| ffprobe cache | 视频音频精确对齐时保存的 MP4 探测结果；用于复用成功记录，也能配合 `--retry-failed-ffprobe` 重试失败记录 | [运行命令和产物](references/commands-and-artifacts.md) |
| smoke embedding | 为验证流水线而生成的基础 embedding；当前不是最终模型效果 | [统一 embedding 契约](modules/embedding-contract.md) |
| `.npz` | NumPy 压缩数组文件；本项目用它保存四个模态 embedding、mask、样本 id 和标签 | [统一 embedding 契约](modules/embedding-contract.md) |
| `modality_mask` | 表示四个模态是否可用的数组，顺序是 `[eeg, wear, face, audio]` | [统一 embedding 契约](modules/embedding-contract.md) |
| `EmbeddingSample` | 单个窗口 embedding 的内存表示；保存前会被批处理堆叠成 `.npz` 数组和 JSON 报告 | [统一 embedding 契约](modules/embedding-contract.md) |
| `basic` profile | 当前唯一可选 encoder profile，用于阶段 5-7 的基础闭环 | [运行命令和产物](references/commands-and-artifacts.md) |
