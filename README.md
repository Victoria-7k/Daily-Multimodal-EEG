# Daily Multimodal Embedding

本项目用于跑通 Daily Multimodal 数据的前半部分流程：从 EEG、PPG、GSR、ACC、面部录像和录音原始数据构建事件级 manifest，并为后续多模态嵌入提取做准备。

当前阶段覆盖：

- 阶段 0：服务器个人工作目录 `/mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding`
- 阶段 1：本地项目脚手架与配置
- 阶段 2：只读构建事件级 manifest 与覆盖率报告

运行约定：

1. 本地创建和修改文件，再同步到服务器 `wzw` 目录。
2. 大规模或长时间任务前先运行小规模可行性测试。
3. 服务器端产生的 manifest、覆盖率报告、日志和其他结果数据，需要同步回本地项目的相应 `outputs/` 子目录，保证本地保留可追踪副本。
