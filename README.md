# Daily Multimodal EEG-Aligned Affective Modeling

本仓库围绕 EEG 对齐的日常情境多模态建模维护两条并列、持续使用的技术路线。两条路线共享 EEG、Wear、Video、Audio 的窗口级表示和协议化数据划分，但采用不同的监督单位与时间聚合方式。

| 当前路线 | 监督与输入单位 | 主要用途 | 技术入口 |
| --- | --- | --- | --- |
| 0814 window | 10 秒窗口；窗口级回归后按实验定义汇总 | 窗口结构、模态组合、融合和表征基线 | [0814 窗口路线](docs/research/current/0814-window/technical_route_20260814.md) |
| 0906 EMA-bag | 一次 EMA event；每个 event 包含 23 个重叠窗口 | event-level 静态/状态/先验/时间核结构 | [0906 EMA-bag 路线](docs/research/current/0906-ema-bag/technical_route_20260906.md) |

两条路线地位并列。0814 提供窗口级结构和融合基线，0906 提供 EMA-event 时间聚合结构；任何比较都应锁定相同协议、输入 token、模态组合、监督边界和下游 seed。

## 当前联合评估

结构 × 11 情绪矩阵同时使用 0814 的单一窗口结构和 0906 的 18 个 EMA-bag 结构变体，分别评估标签专属单任务 EEGPT 路线与 11-label 共同监督的共享多头路线：

- [结构 × 多情绪回归矩阵计划与完成状态](docs/research/current/joint-evaluation/multiemotion_eeg_multitask_experiment_plan_20260913.md)
- [共享 MT11 结果](outputs/server_sync/multiemotion_20260913/structure_matrix_summary/README.md)
- [独立单标签结果](outputs/server_sync/multiemotion_20260913/single_task_structure_matrix_summary/README.md)

## 文档入口

| 需求 | 入口 |
| --- | --- |
| 查看当前研究路线、实验状态和归档边界 | [研究文档索引](docs/research/README.md) |
| 理解仓库如何从数据生成 embedding 并进入训练 | [仓库行为指南](repo-docs/README.md) |
| 查当前脚本入口 | [脚本索引](scripts/README.md) |
| 查命令、字段与产物 | [命令与产物](repo-docs/references/commands-and-artifacts.md) · [字段契约](repo-docs/references/data-contracts.md) |

## 目录约定

```text
docs/research/current/      两条当前路线与联合评估
docs/research/supporting/   Wear 等支撑性实验
docs/research/archive/      已完成的早期工程过程与被替代路线
repo-docs/                  代码行为、数据契约、运行命令和维护记录
scripts/                    按实验路线分组的执行入口
outputs/                    本地结果、轻量同步副本和报告
```

## 本地验证

```powershell
python -m compileall -q src scripts tests
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
python C:\Users\28303\.codex\skills\repo-docs\scripts\validate_repo_docs.py repo-docs --repo-root .
```
