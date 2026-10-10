# 研究文档索引

本目录按“当前路线、联合评估、支撑实验、历史档案”组织研究文档。0814 window 与 0906 EMA-bag 是两条并列的当前路线；多情绪结构矩阵连接两条路线，但不改变各自的监督单位和结论边界。

## 当前路线

| 路线 | 状态 | 核心入口 | 读者应如何理解 |
| --- | --- | --- | --- |
| 0814 window | Current | [技术路线](current/0814-window/technical_route_20260814.md) | 以 10 秒窗口为基本建模单位，承载窗口结构、模态组合、融合器和表征路线的基线。 |
| 0906 EMA-bag | Current | [技术路线](current/0906-ema-bag/technical_route_20260906.md) | 标量回归 head、MSE 训练、validation RMSE 选型、event-level raw r 报告；读取 23 个重叠窗口，比较静态、状态、先验和时间核结构。 |

两条路线保持并列。跨路线比较必须使用相同 event、protocol、token、模态、split 和 seed，并在 event level 评价；不能把窗口级样本数直接当作独立监督样本数。

### 0814 window

- [当前技术路线](current/0814-window/technical_route_20260814.md)
- [MAE 单模态替换路线表](current/0814-window/modality_mae_window_route_20260928.md)
- [MAE 全聊天实验交接手册（2026-10-10）](current/0814-window/experiments/mae_chat_handout_20261010.md)：汇总87回合的决定与工作，封存原始/v2/v4结果、两协议11情绪完整表、两轮修复证据及服务器检查指令。
- [MAE 第一轮输入改进计划（2026-10-10）](current/0814-window/experiments/mae_round1_input_improvement_plan_20261010.md)：独立验证既有2.0×face ROI VideoMAE与扩大过滤池EEG-MAE；cross_day EEG正式SSL、canonical导出和三seed冻结下游已完成并验收，视频cache构建中；同日EEG严格对照按107个原train/holdout信号重叠停止。
- [Cross-Attention 实现说明](current/0814-window/cross_attention_implementation_20260824.md)
- [MAE 第二轮下游适配计划（2026-10-10）](current/0814-window/experiments/mae_round2_downstream_adaptation_plan_20261010.md)：核心A/B/C与D统一test报告均完成，36个唯一融合cell与4个EEG-only模型已验收；[完整有效性解读](../../outputs/server_sync/mae_round2_downstream_20261010/reports/stage_d/abc_effectiveness_report.md)显示A的canonical适配test raw r +0.008106、B的层数val收益在test反向、C新目标在MAE/B0均未改善整体预测。全部val实用候选门槛未通过，保留原C冻结E1与window MSE及同日协议stop；[A](../../outputs/server_sync/mae_round2_downstream_20261010/reports/experiment_a_findings.md)/[B](../../outputs/server_sync/mae_round2_downstream_20261010/reports/experiment_b/experiment_b_findings.md)报告保留各自阶段快照。
- [Fusion、calibration 与 normalization 实验](current/0814-window/experiments/)

`experiments/` 中的 handoff 和结果文档记录已完成实验及其当时的决策。它们为当前路线提供证据和复现信息，不代表每个候选仍处于推进状态。

### 0906 EMA-bag

- [MAE E1基本event聚合实验（2026-10-10）](../../outputs/server_sync/mae_basic_event_pool_20261010/reports/findings.md)：固定原C输入和11-head，以全部有效窗口等权表示聚合接event MSE；三seedtest centered r 0.100470，相对C_EVENT的0.125307下降0.024837，0/3正向。保留预测后平均的C_EVENT为个体内关联候选；本轮没有时间范围或其他结构选型。
- [当前技术路线](current/0906-ema-bag/technical_route_20260906.md)
- [原五分类 / QWK 详细设计](current/0906-ema-bag/daily_affect_multimodal_technical_route_modified.md)
- [五分类 / QWK 路线历史快照](current/0906-ema-bag/experiments/technical_route_20260906_ordinal_snapshot_20261010.md)
- [标量回归、标签置换与相关报告](current/0906-ema-bag/experiments/)

当前标量回归实现、输入组合、三/七 seed 结果和协议级决策以 `technical_route_20260906.md` 为入口；原五分类设计与 QWK 结果单独保留为历史参考。

## 联合评估

- [结构 × 11 情绪回归矩阵](current/joint-evaluation/multiemotion_eeg_multitask_experiment_plan_20260913.md)：0814 单一窗口结构与 0906 十八个结构变体的共同评估入口；ST-11 与 MT-11 两条路线均已完成。

联合评估复用两条当前路线的结构，同时保持上游监督来源、embedding seed 和下游训练 seeds 分离记录。

## 支撑实验

- [Wear 路线](supporting/wear/README.md)：MOMENT、Wear-only foundation model 与相关停止决策。

支撑实验为当前路线提供 token 或组件证据。未通过晋级门槛的分支保留为诊断结果，不自动成为 0814 或 0906 的默认配置。

## 历史档案

- [Embedding 工程闭环](archive/embedding-bringup/README.md)：manifest、窗口、smoke embedding、真实 encoder 与 OpenFace 接入的形成过程。
- [被替代路线](archive/superseded-routes/README.md)：保留概念演进和历史依据，不作为当前执行入口。

## 状态约定

| 状态 | 含义 |
| --- | --- |
| Current | 当前持续使用的路线或入口。 |
| Completed | 实验和判定已完成，作为证据保留。 |
| Supporting | 为当前路线提供组件、token 或诊断证据。 |
| Superseded | 已由新的技术入口承接，保留历史价值。 |
| Archived | 已结束的工程阶段或旧工作流。 |

Evidence status: Confirmed unless noted.
