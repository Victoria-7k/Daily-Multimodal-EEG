# Wear 支撑实验

> Status: Supporting

本目录集中 Wear 模态的模型选择、MOMENT 接入、实验结果、交接说明和 Wear-only foundation-model 诊断。它们为 0814 window 与 0906 EMA-bag 提供 Wear 表征候选和监督边界证据。

| 文档 | 角色 | 当前状态 |
| --- | --- | --- |
| [模型选择方案](wear_model_selection_20260820.md) | 开放模型候选与接入约束 | Completed research |
| [MOMENT 实验方案](wear_moment_experiment_plan_20260820.md) | 实验矩阵与门槛 | Completed |
| [MOMENT 结果](wear_moment_results_20260820.md) | 三 seed 与全维度证据 | Completed evidence |
| [MOMENT handoff](wear_moment_handoff_20260820.md) | 环境、复现与注意事项 | Historical handoff |
| [Wear-only FM 计划与执行记录](wear_only_fm_experiment_plan.md) | 三 encoder 表征矩阵与后续诊断 | Completed diagnostic; Phase-3 stop retained |

Wear 路线的实验结论应按协议、matched mask、上游监督来源和 downstream seed 阅读。诊断性改善不自动改变两条当前路线的默认配置。
