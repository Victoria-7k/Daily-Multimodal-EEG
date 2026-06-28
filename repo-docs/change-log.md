# Repo-docs 变更记录

| Timestamp | Request | Actions | Verification | Result |
| --- | --- | --- | --- | --- |
| 2026-06-29 00:00 Asia/Shanghai | 用 `repo-docs-zh` 为仓库生成中文 repo-docs | 新增中文 `repo-docs/` 包，覆盖 README、主 walkthrough、事件窗口模块、embedding 契约模块、命令产物参考、字段契约、术语表；新增根目录 `AGENTS.md` 维护规则。Synced through 7c75bd3；同时检查了当前未提交工作区状态。 | `python -m pytest tests -q` 通过；repo-docs validator 为 `OK: 0 errors, 6 warning(s)`，剩余 warning 是开头代码名密度和示例数字、示例 `sample_id` 是否要进术语表的阅读提示。 | 完成。 |
| 2026-06-29 07:01 Asia/Shanghai | 更新 repo-docs，确保它和当前代码一致；同时更新项目 README | 同步根 README 当前状态、常用命令和运行约定；更新 repo-docs 对视频音频对齐重跑参数、manifest 汇总验证入口、本地同步产物范围、ffprobe cache 字段的说明。Synced through 6a7cc1c。 | `python -m pytest tests -q` 通过；`python -m compileall -q src scripts tests` 通过；repo-docs validator 为 `OK: 0 errors, 6 warning(s)`，剩余 warning 是示例数字、示例 `sample_id` 和 walkthrough 开头代码名密度的阅读提示。 | 完成。 |
