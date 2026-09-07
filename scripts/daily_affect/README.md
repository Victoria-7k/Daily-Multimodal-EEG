# Daily-Affect Scripts

daily-affect EMA-bag ordinal route、focused diagnostics、window-daily event bridge 和 expected-score Huber screen。该路线读取每 event 23 个 10 秒窗口的 `(N_ema,23,4,256)` bag，不读取 EQL-CAF packed temporal token。

| 脚本 | 用途 |
| --- | --- |
| `73_build_daily_affect_bags.py` | 从 canonical EEG-aligned 窗口级 tokens 构建 EMA bags。 |
| `74_run_daily_affect_phase0_baselines.py` | 训练 `window_replicated` 与 `bag_static` phase0 baselines。 |
| `75_run_daily_affect_state_matrix.py` | 运行 state/prior/global-kernel/dynamic-kernel 矩阵。 |
| `76_summarize_daily_affect_results.py` | 分组汇总、同 seed 配对与 subject-day bootstrap CI。 |
| `77_plot_daily_affect_diagnostics.py` | 输出 seed-aware atlas 和 confusion atlas。 |
| `78_report_daily_affect_focused_diagnostics.py` | 生成 focused candidate/baseline 机制报告。 |
| `79_run_daily_affect_focused_ablation.py` | Focused static/state/prior/kernel 消融。 |
| `80_run_daily_affect_focused_robustness.py` | Frozen checkpoint missing/noise/shuffle 压力测试。 |
| `81_report_daily_affect_prior_guidance.py` | 汇总 state-prior compatibility 与 ordinal difficulty 消融。 |
| `82_run_daily_affect_bottleneck_audit.py` | 固定 look-back/kernel、模态 drift 和 video 覆盖质量审计。 |
| `83_run_daily_affect_routing_factorial.py` | 2x2 拆分 normalization 与 adapter sharing。 |
| `84_compare_window_daily_event_level.py` | 将窗口回归预测聚合到 EMA event，与 daily event 预测配对比较。 |
| `85_plot_daily_affect_event_series.py` | 绘制 daily static/dynamic 的 held-out event 轨迹。 |
| `86_plot_window_daily_event_bridge_series.py` | 绘制窗口主线、daily static、daily dynamic 的 matched event 三面板图。 |
| `87_report_window_daily_range_precision.py` | 报告动态范围、raw r、label separation 和 subject-day bootstrap。 |
| `88_run_daily_affect_expected_score_huber.py` | 扫描 expected-score Huber 辅助损失权重，validation-locked 后输出三 seed exploratory 对照。 |
