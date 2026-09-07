# Calibration Scripts

窗口级 fatigue prediction 的 Phase 1-4 calibration、loss、head 和 trend 诊断入口。

| 脚本 | 用途 |
| --- | --- |
| `39_run_prediction_calibration_phase1.py` | Phase 0 preflight 与 Phase 1 post-hoc calibration。 |
| `40_plot_prediction_calibration_phase1.py` | Phase 1 calibration 趋势图与指标对比表。 |
| `41_run_prediction_calibration_phase1b.py` | 小型分布和预测塌缩检查。 |
| `42_run_prediction_calibration_phase2.py` | Phase 2 loss/sampler 小矩阵。 |
| `43_summarize_prediction_calibration_phase2.py` | Phase 2 汇总、delta gate 与趋势图。 |
| `44_run_prediction_calibration_phase2_paired_gate.py` | Phase 2 收窄候选 3-seed paired gate。 |
| `45_summarize_prediction_calibration_phase2_paired_gate.py` | Phase 2 paired gate 汇总。 |
| `46_run_prediction_calibration_phase3_ordinal_head.py` | Phase 3 ordinal/classification/hybrid head screen。 |
| `47_summarize_prediction_calibration_phase3_ordinal_head.py` | Phase 3 head screen 汇总。 |
| `49_plot_prediction_calibration_phase0_to_phase3.py` | Phase 0-3 动态范围、bias、误差与相关性图。 |
| `50_plot_prediction_calibration_phase0_to_phase3_trends.py` | Phase 0-3 subject-day 趋势图。 |
| `51_run_prediction_calibration_next3_screen.py` | Phase 4 next-three 小筛查。 |
| `52_plot_prediction_calibration_next3_trends.py` | Phase 4 趋势图与 baseline 对比表。 |
