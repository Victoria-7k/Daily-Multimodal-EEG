# Wear-Only FM Scripts

Wear-only foundation-model 表征路线、W3FM formal gate、post-gate 诊断和 ACC 替换筛查入口。

| 脚本 | 用途 |
| --- | --- |
| `wear_fm_phase1_downloads.sh` | Wear-only FM Phase 1 相关模型下载脚本。 |
| `65_run_wear_only_fm_phase1.py` | 审计 `wear_complete_mask`，缓存 PaPaGei-S PPG、HARNet10 ACC、NormWear GSR 三路 label-free frozen embeddings。 |
| `66_run_wear_only_fm_phase2.py` | Phase 2 36-run wear-only fatigue 主矩阵。 |
| `67_summarize_wear_only_fm_phase2.py` | Phase 2 protocol x route 三 seed 汇总。 |
| `68_summarize_wear_only_fm_phase3.py` | Phase 3 subject-day block paired bootstrap gate。 |
| `69_run_wear_only_fm_internal_ablation.py` | Post-Phase3 W3FM leave-one-out 与 partial-FT 诊断。 |
| `70_summarize_wear_fm_training_diagnostics.py` | 汇总 training history，识别 fast-overfit 模式。 |
| `71_summarize_wear_fm_training_sweep.py` | 汇总 W3FM training-parameter sweep。 |
| `72_diagnose_w3fm_failure_modes.py` | 解释 W3FM embedding split shift、feature stability、ridge probe 和 gate/error association。 |
| `89_run_w3fm_acc_handcrafted_screen.py` | 用 label-free handcrafted ACC 特征替代 HARNet10 ACC embedding 的小筛查。 |
