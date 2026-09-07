# Window Fatigue Scripts

当前 EEG-aligned 10 秒窗口级连续 `fatigue` 回归主线，以及 Wear x MOMENT 和 fusion variant 证据入口。

| 脚本 | 用途 |
| --- | --- |
| `16_run_wear_moment_matrix.py` | 训练 MOMENT-1-based wear 256D token，并导出融合兼容 NPZ。 |
| `32_run_eegpt_centered_loss.py` | 将 EEG/Wear/Video/Audio tokens 接入 modality-token attention fusion；支持 `attention`、`concat`、`attention_multihead_pma`、`eeg_anchor`。 |
| `53_summarize_wear_moment_gates.py` | 汇总 Wear x MOMENT fusion matrix 并判定 G1/G2/G3。 |
| `54_summarize_fusion_variants.py` | 汇总 fusion variant 决策切片。 |
| `55_summarize_fusion_variant_full_paired.py` | 做全量 variant 与归档 attention 矩阵的逐 run 配对。 |
| `56_build_fusion_attention_vs_concat_evidence.py` | 从真实 JSON 生成 attention vs concat 证据文档。 |
