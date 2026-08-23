# Current Script Entry Points

当前 `scripts/` 顶层保留最新 EEG-aligned 四模态路线从 embedding 生成开始需要直接调用的入口，以及当前阶段性诊断入口。

## 保留入口

| 脚本 | 用途 |
| --- | --- |
| `12_extract_audio_embeddings.py` | 生成 openSMILE eGeMAPS audio 256D embedding。 |
| `15_extract_wear_embeddings.py` | 生成 Wphysio / Wdeep wearable 256D embedding。 |
| `16_run_wear_moment_matrix.py` | 训练 MOMENT-1-based wear 256D token（`wear_moment_frozen_v1` / `wear_moment_partial_ft_v1`），按 protocol/profile/seed 导出融合兼容 npz；含 `--preflight-only` 数据与推理耗时检查。 |
| `27_extract_dinov2_roi_embeddings.py` | 生成 B0/A1/A2 2x ROI DINOv2 video 256D embedding。 |
| `34_run_eeg_encoder_matrix.py` | 生成并评估 EEG-only encoder 路线，可导出 256D EEG token。 |
| `32_run_eegpt_centered_loss.py` | 将 EEG/Wear/Video/Audio tokens 接入 modality-token attention fusion；`--fusion-variant` 可选 `attention`（默认）/`concat`/`attention_multihead_pma`/`eeg_anchor` 四种融合结构做对照。 |
| `39_run_prediction_calibration_phase1.py` | 读取已冻结 fusion prediction NPZ，执行 fatigue calibration handoff Phase 0 preflight 与 Phase 1 post-hoc calibration；校准器只用 val 拟合，test 只作冻结评估。 |
| `40_plot_prediction_calibration_phase1.py` | 基于 Phase 1 产物生成 calibration 趋势图，并输出 raw r / centered r 的新旧对比表。 |
| `41_run_prediction_calibration_phase1b.py` | 对 Phase 1 代表路线执行小型分布和预测塌缩检查：train/val/test fatigue 分布、高低疲劳覆盖、预测是否集中在 2.x。 |
| `42_run_prediction_calibration_phase2.py` | 准备或执行 Phase 2 loss/sampler 小矩阵：label-bin weighted MSE、extreme-weight Huber、variance regularization 和 label-balanced sampler。 |
| `43_summarize_prediction_calibration_phase2.py` | 读取 Phase 2 report/prediction NPZ，与同路线 raw baseline 重算 val/test 指标、delta gate 和趋势图。 |
| `44_run_prediction_calibration_phase2_paired_gate.py` | 对 Phase 2 收窄候选执行 3-seed paired gate；每个候选与同 route、同 seed raw baseline 配对。 |
| `45_summarize_prediction_calibration_phase2_paired_gate.py` | 汇总 Phase 2 3-seed paired gate，按 handoff 的 Phase 2 -> Phase 3 门槛输出候选是否通过。 |
| `46_run_prediction_calibration_phase3_ordinal_head.py` | 在 Phase 2 未通过后按 raw MSE baseline 设置执行 Phase 3 ordinal/classification/hybrid head 的 3-seed paired screen。 |
| `47_summarize_prediction_calibration_phase3_ordinal_head.py` | 汇总 Phase 3 head screen，输出 `metrics_val.json`、paired delta summary 和 rounded prediction bin/confusion 摘要。 |
| `49_plot_prediction_calibration_phase0_to_phase3.py` | 汇总 Phase 0-3 的代表性动态范围、high fatigue bias、误差和相关性 trade-off，生成可视化对比图。 |
| `50_plot_prediction_calibration_phase0_to_phase3_trends.py` | 生成类似 `fatigue_top3_routes_by_protocol.png` 的 Phase 0-3 subject-day 趋势对比图。 |
| `51_run_prediction_calibration_next3_screen.py` | 小筛查三条后续路线：temporal GRU、subject-day mean+residual、高 fatigue day oversampling；按 split-safe subject-day sequence 训练并输出 val/test delta。 |
| `52_plot_prediction_calibration_next3_trends.py` | 生成 Phase 4 next-three 小筛查的 subject-day 趋势图和 Phase0 baseline 对比表。 |
| `48_run_fixed_tokens_multilabel_fusion.py` | 独立于 fatigue calibration 链路，执行 fixed/frozen EEG/Wear/Video/Audio token 的 11-label fusion baseline，并逐标签报告 RMSE、MAE、raw r、centered r 和 prediction std。 |
| `54_summarize_fusion_variants.py` | 汇总 `32_run_eegpt_centered_loss.py --fusion-variant` 决策切片的多份 report JSON，输出按协议 × 变体的 test 均值表和逐 run 明细（可附加归档矩阵参考列）。 |
| `55_summarize_fusion_variant_full_paired.py` | 把全量矩阵 variant 的 report JSON 与归档 attention 矩阵按 (protocol, experiment, eeg_branch, seed) 逐 run 配对，输出 Δ raw r / Δ centered r / Δ RMSE 与胜/负/平统计。 |
| `56_build_fusion_attention_vs_concat_evidence.py` | 从归档 attention 矩阵、决策切片与全量配对矩阵生成证据文档 `fusion_attention_vs_concat_evidence_20260820.md`（含完整 180 行附表与符号检验），数字全部读自真实 JSON。 |

## 当前复现实验顺序

1. 如需重新生成非 EEG 模态 embedding，先运行 `12`、`15`、`27`；Wear × MOMENT token（2026-08-20 起）用 `16_run_wear_moment_matrix.py`，输出 `wear_tokens/{protocol}/{profile}/seed_{seed}.npz`。
2. 运行 `34_run_eeg_encoder_matrix.py`，输出 protocol/profile/seed 对应的 EEG-only metrics、predictions，并在需要时导出 `eeg_emb (28819,256)`；当前默认包含 EEGPT frozen、DE+MLP、CBraMod frozen/partial、EEGPT partial，以及附件 `networks.py` 改写而来的 `eeg_cnn_dual_branch_v1` EEG-only supervised baseline。
3. 运行 `32_run_eegpt_centered_loss.py --experiment-set video_only --eeg-token-root eeg_encoder_256d_tokens`，完成当前 full/no_audio video-only fusion matrix；Wear × MOMENT 的收窄矩阵用 `--experiment-set custom --experiments` 显式列出 `A1_Wmoment_frozen_full` / `A1_Wmoment_ft_full` 等 route（配对对比时加 `--experiment-seed-fixed` 保证同 seed）。
4. Calibration 诊断按 `39` -> `40` -> `41`；Phase 1 只保留为诊断，Phase 2 小矩阵按 `42 --execute` 后用 `43` 汇总，收窄候选再按 `44 --execute` 和 `45` 做 3-seed paired gate；Phase 3 head screen 用 `46 --execute` 和 `47` 汇总；Phase 0-3 总览图用 `49` 重画，subject-day 趋势图用 `50` 重画；下一组三条路线的小筛查用 `51`，并用 `52` 画图/出表。
5. 多标签 fixed-token 基线用 `48_run_fixed_tokens_multilabel_fusion.py`，只训练 fusion encoder + 11-label head；它不复用 `39-47` calibration 脚本，也不把 fatigue-supervised EEG token 当作 label-free baseline。

历史实验、诊断、旧口径和调参脚本已移动到 `scripts/archive_legacy/`。
