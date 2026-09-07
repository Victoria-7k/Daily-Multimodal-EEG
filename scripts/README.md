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
| `57_run_eql_caf_phase0_baselines.py` | 为 EQL-CAF Phase 0 生成或执行固定基线命令矩阵，包装现有 `32_run_eegpt_centered_loss.py`，锁定 `EEGPT partial FT + B0 video` 等当前四 token 对照。 |
| `58_build_temporal_token_index.py` | 从 canonical window index 生成 5 个 2 秒 temporal token 边界，并输出对齐审计 manifest。 |
| `59_extract_eeg_temporal_tokens.py` | 从现有 EEG 窗口级 256D token 生成 `global_repeat_smoke_v1` temporal NPZ；只用于 58→63→64 管线 smoke，不作为真实 lag-aware 证据。 |
| `60_extract_wear_temporal_tokens.py` | 从现有 Wear 窗口级 256D token 生成 `global_repeat_smoke_v1` temporal NPZ；真实 2 秒 raw Wear encoder 后续替换同一契约。 |
| `61_extract_video_temporal_tokens.py` | 从现有 Video/Face 窗口级 256D token 生成 `global_repeat_smoke_v1` temporal NPZ；支持 `video_emb`，缺省可回退 `face_emb`。 |
| `62_extract_audio_temporal_tokens.py` | 从现有 Audio 窗口级 256D token 生成 `global_repeat_smoke_v1` temporal NPZ；真实 2 秒 audio LLD/functionals 后续替换同一契约。 |
| `63_pack_eql_caf_tokens.py` | 合并 EEG/Wear/Video/Audio 四份 temporal token NPZ，强校验 sample_id、shape、mask 和 quality 字段后输出 EQL-CAF 训练 NPZ。 |
| `64_run_eql_caf_fatigue_matrix.py` | 读取 packed temporal token NPZ，训练 B1/B2/M1-M5 fatigue 矩阵；B1 为 temporal concat，B2 为 temporal self-attention，M1-M5 为 EEG-anchored lag/quality/residual 逐步消融。 |
| `65_run_wear_only_fm_phase1.py` | 执行 Wear-only FM Phase 0-1：审计 `wear_complete_mask`，缓存 PaPaGei-S PPG、HARNet10 ACC、NormWear GSR 三路 label-free frozen embeddings，并输出 manifest/smoke/report。 |
| `66_run_wear_only_fm_phase2.py` | 执行 Wear-only FM Phase 2 36-run 主矩阵：`Wphysio`、`Wdeep`、`Wmoment_frozen`、`W3FM_frozen` 在同一 `wear_complete_mask`、split 和 seed 下训练 wear-only fatigue head。 |
| `67_summarize_wear_only_fm_phase2.py` | 汇总 Wear-only FM Phase 2 的 protocol × route 三 seed mean/std，并输出 W3FM 对当 seed 最强 baseline 的 delta；不执行 Phase 3 bootstrap。 |
| `68_summarize_wear_only_fm_phase3.py` | 对 Wear-only FM Phase 2 的冻结预测执行 Phase 3 判定：选择同 seed 最强 baseline、做 subject-day block paired bootstrap、输出 gate 与图表。 |
| `69_run_wear_only_fm_internal_ablation.py` | 执行 Phase 3 后的诊断扩展：`W3FM_no_ppg/no_acc/no_gsr` 三路 leave-one-out 和 `W3FM_ppg_partial_ft`，用于定位 W3FM 内部拖累来源，不作为正式 promotion gate。 |
| `70_summarize_wear_fm_training_diagnostics.py` | 汇总 Wear-only FM Phase 2 与内部消融的 `val_history.csv`/`metrics.json`，识别 best epoch 偏早、train/val gap 和 fast-overfit 模式，并给出参数 sweep 建议。 |
| `71_summarize_wear_fm_training_sweep.py` | 汇总 Wear-only FM 的 W3FM 训练参数 sweep，对比原始 `W3FM_frozen` 与 Phase 2 最强 baseline，输出配置级 RMSE/raw r/centered r delta 表。 |
| `72_diagnose_w3fm_failure_modes.py` | 对 W3FM 失败来源做解释性诊断：三路 FM embedding split shift、top-feature correlation stability、轻量 ridge probe、checkpoint gate 权重与 test error association；不作为正式 promotion gate。 |
| `73_build_daily_affect_bags.py` | 从 canonical EEG-aligned 窗口级 EEG/Wear/Video/Audio 256D embedding 构建 daily-affect EMA bags，输出 `(N_ema,23,4,256)` tokens、`modality_mask` 和 bag-level split。 |
| `74_run_daily_affect_phase0_baselines.py` | 从不可变的 `20260903/bags` 读取 EMA bags，在独立 routefix root 成对训练 `window_replicated` 和 `bag_static`。两者共享窗口 attention/query-pooling、adapter、normalization 与序数目标，只有复制窗口监督和 EMA-bag 监督不同；显式交叉 train-only normalization 和 adapter mode。 |
| `75_run_daily_affect_state_matrix.py` | 在独立 routefix root 运行 state/prior/global-kernel/dynamic-kernel 矩阵；`--routing-profiles` 显式执行 P0 no-prior、P1 categorical entropy、P2 cumulative entropy、P3 ordinal mix、P4 calibrated probe 与 P5 end-to-end routing。 |
| `76_summarize_daily_affect_results.py` | 按 experiment/normalization/adapter/objective/routing 分开汇总；对同 seed、同目标的 static baseline 做配对，输出 mean+/-std、方向一致性，以及可用时按 subject-day block 的 per-seed bootstrap CI。 |
| `77_plot_daily_affect_diagnostics.py` | 每个 protocol 输出少量 seed-aware atlas；分组键包含 experiment、normalization、adapter、objective 和 routing，避免 P1-P5 与 legacy run 混图；拼图显示 routing weight、probe difficulty、temporal kernel、每模态 Probe 的 Accuracy/NLL/ECE/RPS 和 best-QWK confusion。 |
| `78_report_daily_affect_focused_diagnostics.py` | 对指定 adapter 和 candidate/baseline experiment 的候选生成 kernel/modality 机制小报告，默认聚焦 routefix `cross_day / A1_Wphysio_full`。 |
| `79_run_daily_affect_focused_ablation.py` | 在独立 routefix root 上跑 static/state/prior/global-kernel/fixed-kernel/dynamic-kernel 消融，统一使用当前序数主目标与训练期 modality dropout。 |
| `80_run_daily_affect_focused_robustness.py` | 对 frozen checkpoint 做 clean、单模态 missing、两模态成对 missing、noise、shuffle 压力测试；记录 ordinal/bridge 指标、保护的无剩余模态 event 数以及扰动后的 routing/difficulty 变化。 |
| `81_report_daily_affect_prior_guidance.py` | 在指定 normalization、adapter 和 experiment 下汇总真实 state-prior compatibility、ordinal difficulty 与 dynamic-kernel 的配对消融。 |
| `82_run_daily_affect_bottleneck_audit.py` | 在独立 routefix audit root 比较固定 look-back/fixed-kernel，审计四模态 test-day drift 与 video 覆盖/质量；adapter mode 显式记录。 |
| `83_run_daily_affect_routing_factorial.py` | 在独立 routefix diagnostic root 以 2x2 设计拆开 train-only token normalization 与 adapter sharing；采用当前序数主目标，只在 test event 汇总 routing weight，不参与 promotion。 |
| `84_compare_window_daily_event_level.py` | 将冻结的窗口回归预测按一个 EMA event 的 23 个窗口聚合，与 daily-affect 的 event 级五类预测放入同一 QWK/F1/ordinal-MAE/raw-r/RMSE 表；严格要求每个 event 的 23 个窗口来自同一 val/test leaf split，校准只在 val event 拟合，并写出 primary full-mean 的逐 seed/event `event_predictions.csv`。 |
| `85_plot_daily_affect_event_series.py` | 读取 current daily-affect 的已保存 test prediction，为 static baseline 与 dynamic candidate 输出同一 held-out event 顺序上的真实标签、seed 平均 expected score、10--90% seed 区间及逐 event CSV；横轴显式按 subject-day 分组，避免误读为连续传感器流。 |
| `86_plot_window_daily_event_bridge_series.py` | 读取 `84` 的严格对齐逐 event CSV 与 native metrics，在同一 held-out event 顺序上绘制窗口主线、daily static 和 daily dynamic 三面板 expected-score 图；只用于 matched partial-FT 三 seed bridge。 |
| `87_report_window_daily_range_precision.py` | 在 `84` 的逐 event prediction 上量化窗口与 daily 的 P10--P90/IQR 动态范围、raw r、label-mean separation、within-label variation 与 label-explained variance，并按 subject-day bootstrap 输出 daily-minus-window 配对区间。 |

## 当前复现实验顺序

1. 如需重新生成非 EEG 模态 embedding，先运行 `12`、`15`、`27`；Wear × MOMENT token（2026-08-20 起）用 `16_run_wear_moment_matrix.py`，输出 `wear_tokens/{protocol}/{profile}/seed_{seed}.npz`。
2. 运行 `34_run_eeg_encoder_matrix.py`，输出 protocol/profile/seed 对应的 EEG-only metrics、predictions，并在需要时导出 `eeg_emb (28819,256)`；当前默认包含 EEGPT frozen、DE+MLP、CBraMod frozen/partial、EEGPT partial，以及附件 `networks.py` 改写而来的 `eeg_cnn_dual_branch_v1` EEG-only supervised baseline。
3. 运行 `32_run_eegpt_centered_loss.py --experiment-set video_only --eeg-token-root eeg_encoder_256d_tokens`，完成当前 full/no_audio video-only fusion matrix；`--token-normalization shared` 保持历史默认的跨模态共享 train-only 统计，`--token-normalization per_modality` 为每个模态 slot 单独拟合 train-only mean/std；Wear × MOMENT 的收窄矩阵用 `--experiment-set custom --experiments` 显式列出 `A1_Wmoment_frozen_full` / `A1_Wmoment_ft_full` 等 route（配对对比时加 `--experiment-seed-fixed` 保证同 seed）。
4. Calibration 诊断按 `39` -> `40` -> `41`；Phase 1 只保留为诊断，Phase 2 小矩阵按 `42 --execute` 后用 `43` 汇总，收窄候选再按 `44 --execute` 和 `45` 做 3-seed paired gate；Phase 3 head screen 用 `46 --execute` 和 `47` 汇总；Phase 0-3 总览图用 `49` 重画，subject-day 趋势图用 `50` 重画；下一组三条路线的小筛查用 `51`，并用 `52` 画图/出表。
5. 多标签 fixed-token 基线用 `48_run_fixed_tokens_multilabel_fusion.py`，只训练 fusion encoder + 11-label head；它不复用 `39-47` calibration 脚本，也不把 fatigue-supervised EEG token 当作 label-free baseline。
6. EQL-CAF 路线先用 `57` 复现当前四 token 基线，再用 `58` 生成 5×2s temporal index；`59-62` 当前只提供 `global_repeat_smoke_v1` 管线验证产物，真实 temporal encoder 完成后仍输出同一 NPZ 契约；随后用 `63` 合包，并用 `64` 按 B1/B2/M1-M5 逐阶段跑矩阵。主协议保持 `cross_day` 与 `within_subject_day_strict`，旧 `within_subject_day` 仅作历史诊断。
7. Wear-only FM 路线先用 `65_run_wear_only_fm_phase1.py --mode stage` 生成 10 秒 PPG/ACC/GSR staged inputs，再用 `--mode smoke` 和 `--mode batch` 缓存三路外部 frozen encoder embeddings；这些缓存不读取 fatigue label，后续 `W3FM_frozen` 的投影/gate/head 训练属于监督下游阶段。
8. Wear-only FM Phase 2 用 `66_run_wear_only_fm_phase2.py` 跑 `cross_day`、`within_subject_day`、`cross_subject` × 4 routes × 3 seeds；完成后用 `67_summarize_wear_only_fm_phase2.py` 生成 Phase 2 摘要。Phase 3 的 subject-day block bootstrap 另行执行，避免把 Phase 2 均值表当作显著性结论。
9. Wear-only FM Phase 3 用 `68_summarize_wear_only_fm_phase3.py --phase2-root outputs/wear_fm/phase2 --bootstrap-iters 2000 --out-root outputs/wear_fm/phase3 --plot`。当前 gate 判定为 `stop_after_phase3_do_not_run_second_round_ablation`，因此不进入第二轮消融。
10. 如需解释 W3FM 失败来源，可用 `69_run_wear_only_fm_internal_ablation.py` 在 `cross_day` 与 `within_subject_day` 上跑 `W3FM_no_ppg,W3FM_no_acc,W3FM_no_gsr,W3FM_ppg_partial_ft` 诊断矩阵；结论只用于定位问题，不覆盖 Phase 3 的正式停止判定。
11. 如需检查 Wear-only FM 的早停/过拟合问题，用 `70_summarize_wear_fm_training_diagnostics.py` 汇总本地同步的 Phase 2、internal ablation 与 training sweep history；`66` 和 `69` 的新 run 会额外记录 `val_raw_r`、`val_within_subject_centered_r`、`val_prediction_std` 和 `selection_score`，并可用 `--selection-metric` 做诊断选择。训练参数 sweep 结果用 `71_summarize_wear_fm_training_sweep.py` 汇总；W3FM 的 ACC/embedding/gate failure-mode 解释用 `72_diagnose_w3fm_failure_modes.py` 生成。
12. Daily-affect ordinal route 固定从 canonical 28,819 个 EEG-aligned 10 秒窗口构建每 event 23 窗口 EMA bags，使用 `modality_mask (N_ema,23,4)`，不读取 EQL-CAF `token_mask` 或 packed temporal token。`20260903/bags` 是只读数据输入；当前语义的训练一律写入 `daily_affect_*_routefix_20260905` 根目录。执行顺序为 `74` 的 2x2 static baseline，`75 --routing-profiles` 的 P0-P5/state/kernel 对照，`76 --bootstrap-iters 2000` 的 seed paired 汇总与 subject-day CI，`77` 的少量 atlas；focused line 再依次用 `78-83` 做机制、消融、missing/corruption、prior、bottleneck 与 routing-factorial 查验。promotion 仍看 matched ordinal paired gate，Expected RMSE/raw r/within-subject centered r 仅作为同 1--5 分数轴的横向读数。

历史实验、诊断、旧口径和调参脚本已移动到 `scripts/archive_legacy/`。
