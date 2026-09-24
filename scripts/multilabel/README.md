# Multi-Label Scripts

固定 token、多头回归和 EEGPT 单任务/多任务监督的 11-label 实验入口放在这里；它与 scalar fatigue calibration 链路分开。

| 脚本 | 用途 |
| --- | --- |
| `48_run_fixed_tokens_multilabel_fusion.py` | 训练 fixed/frozen EEG/Wear/Video/Audio token 的 11-label fusion baseline，并逐标签报告指标。 |
| `92_audit_multiemotion_contract.py` | 审计 11 标签、EMA event、canonical split 和 frozen/fatigue token 对齐合同。 |
| `93_run_eegpt_single_label.py` | 以 event-equal loss 和 event-level validation selector 运行 11 套独立 EEGPT partial-FT，并逐 run 导出 256D token。 |
| `94_run_eegpt_multitask_11label.py` | 用同一 EEGPT encoder 共同学习 11 标签并导出一套 multi-task 256D EEG token。 |
| `95_run_multiemotion_fusion.py` | 在 12 条 fixed-token route 上运行 E0/H0/H1 event-aware 多标签回归。 |
| `96_summarize_multiemotion.py` | 只用 validation 锁定 route，并按 matched seeds 判定 H1 相对 H0 的晋级门槛。 |
| `97_run_supervised_eeg_fusion.py` | 在锁定 route 上运行 ST11 frozen/PFT scalar fusion 或 MT11 frozen/PFT/fatigue-transfer H1 fusion。 |
| `98_run_structure_emotion_matrix.py` | 固定 `A1_Wphysio_no_audio__eeg_eegpt_partial_ft_multitask_11label_v1` 的上游 `seed_240800`，将0814单一窗口结构与0906的18种事件结构统一接到 H1 11标签回归头，按两个协议和三个下游 seed 训练；preflight 审计标签、事件、修复 split、EEG token 的 11 标签共同监督来源。 |
| `99_summarize_structure_emotion_matrix.py` | 核对各 run 的 EMA event、标签和输入 provenance，再按协议生成 raw r、standardized RMSE、centered r 的“结构 × 11情绪”表格及 paired-baseline 宏指标 CSV；结构赢家只按 validation macro sRMSE 选。 |
| `100_audit_eegpt_single_label_bank.py` | 逐份核对22个单标签 EEGPT token 的形状、有限值、canonical sample/split、监督边界及对应 metrics，输出可复查的 `bank_audit.json`。 |
| `101_run_single_task_structure_matrix.py` | 第一条完整路线：11标签各自的 EEGPT partial-FT token + A1/Wphysio/no_audio，独立训练0814一条和0906的18条标量结构，固定上游1 seed、下游3 seeds；preflight 审计标签专属 EEG、split 和无音频 bag。 |
| `102_summarize_single_task_structure_matrix.py` | 将第一条路线按协议汇成独立的19结构 × 11情绪表；缺失 run 显示 NA，完整矩阵默认严格检查。旧配置重跑可指定 `--seeds 240729,240730,240731 --pipeline single_task_legacy_eegpt_scalar`。 |
| `103_run_legacy_single_label_eegpt.py` | 用0814旧版 `eegpt_partial_ft_v1` 的窗口 MSE、窗口 RMSE 选模和 `66/104` 编码器参数张量规则，为11标签 × 2协议生成独立 EEG token；cross-day fatigue 直接复用逐元素相同的历史 token。产物另存于 `eeg_encoder_256d_tokens_legacy_20260917/`。 |
| `104_run_legacy_single_task_matrix.py` | 旧配置的独立标量矩阵入口；cross-day fatigue 3-seed 已精确复现 `0.4240490020`，22/22 bag preflight 与正式1254/1254个下游 run 完成。 |
| `106_queue_legacy_eeg_bank.sh` | 等待当前独立微调结束后逐标签、逐协议用独立进程断点续跑旧配置 EEG token bank；失败最多重试3次，不触发下游回归。 |
| `107_queue_legacy_single_task_matrix.sh` | 逐协议/标签独立进程运行旧配置的19结构 × 3下游 seed 标量矩阵，已完成 run 跳过、失败最多重试3次；日志在 `outputs/multiemotion_legacy_replay_20260917/logs/legacy_single_task_matrix.log`。 |
| `108_finalize_legacy_single_task_matrix.sh` | 等待 `107` 成功结束后用 `102` 严格核对1254/1254并生成旧配置独立六张表；运行时传入 `107` 的 PID。 |
| `109_replace_structure_matrix_with_mt11.sh` | 等待 `94` 的两协议 MT11 token 完成并通过审计后，清除当前 fatigue-supervised 共享多头 runs/summary，原位重跑114个结构 run，再由 `99` 生成覆盖后的六张表和 README。 |
| `110_rerun_within_subject_day_splits_new.sh` | 按指定的 `/vePFS-0x0d/DailyEEG/splits_new/within_subject_day` 校验上游结果，预检后替换对应协议的 MT11 或 ST11 bag/run，重算并汇总。 |
| `111_queue_st11_within_subject_day_splits_new.sh` | 等待 MT11 上游训练退出，再串行训练 11 个标签的单任务 EEGPT，并接续 `110` 的 ST11 下游矩阵。 |
| `112_run_modality_mae.py` | 在指定正式 split 下训练 label-free EEG 或 Wear temporal MAE：仅 `pretrain + finetune` 进行重建训练、`val` 选 checkpoint，导出 `(28819,256)` frozen window token 与 valid mask；不读取情绪标签。 |
| `113_queue_modality_mae_stage_a.sh` | 等待 H20 持续低占用后，顺序运行 EEG（cross-day、within-subject-day）再 Wear（同两协议）的全量 Stage-A MAE；每步写独立 checkpoint/token，完成时写 `STAGE_A_COMPLETE`。 |

当前 repaired held-out-day 协议统一命名为 `date_in_order`，解析到 aligned `outputs/splits/date_in_order`。`93`/`94` 的正式 EEG 微调严格只解冻 EEGPT 最后两个 transformer blocks 与 final norm；256D projection、共享 trunk 和任务 head 作为 encoder 外新层训练。
