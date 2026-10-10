# Multi-Label Scripts

固定 token、多头回归和 EEGPT 单任务/多任务监督的 11-label 实验入口放在这里；它与 scalar fatigue calibration 链路分开。

2026-10-10 的 [MAE round1 输入合同](../../docs/research/current/0814-window/experiments/mae_round1_input_improvement_plan_20261010.md) 使用 `133_build_round1_eeg_inputs.py` 和 `134_build_round1_video_inputs.py` 冻结训练池、帧映射与共同mask；`135_run_round1_stages.py` 执行 smoke/formal/冻结下游验收，`137_continue_round1_input_pipeline.py` 在 Windows 一次性接续视频传输与 H20 队列，`136_summarize_round1_inputs.py` 保存每seed配对bootstrap。cross_day EEG 正式SSL、canonical导出及三seed冻结下游已完成并通过一致性验收；within_subject_day EEG 因107个原训练窗与holdout信号交叠按合同停止。视频cache仍在构建，其余完成状态以独立 staging 的门槛标记为准。

| 脚本 | 用途 |
| --- | --- |
| `148_run_basic_event_pool.py` | 原MAE C冻结输入的基本0906式bag_static/uniform：先等权平均窗口表示再接原H1 11-head，event MSE与val平均sRMSE选checkpoint；一项smoke后三seed，与原E1/C_EVENT精确复用对照。先冻结val选择，再独立回放/推理test，逐情绪指标及每seed 2000次subject-day配对bootstrap另存；不扩展时间范围。 |
| `144_run_round2_experiment_c.py` | 原C/B0冻结表征各执行event MSE与λ=0.1弱一致性；核对同event、mask、非EEG槽、归一化和原控制，验证λ=1新旧trainer三epoch权重/历史一致；四项smoke后重初始化跑12个正式seed，支持batch恢复。 |
| `145_select_report_round2_experiment_c.py` | 只读val完成C的同表征联合门槛、同损失B0与损失交互配对区间；冻结全部36个融合cell的A/B/C选择记录，随后允许Stage D test报告。 |
| `146_verify_round2_experiment_c.py` | 从18个C/原控制checkpoint独立回放val/test预测，验证固定token与归一化；生成λ=1原控制的窗口方差/极值诊断，供同损失比较。 |
| `147_report_round2_stage_d.py` | 要求完整val选择先冻结；统一重算36个唯一融合cell的val/test及四个单seed EEG-only模型，保存逐情绪五指标/目标预测SD、配对bootstrap、A/B/C机制与交互、同损失B0差距及窗口/日期诊断。 |
| `141_run_round2_experiment_b.py` | 第二轮仅执行B：原始C初始化，固定Wear/Video与normalization，独立last1/last2切点；末2层标准/低LR及末1层低LR，各10-epoch smoke后重新初始化跑三个seed。保护head后的CPU/CUDA随机流；batch恢复；同日协议stop。 |
| `142_report_round2_experiment_b.py` | B专用val-only报告：两个机制差、三条FT−F_C、联合推进与实用配置选择，每seed共同subject-day bootstrap 2,000次，逐情绪/seed/epoch诊断及全部cell状态。 |
| `143_verify_round2_experiment_b.py` | 独立核对B的源文件hash、冻结encoder层及固定normalization，从全encoder权重重建tail并回放val/test预测；每cell抽1,152个原始信号窗验证prefix等价，检查token健康。test性能留待Stage D。 |
| `139_run_round2_experiment_a.py` | 第二轮仅执行A：审计C/P与原bag/标签/split、生成EEG末两层prefix、四路EEG-only smoke/formal、导出两个适配token，再用原融合trainer训练两个适配来源×三个seed。Linux独占锁与batch恢复保留有效产物；同日严格协议停止；后续B由141独立推进。 |
| `140_report_round2_experiment_a.py` | 只读取A的val叶，冻结融合三seed联合门槛，报告ADAPT−PROBE、同来源融合差和输入×适配交互；每seed按subject-day共同重采样2,000次，保留逐情绪与单上游适配seed边界；test统一报告按完整第二轮Stage D。 |
| `138_report_expanded_e1.py` | 独立汇总已完成的 cross_day 扩池 E1（E_POOL）与最终 v4 E1：验证同 seed 的合同、预测身份与保存指标，输出 val/test 11 情绪的 raw/centered r、RMSE/sRMSE/MAE、三 seed 配对差及逐情绪 subject-day bootstrap；不启动训练或改写 round1 总体完成标记。 |
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
| `112_run_modality_mae.py` | 在指定正式 split 下训练 label-free EEG 或 Wear MAE：仅 `pretrain + finetune` 进行重建训练、`val` 选 checkpoint，导出 `(28819,256)` frozen window token 与 valid mask；不读取情绪标签。`--raw-normalization train_channel` 拟合普通固定尺度；`train_channel_robust` 使用训练行 median center/RMS、按窗口能量平衡损失，并将 EEG 改为通道遮挡（Wear 保留整秒遮挡）。默认保留逐秒归一化与原重建任务。 |
| `113_queue_modality_mae_stage_a.sh` | 等待 H20 持续低占用后，顺序运行 EEG（cross-day、within-subject-day）再 Wear（同两协议）的全量 Stage-A MAE；每步写独立 checkpoint/token，完成时写 `STAGE_A_COMPLETE`。 |
| `114_run_video_mae.py` | 在挂载原始 MP4 的主机上运行 label-free VideoMAE-style tubelet reconstruction；仅训练 `pretrain + finetune`、以 val reconstruction 选择 checkpoint，并导出 canonical 256D video token。cache 可从中断处恢复，无法解码的 clip 会写入 manifest 并设为无效 mask。 |
| `115_queue_video_mae_stage_a.sh` | ncc 专用 VideoMAE 队列：先做 raw-clip 解码与训练 smoke，再在 ncc 本地缓存 clips、分别完成两协议的正式训练；H20 只接收最终小型产物。 |
| `116_run_mae_window_fusion.py` | 将固定 label-free EEG-MAE 或 Wear-MAE window token 接入既有 Wphysio/DINO-B0 三模态 11-label fusion；与 B0 使用同一 head、split 和下游 seed，写出逐标签 test 指标与预测。 |
| `117_queue_mae_window_fusion.sh` | H20 队列：先对 B0/E1/W1 做 3-epoch cross-day smoke，成功后运行 cross-day 与 legacy `within_subject_day` 的三 seed 正式矩阵；不等待 VideoMAE。 |
| `118_run_mae_mt11_event_ablation.py` | 读取完成的 A1+MT11 event-bag 与三 seed B0 metrics；E1/W1/M2 替换 EEG、Wear、或 EEG+Wear 并保留 DINO A1；V1 用 `--video-mae-root` 只替换 video slot 并保留 MT11 EEG/Wphysio，均使用相同的 0814 `window_attention_regression_full_mean`。 |
| `119_queue_mae_mt11_event_ablation.sh` | H20 队列：先跑 A1+MT11 基线口径下 E1/W1 的 cross-day smoke，成功后运行两个协议、三个 seed 的 12 条单模态替换矩阵。 |
| `120_queue_m2_mt11_event_ablation.sh` | H20 队列：先跑 M2（EEG-MAE+Wear-MAE，DINO A1 保持不变）的 cross-day smoke；通过后运行两个协议、三个 seed 的正式 M2 矩阵，并引用同合同的 B0 三 seed metrics。`M2_RUNNER` 与 `M2_OUT` 允许在只读共享实验根上使用可写的用户 staging/输出目录。 |
| `121_queue_v1_mt11_event_ablation.sh` | 新 H20 V1 队列：完成的 VideoMAE token 从 `V1_VIDEO_ROOT` 读取；cross-day smoke 后执行两协议三 seeds，reference 的 MT11 EEG/Wphysio 保持固定。输出默认 `/home/wangzw/outputs/mae_mt11_v1_20261007/`。 |
| `122_queue_remaining_frozen_mae.sh` | M3（EEG+Video）、M4（Wear+Video）、M5-F（三模态）冻结组合：先 smoke，再执行两协议三 seeds 共 18 个候选 run，继续采用 `118` 的 A1+MT11 reference 合同。 |
| `123_export_mae_frozen_prefix.py` | 无标签导出最后两层前的 float32 prefix、tail 初始化、完整初始化及初始 token；EEG/Wear 在 H20、Video 在 ncc 解码缓存上计算，检查 prefix+tail 与完整编码器等价。预训练 prefix 从 checkpoint 读取固定原始信号归一化，随机 EEG/Wear 使用 `--normalization-config`；缓存指纹包含归一化统计量。预训练 EEG/Wear 可用 `--match-stage-a-batch-size` 匹配 checkpoint 导出批大小并记录 manifest，避免 fp32 批次差异，保留原误差门槛。 |
| `124_run_mae_partial_ft.py` | M5-FT/R0 的真实 encoder-tail 联合监督训练：三模态最后两层 + 原 0814 11-head 下游，encoder LR 为下游的 0.1 倍。v3 关闭 encoder dropout 但保留梯度，融合 attention 使用 float32 math 后端；下游 dropout/归一化不变。每 batch 检查 token、预测、loss、梯度与参数，失败写 `failure.json`。保存首次梯度/更新审计、epoch 进度、checkpoint 和 event predictions。 |
| `125_prepare_mae_prefixes.sh` | `h20`/`ncc` 两种角色准备三份初始化：两个 protocol 的 pretrained prefix 和跨 protocol 共享的 random prefix。ncc 新缓存落 `/tmp/wangzw_mae_remaining_20261007/`，原数据不移动。 |
| `126_queue_mae_partial_ft.sh` | v3 等待已验收 prefix 与 frozen formal，先 M5-FT/R0 × 两协议各 10 个全数据 epochs 的四-cell smoke，验收 finite history/梯度更新/metrics 后统一重跑两协议三 seeds 的 12 个正式 cell。默认输出 `.../mae_remaining_20261007/partial_v3/{smoke,formal}`，旧产物保留；`MAE_PARTIAL_OUT` 可指定新 root，错误 trap 写失败标记。 |
| `127_transfer_video_prefixes.ps1` | Windows 本地经 SSH relay 复制 ncc 三份视频 prefix 到 H20，逐份验证 SHA-256；全部通过后解锁 `126`。传输期间此后台 PowerShell 进程需保持运行。 |
| `128_queue_mae_repair_stage_a.sh` | 独立 v2 Stage-A：h20/ncc 各自通过两协议 × 模态的 15-epoch 代表性 smoke 后再全量预训练；位置编码、固定验证掩码、有限值和表征近常量 guard，旧预处理与产物保留。 |
| `129_continue_mae_repair_pipeline.ps1` | 单次本地后台控制器：等待 v2 两台主机正式完成，复制核验 Video 小型产物，启动下游与 video-prefix 准备，复制核验两个 prefix 后解锁 M5-FT。H20 endpoint 显式为 wangzw@124.174.8.252:10022。 |
| `130_queue_mae_repair_downstream.sh` | v2 修复对照按 E1/W1/V1 → M2/M3/M4 → M5-F → M5-FT 执行，共 48 个正式候选 cell；原 B0 与已证明编码不变的 R0 保留。 |
| `131_prepare_mae_repair_prefixes.sh` | 从 v2 正式 checkpoint/token 准备两协议 pretrained prefixes，h20 计算 EEG/Wear、ncc 计算 Video；使用版本指纹拒绝旧缓存误命中。 |
| `132_queue_mae_train_channel_normalization.sh` | H20 独立归一化/重建恢复队列：四个 smoke → EEG/Wear 全量预训练 → 六条冻结路线 → M5-FT/R0。`MAE_NORM_MODE=train_channel_robust` 启用35轮 smoke、典型窗口非塌缩与优于零预测的门槛；`MAE_NORM_STAGING` 指定隔离目录。默认 `train_channel` 保留原15轮尺度对照。Video 沿用 v2，R0 按协议匹配新 EEG/Wear prefix；预训练导出匹配 Stage-A 批大小。重启保留并重验已有冻结组，归档旧失败标记；48 个新正式 cell，加保留 B0/V1 共60行，最后写 `NORMALIZATION_COMPLETE`。 |

当前 repaired held-out-day 协议统一命名为 `date_in_order`，解析到 aligned `outputs/splits/date_in_order`。`93`/`94` 的正式 EEG 微调严格只解冻 EEGPT 最后两个 transformer blocks 与 final norm；256D projection、共享 trunk 和任务 head 作为 encoder 外新层训练。
