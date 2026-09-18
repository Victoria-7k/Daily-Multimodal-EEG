# 字段契约

> 这是查表材料。如果还不理解行为路径，先读 [一条事件如何变成 smoke embedding](../walkthroughs/one-real-run.md)。

## manifest 事件字段

| 字段 | 含义 | 来源 |
| --- | --- | --- |
| `event_id` | 稳定事件编号，来自 subject、session、segment 和评分行号 | [manifest 构建器](../../src/daily_multimodal/manifest/build_manifest.py) |
| `subject_id`、`session_id`、`segment_id` | EEG BIDS 目录和 beh 文件名解析出的身份信息 | [manifest 构建器](../../src/daily_multimodal/manifest/build_manifest.py) |
| `absolute_onset_time` | 评分事件绝对时间，是跨模态对齐主键 | [manifest 构建器](../../src/daily_multimodal/manifest/build_manifest.py) |
| `eeg_recording_start_time`、`eeg_onset_seconds`、`eeg_sampling_frequency` | EEG sidecar 和 beh 行里的时间定位信息 | [manifest 构建器](../../src/daily_multimodal/manifest/build_manifest.py) |
| `wear_ppg_path`、`wear_gsr_path`、`wear_acc_path` | 覆盖事件时间的 wear CSV 路径 | [wear 文件发现](../../src/daily_multimodal/io/wear.py) |
| `candidate_mp4_paths`、`candidate_audio_paths` | 日期目录下的媒体候选 | [manifest 构建器](../../src/daily_multimodal/manifest/build_manifest.py) |
| `has_eeg`、`has_ppg`、`has_gsr`、`has_acc`、`has_video`、`has_audio` | manifest 阶段的可用性布尔值 | [manifest 构建器](../../src/daily_multimodal/manifest/build_manifest.py) |
| `is_complete_wear_event` | PPG、GSR、ACC 都存在 | [manifest 构建器](../../src/daily_multimodal/manifest/build_manifest.py) |
| `is_complete_multimodal_candidate` | EEG、三路 wear、视频和音频日期候选都存在 | [manifest 构建器](../../src/daily_multimodal/manifest/build_manifest.py) |
| `labels` | 情绪评分列集合 | [schema 定义](../../src/daily_multimodal/schema.py) |

## 精确视频候选字段

| 字段 | 含义 | 来源 |
| --- | --- | --- |
| `video_candidates` | 和事件窗口有重叠的 MP4 片段列表 | [视频音频对齐模块](../../src/daily_multimodal/alignment/video_audio_alignment.py) |
| `mp4_start_time`、`mp4_end_time` | 由 `ffprobe` 的 `creation_time` 和 `duration` 得到的本地时间范围 | [视频音频对齐模块](../../src/daily_multimodal/alignment/video_audio_alignment.py) |
| `clip_start_seconds`、`clip_end_seconds` | 事件窗口在 MP4 内的截取位置 | [视频音频对齐模块](../../src/daily_multimodal/alignment/video_audio_alignment.py) |
| `overlap_seconds`、`covers_window` | MP4 与事件窗口的重叠秒数和完整覆盖标记 | [视频音频对齐模块](../../src/daily_multimodal/alignment/video_audio_alignment.py) |
| `has_audio_stream`、`audio_codec`、`audio_sample_rate`、`audio_channels` | MP4 内第一条音频流的信息 | [视频音频对齐模块](../../src/daily_multimodal/alignment/video_audio_alignment.py) |

## ffprobe cache 与对齐报告字段

| 字段 | 含义 | 来源 |
| --- | --- | --- |
| `ok` | 单个 MP4 的 ffprobe 是否成功 | [视频音频对齐模块](../../src/daily_multimodal/alignment/video_audio_alignment.py) |
| `metadata` | 成功探测时保存的 ffprobe JSON 元数据 | [视频音频对齐模块](../../src/daily_multimodal/alignment/video_audio_alignment.py) |
| `error_type`、`error` | 失败探测的异常类型和错误文本 | [视频音频对齐模块](../../src/daily_multimodal/alignment/video_audio_alignment.py) |
| `retry_failed` | 本次运行是否重试 cache 中失败记录 | [视频音频对齐模块](../../src/daily_multimodal/alignment/video_audio_alignment.py) |
| `ffprobe_timeout_seconds` | 传给 ffprobe 的超时秒数；`null` 表示不限时 | [视频音频对齐模块](../../src/daily_multimodal/alignment/video_audio_alignment.py) |
| `events_with_precise_video_overlap`、`events_with_precise_audio_overlap` | 精确视频和音频覆盖事件数 | [视频音频对齐模块](../../src/daily_multimodal/alignment/video_audio_alignment.py) |

## window index 字段

| 字段 | 含义 | 来源 |
| --- | --- | --- |
| `sample_id` | 窗口级稳定样本编号，格式为 `{event_id}_win-0000` | [窗口构建函数](../../src/daily_multimodal/alignment/event_windows.py) |
| `window_start_time`、`window_end_time` | 窗口绝对时间范围 | [窗口构建函数](../../src/daily_multimodal/alignment/event_windows.py) |
| `window_start_offset_seconds`、`window_end_offset_seconds` | 相对事件发生时刻的秒级 offset | [窗口构建函数](../../src/daily_multimodal/alignment/event_windows.py) |
| `event_window_start_seconds`、`event_window_end_seconds` | 当前事件展开范围；默认 `-120` 到 `0`，用于说明同一事件的 12 个 10 秒样本都来自评分前两分钟 | [窗口构建函数](../../src/daily_multimodal/alignment/event_windows.py) |
| `required_history_seconds`、`pre_event_history_seconds` | 事件进入窗口索引所需的前置历史秒数，以及 manifest 中可推断的 EEG 事件前历史秒数 | [窗口构建函数](../../src/daily_multimodal/alignment/event_windows.py) |
| `label_columns` | 从 manifest 的 `labels` 搬到窗口记录里的标签字典 | [窗口构建函数](../../src/daily_multimodal/alignment/event_windows.py) |
| `has_wear`、`has_face`、`has_audio` | 窗口层给 embedding 使用的模态可用性 | [窗口构建函数](../../src/daily_multimodal/alignment/event_windows.py) |

上表描述的是 legacy 03 窗口构建入口的字段。daily-affect ordinal route 读取当前 canonical EEG-aligned index，要求每个 EMA event 恰好有 `23` 个 10 秒窗口，`event_window_id=0..22`，窗口 stride 为 5 秒；这些 event-level bag 字段见下方 daily-affect 契约。

## window index summary 字段

| 字段 | 含义 | 来源 |
| --- | --- | --- |
| `events_total`、`events_selected`、`events_skipped`、`windows_total` | 输入事件数、保留事件数、跳过事件数和展开后的窗口总数 | [窗口索引入口](../../scripts/archive_legacy/03_build_window_index.py) |
| `skip_reasons` | 跳过原因计数；当前包括 `insufficient_pre_event_history` 和 `insufficient_video_coverage` | [窗口构建函数](../../src/daily_multimodal/alignment/event_windows.py) |
| `skipped_events` | 每个被跳过事件的 `event_id`、`subject_id`、`session_id`、`absolute_onset_time`、原因和可用历史秒数 | [窗口构建函数](../../src/daily_multimodal/alignment/event_windows.py) |

## real cache face-filter 字段

| 字段 | 含义 | 来源 |
| --- | --- | --- |
| `selected_window_count` | stage-12 face-presence 过滤后继续准备 cache 的窗口数 | [真实缓存准备模块](../../src/daily_multimodal/embeddings/cache.py) |
| `face_filter.enabled`、`kept_count`、`dropped_count` | 是否启用人脸预检、保留窗口数和剔除窗口数 | [真实缓存准备模块](../../src/daily_multimodal/embeddings/cache.py) |
| `face_filter.dropped_no_face_count`、`dropped_failure_count`、`dropped_windows` | 无脸窗口数、检测失败或源缺失剔除数，以及对应 `sample_id` / `event_id` 列表 | [真实缓存准备模块](../../src/daily_multimodal/embeddings/cache.py) |
| `face_presence` | 写入过滤后窗口索引的检测摘要，含 `detector`、`frame_count`、`detected_frame_count`、检测 clip 秒数、`max_face_count`、`main_face_bbox`、`main_face_area_ratio`、`detected_orientations`、`retained_without_detected_face` 和 `retention_reason`；服务器全量 midpoint 路径的 detector 为 `opencv_haar_frontalface_default_alt_profile_rot180_ffmpeg_midpoint` | [真实缓存准备模块](../../src/daily_multimodal/embeddings/cache.py) |

## embedding 输出字段

| 字段 | 含义 | 来源 |
| --- | --- | --- |
| `eeg_emb`、`wear_emb`、`face_emb`、`audio_emb` | 四个 `(N, 256)` float 数组 | [批处理保存器](../../src/daily_multimodal/embeddings/pipeline.py)、[真实打包器](../../src/daily_multimodal/embeddings/real_pipeline.py) |
| `modality_mask` | `(N, 4)` int 数组，顺序为 `[eeg, wear, face, audio]` | [basic encoder](../../src/daily_multimodal/embeddings/basic.py)、[真实打包器](../../src/daily_multimodal/embeddings/real_pipeline.py) |
| `labels` | 每个样本的标签 JSON 字符串 | [批处理保存器](../../src/daily_multimodal/embeddings/pipeline.py)、[真实打包器](../../src/daily_multimodal/embeddings/real_pipeline.py) |
| `source_paths` | 每个样本使用的源路径 JSON 字符串 | [批处理保存器](../../src/daily_multimodal/embeddings/pipeline.py)、[真实打包器](../../src/daily_multimodal/embeddings/real_pipeline.py) |
| `quality_flags` | 质量信息；basic 路径写在 JSON 报告中，真实 all-real `.npz` 也会按样本写入 JSON 字符串数组 | [批处理保存器](../../src/daily_multimodal/embeddings/pipeline.py)、[真实打包器](../../src/daily_multimodal/embeddings/real_pipeline.py) |
| `encoder_versions` | 真实 all-real `.npz` 中每个样本的四模态 encoder profile JSON 字符串 | [真实打包器](../../src/daily_multimodal/embeddings/real_pipeline.py) |

## EQL-CAF temporal token 字段

| 字段 | 含义 | 来源 |
| --- | --- | --- |
| `token_start_seconds`、`token_end_seconds` | 每个 10 秒窗口内的固定 temporal token 边界；默认 `[0,2,4,6,8]` 与 `[2,4,6,8,10]` | [temporal 切片模块](../../src/daily_multimodal/temporal/window_slicing.py)、[temporal index 入口](../../scripts/eql_caf/58_build_temporal_token_index.py) |
| `eeg_tokens`、`wear_tokens`、`video_tokens`、`audio_tokens` | 四模态 temporal token，shape 均为 `(N,5,256)`；`5` 来自 10 秒窗口按 2 秒切片 | [temporal token 契约](../../src/daily_multimodal/temporal/token_contract.py) |
| `token_mask` | `(N,4,5)` int/bool mask，模态顺序固定为 `[eeg, wear, video, audio]`；训练时按 token 层屏蔽缺失片段 | [temporal token 契约](../../src/daily_multimodal/temporal/token_contract.py) |
| `modality_mask` | `(N,4)` int mask，由每个模态的 `token_mask.any(axis=2)` 得到，用于窗口级模态可用性审计 | [temporal token 契约](../../src/daily_multimodal/temporal/token_contract.py) |
| `quality_features` | `(N,4,5,Q)` float 质量特征矩阵；不同模态原始 Q 可不同，合包时按最大 Q 右侧补零 | [temporal token 契约](../../src/daily_multimodal/temporal/token_contract.py)、[temporal pack 入口](../../scripts/eql_caf/63_pack_eql_caf_tokens.py) |
| `quality_feature_names_json`、`encoder_versions_json`、`source_paths_json` | packed temporal NPZ 的审计 metadata，保存质量特征名、encoder 版本和每个模态来源路径 | [temporal pack 模块](../../src/daily_multimodal/temporal/pack_temporal_tokens.py) |
| `global_repeat_smoke_v1` | `59-62` 当前 smoke 模式写入的 encoder version：把同一个窗口级 256D token 复制到 5 个时间片，只用于验证管线和契约，不用于证明窗口内 temporal fusion 或 lag bias | [global repeat 模块](../../src/daily_multimodal/temporal/global_repeat_tokens.py) |

## `within_subject_day` split 标识

从 2026-09-13 起，`within_subject_day` 是修复后 held-out-day 划分的唯一正式名称。新运行必须读取 `<aligned-root>/outputs/splits/within_subject_day`，并在 manifest 中记录该 split root。

| Split 路径 | 划分单位与含义 | train / val / test 窗口数 | 重叠审计 |
| --- | --- | --- | --- |
| `<aligned-root>/outputs/splits/within_subject_day` | 按 `subject-day pair` 整体划分；每个被试按日期排序，约 60% 日期训练、20% 验证、20% 测试 | `17135 / 5658 / 6026` | train/val/test 的 subject-day 与 event overlap 均为 `0` |
| `/vePFS-0x0d/DailyEEG/splits_new/within_subject_day` | 历史宽松窗口级划分；当前账号无删除权限，已从所有当前入口禁用 | `17243 / 5708 / 5868` | train-val、train-test、val-test 均共享全部 `150` 个 subject-day；train-val 与 val-test 各共享 `107` 个 EMA event |

旧宽松 split 的历史结果只能以 `legacy within-day window split` 标识，不能与 canonical `within_subject_day` 合并。重复的 `within_subject_day_strict` 活动目录已移出 `outputs/splits`；旧报告可保留该名称作为 provenance，新命令和新报告统一使用 `within_subject_day`。构建规则见 [历史 split 构建入口](../../scripts/archive_legacy/33_build_strict_within_subject_day_split.py)，overlap 记录见 [split audit](../../outputs/server_sync/eegpt_centered_improvement/split_audit_subject_day.json)。

## Daily-affect EMA bag 字段

Daily-affect ordinal route 由 `scripts/daily_affect/73_build_daily_affect_bags.py` 调用 [EMA bag 构建模块](../../src/daily_multimodal/daily_affect/ema_bags.py)。它读取窗口级 EEG/Wear/Video/Audio 256D embedding，把同一 EMA event 的 `23` 个 EEG-aligned 窗口聚成一个监督样本；该路线不读取 EQL-CAF packed temporal NPZ，也不使用 `token_mask`。

| 字段 | 含义 | 来源 |
| --- | --- | --- |
| `tokens` | `(N_ema,23,4,256)` float，四模态顺序固定为 `[eeg, wear, video, audio]`；23 个窗口来自同一 event 的 `event_window_id=0..22` | [EMA bag 构建模块](../../src/daily_multimodal/daily_affect/ema_bags.py) |
| `modality_mask` | `(N_ema,23,4)` int/bool 窗口级模态可用性 mask；daily-affect 训练和诊断只用这个 mask 做缺失屏蔽 | [EMA bag 构建模块](../../src/daily_multimodal/daily_affect/ema_bags.py) |
| `label`、`label_zero_based` | event-level ordinal fatigue 标签；`label` 保留原始 `1..5`，`label_zero_based` 转为分类训练用的 `0..4` | [EMA bag 构建模块](../../src/daily_multimodal/daily_affect/ema_bags.py) |
| `event_id`、`subject_id`、`day_id` | bag 级身份字段；构建时要求同一 event 的 23 个窗口共享标签，并投影为一个 bag-level split | [EMA bag 构建模块](../../src/daily_multimodal/daily_affect/ema_bags.py) |
| `sample_id_matrix`、`event_window_id` | 每个 bag 内 23 个窗口的原始 `sample_id` 和窗口序号矩阵，用于回查窗口级 embedding 与 index | [EMA bag 构建模块](../../src/daily_multimodal/daily_affect/ema_bags.py) |
| `split`、`window_leaf_split_counts_json` | bag-level split 名称和该 event 的 23 个窗口 leaf split 组成；单一 leaf split 直接保留，`pretrain/finetune` 混合归入 train，train/val/test 边界混合按窗口多数票投影，并在 report 中记录 `event_split_policy` | [EMA bag 构建模块](../../src/daily_multimodal/daily_affect/ema_bags.py) |
| `pretrain_index`、`finetune_index`、`train_index`、`val_index`、`test_index` | bag-level 训练协议索引；`train_index` 是 `pretrain_index + finetune_index`，模型只用 train/val 选型，test 不参与调参 | [EMA bag 构建模块](../../src/daily_multimodal/daily_affect/ema_bags.py) |
| `route_id`、`source_npz_json`、`supervision_boundary` | 路线名、四模态源 NPZ 路径和监督边界说明；EEG supervised branch 会在有效 `route_id` 中追加 branch 后缀 | [EMA bag 构建模块](../../src/daily_multimodal/daily_affect/ema_bags.py) |

Routefix run 的 `metrics.json` 与 `config.json` 固定记录 `normalization`、`adapter_mode`、`objective_id`、`routing_id` 和 `experiment_id`。`objective_id` 标识 class-weighted CE、soft cumulative ordinal loss 和 ranking 权重；`routing_id` 标识 Probe 类型、难度来源、`beta_ord` 与是否 detach difficulty；`experiment_id` 区分 native/P0-P5、focused、bottleneck 和 routing-factorial。`diagnostics.npz` 仍以 EMA event 为第一维，包含 `modality_weights (N,23,4)`、`modality_difficulty (N,4)`、`temporal_weights (N,23)`、`kernel_mixture (N,3)` 和按 Probe 类型二选一的 `probe_ordinal_logits (N,4,4)` 或 `probe_class_logits (N,4,5)`。

2026-09-03 正式构建使用 `/vePFS-0x0d/DailyEEG/splits_new` 下的 `cross_subject`、`cross_day`、`within_subject_day`。27 个正式 bag 的形状均为 `tokens (1253,23,4,256)` 和 `modality_mask (1253,23,4)`；`cross_subject/cross_day` 各记录 `42` 个 `pretrain/finetune` 训练叶子混合 event，`within_subject_day` 记录 `214` 个 train/val/test 边界投影 event。

## Face ROI 预处理质量字段

| 字段 | 含义 | 来源 |
| --- | --- | --- |
| `face_roi_crop_enabled`、`face_roi_crop_scale`、`face_roi_output_size` | OpenFace 前是否生成主脸 ROI clip、ROI 相对主脸框的放大倍数和输出边长；当前默认 `2.0` 与 `640` | [Face 真实模块](../../src/daily_multimodal/embeddings/face_real.py) |
| `face_roi_frame_count`、`face_roi_detected_frame_count`、`face_roi_filled_missing_frame_count` | ROI clip 的输出帧数、直接检测到主脸 ROI 的帧数、以及沿用上一/后续 ROI 补齐的无脸帧数 | [Face 真实模块](../../src/daily_multimodal/embeddings/face_real.py) |
| `face_roi_full_frame_fallback` | 整个窗口没有可靠主脸 ROI 时是否退回全画面，避免 Haar 小框假阳性把人裁没 | [Face 真实模块](../../src/daily_multimodal/embeddings/face_real.py) |

## fair embedding ablation 输出字段

| 字段 | 含义 | 来源 |
| --- | --- | --- |
| `row_count` | basic 与 real 对齐后参与公平对照的行数 | [Fair ablation 模块](../../src/daily_multimodal/training/fair_embedding_ablation.py) |
| `sample_id_aligned` | basic 和 real `.npz` 的 `sample_id` 是否完全同序 | [Fair ablation 模块](../../src/daily_multimodal/training/fair_embedding_ablation.py) |
| `basic_aligned` | 原始 basic embedding 在 real 对齐样本上的参考实验 | [Fair ablation 模块](../../src/daily_multimodal/training/fair_embedding_ablation.py) |
| `basic_no_path` | 把 EEG/Face/Audio 中路径派生信号置为常量后的 basic 对照 | [Fair ablation 模块](../../src/daily_multimodal/training/fair_embedding_ablation.py) |
| `path_only` | 只用 `sample_id`、`event_id`、`subject_id`、`session_id`、`source_paths` 派生向量的泄漏控制 | [Fair ablation 模块](../../src/daily_multimodal/training/fair_embedding_ablation.py) |
| `real` | 去掉元数据字段后的 real embedding 对照 | [Fair ablation 模块](../../src/daily_multimodal/training/fair_embedding_ablation.py) |
| `modalities` | 本次对照实际使用的模态顺序；默认沿用 full，也可用 `--modalities eeg,wear,audio` 排除覆盖稀疏的 Face | [Fair ablation 模块](../../src/daily_multimodal/training/fair_embedding_ablation.py) |
| `test_pearson_r`、`test_r` | JSON 和 Markdown 表里的测试集 Pearson r，和 RMSE/MAE 一起用于 fatigue 验证 | [Fair ablation 模块](../../src/daily_multimodal/training/fair_embedding_ablation.py) |
| `failure_count`、`failures` | 对齐失败或行数不一致时的失败记录 | [Fair ablation 模块](../../src/daily_multimodal/training/fair_embedding_ablation.py) |

## EEG coverage audit 字段

| 字段 | 含义 | 来源 |
| --- | --- | --- |
| `classification` | EEG 窗口相对 BDF 记录范围的分类：`in_range`、`negative_offset`、`after_recording_end`、`partial_overlap`、`whole_day_shift_candidate` 或 `out_of_range` | [EEG coverage 模块](../../src/daily_multimodal/alignment/eeg_coverage.py) |
| `start_offset_seconds`、`end_offset_seconds` | EEG 窗口相对 BDF 起点的秒级范围 | [EEG coverage 模块](../../src/daily_multimodal/alignment/eeg_coverage.py) |
| `bdf_duration_seconds` | BDF 记录时长，来自窗口字段、cache 字段或 EEG sidecar | [EEG coverage 模块](../../src/daily_multimodal/alignment/eeg_coverage.py) |
| `overlap_seconds` | 窗口与 BDF 记录范围的重叠秒数 | [EEG coverage 模块](../../src/daily_multimodal/alignment/eeg_coverage.py) |
| `whole_day_shift_candidate`、`suggested_shift_seconds` | 是否疑似整天偏移，以及建议尝试的 `-86400` 或 `86400` 秒平移 | [EEG coverage 模块](../../src/daily_multimodal/alignment/eeg_coverage.py) |
| `affected_subject_sessions` | 非 `in_range` 窗口涉及的 `subject/session` 列表 | [EEG coverage audit 入口](../../scripts/archive_legacy/19_audit_eeg_coverage.py) |
| `eeg_window_before_recording`、`eeg_window_after_recording`、`eeg_window_partial_overlap` | EEG real embedding 中由 coverage 分类派生的失败类型 | [EEG 真实模块](../../src/daily_multimodal/embeddings/eeg_real.py) |

## v2 profile 与 subject CV 字段

| 字段 | 含义 | 来源 |
| --- | --- | --- |
| `pooling`、`pooled_feature_dim` | Audio v2 profile 使用的池化方式；`audio_emotion2vec_plus_v1` 使用 `mean_std_max`，`audio_opensmile_egemaps_v1` 使用 `functionals` | [Audio 真实模块](../../src/daily_multimodal/embeddings/audio_real.py) |
| `ppg_rows_in_window`、`gsr_rows_in_window`、`acc_rows_in_window` | Wear v2 每个 10 秒窗口内三路原始 CSV 有效行数 | [Wear 真实模块](../../src/daily_multimodal/embeddings/wear_real.py) |
| `ppg_effective_sampling_rate_hz`、`gsr_effective_sampling_rate_hz`、`acc_effective_sampling_rate_hz` | Wear v2 用窗口内有效行数除以窗口秒数得到的原始有效采样率；与重采样目标不同 | [Wear 真实模块](../../src/daily_multimodal/embeddings/wear_real.py) |
| `*_invalid_rows`、`*_source_rows`、`*_duplicate_timestamps`、`*_duplicate_timestamp_rows`、`*_nonmonotonic_timestamps`、`*_flatline_ratio`、`*_flatline` | Wear v2 每路原始 CSV 的无效行、源行数、重复时间戳窗口标记、重复时间戳行数、非单调时间戳和整窗 flatline 质量字段；重复秒级时间戳会按行顺序摊开用于插值，不再丢弃原始样本 | [Wear 真实模块](../../src/daily_multimodal/embeddings/wear_real.py) |
| `heart_rate`、`heart_rate_plausible`、`heart_rate_plausible_range_bpm`、`ibi_mean`、`ibi_std`、`rmssd`、`peak_count`、`ppg_peak_insufficient` | Wear v2 从 PPG 估计的心率、IBI/HRV、峰值质量字段和默认 `40-180 bpm` 心率合理性标记 | [Wear 真实模块](../../src/daily_multimodal/embeddings/wear_real.py) |
| `tonic_mean`、`phasic_std`、`scr_count`、`gsr_slope` | Wear v2 从 GSR 估计的 tonic/phasic、SCR 和趋势字段 | [Wear 真实模块](../../src/daily_multimodal/embeddings/wear_real.py) |
| `motion_intensity`、`stationary_ratio`、`axis_std`、`spectral_energy` | Wear v2 从 ACC 估计的运动强度、静止比例和频域能量字段 | [Wear 真实模块](../../src/daily_multimodal/embeddings/wear_real.py) |
| `wear_quality_grade`、`wear_quality_label` | Wear v2 每窗 A/B/C 质量分级；A 为 high、B 为 medium、C 为 low。默认只标记，不丢弃；传 `--mask-low-quality-wear` 时 C 类 wear mask 置 0 | [Wear 真实模块](../../src/daily_multimodal/embeddings/wear_real.py)、[Wear 真实入口](../../scripts/embeddings/15_extract_wear_embeddings.py) |
| `ppg_hr_plausible`、`ppg_peak_sufficient`、`gsr_slope_abnormal`、`gsr_scr_abnormal`、`acc_motion_high`、`acc_stable`、`motion_artifact_risk`、`wear_invalid_ratio_zero`、`wear_quality_risk_count` | Wear A/B/C 分级和质量 flags；当前阈值写入每窗 `wear_quality_thresholds`，用于复现实验和质量标记训练 | [Wear 真实模块](../../src/daily_multimodal/embeddings/wear_real.py) |
| `quality_audit` | Wear v2 summary 中的全量质量汇总，聚合 rows/rate、invalid/source、timestamp 异常、flatline、PPG peak/heart-rate、GSR slope/SCR 异常、ACC motion/stationary | [Wear 真实模块](../../src/daily_multimodal/embeddings/wear_real.py) |
| `physio_feature_names`、`physio_feature_values` | Wear v2 写入 `quality_flags` 的原始可解释特征名和值，便于后续分析哪些生理信号起作用 | [Wear 真实模块](../../src/daily_multimodal/embeddings/wear_real.py) |
| `wear_quality_ablation`、`W1_physio_full`、`W2_deep_full`、`W3_physio_high_quality`、`W4_deep_high_quality`、`W5a_deep_full`、`W5b_deep_quality_flags_full`、`W5c_deep_sample_weights_full`、`W5d_deep_quality_flags_sample_weights_full`、`W6_physio_ab_quality`、`W7_deep_ab_quality` | Wear-only W1-W7 与 W5a-W5d 消融输出，报告 RMSE、Pearson r、`pred_std`、`truth_std`、`error_std` 的 fold mean/std，并记录 `quality_subset`、`include_quality_flags`、`use_sample_weight`、`sample_weight_mean/std` | [Wear quality ablation 模块](../../src/daily_multimodal/training/wear_quality_ablation.py) |
| `fold_count`、`subject_leakage`、`modalities`、`rmse_mean`、`rmse_std`、`pearson_r_mean`、`pearson_r_std`、`folds` | Subject-level CV 输出字段；每个 fold 保留 train/val/test subjects 和 RMSE/MAE/Pearson r，Markdown 表用 `test_r` 展示 fold r 值 | [Subject CV 模块](../../src/daily_multimodal/training/subject_cv.py) |
