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

## window index summary 字段

| 字段 | 含义 | 来源 |
| --- | --- | --- |
| `events_total`、`events_selected`、`events_skipped`、`windows_total` | 输入事件数、保留事件数、跳过事件数和展开后的窗口总数 | [窗口索引入口](../../scripts/03_build_window_index.py) |
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
| `affected_subject_sessions` | 非 `in_range` 窗口涉及的 `subject/session` 列表 | [EEG coverage audit 入口](../../scripts/19_audit_eeg_coverage.py) |
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
| `wear_quality_grade`、`wear_quality_label` | Wear v2 每窗 A/B/C 质量分级；A 为 high、B 为 medium、C 为 low。默认只标记，不丢弃；传 `--mask-low-quality-wear` 时 C 类 wear mask 置 0 | [Wear 真实模块](../../src/daily_multimodal/embeddings/wear_real.py)、[Wear 真实入口](../../scripts/15_extract_wear_embeddings.py) |
| `ppg_hr_plausible`、`ppg_peak_sufficient`、`gsr_slope_abnormal`、`gsr_scr_abnormal`、`acc_motion_high`、`acc_stable`、`motion_artifact_risk`、`wear_invalid_ratio_zero`、`wear_quality_risk_count` | Wear A/B/C 分级和质量 flags；当前阈值写入每窗 `wear_quality_thresholds`，用于复现实验和质量标记训练 | [Wear 真实模块](../../src/daily_multimodal/embeddings/wear_real.py) |
| `wear_preprocessing_applied`、`wear_preprocessing_version`、`*_preprocessing_steps`、`*_preprocessing_input_std`、`*_preprocessing_output_std`、`*_preprocessing_mean_abs_delta`、`*_preprocessing_changed` | `wear_physio_features_preprocessed_v1` / `wear_deep_sequence_preprocessed_v1` 写入的显式预处理字段；当前 `wear_signal_preprocessing_v1` 对 PPG 做 robust winsorize + 近似 0.5-5Hz bandpass，对 GSR 做 robust winsorize + 约 1Hz lowpass，对 ACC 做 robust winsorize + median gravity removal + 约 5Hz smoothing | [Wear 真实模块](../../src/daily_multimodal/embeddings/wear_real.py) |
| `quality_audit` | Wear v2 summary 中的全量质量汇总，聚合 rows/rate、invalid/source、timestamp 异常、flatline、PPG peak/heart-rate、GSR slope/SCR 异常、ACC motion/stationary | [Wear 真实模块](../../src/daily_multimodal/embeddings/wear_real.py) |
| `physio_feature_names`、`physio_feature_values` | Wear v2 写入 `quality_flags` 的原始可解释特征名和值，便于后续分析哪些生理信号起作用 | [Wear 真实模块](../../src/daily_multimodal/embeddings/wear_real.py) |
| `wear_quality_ablation`、`W1_physio_full`、`W2_deep_full`、`W3_physio_high_quality`、`W4_deep_high_quality`、`W5a_deep_full`、`W5b_deep_quality_flags_full`、`W5c_deep_sample_weights_full`、`W5d_deep_quality_flags_sample_weights_full`、`W6_physio_ab_quality`、`W7_deep_ab_quality` | Wear-only W1-W7 与 W5a-W5d 消融输出，报告 RMSE、Pearson r、`pred_std`、`truth_std`、`error_std` 的 fold mean/std，并记录 `quality_subset`、`include_quality_flags`、`use_sample_weight`、`sample_weight_mean/std` | [Wear quality ablation 模块](../../src/daily_multimodal/training/wear_quality_ablation.py) |
| `fold_count`、`subject_leakage`、`modalities`、`rmse_mean`、`rmse_std`、`pearson_r_mean`、`pearson_r_std`、`folds` | Subject-level CV 输出字段；每个 fold 保留 train/val/test subjects 和 RMSE/MAE/Pearson r，Markdown 表用 `test_r` 展示 fold r 值 | [Subject CV 模块](../../src/daily_multimodal/training/subject_cv.py) |
