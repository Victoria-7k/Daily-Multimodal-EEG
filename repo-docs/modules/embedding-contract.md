# 统一 embedding 契约

这一页先说明项目为什么需要统一的向量出口：不同原始信号会经历不同的清洗、切窗和编码步骤，但训练和评估入口应该看到稳定、可替换、可追踪的样本表。阅读时可以先把它理解成一层交通规则：每个模态都用固定形状表达可用内容，用同一张可用性表说明缺失或失败原因。

后面的技术小节会列出具体数组名、脚本入口和版本名。那些细节主要服务于排查和复现实验；如果只是想理解整体设计，先抓住“固定维度、固定顺序、失败可审计、版本可区分”这四件事就够了。

## 白话模型

统一 embedding 契约让后续模型不用关心每个模态当前是真实编码器还是 smoke encoder。每个窗口都会尝试产出四个固定维度数组：EEG、wear、face、audio。缺失或不可用的模态用零向量和 `modality_mask` 表示，成功读取的模态用同样长度的向量表示。

这个契约先服务工程闭环。它证明路径、窗口、mask、质量报告和批量保存是通的；模型效果不是当前 `basic` profile 的目标。后续替换真实编码器时，最重要的是保持数组键、样本顺序、mask 顺序和报告字段可读。

## 代码模型

[basic encoder](../../src/daily_multimodal/embeddings/basic.py) 定义 `EMBED_DIM = 256` 和 `MODALITY_ORDER = ("eeg", "wear", "face", "audio")`。`extract_basic_embedding` 读取一个窗口，返回 `EmbeddingSample`。wear 分支会扫描 PPG/GSR/ACC CSV 的窗口内数值，计算均值、标准差、最小值、最大值等基础统计；EEG、face、audio 分支在当前阶段使用文件大小、窗口时长和路径 salt 生成 metadata-derived smoke 向量。

[真实 embedding 契约](../../src/daily_multimodal/embeddings/contracts.py) 是阶段 11 后新增的保护层。`RealEmbeddingResult` 记录一个真实单模态结果的 `sample_id`、`event_id`、`subject_id`、`modality`、`embedding`、`mask_value`、`quality_flags`、`encoder_version` 和 `source_paths`；`validate_embedding_shape` 只接受 `(256,)` 或 `(N, 256)` 的浮点数组，并拒绝 NaN、无限值和非浮点 dtype。真实 encoder 后续替换时，应先通过这层检查，再进入 `.npz` 打包。

[失败清单模块](../../src/daily_multimodal/embeddings/failures.py) 固化了真实 encoder 的可定位失败记录。`EmbeddingFailure` 必须包含 `modality`、`encoder_profile`、`stage`、`error_type` 和 `source_path` 等定位字段；`write_failure_list` 即使没有失败也会写出 JSON 空数组 `[]`，让阶段 12 以后每个失败窗口都能追到模态、文件、依赖或处理阶段。

[批处理保存器](../../src/daily_multimodal/embeddings/pipeline.py) 把多个 `EmbeddingSample` 堆叠成 `.npz`：

```text
eeg_emb, wear_emb, face_emb, audio_emb -> (N, 256)
modality_mask -> (N, 4), order [eeg, wear, face, audio]
sample_id, event_id, subject_id, session_id -> object arrays
labels, source_paths -> JSON strings
```

[embedding 测试](../../tests/test_embedding_pipeline.py) 确认一个只有 wear 可用的窗口会得到 `[0, 1, 0, 0]` mask、非零 `wear_emb` 和零 `eeg_emb`。同一测试也确认保存后的 `.npz` 维度是 `(2, 256)` 和 `(2, 4)`，并确认精确 `video_candidates` 会优先于日期级 `candidate_mp4_paths`。

阶段 12 的 [真实缓存准备模块](../../src/daily_multimodal/embeddings/cache.py) 尚不生成最终真实 embedding；它先用专门的人脸检测器过滤无脸视频窗口，再把保留窗口的切片边界和目标缓存路径固定下来。cache key 使用 `{sample_id}/{modality}/{encoder_profile}`，audio 写 mono 16 kHz wav，face 写 OpenFace CSV 目标路径，EEG 和 wear 写窗口 JSON 描述。这样后续 WavLM、OpenFace、EEG 和 wear sequence encoder 失败时，可以先判断是无人脸样本、缓存/切片问题还是模型问题。

[Audio 真实模块](../../src/daily_multimodal/embeddings/audio_real.py) 是第一个消费真实缓存的 encoder 接入点。它从 `audio_clips/<sample_id>/<encoder_profile>/audio.json` 读取 wav 路径，要求 frozen backend 返回 `[frames, hidden_dim]`，再 mean pooling 并投影到 256 维。后续的 EEG、face、wear 真实模块也沿用同一单模态 `.npz` 形状：只写本模态 embedding，并用 `modality_mask` 标记该模态是否可用。

v2 profile 仍遵守同一 `(N, 256)` 契约，但在 `quality_flags` 中暴露更多可审计信息。`audio_opensmile_egemaps_v1` 把 openSMILE eGeMAPS Functionals 当作单帧功能特征投影；`audio_emotion2vec_plus_v1` 对 frame 特征做 `mean_std_max` pooling 后投影；`wear_physio_features_v2` 用 PPG HR/HRV、GSR slope/SCR 和 ACC motion/stationary 特征投影，并把 `physio_feature_names`/`physio_feature_values` 写入每个样本的质量字段。缺依赖或缺 checkpoint 时，这些 profile 写结构化 failure，不静默退回旧 encoder。

[真实多模态打包器](../../src/daily_multimodal/embeddings/real_pipeline.py) 是阶段 17 的合并入口。它以 window index 为主表保留样本顺序、标签和 source paths，再按 `sample_id` 合并 EEG/Wear/Face/Audio 单模态真实 `.npz`。缺失或质量 mask 为 0 的模态会写零向量并保持 `modality_mask=0`；成功模态保留 `(N, 256)` embedding。输出仍兼容阶段 9/10 训练入口，同时额外写入每个样本的 `quality_flags` 和 `encoder_versions` JSON 字符串，便于定位真实 encoder 的质量和版本。

当前全量 all-real 产物保持同一契约，但可用性不再要求四模态全为 1。服务器 v2 全量打包的 `modality_mask` sum 是 `[738, 781, 207, 781]`，说明 EEG 有 43 行缺失，Face 有 207 行通过 true OpenFace 质量门槛、501 行因低成功率被 mask、73 行仍缺 CSV，Wear 和 Audio 全部可用。这个设计让阶段 18 可以比较 all-real、without-face、single real replacement 等实验，而不是因为某个模态缺失就丢掉整行样本。

## 接下去阅读

在主路径里，[Step 5: basic encoder 写出统一 embedding 契约](../walkthroughs/one-real-run.md#step-5-basic-encoder-写出统一-embedding-契约) 解释这层契约如何被脚本使用。需要查 `.npz`、报告和阶段产物时读 [运行命令和产物](../references/commands-and-artifacts.md)；需要查窗口字段时读 [字段契约](../references/data-contracts.md)。

证据状态：除特别标注外，本页基于当前源码和测试已确认。

## Video V4a DINOv2 contract, 2026-07-04

`src/daily_multimodal/embeddings/dinov2_roi.py` owns the current video V4a embedding contract. It keeps compatibility with the existing multimodal packer by writing video embeddings into `face_emb`, but the encoder version is now `video_v4a_dinov2_2xroi_mean_std_max` rather than an OpenFace profile.

The V4a baseline samples 16 frames per 10-second window by default, runs a frozen DINOv2-style frame encoder, pools the frame sequence with `mean + std + max`, then projects the concatenated vector to `(256,)`. Low-FPS ROI clips are resampled to the requested frame count before encoding, so the current 0.5 FPS server ROI cache still produces 16 frame embeddings per usable window. The public extraction entrypoint is `scripts/27_extract_dinov2_roi_embeddings.py`; the explicit CLI flags are `--num-frames 16` and `--temporal-pooling mean_std_max`. `--frame-sequences-out` writes the aligned `[N, frames, hidden_dim]` frame sequence bundle for V4b. `--max-frames-per-window` is retained as a deprecated compatibility alias.

For region comparison, the same extractor can read `outputs/cache/video_regions/<region>/<sample_id>/window.mp4` via `--region-cache-root` and `--video-region`. The default `2x_face_roi` encoder version remains `video_v4a_dinov2_2xroi_mean_std_max`; region-cache variants use `video_v4a_dinov2_upper_body_mean_std_max` and `video_v4a_dinov2_full_frame_mean_std_max`, with `quality_flags.input_region` recording the selected region.

Per-row `quality_flags` should expose the video contract for later audit: `temporal_pooling`, `pooled_stat_names`, `sampled_frame_count`, `usable_frame_count`, `frame_embedding_dim`, and `projected_from_dim`. Downstream video probes should identify this baseline by `encoder_versions.face == video_v4a_dinov2_2xroi_mean_std_max` and should not treat it as an OpenFace feature stream.

V4d augmented DINOv2 extraction keeps the same `face_emb (N,256)` and optional frame-sequence contracts, but changes the encoder version to `video_v4d_aug_dinov2_<region>_mean_std_max`. The current augmentation profile is `v4d_mild_color_crop_scale`: it applies deterministic brightness, contrast, color jitter, crop jitter, and scale jitter to raw sampled frames, encodes original plus augmented views, averages frame embeddings by frame position, and records `augmentation_profile`, `augmentation_views`, and `augmentation_ops` in `quality_flags`. The default `augmentation_profile=none` path remains the frozen V4a contract.

The stable ROI policy for V4d is explicit rather than global: pass `--region-cache-root outputs/cache/video_regions --video-region upper_body --fallback-video-region full_frame` to prefer upper-body clips while falling back to full-frame cache clips only when the requested upper-body clip is missing. This keeps the R1/2x face ROI baseline unchanged. Policy-enabled encoder versions include the fallback region, for example `video_v4a_dinov2_upper_body_full_frame_fallback_mean_std_max` or `video_v4d_aug_dinov2_upper_body_full_frame_fallback_mean_std_max`; per-row `quality_flags` record `requested_input_region`, `effective_input_region`, `fallback_video_region`, and `video_region_fallback_used`.

`src/daily_multimodal/embeddings/video_domain_robust.py` owns the V4d adversarial-training interface. It does not change the `.npz` embedding contract by itself; instead, training code can use `encode_domain_targets` to build stable subject/session class targets, then attach `DomainAdversarialHeads` to a video embedding tensor. The heads apply gradient reversal before subject/session classifiers, so minimizing their cross-entropy loss pushes the upstream video representation away from subject/session shortcuts while keeping fatigue supervision in the main task loss.

## Video V4b temporal contract, 2026-07-04

`src/daily_multimodal/embeddings/video_temporal.py` owns the first V4b temporal embedding builder. It consumes a DINOv2 frame-sequence bundle with `frame_embeddings` shaped `[N, frames, hidden_dim]`, preserves `sample_id`, `event_id`, `subject_id`, and `labels`, and writes the output through the existing `face_emb (N, 256)` compatibility slot.

The two initial encoder versions are `video_v4b_tcn_dinov2_2xroi` and `video_v4b_temporal_transformer_dinov2_2xroi`. Both expose per-row `quality_flags` with `temporal_encoder`, `input_frame_count`, `frame_embedding_dim`, `source_encoder_version`, `source_mask_value`, and `projected_from_dim` when a row is usable. Non-finite frame-sequence rows are masked with a zero `face_emb`, `modality_mask[:,2]=0`, and `quality_flags.masked_reason=nonfinite_frame_embeddings`; source rows with `modality_mask[:,2]=0` are also masked with `masked_reason=source_video_mask_zero`.

## Video region cache contract, 2026-07-04

`src/daily_multimodal/embeddings/video_regions.py` prepares the three planned video input regions under one cache root: `2x_face_roi`, `upper_body`, and `full_frame`. Each successful region writes `<out_root>/<region>/<sample_id>/window.mp4` plus a `region.json` sidecar, and the full run writes `video_regions_manifest.jsonl` and `video_regions_failures.json`.

The upper-body path may use injected MediaPipe/pose localization or existing bbox metadata. If no upper-body box is available, the sidecar keeps `region=upper_body` but records `effective_region=full_frame` and `upper_body_fallback_full_frame=true`, making the fallback explicit for later embedding extraction and ROI comparisons.

The default region writer emits short, DINO-oriented clips rather than copying full source MP4 files: it samples 16 frames from the requested window, applies any crop bbox, scales frames to 640px width, and writes a compact MP4. It reads each source window sequentially and reuses the sampled raw frames when both `upper_body` and `full_frame` are requested for the same window. `face_presence.main_face_bbox` from the current window index is OpenCV-style `[x,y,w,h]`, and the upper-body fallback expands that face box before cropping. This keeps region cache artifacts small enough for R1/R2/R3 extraction while preserving the region content used by the visual encoder.

When the default writer is asked for only `upper_body` and `full_frame`, it further groups rows by `(source_video_path, event_id)` and decodes each source/event group once. Sidecars written through this faster path use `region_source=source_video_event_group`; the per-window output path and fallback fields stay the same, so downstream DINOv2 extraction continues to read the cache by `sample_id` and `video_region`.
