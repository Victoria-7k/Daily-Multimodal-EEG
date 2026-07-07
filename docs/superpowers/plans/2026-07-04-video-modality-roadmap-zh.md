# 视频模态后续计划实施方案

> **给执行代理：** 必须按任务逐项执行本方案。推荐使用 `superpowers:subagent-driven-development`，或使用 `superpowers:executing-plans`。步骤用 checkbox (`- [ ]`) 跟踪。

**总目标：** 将视频主线从 OpenFace 统计特征切换为自然视频深度视觉路线：先固定 V4a，再用 probe 诊断 subject/session shortcut，随后推进 V4b 时序建模、ROI 对照、V4d 泛化增强和 V4c 原生视频模型对照。

**架构原则：** 短期继续复用现有 `face_emb (N, 256)` 存储契约和 face-slot mask，以保持 `scripts/26_run_video_variant_ablation.py` 等训练入口兼容；实际语义改为 video embedding，并通过 `encoder_version` 区分模型版本。MediaPipe 不进入主 embedding 表征，只保留为 ROI 辅助、行为 audit、quality flags 和结果解释工具。

**技术栈：** Python 3.10+, NumPy, PyTorch, Transformers DINOv2, OpenCV/ffmpeg, scikit-learn, existing NPZ/JSONL/Markdown artifacts, pytest。

---

## 当前约束和命名

- 主线目标：自然视频 -> 深度视觉编码 -> 时序建模 -> 减少 subject/session shortcut -> 256D video embedding。
- 主线不再以 OpenFace 作为对照目标；OpenFace V1/V2 仅作为 archived legacy reference。
- 当前仓库兼容契约仍是 `face_emb (N, 256)`，mask 顺序仍为 `[eeg, wear, face, audio]`。后续文档中可称为 video embedding，但落盘字段短期不改。
- `scripts/28_run_video_embedding_probes.py` 后续会修改，fatigue Ridge probe 需要支持 LOSO/S1/S4/S2，而不是只用随机 KFold。
- 固定评估口径：
  - LOSO: `leave_one_subject_out`
  - S1: `within_subject_event_split`
  - S4: `within_subject_session_leave_out`
  - S2: `within_subject_chronological_split`

## 文件结构

- Modify: `src/daily_multimodal/embeddings/dinov2_roi.py`
  - 将当前 DINOv2 ROI 实现固定为 V4a：16 帧、Frozen DINOv2、frame sequence、`mean + std + max` temporal pooling、256D 投影。
- Modify: `scripts/27_extract_dinov2_roi_embeddings.py`
  - 暴露 V4a 参数：`--num-frames 16`、`--temporal-pooling mean_std_max`、稳定 `encoder_version`。
- Modify: `tests/test_dinov2_roi_embeddings.py`
  - 覆盖 16 帧采样、pooling metadata、mask 契约和缺失 ROI video 行为。
- Modify: `src/daily_multimodal/training/video_embedding_probes.py`
  - subject/session probe 保留；fatigue Ridge probe 增加 LOSO/S1/S4/S2 fold strategy。
- Modify: `scripts/28_run_video_embedding_probes.py`
  - 增加 `--fold-strategy`、`--n-splits`、输出四类 probe 结果。
- Modify: `tests/test_video_embedding_probes.py`
  - 覆盖 subject/session Logistic 与 fatigue Ridge 的四种 split。
- Reuse: `src/daily_multimodal/training/video_variant_ablation.py`
  - 复用 `_build_video_folds` 和现有 split 语义。
- Reuse: `scripts/26_run_video_variant_ablation.py`
  - 跑 V4a/V4b/V4d/V4c 的 LOSO/S1/S4/S2 下游指标。
- Create: `src/daily_multimodal/embeddings/video_temporal.py`
  - V4b-TCN 与 V4b-TemporalTransformer 的时序编码实现。
- Create: `scripts/31_train_video_temporal_encoder.py`
  - 训练或抽取 V4b 256D embedding 的 CLI。
- Test: `tests/test_video_temporal_embeddings.py`
  - 覆盖 sequence 输入、输出 shape、mask、`encoder_version`。
- Create: `src/daily_multimodal/embeddings/video_regions.py`
  - 统一 2x face ROI、upper-body、full-frame region cache。
- Create: `scripts/29_prepare_video_regions.py`
  - 生成 `outputs/cache/video_regions/...` 下的区域视频片段。
- Test: `tests/test_video_regions.py`
  - 覆盖 upper-body fallback 到 full frame、region metadata 和 quality flags。
- Optional Create: `src/daily_multimodal/embeddings/video_domain_robust.py`
  - V4d subject/session adversarial head 和 gradient reversal。
- Optional Create: `src/daily_multimodal/embeddings/native_video.py`
  - V4c VideoMAE、Video Swin、TimeSformer 对照入口。
- Docs: `repo-docs/modules/embedding-contract.md`
  - 代码落地后同步说明 video embedding 暂存于 `face_emb` 槽。
- Docs: `repo-docs/references/commands-and-artifacts.md`
  - 代码落地后补充 scripts 27/28/29/31/33 的命令和产物。
- Docs: `repo-docs/change-log.md`
  - 记录本轮视频主线接口变更和验证。

## Task 1: 固定 V4a DINOv2 Spatial Baseline

**Files:**
- Modify: `src/daily_multimodal/embeddings/dinov2_roi.py`
- Modify: `scripts/27_extract_dinov2_roi_embeddings.py`
- Test: `tests/test_dinov2_roi_embeddings.py`

- [x] Step 1: 在 `tests/test_dinov2_roi_embeddings.py` 新增失败测试，要求 fake frame encoder 返回 16 个 frame embeddings，并断言输出 metadata 包含 `sampled_frame_count=16`、`temporal_pooling=mean_std_max`、`encoder_version=video_v4a_dinov2_2xroi_mean_std_max`。
- [x] Step 2: 运行 `python -m pytest tests/test_dinov2_roi_embeddings.py -q`，预期新增测试失败，失败点应指向当前实现仍使用单一 mean pooling 或旧 `encoder_version`。
- [x] Step 3: 修改 `dinov2_roi.py`，使 DINOv2 backend 返回 `[frames, hidden_dim]` frame sequence，并在主 builder 中执行 `mean + std + max` pooling 后投影到 256D。
- [x] Step 4: 修改 `scripts/27_extract_dinov2_roi_embeddings.py`，将默认抽帧固定为 16 帧，CLI 参数保留 `--fps` 但新增或改名为 `--num-frames 16`，避免继续依赖 `max_frames_per_window=20` 作为 V4a 默认。
- [x] Step 5: 运行 `python -m pytest tests/test_dinov2_roi_embeddings.py -q`，预期通过。
- [x] Step 6: 生成 V4a 产物：

```powershell
python scripts/27_extract_dinov2_roi_embeddings.py `
  --window-index outputs/window_index/real_cache_face_detected_full_v2_mainface.jsonl `
  --openface-cache-root outputs/cache/real_stage12_face_filter_full_v2_mainface `
  --openface-encoder-profile openface_temporal_v1 `
  --out outputs/embeddings/video_v4a_dinov2_2xroi_embeddings.npz `
  --frame-sequences-out outputs/embeddings/video_v4a_dinov2_2xroi_frame_sequences.npz `
  --num-frames 16 `
  --temporal-pooling mean_std_max `
  --model-name facebook/dinov2-base `
  --progress-out outputs/reports/video_v4a_dinov2_2xroi_progress.log `
  --failures-out outputs/reports/video_v4a_dinov2_2xroi_failures.json
```

执行记录，2026-07-04：完整 V4a 产物已在 `ncc_serve_4090` 的服务器/cache 环境完成生成，使用 `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`。输出为 `outputs/embeddings/video_v4a_dinov2_2xroi_embeddings.npz` 和 `outputs/embeddings/video_v4a_dinov2_2xroi_frame_sequences.npz`。验证结果：`face_emb.shape=(8328, 256)`，`frame_embeddings.shape=(8328, 16, 768)`，`face_mask_sum=8328`，`subject_count=14`，无 NaN，failures 为 `[]`。

## Task 2: 修改并运行 V4a Probe

**Files:**
- Modify: `src/daily_multimodal/training/video_embedding_probes.py`
- Modify: `scripts/28_run_video_embedding_probes.py`
- Test: `tests/test_video_embedding_probes.py`
- Reuse: `src/daily_multimodal/training/video_variant_ablation.py`

- [x] Step 1: 在 `tests/test_video_embedding_probes.py` 增加测试，构造多 subject、多 session、多 event 的 synthetic `.npz`，断言 fatigue Ridge 可分别在 `leave_one_subject_out`、`within_subject_event_split`、`within_subject_session_leave_out`、`within_subject_chronological_split` 下运行。
- [x] Step 2: 运行 `python -m pytest tests/test_video_embedding_probes.py -q`，预期新增 split 测试失败。
- [x] Step 3: 修改 `video_embedding_probes.py`，subject/session Logistic probe 保持现有逻辑；fatigue Ridge probe 增加 `fold_strategy`，并复用 `video_variant_ablation._build_video_folds` 生成 train/val/test。
- [x] Step 4: 修改 `scripts/28_run_video_embedding_probes.py`，新增 `--fold-strategy`，choices 与 `scripts/26_run_video_variant_ablation.py` 保持一致。
- [x] Step 5: 运行 `python -m pytest tests/test_video_embedding_probes.py tests/test_video_variant_ablation.py -q`，预期通过。
- [x] Step 6: 对 V4a 跑 subject/session probe 和四套 fatigue Ridge：

```powershell
python scripts/28_run_video_embedding_probes.py `
  --embeddings outputs/embeddings/video_v4a_dinov2_2xroi_embeddings.npz `
  --target-label fatigue `
  --fold-strategy leave_one_subject_out `
  --out-json outputs/reports/video_probes/v4a_loso_probes.json `
  --out-table outputs/reports/video_probes/v4a_loso_probes.md

python scripts/28_run_video_embedding_probes.py `
  --embeddings outputs/embeddings/video_v4a_dinov2_2xroi_embeddings.npz `
  --target-label fatigue `
  --fold-strategy within_subject_event_split `
  --out-json outputs/reports/video_probes/v4a_s1_probes.json `
  --out-table outputs/reports/video_probes/v4a_s1_probes.md

python scripts/28_run_video_embedding_probes.py `
  --embeddings outputs/embeddings/video_v4a_dinov2_2xroi_embeddings.npz `
  --target-label fatigue `
  --fold-strategy within_subject_session_leave_out `
  --out-json outputs/reports/video_probes/v4a_s4_probes.json `
  --out-table outputs/reports/video_probes/v4a_s4_probes.md

python scripts/28_run_video_embedding_probes.py `
  --embeddings outputs/embeddings/video_v4a_dinov2_2xroi_embeddings.npz `
  --target-label fatigue `
  --fold-strategy within_subject_chronological_split `
  --out-json outputs/reports/video_probes/v4a_s2_probes.json `
  --out-table outputs/reports/video_probes/v4a_s2_probes.md
```

执行记录，2026-07-04：Step 6 已在 `ncc_serve_4090` 完成，输入为包含 8328 行的 `outputs/embeddings/video_v4a_dinov2_2xroi_embeddings.npz`。Subject probe accuracy 为 0.9777，within-subject session probe accuracy 为 0.9691，说明 V4a 中存在很强的身份/场次信息。Fatigue Ridge 结果：LOSO RMSE 1.0333，Pearson r -0.0460；S1 RMSE 0.9188，Pearson r 0.2746；S4 RMSE 0.9620，Pearson r 0.1702；S2 RMSE 1.0172，Pearson r 0.2254。报告位于 `outputs/reports/video_probes/v4a_*_probes.{json,md}`。

## Task 3: 固定 V4a 下游评估表

**Files:**
- Reuse: `scripts/26_run_video_variant_ablation.py`
- Reuse: `src/daily_multimodal/training/video_variant_ablation.py`
- Optional Docs: `outputs/reports/video_variants/v4a/v4a_split_summary.md`

- [x] Step 1: 用 `scripts/26_run_video_variant_ablation.py` 跑 V4a 的 LOSO/S1/S4/S2。
- [x] Step 2: 每次运行使用 `--sample-mode behavior_retained`，因为 V4a 的可用性由 ROI video 是否可读决定，不继承 OpenFace face-detection mask。
- [x] Step 3: 将四个 JSON 汇总成 `outputs/reports/video_variants/v4a/v4a_split_summary.md`，表格至少包含 rows、folds、RMSE mean/std、Pearson r mean/std。
- [x] Step 4: 判读标准：
  - 如果 S1 明显好、LOSO/S4/S2 明显差，并且 subject/session probe 准确率高，则进入 V4b 和 ROI 对照后再考虑 V4d。
  - 如果 V4a 在 LOSO 也稳定提升，则 V4b 只需轻量时序模型，不优先做 adversarial。

执行记录，2026-07-04：Task 3 已在 `ncc_serve_4090` 完成，使用 `--sample-mode behavior_retained`。汇总报告为 `outputs/reports/video_variants/v4a/v4a_split_summary.md`。V4a 下游 MLP 结果：LOSO RMSE 1.0083，Pearson r -0.0517；S1 RMSE 0.8801，Pearson r 0.3258；S4 RMSE 0.9477，Pearson r 0.1788；S2 RMSE 0.9703，Pearson r 0.2272。判读：S1 明显强于 LOSO/S4/S2，且 subject/session probes 很高，因此先继续 V4b 时序建模和 ROI 对照，再考虑 V4d adversarial/domain robustness。

## Task 4: 开发 V4b 时序模型

**Files:**
- Create: `src/daily_multimodal/embeddings/video_temporal.py`
- Create: `scripts/31_train_video_temporal_encoder.py`
- Test: `tests/test_video_temporal_embeddings.py`

- [x] Step 1: 新增测试，构造 `[N, 16, D]` synthetic frame sequence，断言 TCN 和 Temporal Transformer 均输出 `[N, 256]`。
- [x] Step 2: 实现 `V4b-TCN`，输入 DINOv2 frame sequence，输出 256D embedding，`encoder_version=video_v4b_tcn_dinov2_2xroi`。
- [x] Step 3: 实现 `V4b-TemporalTransformer`，输入同一 frame sequence，输出 256D embedding，`encoder_version=video_v4b_temporal_transformer_dinov2_2xroi`。
- [x] Step 4: 输出 `.npz` 时保持 `sample_id`、`event_id`、`subject_id`、`labels`、`face_emb`、`modality_mask`、`quality_flags`、`encoder_version`。
- [x] Step 5: 运行 `python -m pytest tests/test_video_temporal_embeddings.py -q`。
- [x] Step 6: 用 `scripts/26_run_video_variant_ablation.py` 对 V4a、V4b-TCN、V4b-TemporalTransformer 跑 LOSO/S1/S4/S2，并记录是否是时序建模带来的提升。

执行记录，2026-07-04：Task 4 已在 `ncc_serve_4090` 完成。V4b 产物为 `outputs/embeddings/video_v4b_tcn_dinov2_2xroi_embeddings.npz` 和 `outputs/embeddings/video_v4b_temporal_transformer_dinov2_2xroi_embeddings.npz`；二者均满足 `face_emb.shape=(8328, 256)`、`face_mask_sum=8328`，且无 NaN。汇总报告为 `outputs/reports/video_variants/v4b/v4a_v4b_temporal_summary.md`。时序建模只带来有限收益：V4b-TCN 略微改善 LOSO RMSE，但 Pearson r 仍接近 0；Temporal Transformer 在 S1/S4 上略优；两个 V4b 变体在 S2 上均弱于 V4a。

## Task 5: 比较视频输入区域

**Files:**
- Create: `src/daily_multimodal/embeddings/video_regions.py`
- Create: `scripts/29_prepare_video_regions.py`
- Test: `tests/test_video_regions.py`
- Reuse: `src/daily_multimodal/embeddings/video_behavior_flags.py`
- Reuse: `scripts/23_extract_video_behavior_flags.py`
- Reuse: `scripts/24_audit_video_behavior_flags.py`

- [x] Step 1: 实现 region cache 输出路径：
  - `outputs/cache/video_regions/2x_face_roi/<sample_id>/window.mp4`
  - `outputs/cache/video_regions/upper_body/<sample_id>/window.mp4`
  - `outputs/cache/video_regions/full_frame/<sample_id>/window.mp4`
- [x] Step 2: upper-body ROI 使用 MediaPipe pose/face landmarks 或已存在的检测元数据辅助定位；无法定位 upper-body 时写 `quality_flags.upper_body_fallback_full_frame=true` 并使用 full frame。
- [x] Step 3: 分别抽取 R1/R2/R3 的 V4a 或最佳 V4b embedding。
- [x] Step 4: 在 LOSO/S1/S4/S2 下比较 R1、R2、R3。
- [x] Step 5: 主推规则：若 upper-body 在 S4/S2 不差于 2x ROI 且 LOSO 更稳，则后续默认使用 upper-body；否则继续保留 2x ROI 作为 baseline。

执行记录，2026-07-04/05：`video_regions.py` 与 `scripts/29_prepare_video_regions.py` 已能生成 region cache 路径、manifest、sidecars 和 fallback flags。`scripts/27_extract_dinov2_roi_embeddings.py` 已支持 `--region-cache-root` 和 `--video-region {2x_face_roi,upper_body,full_frame}`，因此 R1/R2/R3 region cache 可进入同一 V4a extractor；同时支持 `--direct-video-region-from-window`，可在完整 cache 完成前直接从源视频抽取 `upper_body`/`full_frame` 做 smoke/debug。服务器 smoke 状态：单窗口 region cache 已写出三种 region 且 failures 为 `[]`；从 `upper_body` 与 `full_frame` region cache 抽取 DINOv2 embedding 可得到 `(1,256)`；直接源视频 smoke 可得到 `(2,256)`、mask sum 2、无 NaN、failures `[]`，并带有 region-specific encoder versions。完整 grouped region cache 已在 `ncc_serve_4090` 完成：`upper_body` 与 `full_frame` 各有 8328 个 `window.mp4` 和 8328 个 `region.json`，manifest 合计 16656 行，failures 合计 0。R2/R3 V4a 产物已抽取为 `outputs/embeddings/video_v4a_dinov2_upper_body_embeddings.npz` 与 `outputs/embeddings/video_v4a_dinov2_full_frame_embeddings.npz`；二者均验证为 `face_emb.shape=(8328,256)`、frame sequences `(8328,16,768)`、mask sum 8328、无 NaN、failures `[]`。区域比较报告为 `outputs/reports/video_variants/regions/region_comparison_summary.md`。决策：当前默认仍保留 R1 / 2x face ROI，因为 R2 upper-body 虽然改善 LOSO/S1/S4 的 RMSE，但 S2 更差（R2 S2 RMSE 1.0049、r 0.1435；R1 S2 RMSE 0.9703、r 0.2272）；R3 full-frame 也不设为默认，因为它虽然提高 S1/S4 Pearson r，但损害 LOSO/S2 RMSE。

## Task 6: V4d 泛化增强

**Files:**
- Modify: `src/daily_multimodal/embeddings/video_temporal.py`
- Optional Create: `src/daily_multimodal/embeddings/video_domain_robust.py`
- Test: `tests/test_video_domain_robust.py`

- [x] Step 1: 触发条件确认：S1 明显好于 LOSO/S4/S2，且 V4a/V4b 的 subject 或 session probe 准确率显著高。
- [x] Step 2: 先接入第一版 appearance augmentation 接口。
- [ ] Step 3: 将 appearance augmentation 设为 V4d 第一优先级：upper-body ROI + brightness jitter、contrast jitter、color jitter、random grayscale、轻度 blur、crop/scale jitter。augmentation 只能用于训练 fold；验证/测试行必须使用原始 deterministic upper-body 输入。
- [ ] Step 4: appearance 路径可跑通后，再做轻量 ROI stabilization：ROI temporal smoothing、限制异常 crop scale、检查高 fallback session、对异常严重 subject/session 做 QC。
- [ ] Step 5: augmentation 后重新跑 Subject Probe、Session Probe、LOSO、S1、S4、S2。
- [ ] Step 6: 只有当 appearance augmentation 后 subject/session probe 仍然很高且 LOSO 仍接近 0 时，才进入 GRL：加入 GRL subject head 和 GRL session head。
- [ ] Step 7: 成功标准：subject/session probe 下降，同时 fatigue 的 LOSO/S4/S2 RMSE 不升高、Pearson r 不下降；S1 应保持。

### ROI Geometry Audit 后的 V4d 优先级更新

当前 V4d 优先级按证据调整为：

1. **第一优先：Appearance augmentation。** 使用 `upper_body` ROI，并加入 brightness jitter、contrast jitter、color jitter、random grayscale、轻度 blur、crop/scale jitter。理由是 DINOv2 Session Probe 约为 `0.969`，而 `outputs/reports/roi_audit/geometry_session_probe.json` 中的 Geometry-only Session Probe 只有 `0.364`；剩余的大量 session 信息不是简单 ROI geometry 能解释的，更可能来自衣服、光照、背景和其他 appearance cues。
2. **第二优先：轻量 ROI stabilization。** 不马上上完整 Pose pipeline，也不一开始重做全量重型 ROI。先做 ROI temporal smoothing、异常 crop scale 限制、高 fallback session 检查，以及严重 subject/session 的 QC。ROI audit 显示部分 subject/session 几何漂移明显，但差异不均匀，所以先做低成本稳定化。
3. **第三优先：重新 probe 和评估。** augmentation 后重跑 Subject Probe、Session Probe、LOSO、S1、S4、S2，但必须保持严格 split hygiene：训练 fold 可以使用随机增强视图，验证/测试 fold 必须使用原始 deterministic upper-body embedding。期望看到 Session Probe 下降、Subject Probe 下降、LOSO/S4/S2 改善，同时 S1 保持。
4. **第四优先：GRL。** 只有当 appearance augmentation 后 session/subject probe 仍然很高，且 LOSO 仍接近 0 时，才加入 gradient reversal 的 subject/session heads。

执行记录，2026-07-05：V4d 触发条件成立：V4a subject/session probe 很高（subject 0.9777、session 0.9691），S1 明显强于 LOSO，ROI 对比也没有消除 S2/LOSO 弱点。`scripts/27_extract_dinov2_roi_embeddings.py` 与 `dinov2_roi.py` 已加入 `--augmentation-profile v4d_mild_color_crop_scale` 和 `--augmentation-views`；该 profile 在原始帧层面做 deterministic brightness、contrast、color jitter、crop jitter、scale jitter，平均 original 与 augmented DINOv2 frame embeddings，并写出 `video_v4d_aug_dinov2_2xroi_mean_std_max` 等 encoder version。服务器 1 个真实 2x ROI 窗口 smoke 已写出 `outputs/embeddings/video_v4d_aug_dinov2_2xroi_smoke1_embeddings.npz`，验证为 `face_emb.shape=(1,256)`、frame sequence `(1,16,768)`、mask `[[0,0,1,0]]`、无 NaN、failures `[]`，quality flags 记录 5 类 augmentation ops。完整 V4d 2xROI extraction 曾启动，但共享 GPU 满载下 6 小时仅推进到约 35.6%，因此已停止；目前不声称已有完整 V4d 产物或 downstream evaluation。为让后续全量抽取可恢复，`scripts/27_extract_dinov2_roi_embeddings.py` 现在除 `--max-windows` 外还支持 `--start-index`；服务器真实 chunk smoke 使用 `--start-index 10 --max-windows 1` 写出 `outputs/embeddings/video_v4d_aug_dinov2_2xroi_chunk_smoke_start10_embeddings.npz` 和对应 frame-sequence bundle，验证输出 sample id 为 `sub-02_ses-01_00_row-0001_win-0010`，输出 `(1,256)` / `(1,16,768)`、mask `[[0,0,1,0]]`、无 NaN、failures `[]`，并保留 augmentation flags。稳定 ROI 策略现在以显式 policy 形式冻结，而不是覆盖全局默认：`--video-region upper_body --fallback-video-region full_frame` 保留当前 R1/2xROI baseline，同时允许 V4d 运行优先使用 upper-body，并在 upper-body clip 缺失时回退到 full-frame cache。本地和远端测试覆盖了真实 fallback 分支；服务器 1 个真实窗口 smoke 已写出 `outputs/embeddings/video_v4d_roi_policy_upper_full_smoke1_embeddings.npz`，验证为 `face_emb.shape=(1,256)`、mask `[[0,0,1,0]]`、无 NaN、failures `[]`，encoder version 为 `video_v4a_dinov2_upper_body_full_frame_fallback_mean_std_max`，quality flags 记录 requested/effective/fallback region。`src/daily_multimodal/embeddings/video_domain_robust.py` 现已提供稳定的 subject/session target 编码、gradient reversal，以及 PyTorch subject/session adversarial heads 和联合 adversarial loss。本地测试覆盖无 PyTorch 环境下的 label 契约；服务器 `lzs` 环境测试覆盖 gradient sign reversal、head 输出和 loss shape。`scripts/32_check_video_v4d_success.py` 现已把最终成功 gate 固化为自动判定：subject/session probe accuracy 必须严格下降，同时 LOSO/S4/S2 fatigue RMSE 不升高、Pearson r 不下降。服务器 V4a self-check 会按预期失败，并写出 `outputs/reports/video_v4d_success/v4a_self_check.{json,md}`，证明 subject/session probe 未下降不会被接受。最终 V4d 成功步骤仍需等完整 V4d 产物训练/评估后，用真实 V4d probe 与 downstream reports 跑通该 gate。

执行记录，2026-07-05 更新：`dinov2_roi.py` 与 `scripts/27_extract_dinov2_roi_embeddings.py` 现在新增 appearance-specific profile：`--augmentation-profile v4d_appearance_mild`。该 profile 在原有 brightness、contrast、color、crop、scale jitter 基础上补齐 `random_grayscale` 和 `light_blur`，写出 `video_v4d_appearance_aug_dinov2_upper_body_full_frame_fallback_mean_std_max` 等 encoder version，并在 quality flags 中记录 7 类 augmentation ops。本地和服务器 `tests/test_dinov2_roi_embeddings.py` 均通过 15 tests，服务器 compileall 通过。服务器真实 upper-body ROI + full-frame fallback smoke 已验证 1-window `(1,256)` 和 10-window `(10,256)` 输出，无 NaN、failures `[]`，且 encoder version/quality flags 符合预期。初始 NumPy blur/resize 路径 CPU 开销过高，现已改为优先使用 OpenCV `blur`/`resize` 快路径并保留 NumPy fallback；10-window timing smoke 在 `--batch-size 32` 下从约 251 秒改善到约 41 秒。重要修正：第一版 full upper-body appearance-augmented extraction 已停止，因为若先把全体样本随机增强后写成全局 `.npz`，直接用于 probes 或 split evaluation 时会让验证/测试行也被增强，违反 V4d 评估协议。后续 random appearance augmentation 必须 fold-aware 且仅用于训练 fold；验证/测试评估必须使用原始 deterministic upper-body embedding。Task 6 Step 3 仍处于进行中，直到实现并验证 train-only augmentation 路径。

执行记录，2026-07-05 A0-A3 消融更新：V4d appearance 消融网格已明确。A0 是原始 deterministic upper-body embedding：`outputs/embeddings/video_v4a_dinov2_upper_body_embeddings.npz`。A1 为 `v4d_a1_color_brightness`，只包含 brightness + color jitter。A2 为 `v4d_a2_color_brightness_grayscale`，在 A1 基础上加入 deterministic random-grayscale probability。A3 为 `v4d_a3_color_brightness_grayscale_crop_scale`，在 A2 基础上加入 crop/scale jitter。`scripts/26_run_video_variant_ablation.py` 现在支持 `NAME=eval_embeddings.npz::train_embeddings.npz` 的 train-only embedding override；`scripts/28_run_video_embedding_probes.py` 现在支持 `--train-embeddings`。两条路径都保证训练 fold 使用 train override，而验证/测试 fold 使用 deterministic eval embedding。本地和服务器 focused tests 均通过 37 tests，覆盖该 split hygiene。服务器 A1/A2/A3 单窗口 train embedding smoke 已验证 encoder version 和 augmentation ops 符合定义。当前服务器串行任务 PID 为 `2211748`，正在生成 full train-only artifacts：`outputs/embeddings/video_v4d_A{1,2,3}_upper_body_train_embeddings.npz`；最近一次检查显示 A1 正在运行，failures `[]`。后续评估 runner 已在服务器以 PID `1779027` 排队，脚本为 `outputs/reports/video_v4d_ablation/run_a0_a3_train_only_eval.sh`；它会等待 A1/A2/A3 train-only artifacts 完成，先与 deterministic A0 校验 sample/label 对齐，再按验证/测试始终使用原始 upper-body embedding 的协议运行 A0-A3 的 LOSO/S1/S4/S2 downstream 与 probes。

执行记录，2026-07-06 A3 停止，已被后续修正覆盖：A3 当时是因为第一版 A1/A2 audit 看起来显示同一 window embedding 与 A0 几乎正交而停止。但该解释已经被 2026-07-07 projection-salt 修正覆盖：旧 A1/A2 audit 混入了不同随机投影矩阵，不能作为 augmentation 强度证据引用。服务器 PID `3403028`（A3 extractor）、`2211748`（A1/A2/A3 generator）和 `1779027`（等待 A3 的 A0-A3 eval runner）已于 2026-07-06 23:00 Asia/Shanghai 停止。A3 当时约 63.9%，但未写出最终 A3 `.npz`。

执行记录，2026-07-06 Weak-Aug：新增弱增强 profile `v4d_weak_color_brightness_contrast`，用于更温和的 V4d 分支。该 profile 只保留 weak brightness、weak contrast、weak color jitter，去掉 grayscale、blur、crop/scale jitter。目标不是把 subject/session probe 从约 `0.95` 打到 `0.07`，而是温和降到约 `0.60-0.80`，同时希望 LOSO/S4/S2 提升，S1 基本保持。本地和服务器 focused tests 已通过；服务器 1-window smoke 写出 `outputs/embeddings/video_v4d_weak_upper_body_smoke1_train_embeddings.npz`，quality flags 符合预期。第一轮 full Weak-Aug run 在发现 projection-salt 问题后已于最终 `.npz` 写出前停止，因此目前不要引用 full Weak-Aug `.npz`。

执行记录，2026-07-07 projection-salt 修正：此前 A1/A2 paired embedding audit 不能用于判断 augmentation 强度，因为 A1/A2 projected artifacts 是 projection salt 修复前生成的；A0/A1/A2 可能使用了不同随机投影矩阵。当前代码已改为不同 augmentation profile 共享 deterministic V4a region salt，例如 upper-body 使用 `video_v4a_dinov2_upper_body_mean_std_max`，但 `encoder_version` 仍保留 variant 区分。旧 A1/A2 artifacts 和旧 A0-A2 reports 已归档到 `outputs/archive/invalid_projection_salt_20260707_0125/`。fixed-salt A1/A2 已重新生成完成，并在 `outputs/reports/video_v4d_fixed_salt/paired_embedding_audit/` 写出新 audit：A0-vs-A1 cosine mean `0.9990`、L2 mean `0.0408`；A0-vs-A2 cosine mean `0.9938`、L2 mean `0.1076`。这说明在共享投影矩阵下 A1/A2 只是轻到中等扰动，必须重新跑下游评估后再判断效果。

## Task 7: V4c 原生视频模型对照

**Files:**
- Create: `src/daily_multimodal/embeddings/native_video.py`
- Create: `scripts/33_extract_native_video_embeddings.py`
- Test: `tests/test_native_video_embeddings.py`

- [ ] Step 1: 仅在 V4a/V4b/ROI/V4d 主要结论稳定后启动。
- [ ] Step 2: 接入 VideoMAE、Video Swin、TimeSformer 中的至少一个 frozen encoder。
- [ ] Step 3: 输出同样的 `(N, 256)` 契约，写入 `encoder_version` 区分模型。
- [ ] Step 4: 与最佳 V4b/V4d 在同一 LOSO/S1/S4/S2 split 上比较。

## Task 8: 文档同步

**Files:**
- Modify: `repo-docs/modules/embedding-contract.md`
- Modify: `repo-docs/references/commands-and-artifacts.md`
- Modify: `repo-docs/change-log.md`

- [x] Step 1: 在 `embedding-contract.md` 说明视频深度模型短期仍写入 `face_emb`，但语义是 video embedding。
- [x] Step 2: 在 `commands-and-artifacts.md` 增加 V4a/V4b/ROI/probe 命令和产物路径。
- [x] Step 3: 在 `change-log.md` 记录视频主线从 OpenFace legacy reference 转向 DINOv2/V4a/V4b 的接口变更。
- [x] Step 4: 运行 repo-docs validator：

```powershell
python C:\Users\28303\.codex\skills\repo-docs\scripts\validate_repo_docs.py repo-docs --repo-root .
```

## 执行优先级

1. 固定 V4a 实现口径：16 帧、Frozen DINOv2、mean/std/max pooling、256D。
2. 修改并运行 `28_run_video_embedding_probes.py`，拿到 subject/session/fatigue 四口径诊断。
3. 用 `26_run_video_variant_ablation.py` 固定 V4a 的 LOSO/S1/S4/S2 表。
4. 开发 V4b-TCN 与 V4b-TemporalTransformer。
5. 进行 ROI 对照：2x face ROI、upper-body、full frame。
6. 只有在诊断支持时推进 V4d。
7. 最后做 V4c 原生视频模型。
