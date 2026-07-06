# Video Modality Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the video branch away from OpenFace statistical features and toward natural-video deep visual encoding: freeze V4a, diagnose subject/session shortcut with probes, then build V4b temporal models, ROI comparisons, V4d robustness, and V4c native-video baselines.

**Architecture:** Keep the current `face_emb (N, 256)` storage contract and face-slot mask for short-term compatibility with existing training entrypoints such as `scripts/26_run_video_variant_ablation.py`; treat the actual content as video embedding and distinguish model versions with `encoder_version`. MediaPipe stays auxiliary only: ROI assistance, behavior audit, quality flags, and interpretation, not the main embedding representation.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, Transformers DINOv2, OpenCV/ffmpeg, scikit-learn, existing NPZ/JSONL/Markdown artifacts, pytest.

---

## Current Constraints and Naming

- Main objective: natural video -> deep visual encoding -> temporal modeling -> reduced subject/session shortcut -> 256D video embedding.
- OpenFace is no longer a mainline comparison target. OpenFace V1/V2 should be treated as archived legacy references.
- The current repository contract remains `face_emb (N, 256)`, with mask order `[eeg, wear, face, audio]`. Documentation may call it video embedding, but the short-term on-disk field should remain unchanged.
- `scripts/28_run_video_embedding_probes.py` will be modified. The fatigue Ridge probe must support LOSO/S1/S4/S2 instead of random KFold only.
- Fixed evaluation protocol:
  - LOSO: `leave_one_subject_out`
  - S1: `within_subject_event_split`
  - S4: `within_subject_session_leave_out`
  - S2: `within_subject_chronological_split`

## File Structure

- Modify: `src/daily_multimodal/embeddings/dinov2_roi.py`
  - Freeze the DINOv2 ROI implementation as V4a: 16 frames, frozen DINOv2, frame sequence, `mean + std + max` temporal pooling, 256D projection.
- Modify: `scripts/27_extract_dinov2_roi_embeddings.py`
  - Expose V4a parameters: `--num-frames 16`, `--temporal-pooling mean_std_max`, and a stable `encoder_version`.
- Modify: `tests/test_dinov2_roi_embeddings.py`
  - Cover 16-frame sampling, pooling metadata, mask contract, and missing ROI video behavior.
- Modify: `src/daily_multimodal/training/video_embedding_probes.py`
  - Keep subject/session probes; add LOSO/S1/S4/S2 fold strategies for the fatigue Ridge probe.
- Modify: `scripts/28_run_video_embedding_probes.py`
  - Add `--fold-strategy`, `--n-splits`, and report the four probe protocols.
- Modify: `tests/test_video_embedding_probes.py`
  - Cover subject/session Logistic probes and fatigue Ridge under the four split protocols.
- Reuse: `src/daily_multimodal/training/video_variant_ablation.py`
  - Reuse `_build_video_folds` and existing split semantics.
- Reuse: `scripts/26_run_video_variant_ablation.py`
  - Run LOSO/S1/S4/S2 downstream metrics for V4a/V4b/V4d/V4c.
- Create: `src/daily_multimodal/embeddings/video_temporal.py`
  - Implement V4b-TCN and V4b-TemporalTransformer temporal encoders.
- Create: `scripts/31_train_video_temporal_encoder.py`
  - CLI to train or extract V4b 256D embeddings.
- Test: `tests/test_video_temporal_embeddings.py`
  - Cover sequence input, output shape, mask, and `encoder_version`.
- Create: `src/daily_multimodal/embeddings/video_regions.py`
  - Unified cache for 2x face ROI, upper-body, and full-frame regions.
- Create: `scripts/29_prepare_video_regions.py`
  - Generate region clips under `outputs/cache/video_regions/...`.
- Test: `tests/test_video_regions.py`
  - Cover upper-body fallback to full frame, region metadata, and quality flags.
- Optional Create: `src/daily_multimodal/embeddings/video_domain_robust.py`
  - V4d subject/session adversarial heads and gradient reversal.
- Optional Create: `src/daily_multimodal/embeddings/native_video.py`
  - V4c entrypoint for VideoMAE, Video Swin, and TimeSformer.
- Docs: `repo-docs/modules/embedding-contract.md`
  - After implementation, document that video embeddings are temporarily stored in the `face_emb` slot.
- Docs: `repo-docs/references/commands-and-artifacts.md`
  - After implementation, add commands and artifacts for scripts 27/28/29/31/33.
- Docs: `repo-docs/change-log.md`
  - Record the video-mainline interface changes and verification.

## Task 1: Freeze V4a DINOv2 Spatial Baseline

**Files:**
- Modify: `src/daily_multimodal/embeddings/dinov2_roi.py`
- Modify: `scripts/27_extract_dinov2_roi_embeddings.py`
- Test: `tests/test_dinov2_roi_embeddings.py`

- [x] Step 1: Add a failing test in `tests/test_dinov2_roi_embeddings.py` where the fake frame encoder returns 16 frame embeddings; assert `sampled_frame_count=16`, `temporal_pooling=mean_std_max`, and `encoder_version=video_v4a_dinov2_2xroi_mean_std_max`.
- [x] Step 2: Run `python -m pytest tests/test_dinov2_roi_embeddings.py -q`; the new test should fail because the current implementation still uses single mean pooling or the old `encoder_version`.
- [x] Step 3: Modify `dinov2_roi.py` so the DINOv2 backend returns a `[frames, hidden_dim]` frame sequence and the main builder applies `mean + std + max` pooling before projection to 256D.
- [x] Step 4: Modify `scripts/27_extract_dinov2_roi_embeddings.py`; freeze the default frame count at 16, keep `--fps`, and add or rename the frame-count parameter to `--num-frames 16` so V4a no longer depends on `max_frames_per_window=20`.
- [x] Step 5: Run `python -m pytest tests/test_dinov2_roi_embeddings.py -q`; it should pass.
- [x] Step 6: Generate the V4a artifact:

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

Execution note, 2026-07-04: full V4a artifact generation completed on `ncc_serve_4090` in the server/cache environment with `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`. Outputs are `outputs/embeddings/video_v4a_dinov2_2xroi_embeddings.npz` and `outputs/embeddings/video_v4a_dinov2_2xroi_frame_sequences.npz`. Verification: `face_emb.shape=(8328, 256)`, `frame_embeddings.shape=(8328, 16, 768)`, `face_mask_sum=8328`, `subject_count=14`, no NaN values, and failures `[]`.

## Task 2: Update and Run V4a Probes

**Files:**
- Modify: `src/daily_multimodal/training/video_embedding_probes.py`
- Modify: `scripts/28_run_video_embedding_probes.py`
- Test: `tests/test_video_embedding_probes.py`
- Reuse: `src/daily_multimodal/training/video_variant_ablation.py`

- [x] Step 1: Add tests in `tests/test_video_embedding_probes.py` using a synthetic `.npz` with multiple subjects, sessions, and events; assert that the fatigue Ridge probe runs under `leave_one_subject_out`, `within_subject_event_split`, `within_subject_session_leave_out`, and `within_subject_chronological_split`.
- [x] Step 2: Run `python -m pytest tests/test_video_embedding_probes.py -q`; the new split tests should fail.
- [x] Step 3: Modify `video_embedding_probes.py`; keep the subject/session Logistic probe logic, add `fold_strategy` to the fatigue Ridge probe, and reuse `video_variant_ablation._build_video_folds` for train/val/test splits.
- [x] Step 4: Modify `scripts/28_run_video_embedding_probes.py`; add `--fold-strategy` with choices aligned to `scripts/26_run_video_variant_ablation.py`.
- [x] Step 5: Run `python -m pytest tests/test_video_embedding_probes.py tests/test_video_variant_ablation.py -q`; it should pass.
- [x] Step 6: Run subject/session probes and the four fatigue Ridge protocols on V4a:

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

Execution note, 2026-07-04: Step 6 completed on `ncc_serve_4090` using `outputs/embeddings/video_v4a_dinov2_2xroi_embeddings.npz` with 8328 rows. Subject probe accuracy is 0.9777 and within-subject session probe accuracy is 0.9691, showing strong identity/session information in V4a. Fatigue Ridge results: LOSO RMSE 1.0333, Pearson r -0.0460; S1 RMSE 0.9188, Pearson r 0.2746; S4 RMSE 0.9620, Pearson r 0.1702; S2 RMSE 1.0172, Pearson r 0.2254. Reports are under `outputs/reports/video_probes/v4a_*_probes.{json,md}`.

## Task 3: Freeze the V4a Downstream Evaluation Table

**Files:**
- Reuse: `scripts/26_run_video_variant_ablation.py`
- Reuse: `src/daily_multimodal/training/video_variant_ablation.py`
- Optional Docs: `outputs/reports/video_variants/v4a/v4a_split_summary.md`

- [x] Step 1: Run V4a under LOSO/S1/S4/S2 with `scripts/26_run_video_variant_ablation.py`.
- [x] Step 2: Use `--sample-mode behavior_retained` for each run because V4a usability is defined by readable ROI video, not by the inherited OpenFace face-detection mask.
- [x] Step 3: Summarize the four JSON outputs into `outputs/reports/video_variants/v4a/v4a_split_summary.md`; include rows, folds, RMSE mean/std, and Pearson r mean/std.
- [x] Step 4: Apply the interpretation rule:
  - If S1 is clearly strong but LOSO/S4/S2 are weak, and subject/session probes are high, continue to V4b and ROI comparison before considering V4d.
  - If V4a is also stable under LOSO, keep V4b lightweight and do not prioritize adversarial training.

Execution note, 2026-07-04: Task 3 completed on `ncc_serve_4090` with `--sample-mode behavior_retained`. Summary report: `outputs/reports/video_variants/v4a/v4a_split_summary.md`. V4a downstream MLP results: LOSO RMSE 1.0083, Pearson r -0.0517; S1 RMSE 0.8801, Pearson r 0.3258; S4 RMSE 0.9477, Pearson r 0.1788; S2 RMSE 0.9703, Pearson r 0.2272. Interpretation: S1 is clearly stronger than LOSO/S4/S2 and subject/session probes are high, so continue to V4b temporal modeling and ROI comparison before V4d adversarial/domain robustness.

## Task 4: Build V4b Temporal Models

**Files:**
- Create: `src/daily_multimodal/embeddings/video_temporal.py`
- Create: `scripts/31_train_video_temporal_encoder.py`
- Test: `tests/test_video_temporal_embeddings.py`

- [x] Step 1: Add tests with synthetic `[N, 16, D]` frame sequences and assert that both TCN and Temporal Transformer produce `[N, 256]`.
- [x] Step 2: Implement `V4b-TCN`; input is the DINOv2 frame sequence, output is a 256D embedding, and `encoder_version=video_v4b_tcn_dinov2_2xroi`.
- [x] Step 3: Implement `V4b-TemporalTransformer`; input is the same frame sequence, output is a 256D embedding, and `encoder_version=video_v4b_temporal_transformer_dinov2_2xroi`.
- [x] Step 4: Keep `.npz` output compatible with existing loaders: `sample_id`, `event_id`, `subject_id`, `labels`, `face_emb`, `modality_mask`, `quality_flags`, and `encoder_version`.
- [x] Step 5: Run `python -m pytest tests/test_video_temporal_embeddings.py -q`.
- [x] Step 6: Evaluate V4a, V4b-TCN, and V4b-TemporalTransformer with `scripts/26_run_video_variant_ablation.py` under LOSO/S1/S4/S2; record whether temporal modeling is the source of any gain.

Execution note, 2026-07-04: Task 4 completed on `ncc_serve_4090`. V4b artifacts are `outputs/embeddings/video_v4b_tcn_dinov2_2xroi_embeddings.npz` and `outputs/embeddings/video_v4b_temporal_transformer_dinov2_2xroi_embeddings.npz`; both have `face_emb.shape=(8328, 256)`, `face_mask_sum=8328`, and no NaN values. Summary report: `outputs/reports/video_variants/v4b/v4a_v4b_temporal_summary.md`. Temporal modeling gives only limited benefit: V4b-TCN slightly improves LOSO RMSE but Pearson r remains near zero; Temporal Transformer is slightly best on S1/S4; both V4b variants degrade S2 relative to V4a.

## Task 5: Compare Video Input Regions

**Files:**
- Create: `src/daily_multimodal/embeddings/video_regions.py`
- Create: `scripts/29_prepare_video_regions.py`
- Test: `tests/test_video_regions.py`
- Reuse: `src/daily_multimodal/embeddings/video_behavior_flags.py`
- Reuse: `scripts/23_extract_video_behavior_flags.py`
- Reuse: `scripts/24_audit_video_behavior_flags.py`

- [x] Step 1: Implement region cache output paths:
  - `outputs/cache/video_regions/2x_face_roi/<sample_id>/window.mp4`
  - `outputs/cache/video_regions/upper_body/<sample_id>/window.mp4`
  - `outputs/cache/video_regions/full_frame/<sample_id>/window.mp4`
- [x] Step 2: Use MediaPipe pose/face landmarks or existing detection metadata to assist upper-body localization; when upper-body cannot be localized, write `quality_flags.upper_body_fallback_full_frame=true` and use full frame.
- [x] Step 3: Extract V4a or the best V4b embedding for R1/R2/R3.
- [x] Step 4: Compare R1/R2/R3 under LOSO/S1/S4/S2.
- [x] Step 5: Decision rule: if upper-body is no worse than 2x ROI on S4/S2 and more stable on LOSO, use upper-body as the default going forward; otherwise keep 2x ROI as the baseline.

Execution note, 2026-07-04/05: `video_regions.py` and `scripts/29_prepare_video_regions.py` now build the region cache paths, manifest, sidecars, and fallback flags. `scripts/27_extract_dinov2_roi_embeddings.py` accepts `--region-cache-root` and `--video-region {2x_face_roi,upper_body,full_frame}` so R1/R2/R3 region caches can feed the same V4a extractor. It also accepts `--direct-video-region-from-window` for `upper_body` and `full_frame`, which samples region frames directly from the source videos and can be used for smoke/debug extraction without waiting for the full audit cache. Server smoke status: one-window region cache wrote all three regions with failures `[]`; DINOv2 extraction from `upper_body` and `full_frame` region caches produced `(1,256)` embeddings; direct source-video DINOv2 smoke over two windows produced `(2,256)` embeddings for both `upper_body` and `full_frame`, with mask sum 2, no NaN values, failures `[]`, and region-specific encoder versions. The full grouped region cache completed on `ncc_serve_4090`: `upper_body` and `full_frame` each have 8328 `window.mp4` files and 8328 `region.json` sidecars, manifest rows total 16656, and failures total 0. R2 and R3 V4a artifacts were extracted as `outputs/embeddings/video_v4a_dinov2_upper_body_embeddings.npz` and `outputs/embeddings/video_v4a_dinov2_full_frame_embeddings.npz`; both verify as `face_emb.shape=(8328,256)`, frame sequences `(8328,16,768)`, mask sum 8328, no NaN values, and failures `[]`. Region comparison report: `outputs/reports/video_variants/regions/region_comparison_summary.md`. Decision: keep R1 / 2x face ROI as the current default baseline because R2 upper-body improves LOSO/S1/S4 RMSE but is worse on S2 (R2 S2 RMSE 1.0049, r 0.1435 vs R1 S2 RMSE 0.9703, r 0.2272); R3 full-frame is not default because it hurts LOSO/S2 RMSE despite stronger S1/S4 Pearson r.

## Task 6: V4d Generalization Enhancement

**Files:**
- Modify: `src/daily_multimodal/embeddings/video_temporal.py`
- Optional Create: `src/daily_multimodal/embeddings/video_domain_robust.py`
- Test: `tests/test_video_domain_robust.py`

- [x] Step 1: Confirm the trigger condition: S1 is much stronger than LOSO/S4/S2, and V4a/V4b subject or session probe accuracy is high.
- [x] Step 2: Add the first appearance-augmentation interface.
- [ ] Step 3: Make appearance augmentation the first V4d priority: upper-body ROI plus brightness jitter, contrast jitter, color jitter, random grayscale, light blur, and crop/scale jitter. Augmentation is train-fold only; validation/test rows must use deterministic original upper-body inputs.
- [ ] Step 4: Add lightweight ROI stabilization only after the appearance path is runnable: ROI temporal smoothing, abnormal crop-scale limits, high-fallback-session checks, and QC for severe subject/session outliers.
- [ ] Step 5: Re-run probes and downstream evaluation after augmentation: Subject Probe, Session Probe, LOSO, S1, S4, and S2.
- [ ] Step 6: Enter GRL only if appearance augmentation still leaves high subject/session probes and LOSO remains near zero; then add GRL subject head and GRL session head.
- [ ] Step 7: Success criterion: subject/session probe scores drop while fatigue LOSO/S4/S2 RMSE does not increase and Pearson r does not decrease; S1 should be retained.

### V4d Priority Update After ROI Geometry Audit

The current V4d priority is evidence-driven:

1. **First priority: appearance augmentation.** Use `upper_body` ROI with brightness jitter, contrast jitter, color jitter, random grayscale, light blur, and crop/scale jitter. Rationale: DINOv2 Session Probe is about `0.969`, while the geometry-only Session Probe from `outputs/reports/roi_audit/geometry_session_probe.json` is only `0.364`; the remaining session signal is therefore not explained by simple ROI geometry alone and likely comes from clothing, lighting, background, and other appearance cues.
2. **Second priority: lightweight ROI stabilization.** Do not immediately replace the region cache with a heavy full Pose pipeline. Start with ROI temporal smoothing, limits on abnormal crop scale, high-fallback-session checks, and QC for severe subject/session outliers. The ROI audit shows large geometry drift for some subjects/sessions, but the drift is uneven, so full-data heavy ROI rebuilding is not the first move.
3. **Third priority: re-probe and re-evaluate.** After augmentation, re-run Subject Probe, Session Probe, LOSO, S1, S4, and S2 with strict split hygiene: training folds may use random augmented views, while validation/test folds must use original deterministic upper-body embeddings. Desired pattern: Session Probe down, Subject Probe down, LOSO/S4/S2 up, and S1 retained.
4. **Fourth priority: GRL.** Only add gradient reversal if appearance augmentation still leaves high session/subject probes and LOSO remains near zero. At that point add GRL subject and session heads.

Execution note, 2026-07-05: The V4d trigger condition is met: V4a has high subject/session probe accuracy (subject 0.9777, session 0.9691), S1 remains substantially stronger than LOSO, and ROI comparison does not remove the S2/LOSO weakness. `scripts/27_extract_dinov2_roi_embeddings.py` and `dinov2_roi.py` now expose `--augmentation-profile v4d_mild_color_crop_scale` and `--augmentation-views`; the profile applies deterministic brightness, contrast, color jitter, crop jitter, and scale jitter at the raw-frame level, averages original plus augmented DINOv2 frame embeddings, and writes encoder versions such as `video_v4d_aug_dinov2_2xroi_mean_std_max`. Server smoke over one real 2x ROI window wrote `outputs/embeddings/video_v4d_aug_dinov2_2xroi_smoke1_embeddings.npz` with `face_emb.shape=(1,256)`, frame sequence `(1,16,768)`, mask `[[0,0,1,0]]`, no NaN values, failures `[]`, and quality flags recording the five augmentation ops. A full V4d 2xROI extraction was started but stopped after 6 hours because the shared GPUs were saturated and progress was only about 35.6%; no full V4d artifact or downstream evaluation is claimed yet. To make the full extraction recoverable, `scripts/27_extract_dinov2_roi_embeddings.py` now supports `--start-index` in addition to `--max-windows`; a real server chunk smoke with `--start-index 10 --max-windows 1` wrote `outputs/embeddings/video_v4d_aug_dinov2_2xroi_chunk_smoke_start10_embeddings.npz` and the matching frame-sequence bundle, verified the output sample id as `sub-02_ses-01_00_row-0001_win-0010`, and produced `(1,256)` / `(1,16,768)` outputs with mask `[[0,0,1,0]]`, no NaNs, failures `[]`, and augmentation flags. The stable ROI policy is now explicit rather than global: `--video-region upper_body --fallback-video-region full_frame` keeps R1/2xROI as the current default baseline while allowing V4d runs to use upper-body first and full-frame cache fallback if an upper-body clip is missing. Local and remote tests cover the true fallback path; a real one-window server smoke wrote `outputs/embeddings/video_v4d_roi_policy_upper_full_smoke1_embeddings.npz` with `face_emb.shape=(1,256)`, mask `[[0,0,1,0]]`, no NaN values, failures `[]`, encoder version `video_v4a_dinov2_upper_body_full_frame_fallback_mean_std_max`, and quality flags recording requested/effective/fallback regions. `src/daily_multimodal/embeddings/video_domain_robust.py` now provides stable subject/session target encoding, gradient reversal, and PyTorch subject/session adversarial heads with combined adversarial loss. Local tests cover the non-PyTorch label contract; server tests in the `lzs` environment cover gradient sign reversal and head/loss shapes. `scripts/32_check_video_v4d_success.py` now formalizes the final success gate: subject and session probe accuracy must strictly drop, while LOSO/S4/S2 fatigue RMSE must not increase and Pearson r must not decrease. A server V4a self-check intentionally fails this gate and writes `outputs/reports/video_v4d_success/v4a_self_check.{json,md}`, proving that unchanged subject/session probes are not accepted. The final V4d success step remains pending until full V4d artifacts can be trained/evaluated and this gate passes against real V4d probe and downstream reports.

Execution note, 2026-07-05 update: `dinov2_roi.py` and `scripts/27_extract_dinov2_roi_embeddings.py` now add the appearance-specific profile `--augmentation-profile v4d_appearance_mild`, which extends the first V4d augmentation path with `random_grayscale` and `light_blur` in addition to brightness, contrast, color, crop, and scale jitter. The new profile writes encoder versions such as `video_v4d_appearance_aug_dinov2_upper_body_full_frame_fallback_mean_std_max` and quality flags listing all seven augmentation ops. Local and server `tests/test_dinov2_roi_embeddings.py` now pass 15 tests; server compileall passes. Real server smokes on `ncc_serve_4090` verified upper-body ROI plus full-frame fallback with `face_emb.shape=(1,256)` for one window and `(10,256)` for 10-window timing runs, failures `[]`, no NaNs, and the expected quality flags. The initial NumPy blur/resize path was too CPU-heavy, so the implementation now uses OpenCV `blur`/`resize` fast paths with NumPy fallbacks; the 10-window timing smoke improved from about 251 seconds to about 41 seconds with `--batch-size 32`. Important correction: the first attempted full upper-body appearance-augmented extraction was stopped because applying random augmentation before writing a global `.npz` would also augment validation/test rows if used directly for probes or split evaluation. That violates the V4d protocol. Going forward, random appearance augmentation must be fold-aware and train-only; validation/test evaluation must use original deterministic upper-body embeddings. Task 6 Step 3 remains in progress until that train-only augmentation path is implemented and evaluated without test-time augmentation.

Execution note, 2026-07-05 A0-A3 ablation update: The V4d appearance ablation grid is now explicit. A0 is the original deterministic upper-body embedding, `outputs/embeddings/video_v4a_dinov2_upper_body_embeddings.npz`. A1 is `v4d_a1_color_brightness` with brightness plus color jitter. A2 is `v4d_a2_color_brightness_grayscale` with A1 plus deterministic random-grayscale probability. A3 is `v4d_a3_color_brightness_grayscale_crop_scale` with A2 plus crop/scale jitter. `scripts/26_run_video_variant_ablation.py` now supports train-only embedding overrides through `NAME=eval_embeddings.npz::train_embeddings.npz`; `scripts/28_run_video_embedding_probes.py` now supports `--train-embeddings`. In both paths, train folds use the train override while validation/test folds use the deterministic eval embedding. Local and server tests cover this split hygiene and pass 37 focused tests. Server smokes verified A1/A2/A3 one-window train embeddings with expected encoder versions and augmentation ops. A sequential server job is running as PID `2211748` to generate full train-only artifacts `outputs/embeddings/video_v4d_A{1,2,3}_upper_body_train_embeddings.npz`; the latest check showed A1 running with failures `[]`. A follow-on server runner is queued as PID `1779027` via `outputs/reports/video_v4d_ablation/run_a0_a3_train_only_eval.sh`; it waits for A1/A2/A3 train-only artifacts, validates alignment against deterministic A0, then runs A0-A3 downstream LOSO/S1/S4/S2 and probes using deterministic upper-body embeddings for all validation/test rows.

Execution note, 2026-07-06 A3 stop: A1 and A2 completed and were analyzed first under the train-only override protocol. A paired embedding audit showed A1/A2 moved same-window embeddings nearly orthogonally from A0 (`cosine` near 0 and `L2` near `sqrt(2)`), explaining why subject/session probes dropped while fatigue downstream performance degraded. Based on that evidence, A3 was stopped before completion. Server PIDs `3403028` (A3 extractor), `2211748` (A1/A2/A3 generator), and `1779027` (A0-A3 eval runner waiting for A3) were terminated on 2026-07-06 23:00 Asia/Shanghai. A3 had reached about 63.9% but no final A3 `.npz` was written. Keep the completed A1/A2 artifacts and A0-A2 reports; do not wait for or cite A3 results.

Execution note, 2026-07-06 Weak-Aug: A new weak profile `v4d_weak_color_brightness_contrast` was added for a gentler V4d branch. It keeps only weak brightness, weak contrast, and weak color jitter, and removes grayscale, blur, and crop/scale jitter. The intended probe target is moderate shortcut reduction, not collapse: subject/session probes should move from about `0.95` toward roughly `0.60-0.80`, while LOSO/S4/S2 should improve and S1 should be mostly retained. Local and server focused tests pass for the new profile, and a one-window server smoke wrote `outputs/embeddings/video_v4d_weak_upper_body_smoke1_train_embeddings.npz` with expected quality flags. Full train-only Weak-Aug extraction plus paired audit/downstream/probe evaluation is running on `ncc_serve_4090` as runner PID `380378` with extractor PID `380420`; outputs will use `outputs/embeddings/video_v4d_weak_upper_body_train_embeddings.npz`, `outputs/reports/video_v4d_weak/`, `outputs/reports/video_variants/v4d_weak_train_only/`, and `outputs/reports/video_probes/v4d_weak_train_only/`.

## Task 7: V4c Native Video Model Baselines

**Files:**
- Create: `src/daily_multimodal/embeddings/native_video.py`
- Create: `scripts/33_extract_native_video_embeddings.py`
- Test: `tests/test_native_video_embeddings.py`

- [ ] Step 1: Start only after the V4a/V4b/ROI/V4d conclusions are stable.
- [ ] Step 2: Integrate at least one frozen encoder among VideoMAE, Video Swin, and TimeSformer.
- [ ] Step 3: Output the same `(N, 256)` contract and use `encoder_version` to identify the model.
- [ ] Step 4: Compare against the best V4b/V4d model under the same LOSO/S1/S4/S2 splits.

## Task 8: Documentation Sync

**Files:**
- Modify: `repo-docs/modules/embedding-contract.md`
- Modify: `repo-docs/references/commands-and-artifacts.md`
- Modify: `repo-docs/change-log.md`

- [x] Step 1: Update `embedding-contract.md` to explain that deep video embeddings are temporarily stored in the `face_emb` slot.
- [x] Step 2: Update `commands-and-artifacts.md` with V4a/V4b/ROI/probe commands and artifact paths.
- [x] Step 3: Update `change-log.md` to record the video mainline interface move from OpenFace legacy reference to DINOv2/V4a/V4b.
- [x] Step 4: Run the repo-docs validator:

```powershell
python C:\Users\28303\.codex\skills\repo-docs\scripts\validate_repo_docs.py repo-docs --repo-root .
```

## Execution Priority

1. Freeze the V4a implementation: 16 frames, frozen DINOv2, mean/std/max pooling, 256D.
2. Modify and run `28_run_video_embedding_probes.py` for subject/session/fatigue diagnostics.
3. Use `26_run_video_variant_ablation.py` to freeze the V4a LOSO/S1/S4/S2 result table.
4. Build V4b-TCN and V4b-TemporalTransformer.
5. Compare input regions: 2x face ROI, upper-body, full frame.
6. Start V4d only if diagnostics support it.
7. Run V4c native video models last.
