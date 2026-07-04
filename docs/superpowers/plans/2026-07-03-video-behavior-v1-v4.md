# Video Behavior V1-V4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the current face/OpenFace video branch into four comparable `face_emb` variants: OpenFace only, OpenFace plus behavior flags, full-frame deep only, and OpenFace plus deep behavior.

**Architecture:** Keep the existing repository contract unchanged: video variants continue to occupy the `face_emb (N, 256)` slot and the mask column remains `[eeg, wear, face, audio]`. Behavior flags are extracted as a separate window-level JSONL, audited before training, then projected or fused into the `face_emb` vector for V2/V3. All comparisons use identical `sample_id` order where possible, with explicit aligned and behavior-retained evaluation modes.

**Tech Stack:** Python 3.10+, numpy, OpenCV, ffmpeg/ffprobe, OpenFace CSV, MediaPipe Holistic or Hands/Pose, YOLO only when the selected weights expose the required classes, pytest, JSONL/NPZ/Markdown reports.

---

## Incorporated Plan Corrections

This plan already includes the six execution corrections from the review:

1. Do not create a new `video_emb_*` contract. Every variant writes `face_emb` and uses `encoder_version` for `openface_only_v1`, `openface_behavior_flags_v2`, `fullframe_deep_v4`, or `openface_deep_behavior_v3`.
2. Do not run behavior extraction from a face-filtered index. Use the original `window_index` or run cache preparation with `--skip-face-presence-filter`, so low-head, occlusion, side-turn, and offscreen windows are not deleted before measurement.
3. Treat OpenFace `pose_Rx`, `pose_Ry`, and `pose_Rz` as radians in code. Human-facing thresholds may be written as degrees, but implementation must compare to `math.radians(...)`.
4. For V2, first run a minimal fixed-projection validation: behavior ratios only, then OpenFace plus behavior ratios. Do not train a new representation layer before proving the flags carry signal.
5. Make the behavior audit reproducible by writing the exact `sample_id`, source MP4, clip seconds, flag values, and OpenFace mask/quality state for each reviewed window.
6. Compare V1/V2 on two explicit sample-set modes: `strict_aligned` for rows where both variants are usable, and `behavior_retained` for rows where V2 keeps valid behavior windows even if OpenFace quality is low.

## File Structure

- Create: `src/daily_multimodal/embeddings/video_behavior_flags.py`
  - Extract frame-level detections and aggregate the 8 behavior ratios per `sample_id`.
  - Keep detector output and rule code independent from OpenFace embedding code.
- Create: `scripts/23_extract_video_behavior_flags.py`
  - CLI wrapper that reads `window_index.jsonl`, optionally limits rows, writes `outputs/cache/video_behavior_flags/video_behavior_flags.jsonl`, and writes failures.
- Create: `scripts/24_audit_video_behavior_flags.py`
  - CLI wrapper that summarizes ratio distributions and writes top-window review manifests.
- Create: `src/daily_multimodal/embeddings/video_variants.py`
  - Build V1/V2/V4/V3 `face_emb` bundles from OpenFace CSV, behavior flags, and optional full-frame features.
- Create: `scripts/25_build_video_variant_embeddings.py`
  - CLI wrapper that writes one `.npz` per video variant while preserving the existing `face_emb` contract.
- Create: `src/daily_multimodal/training/video_variant_ablation.py`
  - Run video-only and fixed-multimodal comparisons with paired fold output.
- Create: `scripts/26_run_video_variant_ablation.py`
  - CLI wrapper for V0/V1/V2/V4/V3 and M0/M1/M2/M3/M4 experiments.
- Modify: `configs/encoders.yaml`
  - Add the four video/face profiles and their thresholds.
- Modify: `repo-docs/modules/embedding-contract.md`
  - Only after code lands, document that behavior-video variants still write to `face_emb`.
- Modify: `repo-docs/references/commands-and-artifacts.md`
  - Only after code lands, add the new stage commands and artifacts.
- Test: `tests/test_video_behavior_flags.py`
- Test: `tests/test_video_variant_embeddings.py`
- Test: `tests/test_video_variant_ablation.py`

## Execution Order

### Task 1: Lock The Contract And Profiles

**Files:**
- Modify: `configs/encoders.yaml`
- Test: `tests/test_video_variant_embeddings.py`

- [ ] **Step 1: Add profile names to config**

Add these profile records under `face_real_profiles`:

```yaml
  openface_only_v1:
    output_dim: 256
    source: openface_csv
    extractor: openface
    contract_key: face_emb
    min_success_rate: 0.50
  openface_behavior_flags_v2:
    output_dim: 256
    source: openface_csv_plus_behavior_flags
    contract_key: face_emb
    behavior_flag_count: 8
    projection: deterministic_fixed_random
  fullframe_deep_v4:
    output_dim: 256
    source: full_window_video
    contract_key: face_emb
    frame_count: 8
    pooling: mean_std
  openface_deep_behavior_v3:
    output_dim: 256
    source: openface_csv_plus_fullframe_deep
    contract_key: face_emb
    optional_behavior_flags: true
```

- [ ] **Step 2: Write a contract test**

Add a test that builds a tiny V2 bundle and asserts:

```python
assert "face_emb" in loaded.files
assert "video_emb_v2" not in loaded.files
assert loaded["face_emb"].shape == (2, 256)
assert loaded["modality_mask"].tolist() == [[0, 0, 1, 0], [0, 0, 1, 0]]
assert set(loaded["encoder_version"].astype(str)) == {"openface_behavior_flags_v2"}
```

- [ ] **Step 3: Run the focused test**

Run:

```powershell
python -m pytest tests/test_video_variant_embeddings.py -q
```

Expected first result: failure because the variant builder does not exist yet.

### Task 2: Extract Window-Level Behavior Flags Without Face Filtering

**Files:**
- Create: `src/daily_multimodal/embeddings/video_behavior_flags.py`
- Create: `scripts/23_extract_video_behavior_flags.py`
- Test: `tests/test_video_behavior_flags.py`

- [ ] **Step 1: Define the output schema**

Each JSONL row must include:

```json
{
  "sample_id": "sub-02_ses-03_00_row-0012_win-0003",
  "event_id": "sub-02_ses-03_00_row-0012",
  "subject_id": "sub-02",
  "source_mp4_path": "absolute-or-server-path.mp4",
  "clip_start_seconds": 92.0,
  "clip_end_seconds": 102.0,
  "sampled_frame_count": 20,
  "usable_frame_count": 20,
  "face_visible_ratio": 0.75,
  "low_confidence_ratio": 0.25,
  "head_down_ratio": 0.40,
  "side_turn_ratio": 0.10,
  "hand_near_face_ratio": 0.30,
  "hand_occlusion_ratio": 0.20,
  "large_motion_ratio": 0.55,
  "offscreen_ratio": 0.05,
  "detectors": {
    "face": "openface_or_mediapipe_or_yolo",
    "hand": "mediapipe_hands_or_yolo_hand",
    "person": "yolo_person_or_mediapipe_pose"
  }
}
```

- [ ] **Step 2: Write rule tests with synthetic frames**

Cover these cases:

```python
def test_pose_thresholds_use_radians():
    rows = [{"pose_Rx": math.radians(21), "pose_Ry": math.radians(31), "confidence": 0.9, "success": 1}]
    flags = frame_flags_from_openface_rows(rows)
    assert flags[0]["head_down"] is True
    assert flags[0]["side_turn"] is True

def test_behavior_rows_do_not_require_face_presence_filter():
    window = {"sample_id": "s1", "event_id": "e1", "subject_id": "sub-01"}
    row = aggregate_behavior_window(window, frame_flags=[{"offscreen": True}] * 20, source={})
    assert row["offscreen_ratio"] == 1.0
    assert row["sampled_frame_count"] == 20
```

- [ ] **Step 3: Implement the first-pass rules**

Use these exact first-pass rules:

- `face_visible = 1` when OpenFace success, a face bbox, or MediaPipe face landmarks exist.
- `low_confidence = 1` when OpenFace confidence `< 0.80`, YOLO face confidence `< 0.50`, or MediaPipe face tracking fails.
- `head_down = 1` when `pose_Rx > math.radians(20)`.
- `side_turn = 1` when `abs(pose_Ry) > math.radians(30)`.
- `hand_near_face = 1` when hand center distance to face bbox center `< 0.75 * face_bbox_width`.
- `hand_occlusion = 1` when hand-face intersection over face area `> 0.10` or more than 20% of hand landmarks lie inside face bbox.
- `large_motion = 1` when adjacent person bbox center displacement over person bbox height `> 0.10`.
- `offscreen = 1` when there is no person bbox, no face bbox, and MediaPipe pose/face both fail.

- [ ] **Step 4: Add the extraction CLI**

The CLI must default to the unfiltered window index:

```powershell
python scripts/23_extract_video_behavior_flags.py `
  --window-index outputs/window_index/window_index.jsonl `
  --out outputs/cache/video_behavior_flags/video_behavior_flags.jsonl `
  --failures-out outputs/reports/video_behavior_flags_failures.json `
  --fps 2 `
  --max-windows 10
```

The help text must warn that face-filtered inputs such as `real_cache_face_detected...jsonl` are not valid for the main behavior audit.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests/test_video_behavior_flags.py -q
```

Expected: all behavior rule tests pass.

### Task 3: Audit Behavior Flags Before Embedding

**Files:**
- Create: `scripts/24_audit_video_behavior_flags.py`
- Test: `tests/test_video_behavior_flags.py`

- [ ] **Step 1: Write audit aggregation tests**

Assert the report contains:

```python
assert report["window_count"] == 4
assert report["success_count"] == 4
assert "face_visible_ratio" in report["ratios"]
assert report["ratios"]["head_down_ratio"]["mean"] == 0.5
assert len(report["review_sets"]["top_head_down_ratio"]) == 2
```

- [ ] **Step 2: Implement the summary**

The audit JSON must include total windows, successful rows, missing rows, mean and median for all 8 ratios, plus review sets for:

- `top_head_down_ratio`
- `top_hand_occlusion_ratio`
- `top_offscreen_ratio`
- `top_large_motion_ratio`
- `random_windows`

Each review row must include `sample_id`, `event_id`, `subject_id`, `source_mp4_path`, `clip_start_seconds`, `clip_end_seconds`, all 8 ratios, `openface_mask_value` when available, and `openface_quality_flags` when available.

- [ ] **Step 3: Run the audit on a small sample**

Run:

```powershell
python scripts/24_audit_video_behavior_flags.py `
  --flags outputs/cache/video_behavior_flags/video_behavior_flags.jsonl `
  --openface-embeddings outputs/embeddings/face_openface_real_full_mainface_roi_embeddings.npz `
  --out-json outputs/reports/video_behavior_flags_audit.json `
  --out-table outputs/reports/video_behavior_flags_audit.md `
  --top-k 20
```

Expected: the Markdown table contains clip coordinates that a reviewer can open directly.

### Task 4: Build V1 And V2 `face_emb` Bundles

**Files:**
- Create: `src/daily_multimodal/embeddings/video_variants.py`
- Create: `scripts/25_build_video_variant_embeddings.py`
- Test: `tests/test_video_variant_embeddings.py`

- [ ] **Step 1: Implement V1 as the current OpenFace stats baseline**

V1 reads OpenFace-compatible CSV or an existing OpenFace `.npz`, writes:

```text
face_emb: (N, 256)
modality_mask: [0, 0, face_mask, 0]
encoder_version: openface_only_v1
quality_flags: original OpenFace quality plus variant name
```

- [ ] **Step 2: Implement V2 behavior-only smoke mode**

Before fusing with OpenFace, support a behavior-only bundle for validation:

```text
encoder_version = behavior_flags_only_v2_probe
face_emb = deterministic_projection([8 behavior ratios], 256)
modality_mask = [0, 0, 1, 0] when the behavior row exists and has at least one usable frame
```

- [ ] **Step 3: Implement V2 OpenFace plus behavior flags**

Use:

```text
feature_vector = concat(openface_stats, 8 behavior ratios)
face_emb = deterministic_projection(feature_vector, 256)
encoder_version = openface_behavior_flags_v2
```

Rows with behavior extraction success but low OpenFace quality must be preserved in `behavior_retained` mode and marked in `quality_flags`; rows missing the video source or having no readable frames must use `modality_mask=0`.

- [ ] **Step 4: Run V1/V2 build commands**

Run:

```powershell
python scripts/25_build_video_variant_embeddings.py `
  --variant openface_only_v1 `
  --window-index outputs/window_index/window_index.jsonl `
  --openface-embeddings outputs/embeddings/face_openface_real_full_mainface_roi_embeddings.npz `
  --out outputs/embeddings/face_openface_only_v1_embeddings.npz

python scripts/25_build_video_variant_embeddings.py `
  --variant behavior_flags_only_v2_probe `
  --window-index outputs/window_index/window_index.jsonl `
  --behavior-flags outputs/cache/video_behavior_flags/video_behavior_flags.jsonl `
  --out outputs/embeddings/face_behavior_flags_only_v2_probe_embeddings.npz

python scripts/25_build_video_variant_embeddings.py `
  --variant openface_behavior_flags_v2 `
  --window-index outputs/window_index/window_index.jsonl `
  --openface-embeddings outputs/embeddings/face_openface_real_full_mainface_roi_embeddings.npz `
  --behavior-flags outputs/cache/video_behavior_flags/video_behavior_flags.jsonl `
  --sample-mode strict_aligned `
  --out outputs/embeddings/face_openface_behavior_flags_v2_strict_embeddings.npz
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests/test_video_variant_embeddings.py -q
```

Expected: V1/V2 outputs preserve `sample_id` order, `face_emb` shape, and the existing `modality_mask` shape.

### Task 5: Run Video-Only V1/V2 Ablation First

**Files:**
- Create: `src/daily_multimodal/training/video_variant_ablation.py`
- Create: `scripts/26_run_video_variant_ablation.py`
- Test: `tests/test_video_variant_ablation.py`

- [ ] **Step 1: Implement V0/V1/V2 loading**

The ablation runner must accept a list of face/video `.npz` files and evaluate only the face slot. It must output:

- RMSE mean and std.
- Pearson r mean and std.
- `pred_std` mean and std.
- `truth_std` mean and std.
- `error_std` mean and std.
- Per-fold metrics.
- Paired fold deltas for V2 vs V1.

- [ ] **Step 2: Implement both sample-set modes**

The runner must support:

```text
strict_aligned: compare only identical sample_id rows with usable V1 and usable V2 masks.
behavior_retained: compare V1 on usable OpenFace rows and V2 on behavior-usable rows, with sample counts reported separately.
```

- [ ] **Step 3: Run the first decisive comparison**

Run:

```powershell
python scripts/26_run_video_variant_ablation.py `
  --target-label fatigue `
  --mode video_only `
  --sample-mode strict_aligned `
  --variants `
    V0=mean_baseline `
    V1=outputs/embeddings/face_openface_only_v1_embeddings.npz `
    V2_probe=outputs/embeddings/face_behavior_flags_only_v2_probe_embeddings.npz `
    V2=outputs/embeddings/face_openface_behavior_flags_v2_strict_embeddings.npz `
  --out-json outputs/reports/video_variant_v1_v2_strict_fatigue_metrics.json `
  --out-table outputs/reports/video_variant_v1_v2_strict_fatigue_table.md
```

Decision rule: continue to V4 only after the audit is reviewed and V2 is not obviously broken by low variance, empty folds, or sample mismatch.

### Task 6: Add V4 Full-Frame Deep Only

**Files:**
- Modify: `src/daily_multimodal/embeddings/video_variants.py`
- Modify: `scripts/25_build_video_variant_embeddings.py`
- Test: `tests/test_video_variant_embeddings.py`

- [ ] **Step 1: Implement the light first version**

Use a frozen image encoder if available; otherwise implement the same deterministic feature path used elsewhere in this repo:

```text
sample 8 frames per 10-second window
resize frames
extract frame features
pool mean and std across time
project to 256
write face_emb with encoder_version fullframe_deep_v4
```

This task must not introduce VideoMAE, SlowFast, or another heavy video model.

- [ ] **Step 2: Run V1/V2/V4 ablation**

Run:

```powershell
python scripts/26_run_video_variant_ablation.py `
  --target-label fatigue `
  --mode video_only `
  --sample-mode strict_aligned `
  --variants `
    V1=outputs/embeddings/face_openface_only_v1_embeddings.npz `
    V2=outputs/embeddings/face_openface_behavior_flags_v2_strict_embeddings.npz `
    V4=outputs/embeddings/face_fullframe_deep_v4_embeddings.npz `
  --out-json outputs/reports/video_variant_v1_v2_v4_fatigue_metrics.json `
  --out-table outputs/reports/video_variant_v1_v2_v4_fatigue_table.md
```

### Task 7: Add V3 OpenFace Plus Deep Behavior

**Files:**
- Modify: `src/daily_multimodal/embeddings/video_variants.py`
- Modify: `scripts/25_build_video_variant_embeddings.py`
- Test: `tests/test_video_variant_embeddings.py`

- [ ] **Step 1: Fuse OpenFace and V4**

Build:

```text
feature_vector = concat(openface_stats, fullframe_deep_v4_features)
optional_feature_vector = concat(openface_stats, fullframe_deep_v4_features, 8 behavior ratios)
face_emb = deterministic_projection(feature_vector, 256)
encoder_version = openface_deep_behavior_v3
```

- [ ] **Step 2: Run the full video-only comparison**

Run:

```powershell
python scripts/26_run_video_variant_ablation.py `
  --target-label fatigue `
  --mode video_only `
  --sample-mode strict_aligned `
  --variants `
    V0=mean_baseline `
    V1=outputs/embeddings/face_openface_only_v1_embeddings.npz `
    V2=outputs/embeddings/face_openface_behavior_flags_v2_strict_embeddings.npz `
    V4=outputs/embeddings/face_fullframe_deep_v4_embeddings.npz `
    V3=outputs/embeddings/face_openface_deep_behavior_v3_embeddings.npz `
  --out-json outputs/reports/video_variant_v1_v2_v4_v3_fatigue_metrics.json `
  --out-table outputs/reports/video_variant_v1_v2_v4_v3_fatigue_table.md
```

### Task 8: Run Fixed-Multimodal Verification

**Files:**
- Modify: `src/daily_multimodal/training/video_variant_ablation.py`
- Modify: `scripts/26_run_video_variant_ablation.py`
- Test: `tests/test_video_variant_ablation.py`

- [ ] **Step 1: Add multimodal mode**

Use fixed EEG, Audio, and Wear embeddings. Replace only `face_emb` for:

```text
M0: no-video
M1: V1 OpenFace only
M2: V2 OpenFace + behavior flags
M3: V4 full-frame deep only
M4: V3 OpenFace + deep behavior
```

- [ ] **Step 2: Run fixed multimodal comparison**

Run:

```powershell
python scripts/26_run_video_variant_ablation.py `
  --target-label fatigue `
  --mode fixed_multimodal `
  --base-embeddings outputs/embeddings/all_complete_real_v2_embeddings.npz `
  --fixed-modalities eeg,wear,audio `
  --variants `
    M0=no_video `
    M1=outputs/embeddings/face_openface_only_v1_embeddings.npz `
    M2=outputs/embeddings/face_openface_behavior_flags_v2_strict_embeddings.npz `
    M3=outputs/embeddings/face_fullframe_deep_v4_embeddings.npz `
    M4=outputs/embeddings/face_openface_deep_behavior_v3_embeddings.npz `
  --out-json outputs/reports/video_variant_fixed_multimodal_fatigue_metrics.json `
  --out-table outputs/reports/video_variant_fixed_multimodal_fatigue_table.md
```

Decision rule: report whether video helps beyond EEG/Wear/Audio, whether V2 beats V1, whether V4 beats V1, and whether V3 beats both V2 and V4 on the same fold protocol.

### Task 9: Only Then Compare MediaPipe/YOLO Against MMPose Or RTMW

**Files:**
- Create only if the audit shows many detector mistakes: `scripts/27_compare_pose_detectors_for_video_behavior.py`
- Test only if created: `tests/test_video_behavior_detector_comparison.py`

- [ ] **Step 1: Sample windows for detector comparison**

Use 200 to 500 windows, stratified by high `head_down_ratio`, high `hand_occlusion_ratio`, high `side_turn_ratio`, high `offscreen_ratio`, and random controls.

- [ ] **Step 2: Compare flags, not raw detector confidence only**

The report must compare final behavior decisions:

```text
MediaPipe/YOLO head_down vs MMPose/RTMW head_down
MediaPipe/YOLO hand_occlusion vs MMPose/RTMW hand_occlusion
MediaPipe/YOLO side_turn vs MMPose/RTMW side_turn
```

Upgrade to MMPose/RTMW only if the current pipeline has repeated, manually verified errors on the behavior states that affect V2/V3 results.

### Task 10: Sync Docs After Behavior Changes

**Files:**
- Modify: `repo-docs/modules/embedding-contract.md`
- Modify: `repo-docs/references/commands-and-artifacts.md`
- Modify: `repo-docs/change-log.md`

- [ ] **Step 1: Run the Understanding Sync check**

Ask what a new reader would misunderstand after the video behavior branch lands. At minimum, confirm whether the guide still says video is only OpenFace face stats.

- [ ] **Step 2: Patch the smallest owning pages**

Update `embedding-contract.md` only to say that V1/V2/V3/V4 are still `face_emb` variants. Update `commands-and-artifacts.md` with the new scripts, outputs, and verification commands.

- [ ] **Step 3: Run validation**

Run:

```powershell
python C:\Users\28303\.codex\skills\repo-docs\scripts\validate_repo_docs.py repo-docs --repo-root .
```

Expected: validator passes, or the failure is a concrete link/source issue fixed before finishing.

## Completion Checks

- [ ] Behavior flags were extracted from an unfiltered window index.
- [ ] The behavior audit was manually reviewed before V2 was treated as evidence.
- [ ] Every variant writes `face_emb (N, 256)`, not `video_emb_*`.
- [ ] `encoder_version` distinguishes V1/V2/V4/V3.
- [ ] OpenFace pose thresholds compare radians to `math.radians(...)`.
- [ ] V1/V2 were compared in both `strict_aligned` and `behavior_retained` modes.
- [ ] Video-only ablation finished before fixed-multimodal ablation.
- [ ] MMPose/RTMW was considered only after the lightweight detector audit.
- [ ] Repo docs were synced after code, config, scripts, tests, or behavior changed.

