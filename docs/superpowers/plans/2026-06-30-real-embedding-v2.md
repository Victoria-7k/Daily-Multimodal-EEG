# Real Embedding V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fair second-round real-embedding workflow that removes metadata shortcut bias, adds true OpenFace, fixes EEG coverage failures, improves Audio/Wear emotion features, and reruns stage 17/18 with comparable evidence.

**Architecture:** Keep the existing 256-dimensional per-modality embedding contract and stage 17 packer. Add narrowly scoped v2 profiles and audit scripts around the current pipeline instead of replacing it wholesale. Each modality upgrade must pass 10-window, single-subject, full, pack, and ablation gates before it can be treated as an accepted branch.

**Tech Stack:** Python, NumPy, pandas, scipy when available, MNE/Braindecode EEGPT, ffmpeg, Apptainer OpenFace `FeatureExtraction`, Hugging Face/ModelScope speech emotion checkpoints, openSMILE/eGeMAPS, existing stage 17/18 training utilities.

---

## Evidence Snapshot

Current accepted evidence:

- Full real pack exists at `outputs/embeddings/all_complete_real_embeddings.npz`.
- Full real row count is 781.
- Full real mask sum is `[738, 781, 657, 781]` for `[eeg, wear, face, audio]`.
- Stage 18 full ablation produced 13 experiments and 0 failures.
- Current strongest reference is `stage10_modality_token_attention`, RMSE `0.6968`.
- Current all-real branches are rollback.
- Current Face is OpenCV dirty fallback, not true OpenFace.
- Current EEG failures are 43 `shape_mismatch` windows caused by EEG time coverage mismatch.
- Current Wear sequence encoder is deterministic/lightweight.
- Current Audio model is wav2vec2 ASR-oriented, not emotion-specialized.

Main risk:

- The old basic embedding uses `sample_id`, file paths, file size, and session structure. It may encode subject/session metadata shortcuts, so all v2 comparisons must use aligned rows and explicit leakage controls.

---

## File Map

Planned new files:

- `scripts/18_run_fair_embedding_ablation.py`: run 781-row aligned fair baseline, path-leakage controls, and real-vs-basic comparisons.
- `src/daily_multimodal/training/fair_embedding_ablation.py`: construct aligned baseline variants and reuse the existing MLP evaluation helpers.
- `tests/test_fair_embedding_ablation.py`: verify sample alignment, metadata masking, and comparison output.
- `scripts/19_audit_eeg_coverage.py`: report EEG window offsets against BDF durations and propose explicit correction candidates.
- `src/daily_multimodal/alignment/eeg_coverage.py`: pure audit helpers for EEG offset, duration, and correction classification.
- `tests/test_eeg_coverage.py`: verify out-of-range, negative-offset, in-range, and whole-day-shift classification.
- `configs/eeg_time_corrections.yaml`: optional explicit per-subject/session corrections; starts empty except comments.
- `scripts/20_run_subject_cv.py`: subject-level cross-validation for final candidate comparisons.
- `src/daily_multimodal/training/subject_cv.py`: leave-one-subject-out or grouped subject split evaluation.
- `tests/test_subject_cv.py`: verify folds do not leak subjects.

Planned modified files:

- `src/daily_multimodal/embeddings/face_real.py`: add true OpenFace window-level clip execution before `FeatureExtraction`.
- `tests/test_face_real_embedding.py`: cover OpenFace clip generation and no-fallback behavior.
- `scripts/13_extract_face_embeddings.py`: expose any new clip-cache options if needed.
- `src/daily_multimodal/embeddings/eeg_real.py`: improve failure classification after EEG coverage audit.
- `src/daily_multimodal/embeddings/audio_real.py`: add emotion-specific and openSMILE backends behind explicit profiles.
- `scripts/12_extract_audio_embeddings.py`: accept new audio profiles and dependencies.
- `tests/test_audio_real_embedding.py`: cover openSMILE/emotion backend selection and dependency failures.
- `src/daily_multimodal/embeddings/wear_real.py`: add `wear_physio_features_v2` feature extraction.
- `tests/test_wear_real_embedding.py`: cover HR/HRV, EDA, ACC, quality flags, and deterministic output.
- `configs/encoders.yaml`: register v2 profiles without removing existing accepted profiles.
- `真实多模态完整embedding接入计划.md`: update phase status after each accepted v2 milestone.
- `真实多模态完整embedding执行报告.md`: append final v2 evidence only after full validation.

---

### Task 1: Fair Baseline and Metadata Leakage Audit

**Files:**
- Create: `src/daily_multimodal/training/fair_embedding_ablation.py`
- Create: `scripts/18_run_fair_embedding_ablation.py`
- Create: `tests/test_fair_embedding_ablation.py`
- Modify: `configs/encoders.yaml`

- [ ] **Step 1: Write failing tests for aligned comparison**

Add tests that build tiny basic/real NPZ files with identical `sample_id` order and assert:

```python
result = run_fair_embedding_ablation(
    basic_embeddings=basic_npz,
    real_embeddings=real_npz,
    target_label="alert",
    out_json=metrics_json,
    out_table=table_md,
)
assert result["row_count"] == 6
assert result["sample_id_aligned"] is True
assert {"basic_aligned", "basic_no_path", "path_only", "real"} <= set(result["experiments"])
```

Run:

```powershell
python -m pytest tests/test_fair_embedding_ablation.py -q
```

Expected: fails because the module and script do not exist.

- [ ] **Step 2: Implement aligned dataset variants**

Implement four variants:

```text
basic_aligned: existing aligned basic embeddings on the same 781 rows
basic_no_path: same labels/masks, but replace EEG/Face/Audio metadata embeddings with path-neutral deterministic constants and keep Wear statistics
path_only: path/sample/session derived metadata features only, no real signal arrays
real: current all-real embedding pack
```

Decision rule:

```text
No v2 real encoder is called accepted unless it beats basic_no_path and is competitive with basic_aligned on the same 781 rows.
```

- [ ] **Step 3: Add CLI**

Run on server:

```bash
cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding
PYTHONPATH=src python scripts/18_run_fair_embedding_ablation.py \
  --basic-embeddings outputs/embeddings/all_complete_basic_real_aligned_embeddings.npz \
  --real-embeddings outputs/embeddings/all_complete_real_embeddings.npz \
  --target-label alert \
  --out-json outputs/reports/fair_embedding_ablation_metrics.json \
  --out-table outputs/reports/fair_embedding_ablation_table.md
```

Expected:

```text
row_count=781
sample_id_aligned=True
experiments include basic_aligned/basic_no_path/path_only/real
failure_count=0
```

- [ ] **Step 4: Commit**

```bash
git add src/daily_multimodal/training/fair_embedding_ablation.py scripts/18_run_fair_embedding_ablation.py tests/test_fair_embedding_ablation.py configs/encoders.yaml
git commit -m "Add fair embedding leakage audit"
```

---

### Task 2: True OpenFace Window-Level Extraction

**Files:**
- Modify: `src/daily_multimodal/embeddings/face_real.py`
- Modify: `scripts/13_extract_face_embeddings.py`
- Modify: `tests/test_face_real_embedding.py`
- Reuse: `OpenFace真实接入方案计划.md`

- [ ] **Step 1: Install or mount OpenFace through Apptainer**

On server:

```bash
mkdir -p /mnt/dataset4/sitian/wzw/tools/openface
cd /mnt/dataset4/sitian/wzw/tools/openface
apptainer pull openface.sif docker://algebr/openface:latest
apptainer exec openface.sif find / -name FeatureExtraction 2>/dev/null | head
```

Create wrapper:

```bash
cat >/mnt/dataset4/sitian/wzw/tools/openface/feature_extraction.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
OPENFACE_SIF=/mnt/dataset4/sitian/wzw/tools/openface/openface.sif
OPENFACE_BIN=/OpenFace/build/bin/FeatureExtraction
exec apptainer exec --cleanenv --bind /mnt/dataset1,/mnt/dataset4 "$OPENFACE_SIF" "$OPENFACE_BIN" "$@"
SH
chmod +x /mnt/dataset4/sitian/wzw/tools/openface/feature_extraction.sh
/mnt/dataset4/sitian/wzw/tools/openface/feature_extraction.sh -help
```

Expected: help text from `FeatureExtraction`.

- [ ] **Step 2: Write failing test for window clip generation**

Test requirement:

```text
When true OpenFace executable is present, the pipeline creates a 10-second window MP4 under the sample cache and calls FeatureExtraction on that clip, not the source full MP4.
```

Expected fake call:

```text
FeatureExtraction -f <cache>/window.mp4 -out_dir <cache> -of openface
```

- [ ] **Step 3: Implement clip-first OpenFace**

Add helper responsibilities:

```text
_ensure_window_clip(source_path, cache, clip_path)
  read clip_start_seconds and clip_end_seconds from cache/window metadata
  call ffmpeg to write cache-local window.mp4
  reuse existing clip if non-empty

_run_openface(executable, clip_path, csv_path)
  pass -f clip_path
  pass -out_dir csv_path.parent
  pass -of csv_path.stem
```

Never silently fall back to OpenCV when `--openface-executable` was provided and execution fails. Write `extraction_failed`.

- [ ] **Step 4: Server validation**

Run:

```bash
PYTHONPATH=src python scripts/13_extract_face_embeddings.py \
  --window-index outputs/window_index/real_cache_complete_10.jsonl \
  --cache-root outputs/cache/real_stage12_openface_real_10 \
  --encoder-profile face_raw_openface_stats_v1 \
  --openface-executable /mnt/dataset4/sitian/wzw/tools/openface/feature_extraction.sh \
  --out outputs/embeddings/face_openface_real_10_embeddings.npz \
  --failures-out outputs/reports/face_openface_real_10_failures.json \
  --summary-out outputs/reports/face_openface_real_10_quality_summary.json \
  --decision-out outputs/reports/face_preprocessing_decision_openface_real_10.json
```

Expected:

```text
face_emb.shape=(10, 256)
OpenFace CSV has frame/confidence/success/pose/AU columns
failure_count is explainable
nan_count=0
```

- [ ] **Step 5: Expand to sub-12 and full**

Run the same command on `sub-12`, then full 781. Do not proceed to stage 17 until failures and masks are understood.

- [ ] **Step 6: Commit**

```bash
git add src/daily_multimodal/embeddings/face_real.py scripts/13_extract_face_embeddings.py tests/test_face_real_embedding.py
git commit -m "Use window clips for true OpenFace extraction"
```

---

### Task 3: EEG Coverage Audit and Correction Map

**Files:**
- Create: `src/daily_multimodal/alignment/eeg_coverage.py`
- Create: `scripts/19_audit_eeg_coverage.py`
- Create: `tests/test_eeg_coverage.py`
- Create: `configs/eeg_time_corrections.yaml`
- Modify: `src/daily_multimodal/embeddings/eeg_real.py`

- [ ] **Step 1: Write coverage classification tests**

Required classifications:

```text
in_range: 0 <= start_offset and end_offset <= bdf_duration
negative_offset: end_offset <= 0
after_recording_end: start_offset >= bdf_duration
partial_overlap: overlaps but not full 10-second window
whole_day_shift_candidate: adding or subtracting 86400 seconds makes it in_range
```

Run:

```powershell
python -m pytest tests/test_eeg_coverage.py -q
```

Expected: fails before implementation.

- [ ] **Step 2: Implement audit script**

Server command:

```bash
PYTHONPATH=src python scripts/19_audit_eeg_coverage.py \
  --window-index outputs/window_index/real_cache_complete_full.jsonl \
  --out-json outputs/reports/eeg_coverage_audit_full.json \
  --out-table outputs/reports/eeg_coverage_audit_full.md
```

Expected output includes:

```text
total_windows=781
in_range_count
negative_offset_count
after_recording_end_count
whole_day_shift_candidate_count
affected_subject_sessions
```

- [ ] **Step 3: Create explicit correction map**

Start with an empty safe file:

```yaml
# Explicit EEG time corrections.
# Keys are "<subject_id>/<session_id>".
# Do not add a correction unless eeg_coverage_audit shows that it brings the full window inside the BDF duration.
corrections: {}
```

Only after audit, add corrections such as:

```yaml
corrections:
  sub-10/ses-04:
    shift_seconds: -86400
    reason: "behavior timestamps appear one day after BDF recording start"
```

- [ ] **Step 4: Improve EEG failure type**

Change EEG failures from generic `shape_mismatch` to specific categories:

```text
eeg_window_before_recording
eeg_window_after_recording
eeg_window_partial_overlap
eeg_window_shape_mismatch
```

Keep these recoverable.

- [ ] **Step 5: Rerun EEGPT after approved corrections**

Run:

```bash
PYTHONPATH=src python scripts/14_extract_eeg_embeddings.py \
  --window-index outputs/window_index/real_cache_complete_full.jsonl \
  --cache-root outputs/cache/real_stage12_full \
  --encoder-profile eeg_deep_frozen_v1 \
  --checkpoint outputs/checkpoints/eegpt-pretrained \
  --device cpu \
  --out outputs/embeddings/eeg_real_eegpt_full_v2_embeddings.npz \
  --failures-out outputs/reports/eeg_real_eegpt_full_v2_failures.json \
  --summary-out outputs/reports/eeg_real_eegpt_full_v2_quality_summary.json
```

Expected: failures are reduced or explicitly justified by coverage audit.

- [ ] **Step 6: Commit**

```bash
git add src/daily_multimodal/alignment/eeg_coverage.py scripts/19_audit_eeg_coverage.py tests/test_eeg_coverage.py configs/eeg_time_corrections.yaml src/daily_multimodal/embeddings/eeg_real.py
git commit -m "Audit EEG coverage before real embedding extraction"
```

---

### Task 4: Audio Emotion Embedding Profiles

**Files:**
- Modify: `src/daily_multimodal/embeddings/audio_real.py`
- Modify: `scripts/12_extract_audio_embeddings.py`
- Modify: `tests/test_audio_real_embedding.py`
- Modify: `configs/encoders.yaml`

- [ ] **Step 1: Add profile definitions**

Register:

```yaml
audio_real_profiles:
  audio_opensmile_egemaps_v1:
    backend: opensmile_egemaps
    feature_set: eGeMAPSv02
    feature_level: Functionals
    output_dim: 256
  audio_emotion2vec_plus_v1:
    backend: emotion2vec
    checkpoint_required: true
    output_dim: 256
    pooling: mean_std_max
```

- [ ] **Step 2: Write backend tests**

Required behavior:

```text
audio_opensmile_egemaps_v1 returns (N, 256) with deterministic projection.
audio_emotion2vec_plus_v1 writes checkpoint_missing if checkpoint is absent.
audio_emotion2vec_plus_v1 writes dependency_missing if backend dependency is absent.
mean_std_max pooling changes feature size before projection and remains deterministic.
```

- [ ] **Step 3: Implement openSMILE eGeMAPS backend first**

Use Python `opensmile` when available:

```python
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
features = smile.process_file(str(wav_path)).to_numpy().reshape(-1)
```

If unavailable, write `dependency_missing` and do not fallback silently.

- [ ] **Step 4: Implement emotion2vec backend**

Use a local checkpoint directory under:

```text
outputs/checkpoints/emotion2vec_plus_large
```

Allowed dependency behavior:

```text
checkpoint missing -> checkpoint_missing
backend import missing -> dependency_missing
runtime failure -> extraction_failed
```

- [ ] **Step 5: Server validation**

Run 10-window and sub-12 for both profiles:

```bash
PYTHONPATH=src python scripts/12_extract_audio_embeddings.py \
  --window-index outputs/window_index/real_cache_complete_10.jsonl \
  --cache-root outputs/cache/real_stage12_full \
  --encoder-profile audio_opensmile_egemaps_v1 \
  --out outputs/embeddings/audio_opensmile_egemaps_10_embeddings.npz \
  --failures-out outputs/reports/audio_opensmile_egemaps_10_failures.json \
  --summary-out outputs/reports/audio_opensmile_egemaps_10_quality_summary.json
```

Expected:

```text
audio_emb.shape=(10, 256)
nan_count=0
failure_count=0 or dependency_missing with clear message
```

- [ ] **Step 6: Commit**

```bash
git add src/daily_multimodal/embeddings/audio_real.py scripts/12_extract_audio_embeddings.py tests/test_audio_real_embedding.py configs/encoders.yaml
git commit -m "Add emotion-oriented audio embedding profiles"
```

---

### Task 5: Wear Physiological Features V2

**Files:**
- Modify: `src/daily_multimodal/embeddings/wear_real.py`
- Modify: `tests/test_wear_real_embedding.py`
- Modify: `configs/encoders.yaml`

- [ ] **Step 1: Add profile**

Register:

```yaml
wear_real_profiles:
  wear_physio_features_v2:
    output_dim: 256
    ppg_features: [heart_rate, ibi_mean, ibi_std, rmssd, peak_count, missing_ratio]
    gsr_features: [tonic_mean, phasic_std, scr_count, slope, missing_ratio]
    acc_features: [motion_intensity, stationary_ratio, axis_std, spectral_energy]
```

- [ ] **Step 2: Write feature tests**

Use synthetic windows:

```text
PPG with regular peaks -> stable heart_rate and low RMSSD
flat PPG -> peak_count=0 and quality flag ppg_peak_insufficient=True
GSR ramp -> positive slope
ACC static -> high stationary_ratio
ACC movement -> higher motion_intensity
```

- [ ] **Step 3: Implement features**

Keep deterministic projection to 256 dimensions. Store raw feature names and values in `quality_flags` so future analysis can identify which physiology signals matter.

- [ ] **Step 4: Server validation**

Run:

```bash
PYTHONPATH=src python scripts/15_extract_wear_embeddings.py \
  --window-index outputs/window_index/real_cache_complete_10.jsonl \
  --cache-root outputs/cache/real_stage12_full \
  --encoder-profile wear_physio_features_v2 \
  --out outputs/embeddings/wear_physio_features_v2_10_embeddings.npz \
  --failures-out outputs/reports/wear_physio_features_v2_10_failures.json \
  --summary-out outputs/reports/wear_physio_features_v2_10_quality_summary.json
```

Expected:

```text
wear_emb.shape=(10, 256)
nan_count=0
feature quality fields present
```

- [ ] **Step 5: Commit**

```bash
git add src/daily_multimodal/embeddings/wear_real.py tests/test_wear_real_embedding.py configs/encoders.yaml
git commit -m "Add physiological wearable features v2"
```

---

### Task 6: V2 Pack and Fair Ablation

**Files:**
- Modify: `真实多模态完整embedding接入计划.md`
- Modify: `真实多模态完整embedding执行报告.md`

- [x] **Step 1: Produce v2 single-modality full files**

Expected full files:

```text
outputs/embeddings/eeg_real_eegpt_full_v2_embeddings.npz
outputs/embeddings/wear_physio_features_v2_full_embeddings.npz
outputs/embeddings/face_openface_real_full_embeddings.npz
outputs/embeddings/audio_<best_profile>_full_embeddings.npz
```

- [x] **Step 2: Pack v2 all-real**

Run:

```bash
PYTHONPATH=src python scripts/16_extract_all_real_embeddings.py \
  --window-index outputs/window_index/real_cache_complete_full.jsonl \
  --eeg outputs/embeddings/eeg_real_eegpt_full_v2_embeddings.npz \
  --wear outputs/embeddings/wear_physio_features_v2_full_embeddings.npz \
  --face outputs/embeddings/face_openface_real_full_embeddings.npz \
  --audio outputs/embeddings/audio_<best_profile>_full_embeddings.npz \
  --out outputs/embeddings/all_complete_real_v2_embeddings.npz \
  --report-out outputs/reports/all_complete_real_v2_embedding_report.json \
  --failures-out outputs/reports/all_complete_real_v2_embedding_failures.json
```

Expected:

```text
selected_windows=781
all embedding shapes are (781, 256)
nan_count=0 for all modalities
mask distribution is reported
```

- [x] **Step 3: Run fair v2 ablation**

Run:

```bash
PYTHONPATH=src python scripts/18_run_fair_embedding_ablation.py \
  --basic-embeddings outputs/embeddings/all_complete_basic_real_aligned_embeddings.npz \
  --real-embeddings outputs/embeddings/all_complete_real_v2_embeddings.npz \
  --target-label fatigue \
  --modalities eeg,wear,audio \
  --out-json outputs/reports/fair_embedding_ablation_v2_fatigue_ewa_metrics.json \
  --out-table outputs/reports/fair_embedding_ablation_v2_fatigue_ewa_table.md
```

Decision:

```text
Accept a v2 modality only if it beats basic_no_path and does not depend on path_only leakage.
Keep stage10 accepted if all-real v2 still underperforms it.
```

- [x] **Step 4: Update reports**

Update only after server validation:

```text
真实多模态完整embedding接入计划.md
真实多模态完整embedding执行报告.md
outputs/reports/real_embedding_quality_summary_v2.json
```

- [x] **Step 5: Commit**

```bash
git add 真实多模态完整embedding接入计划.md 真实多模态完整embedding执行报告.md outputs/reports/real_embedding_quality_summary_v2.json
git commit -m "Report real embedding v2 validation"
```

---

### Task 7: Subject-Level Cross-Validation

**Files:**
- Create: `src/daily_multimodal/training/subject_cv.py`
- Create: `scripts/20_run_subject_cv.py`
- Create: `tests/test_subject_cv.py`

- [x] **Step 1: Write fold leakage tests**

Assert:

```python
for fold in folds:
    assert set(subjects[fold.train]).isdisjoint(set(subjects[fold.test]))
```

- [x] **Step 2: Implement grouped subject CV**

Support:

```text
leave_one_subject_out
grouped_k_fold with deterministic seed
minimum train/val/test non-empty checks
```

- [x] **Step 3: Run on accepted candidates**

Run:

```bash
PYTHONPATH=src python scripts/20_run_subject_cv.py \
  --embeddings outputs/embeddings/all_complete_real_v2_embeddings.npz \
  --target-label fatigue \
  --modalities eeg,wear,audio \
  --out-json outputs/reports/subject_cv_real_v2_fatigue_ewa_metrics.json \
  --out-table outputs/reports/subject_cv_real_v2_fatigue_ewa_table.md
```

Expected:

```text
fold_count >= 10
no subject leakage
mean/std RMSE and Pearson r reported
```

- [x] **Step 4: Commit**

```bash
git add src/daily_multimodal/training/subject_cv.py scripts/20_run_subject_cv.py tests/test_subject_cv.py
git commit -m "Add subject-level cross validation"
```

---

## Execution Order

Recommended order:

```text
1. Fair baseline and metadata leakage audit
2. True OpenFace window-level extraction
3. EEG coverage audit and correction map
4. Audio emotion profiles
5. Wear physiological features v2
6. V2 pack and fair ablation
7. Subject-level cross-validation
```

Do not start full v2 reruns until Task 1 is complete. The fair baseline decides whether later changes are genuinely better than metadata shortcuts.

---

## External References

- OpenFace command-line arguments: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments
- OpenFace output format: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format
- emotion2vec paper page: https://aclanthology.org/2024.findings-acl.931/
- emotion2vec plus large model page: https://huggingface.co/emotion2vec/emotion2vec_plus_large
- audeering wav2vec2 emotion model guide: https://github.com/audeering/w2v2-how-to
- openSMILE documentation: https://audeering.github.io/opensmile-python/

---

## Self-Review

Spec coverage:

- Fair comparison against path/metadata leakage is Task 1.
- True OpenFace is Task 2.
- EEG 43-window root cause and remediation is Task 3.
- Emotion-oriented Audio is Task 4.
- Stronger Wear features are Task 5.
- Full pack, quality summary, execution report, and ablation are Task 6.
- Small subject count and robustness are addressed by Task 7.

Placeholder scan:

- No task depends on an unspecified output path.
- Every full run has expected files and validation output.
- Fallback behavior is explicit: missing dependency/checkpoint writes structured failure and does not silently generate fake real embeddings.

Type consistency:

- Existing modality keys remain `eeg_emb`, `wear_emb`, `face_emb`, `audio_emb`.
- Existing mask order remains `[eeg, wear, face, audio]`.
- Existing target label remains `alert`.
