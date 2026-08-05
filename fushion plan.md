# Learnable Lightweight Cross-Attention Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the final multimodal fusion step that consumes fixed window-level EEG, wear, video, and audio embeddings, then evaluates whether learnable modality-token attention improves fatigue prediction under subject-level validation.

**Architecture:** The fusion module does not change upstream encoders. It reads aligned `(N, 256)` modality embeddings, treats each available modality as one token, applies a small mask-aware PyTorch attention block, and predicts `fatigue`. Video still occupies the current `face_emb` contract slot, but all new fusion metadata should call it `video`.

**Tech Stack:** Python 3.10+, NumPy, optional PyTorch lazy import, existing `.npz` embedding contract, existing subject CV / fair ablation patterns, pytest.

---

## Fixed Modality Inputs

The first fusion release must use exactly these modality sources.

| Modality | Contract key | Source profiles / artifacts | Notes |
| --- | --- | --- | --- |
| EEG | `eeg_emb` | `outputs/embeddings/eeg_real_eegpt_full_v2_mainface_120s10s_embeddings.npz`, profile `eeg_deep_frozen_v1`, cache profile `eeg_real_frozen_v1` | Regenerated on the face-filtered 120s-before-event / 10s-window index. Verified `8124 x 256`, NaN `0`, failures `204 eeg_window_after_recording`. |
| Wear physio | `wear_emb` | `wear_physio_features_preprocessed_v1` | Preprocessed PPG/GSR/ACC physiology features using `wear_signal_preprocessing_v1`. |
| Wear deep | `wear_emb` | `wear_deep_sequence_preprocessed_v1` | Preprocessed sequence-based wear embedding using the fixed deep sequence projection path. |
| Video `full_sweep/B0` | `face_emb` | 2xROI B0 base embedding | Fixed video input for train/validation/test. |
| Video `full_sweep/B3_lam0.05` | `face_emb` | B3 session-GRL adapter on B0 inputs | Fit inside each fusion training fold; the fitted fold route encodes validation/test. |
| Video `a1_a2_train_only/A2` | `face_emb` | A2 2xROI augmentation embedding | Replaces only fusion training rows; validation/test retain B0. |
| Video `b5_a1/B5_A1_lam0.001` | `face_emb` | B5 A1-input adapter/GRL | Fit only with A1 training rows; validation/test use the B0 base route. |
| Audio | `audio_emb` | `outputs/embeddings/audio_opensmile_egemaps_full_v2_mainface_120s10s_embeddings.npz`, profile `audio_opensmile_egemaps_v1`, cache profile `wavlm_frozen_v1` | Regenerated on the same 120s/10s index. Verified `8256 x 256`, NaN `0`, failures `72 source_missing`. Quality remains uncertain, so every full run must have a matched `no_audio` run. |

Label/metadata source: the B0 base bundle supplies `labels`, `event_id`, and `subject_id` for experiments whose enabled branch files are embedding-only, such as `no_video` and `bio_only`. It is not treated as a video token unless a video route is explicitly enabled.

## Experiment Matrix

The main fusion grid is paired by `sample_id` and fold assignment. For each row, use the same aligned sample set across `full`, `no_audio`, and any applicable comparison so changes reflect token removal rather than row-count changes.

| Experiment family | Wear input | Video input | Enabled modalities |
| --- | --- | --- | --- |
| `fusion_<wear>_<route>_full` | either preprocessed wear profile | one of the four routes | EEG + Wear + Video + Audio |
| `fusion_<wear>_<route>_no_audio` | either preprocessed wear profile | matched route | EEG + Wear + Video |
| `fusion_<wear>_no_video` | either preprocessed wear profile | disabled | EEG + Wear + Audio |
| `fusion_<wear>_bio_only` | either preprocessed wear profile | disabled | EEG + Wear |

This is a 20-experiment matrix: `2 wear x 4 video x (full,no_audio)` plus
two no-video and two bio-only controls. Route names are `FullSweepB0`,
`FullSweepB3Lam005`, `A1A2TrainOnlyA2`, and `B5A1Lam0001`.

## Video Route Validation Lock

The four selected routes were validated on the valid 2xROI 8328-window bundle
with target `fatigue`. The route-specific reports are under
`outputs/reports/video_2xroi_long_runs/{full_sweep,a1_a2_train_only,b5_a1}/`.

| Route | LOSO RMSE / r | S1 RMSE / r | S2 RMSE / r | S4 RMSE / r |
| --- | --- | --- | --- | --- |
| `full_sweep/B0` | `0.9409 / -0.0094` | `0.8390 / 0.3518` | `0.9514 / 0.1839` | `0.9198 / 0.1933` |
| `full_sweep/B3_lam0.05` | `0.9864 / -0.0539` | `0.8327 / 0.3938` | `0.9132 / 0.2969` | `0.9217 / 0.2137` |
| `a1_a2_train_only/A2` | `1.0117 / -0.0653` | `0.8837 / 0.3294` | `0.9609 / 0.2549` | `0.9458 / 0.1956` |
| `b5_a1/B5_A1_lam0.001` | `0.9757 / -0.0345` | `0.8508 / 0.3712` | `0.9808 / 0.1575` | `0.9117 / 0.2358` |

Lock interpretation: B0 remains the LOSO reference. B3 is strongest for the
same-subject event (S1) and chronological (S2) views. B5 is strongest for
session-held-out S4 among the selected routes. A2 improves S2 correlation over
B0 but has weaker RMSE. No selected route has positive LOSO correlation, so
the four-route fusion experiment is within-subject-only and must not be used
as cross-subject promotion evidence.

Interpretation rules:

- `no_audio` answers whether questionable audio hurts or helps once video is present.
- `no_video` answers whether video adds value beyond EEG + wear + audio.
- `bio_only` is a sanity baseline for EEG + wear without either uncertain audio or video.
- Promote a default fusion only if it beats the matching `no_audio`/`no_video` controls under the same sample set and folds.

## Model And Data Contract

- Add `learnable_cross_attention` as a model option beside existing `concat_mlp` and non-learned `modality_token_attention`.
- Token order inside the fusion model must be canonical: `eeg`, `wear`, `video`, `audio`.
- On disk, map token `video` to `face_emb` and mask column 2. Preserve the existing mask order `[eeg, wear, face, audio]`.
- Default model: per-token `256 -> 128` projection, learned modality embedding, one modality self-attention block, learned query attention pooling, and MLP regression head.
- Default training: AdamW, `lr=1e-3`, `weight_decay=1e-4`, `batch_size=64`, `epochs=200`, dropout `0.1`, early stopping patience `25`.
- Default filtering: require at least two enabled modalities after masking. For paired runs, compute eligible rows once for the comparison family and reuse them across variants.
- PyTorch must be lazy-imported. In environments without torch, raise `learnable_cross_attention requires torch` without breaking existing NumPy tests.

## Implementation Tasks

### Task 1: Fusion Dataset Builder

**Files:**
- Create: `src/daily_multimodal/training/cross_attention_fusion.py`
- Test: `tests/test_cross_attention_fusion.py`

- [x] Add tests that build small synthetic `.npz` bundles for EEG, two wear variants, two video variants, and audio.
- [x] Implement alignment by `sample_id`; fail with a structured error when any required branch has duplicate or missing `sample_id`.
- [x] Implement modality selection so `full`, `no_audio`, `no_video`, and `bio_only` produce token tensors, token masks, labels, subjects, and sample ids from the same base rows.
- [x] Test that `video` uses `face_emb` and mask column 2, while output metadata still names the token `video`.

### Task 2: Learnable Attention Regressor

**Files:**
- Modify: `src/daily_multimodal/training/cross_attention_fusion.py`
- Test: `tests/test_cross_attention_fusion.py`

- [x] Add a non-torch test that verifies calling the learnable model without PyTorch raises `learnable_cross_attention requires torch`.
- [x] Add torch-enabled tests, skipped when torch is missing, that verify forward shape, masked token exclusion, and loss decrease on a tiny synthetic dataset.
- [x] Implement the mask-aware attention regressor with saved config, modality order, normalization statistics, and target statistics.
- [x] Save model checkpoints as `.pt`; metrics remain JSON and tables remain Markdown.

### Task 3: Subject CV Integration

**Files:**
- Modify: `src/daily_multimodal/training/subject_cv.py`
- Modify: `scripts/20_run_subject_cv.py`
- Test: `tests/test_subject_cv.py`

- [x] Add `--model concat_mlp|learnable_cross_attention`.
- [x] Add `--min-available-modalities`, `--device`, and an optional `--fusion-spec` JSON path describing the branch files for the experiment matrix.
- [x] Preserve existing `concat_mlp` behavior by default.
- [x] For `learnable_cross_attention`, run the same leave-one-subject-out or grouped-k-fold splits already used by `subject_cv`.
- [x] Write per-fold `sample_counts`, RMSE, MAE, Pearson r, and attention summaries.

### Task 4: Paired Fusion Matrix Runner

**Files:**
- Create: `scripts/43_run_fusion_matrix.py`
- Create: `configs/fusion_matrix.yaml`
- Test: `tests/test_cross_attention_fusion.py`

- [ ] Replace the static video branch registry with the four fold-safe routes in `configs/within_subject_video_routes.yaml`.
- [ ] Implement the 20-experiment within-subject route runner using one paired sample set across the whole matrix.
- [x] Write outputs under `outputs/reports/fusion_matrix/` and models under `outputs/models/fusion_matrix/`.
- [x] Write a summary Markdown table with experiment name, enabled modalities, row count, fold count, RMSE mean/std, Pearson r mean/std, and decision.

### Task 5: Fair And Leakage Controls

**Files:**
- Modify: `src/daily_multimodal/training/fair_embedding_ablation.py`
- Modify: `scripts/18_run_fair_embedding_ablation.py`
- Test: `tests/test_fair_embedding_ablation.py`

- [x] Add `--model learnable_cross_attention` support while keeping `concat_mlp` as the default.
- [x] Run fair controls for the strongest fusion candidate and its `no_audio` / `no_video` controls.
- [x] Require `path_only` to stay clearly below the real signal runs before promoting any fusion result.
- [x] Keep the decision rule: real branches must beat `basic_no_path` on identical aligned rows.

### Task 6: Documentation Sync

**Files:**
- Modify: `repo-docs/modules/embedding-contract.md`
- Modify: `repo-docs/references/commands-and-artifacts.md`
- Modify: `repo-docs/change-log.md`

- [x] Document that fusion names the third token `video` while reading/writing the current `face_emb` slot.
- [x] Add commands for `scripts/43_run_fusion_matrix.py`, full/no-audio/no-video runs, and fair controls.
- [x] Record the chosen wear/video/audio branches and the final recommended fusion configuration.
- [x] Run the repo-docs validator:

```powershell
python C:\Users\28303\.codex\skills\repo-docs\scripts\validate_repo_docs.py repo-docs --repo-root .
```

## Validation Commands

Focused local tests:

```powershell
python -m pytest tests/test_cross_attention_fusion.py tests/test_subject_cv.py tests/test_fair_embedding_ablation.py -q
```

Full regression before reporting completion:

```powershell
python -m pytest tests -q
python -m compileall -q src scripts tests
```

Server-side fusion run, after branch artifacts are available:

```powershell
python scripts/43_run_fusion_matrix.py `
  --config configs/fusion_matrix.yaml `
  --target-label fatigue `
  --model learnable_cross_attention `
  --out-dir outputs/reports/fusion_matrix `
  --model-dir outputs/models/fusion_matrix
```

## Prior Formal Evidence (Superseded For The Four-Route Plan)

- Local environment: PyTorch is not installed and the full branch artifacts are not present, so local validation is limited to unit tests, compile checks, and matrix dry-runs.
- Server environment `ncc_serve_4090`: PyTorch CPU is available; the fixed EEG, wear physio/deep, legacy V4a/B1, and audio artifacts are available. The four-route provider must be materialized fold-by-fold before its first matrix run.
- Server smoke: `fusion_WphysioPre_no_video` ran successfully with `epochs=2`, proving real branch alignment, label metadata fallback, subject folds, checkpoint writing, metrics JSON, and summary JSON work end-to-end for a no-video control.
- Server formal 120s/10s matrix: `outputs/reports/fusion_matrix_120s10s/` ran the earlier 12-experiment V4aUpper/B1 matrix with `epochs=200`, `hidden_dim=128`, CPU. Its result does not rank the new four-route within-subject matrix.

| Experiment | Rows | RMSE mean | RMSE std | Pearson r mean | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `fusion_WphysioPre_B1_full` | 7368 | 0.9538 | 0.2540 | 0.0258 | `accepted_candidate` |
| `fusion_WphysioPre_V4aUpper_full` | 7368 | 0.9578 | 0.2483 | -0.0491 | `rollback` |
| `fusion_WphysioPre_V4aUpper_no_audio` | 7368 | 0.9567 | 0.2208 | -0.0586 | `audio_control` |
| `fusion_WphysioPre_B1_no_audio` | 7368 | 0.9707 | 0.2734 | -0.0482 | `audio_control` |
| `fusion_WphysioPre_no_video` | 7368 | 1.0091 | 0.2522 | -0.0075 | `video_control` |
| `fusion_WphysioPre_bio_only` | 7440 | 0.9680 | 0.2646 | -0.0193 | `bio_only_control` |
| `fusion_WdeepPre_V4aUpper_full` | 7368 | 0.9695 | 0.2331 | -0.0048 | `accepted_candidate` |
| `fusion_WdeepPre_B1_full` | 7368 | 0.9854 | 0.1885 | -0.0639 | `rollback` |
| `fusion_WdeepPre_no_video` | 7368 | 0.9743 | 0.2140 | -0.0454 | `video_control` |
| `fusion_WdeepPre_B1_no_audio` | 7368 | 0.9745 | 0.2307 | -0.0419 | `audio_control` |
| `fusion_WdeepPre_bio_only` | 7440 | 0.9780 | 0.2374 | -0.0772 | `bio_only_control` |
| `fusion_WdeepPre_V4aUpper_no_audio` | 7368 | 1.0385 | 0.2632 | -0.0623 | `audio_control` |

Current interpretation:

- Best RMSE and the only positive mean Pearson r in the new matrix is `fusion_WphysioPre_B1_full`, using EEG + `wear_physio_features_preprocessed_v1` + B1 video + audio.
- `fusion_WdeepPre_V4aUpper_full` also passes the implemented paired-control rule, but its mean Pearson r is still slightly negative.
- Audio is not uniformly harmful in this matrix: for B1 + physio, adding audio improves both RMSE and r relative to `no_audio`; for V4a + physio, audio slightly improves r but not RMSE; for V4a + deep, audio improves both relative to `no_audio`.
- Video evidence is mixed: the best candidate uses B1 video, but both `no_video` controls are close enough that a dedicated fair audit is required before promotion.
- Do not treat the matrix `accepted_candidate` flag as final promotion. Next required step is fusion-spec fair controls for `fusion_WphysioPre_B1_full`, its matched `fusion_WphysioPre_B1_no_audio`, and `fusion_WphysioPre_no_video`, with `path_only` required to stay below the real branch result.

## Superseded Server Evidence

- The previous formal server matrix under `outputs/reports/fusion_matrix/` used EEG/audio from `outputs/embeddings/all_complete_real_v2_embeddings.npz`, which was the old 781-row complete-event pack rather than four independent 120s/10s branch embeddings. Treat those numbers as superseded for final fusion selection.
- The previous dedicated fusion fair audits under `outputs/reports/fusion_fair/` also audited those old row sets and should be used only as a reference for the audit procedure, not as final evidence for the regenerated 120s/10s branches.

## Assumptions

- EEG remains frozen for this work.
- Wear comparison is limited to the two preprocessed profiles named above.
- Video comparison uses four within-subject routes: `full_sweep/B0`,
  `full_sweep/B3_lam0.05`, `a1_a2_train_only/A2`, and
  `b5_a1/B5_A1_lam0.001`. B0 is fixed; B3/B5 are fitted within each fusion
  training fold; A2 replaces train-only inputs and validation/test use B0.
- Audio remains included only as the current packed `audio_emb`; no audio profile sweep happens in this plan.
- The first release uses window-level 256D embeddings only; no internal time-series tokens are introduced.
- Main promotion is based on `fatigue` with subject-level CV and paired full/no-audio/no-video controls.
