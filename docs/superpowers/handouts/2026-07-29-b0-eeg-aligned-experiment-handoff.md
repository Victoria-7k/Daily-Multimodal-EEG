# B0 EEG-Aligned Experiment Handoff

## Purpose

Start or continue EEG-aligned multimodal experiments on the new server. B0, A1, and A2 video embeddings have now all been copied and verified; B0 remains the simplest baseline route.

This handoff is for the next Codex conversation. It should prevent two mistakes:

- Do not use the older `7368`-window `within_subject_fusion_120s10s` experiment as if it were the new EEG-aligned task.
- Do not use non-EEG-aligned 2xROI files; use the `eeg23win` video files listed below.

## Remote Hosts

Old server:

```text
ssh -i C:\Users\28303\.ssh\lzs@nccserv1 -p 1022 lzs@10.20.37.212
```

New server:

```text
ssh -i C:\Users\28303\.ssh\id_ed25519 -p 36083 wangziwei@124.174.8.252
```

Use PowerShell-safe SSH quoting. Prefer a single-quoted PowerShell here-string piped to remote Bash:

```powershell
@'
cd /some/remote/path || exit 1
pwd
'@ | ssh -i C:\Users\28303\.ssh\id_ed25519 -p 36083 wangziwei@124.174.8.252 'bash -s'
```

## New EEG-Aligned Baseline

Authoritative EEG-aligned data on the new server:

```text
/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new/
  X.npy
  y.npy
  sub.npy
  d.npy
  ts.npy

/vePFS-0x0d/DailyEEG/splits_new/

/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/index/
  eeg_aligned_window_index.jsonl
```

Expected index facts:

```text
rows = 28819
events = 1253
windows per event = 23
window length = 10s
stride = 5s
sample_id = eeg_{eeg_sample_index:06d}
```

## Completed Modality Embeddings

Video B0:

```text
/vePFS-0x0d/DailyEEG_multimodal/embeddings/video/video_B0_2xroi_eeg23win_embeddings.npz
```

Known status:

```text
rows = 28819
video_emb shape = (28819, 256)
video_mask_sum = 18017
sha256 = 133b36d6af45c2eed5b2945b223490b3b243eea2a404c9ffedc38fe2601581b3
```

Video A1:

```text
/vePFS-0x0d/DailyEEG_multimodal/embeddings/video/video_A1_2xroi_eeg23win_embeddings.npz
```

Known status:

```text
rows = 28819
video_emb shape = (28819, 256)
video_mask_sum = 18021
sha256 = a82485929f5ae86bc18de394f8ea407e7a8d90a727e05aa898ff93d2379fee14
```

Video A2:

```text
/vePFS-0x0d/DailyEEG_multimodal/embeddings/video/video_A2_2xroi_eeg23win_embeddings.npz
```

Known status:

```text
rows = 28819
video_emb shape = (28819, 256)
video_mask_sum = 17992
sha256 = d61bc37e808722bba35ec72573555dfce50803abaeab6e95bb616878ac2d0cea
```

Audio:

```text
/vePFS-0x0d/DailyEEG_multimodal/embeddings/audio/audio_opensmile_eeg23win_embeddings.npz
```

Known status:

```text
rows = 28819
audio_emb shape = (28819, 256)
audio_mask_sum = 17924
sha256 = 110d01aa08b5691b4c6a1ed5ccf65c5e9dff2a3f2738d015b0175b3342080b73
```

Wear physio:

```text
/vePFS-0x0d/DailyEEG_multimodal/embeddings/wear/wear_physio_preprocessed_eeg23win_embeddings.npz
```

Known status:

```text
rows = 28819
wear_emb shape = (28819, 256)
wear_mask_sum = 24127
sha256 = 311c86ee4f4d8706d975b55d6719be1c7afe7a6e9bbeb073cdc1a211ae97a5f4
```

Wear deep:

```text
/vePFS-0x0d/DailyEEG_multimodal/embeddings/wear/wear_deep_sequence_preprocessed_eeg23win_embeddings.npz
```

Known status:

```text
rows = 28819
wear_emb shape = (28819, 256)
wear_mask_sum = 24127
sha256 = b8783d15859e0dd1f7202e500d2faf610004bea0e68d0a40fb0e8aa5df8279fd
```

## Completed Transfer Reports

Final reports on the new server:

```text
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/reports/final_multimodal_alignment_report.md
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/reports/final_missing_windows_by_modality.json
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/checksums/sha256_manifest_video_audio_wear_final.txt
```

The old-server generated embeddings remain in place under `/mnt/dataset4/sitian/wzw/DailyEEG_multimodal_eeg_aligned_export/`. The transfer was copy-only.

## Key Blocker Before Training

The new server currently does not expose a Python runtime:

```text
python:  command not found
python3: command not found
conda:   command not found
```

The next executor must first prepare or locate a Python/PyTorch environment on the new server. Install or place it under:

```text
/vePFS-0x0d/home/wangzw/
```

Recommended work root:

```text
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/
```

Keep any new scripts, configs, temporary files, and logs under this root.

## B0 Experiment Scope

Start with B0-only experiments over the EEG-aligned `28819` rows if the immediate goal is the first smoke/fusion baseline. A1/A2 are now available for later video-route comparisons, but B0 remains the simplest starting point.

Minimum matrix:

| Experiment | Modalities |
| --- | --- |
| `B0_Wphysio_full` | EEG + wear physio + video B0 + audio |
| `B0_Wphysio_no_audio` | EEG + wear physio + video B0 |
| `B0_Wphysio_no_video` | EEG + wear physio + audio |
| `B0_Wphysio_bio_only` | EEG + wear physio |
| `B0_Wdeep_full` | EEG + wear deep + video B0 + audio |
| `B0_Wdeep_no_audio` | EEG + wear deep + video B0 |
| `B0_Wdeep_no_video` | EEG + wear deep + audio |
| `B0_Wdeep_bio_only` | EEG + wear deep |

Use EEG `splits_new` exactly. Do not recompute splits from video/audio/wear availability. Missing modalities must remain as mask values.

## Important Design Decision: EEG Input

The existing B0/audio/wear files are 256-dimensional embeddings. The EEG baseline folder contains sliced EEG signal arrays:

```text
/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new/X.npy
```

Before running cross-attention fusion, the next executor must choose and document one of these two paths:

1. Generate an EEG embedding file aligned to the same `28819` rows:

```text
eeg_emb shape = (28819, 256)
eeg_mask shape = (28819,)
sample_id / eeg_sample_index aligned to the canonical index
```

2. Implement an EEG branch encoder inside the B0 experiment runner that converts `X.npy` windows into a 256-dimensional token during training.

For the current project code, option 1 is safer because the existing fusion dataset expects modality token arrays shaped like `(N, 256)`.

Suggested EEG embedding target:

```text
/vePFS-0x0d/DailyEEG_multimodal/embeddings/eeg/eeg_eegpt_eeg23win_embeddings.npz
```

If generating EEG embeddings is too slow, start with a smoke experiment that excludes EEG only as a plumbing check, but do not report that as the actual B0 multimodal result.

## Recommended New-Server File Layout

```text
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/
  scripts/
    prepare_b0_fusion_inputs.py
    run_b0_smoke.py
    run_b0_fusion_matrix.py
    validate_b0_alignment.py
  configs/
    b0_fusion_matrix.json
    b0_split_config.json
  index/
    eeg_aligned_window_index.jsonl
    eeg_aligned_event_index.jsonl
    eeg_aligned_split_membership.json
  reports/
    b0_alignment_preflight.md
    b0_smoke_report.md
    b0_fusion_matrix_summary.json
    b0_fusion_matrix_summary.md
  intermediate/
    eeg_embedding_cache/
    packed_b0_inputs/
  checksums/
    b0_input_sha256_manifest.txt
  tmp/
```

Final or reusable embeddings should live under:

```text
/vePFS-0x0d/DailyEEG_multimodal/embeddings/
  eeg/
  video/
  audio/
  wear/
```

## First Commands for the Next Executor

### 1. Re-check files

```powershell
@'
set -e
echo "time $(date '+%F %T')"
for p in \
  /vePFS-0x0d/DailyEEG/processed_cadt_addtime_new/X.npy \
  /vePFS-0x0d/DailyEEG/processed_cadt_addtime_new/y.npy \
  /vePFS-0x0d/DailyEEG/splits_new \
  /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/index/eeg_aligned_window_index.jsonl \
  /vePFS-0x0d/DailyEEG_multimodal/embeddings/video/video_B0_2xroi_eeg23win_embeddings.npz \
  /vePFS-0x0d/DailyEEG_multimodal/embeddings/audio/audio_opensmile_eeg23win_embeddings.npz \
  /vePFS-0x0d/DailyEEG_multimodal/embeddings/wear/wear_physio_preprocessed_eeg23win_embeddings.npz \
  /vePFS-0x0d/DailyEEG_multimodal/embeddings/wear/wear_deep_sequence_preprocessed_eeg23win_embeddings.npz
do
  if [ -e "$p" ]; then
    du -h "$p" 2>/dev/null || echo "OK $p"
  else
    echo "MISSING $p"
  fi
done
'@ | ssh -i C:\Users\28303\.ssh\id_ed25519 -p 36083 wangziwei@124.174.8.252 'bash -s'
```

### 2. Locate or install Python

```powershell
@'
set +e
command -v python
python --version
command -v python3
python3 --version
command -v conda
conda --version
'@ | ssh -i C:\Users\28303\.ssh\id_ed25519 -p 36083 wangziwei@124.174.8.252 'bash -s'
```

If no Python exists, create an environment under `/vePFS-0x0d/home/wangzw/`. Do not write environment files into `/vePFS-0x0d/DailyEEG_multimodal/embeddings/`.

### 3. Validate B0/audio/wear alignment once Python exists

Write a validator under:

```text
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/scripts/validate_b0_alignment.py
```

It must assert:

```text
all files have row_count = 28819
eeg_sample_index equals 0..28818 in every embedding file
sample_id order matches eeg_aligned_window_index.jsonl
labels match y.npy or index labels
missing values are represented only by *_mask
splits_new indices are fully within 0..28818
```

### 4. Start a tiny smoke before full training

Smoke should use:

```text
one split family from splits_new
one wear branch, preferably wear_physio
one model seed
very small epochs
```

Output smoke artifacts to:

```text
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/reports/b0_smoke/
```

Only after the smoke verifies split consumption and masks should the executor start the B0 matrix.

## Expected Outputs

Full B0 experiment outputs should stay under:

```text
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/reports/b0_fusion_matrix/
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/intermediate/b0_fusion_matrix/
```

Required final report:

```text
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/reports/b0_fusion_matrix_summary.md
```

The report should include:

```text
experiment name
enabled modalities
row count
mask coverage by split
RMSE
MAE
pooled raw Pearson r
within-subject centered r
per-subject r mean/std
which splits_new protocol was used
```

## Do Not Do

- Do not move original data from either server.
- Do not delete old-server A1/A2 jobs unless the user explicitly approves a rerun strategy.
- Do not treat `video_v4a_dinov2_2xroi_embeddings.npz` as the EEG-aligned B0 file; the EEG-aligned file is `video_B0_2xroi_eeg23win_embeddings.npz`.
- Do not silently drop rows with missing video/audio/wear. Keep all `28819` EEG rows and use masks.
- Do not use the old `7368`-row `within_subject_fusion_120s10s` split/cohort as the new EEG-aligned B0 result.

## Status Snapshot

Last live checks before this handoff:

```text
New server check:
  EEG X/y present
  splits_new present
  eeg_aligned_window_index present
  video B0 present
  audio present
  wear physio present
  wear deep present under wear_deep_sequence_preprocessed_eeg23win_embeddings.npz
  python/python3/conda not found

Old/new transfer check:
  B0/A1/A2 complete
  B0/A1/A2 copied to new server
  B0/A1/A2 sha256 match old-server files
  final_multimodal_alignment_report.md written on new server
```

Evidence status: confirmed by live SSH checks in the current conversation unless noted.
