# EEGPT Multimodal Cross-Attention Handoff

## Purpose

This handoff is for the next worker who will run EEG-aligned multimodal cross-attention analysis on the new server.

Use the newly generated EEGPT EEG embedding as the EEG branch. Keep the previous B0 EEG-aligned cross-attention scheme, train under the three EEG `splits_new` protocols, and evaluate the `fatigue` label by default.

## Execution Update: B0/A1/A2 Route-Aware Run Completed

On 2026-08-01 Asia/Shanghai, the run was expanded from B0-only to a B0/A1/A2 route-aware matrix because A1 and A2 video embeddings were ready.

Remote runner changes:

```text
scripts/run_b0_fusion_matrix.py
scripts/validate_b0_alignment.py
```

Both scripts now point the EEG branch to:

```text
/vePFS-0x0d/DailyEEG_multimodal/embeddings/eeg/eeg_eegpt_eeg23win_embeddings.npz
```

The validator also checks `video_A1_2xroi_eeg23win_embeddings.npz` and `video_A2_2xroi_eeg23win_embeddings.npz`. The route-aware matrix uses the original 8 B0 controls plus 4 A1 and 4 A2 video-using experiments per protocol. `no_video` and `bio_only` controls remain route-independent to avoid duplicate non-video results.

Completed outputs:

```text
reports/eegpt_allvideo_alignment_preflight.json
reports/eegpt_allvideo_alignment_preflight.md
reports/eegpt_allvideo_fusion_matrix_cross_subject_summary.json
reports/eegpt_allvideo_fusion_matrix_cross_subject_summary.md
reports/eegpt_allvideo_fusion_matrix_cross_day_summary.json
reports/eegpt_allvideo_fusion_matrix_cross_day_summary.md
reports/eegpt_allvideo_fusion_matrix_within_subject_day_summary.json
reports/eegpt_allvideo_fusion_matrix_within_subject_day_summary.md
reports/eegpt_allvideo_fusion_matrix_all_protocols_summary.json
reports/eegpt_allvideo_fusion_matrix_all_protocols_summary.md
```

Completion check:

```text
protocol_summaries = 3/3
metrics_files = 48/48
target_label = fatigue
EEG branch path verified in every metrics JSON
validation errors = 0
```

Best RMSE by protocol in the merged report:

| Protocol | Best experiment | RMSE | Raw r |
| --- | --- | ---: | ---: |
| `cross_subject` | `B0_Wphysio_bio_only` | `0.9012` | `-0.0207` |
| `cross_day` | `A1_Wphysio_no_audio` | `0.9298` | `0.2594` |
| `within_subject_day` | `A1_Wphysio_no_audio` | `0.9348` | `0.3032` |

## Remote Host

Run the analysis on the new server:

```text
ssh -i C:\Users\28303\.ssh\id_ed25519 -p 36083 wangziwei@124.174.8.252
```

Recommended working root:

```text
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
```

Use PowerShell-safe SSH quoting from the project instructions. For nontrivial remote Bash, prefer:

```powershell
@'
cd /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned || exit 1
pwd
'@ | ssh -i C:\Users\28303\.ssh\id_ed25519 -p 36083 wangziwei@124.174.8.252 'bash -s'
```

## Canonical Dataset

Canonical EEG-aligned index:

```text
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/index/eeg_aligned_window_index.jsonl
```

Raw EEG arrays:

```text
/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new/
  X.npy
  y.npy
  sub.npy
  d.npy
  ts.npy
```

Expected facts:

```text
row_count = 28819
event_count = 1253
windows_per_event = 23
window_length = 10s
stride = 5s
sample_id = eeg_{eeg_sample_index:06d}
eeg_sample_index = 0..28818
```

Splits are authoritative and must be used as-is:

```text
/vePFS-0x0d/DailyEEG/splits_new/
  cross_subject/
  cross_day/
  within_subject_day/
```

Each protocol has:

```text
pretrain.json
finetune.json
val.json
test.json
split_info.json
```

For supervised training, use:

```text
train = pretrain + finetune
validation = val
test = test
```

Do not rewrite or regenerate `splits_new`.

## Embedding Branches

All branch files are on the new server under:

```text
/vePFS-0x0d/DailyEEG_multimodal/embeddings/
```

Use these branches for the main B0 multimodal analysis:

| Branch | Path | Embedding key | Mask key | Known status |
| --- | --- | --- | --- | --- |
| EEGPT EEG | `/vePFS-0x0d/DailyEEG_multimodal/embeddings/eeg/eeg_eegpt_eeg23win_embeddings.npz` | `eeg_emb` | `eeg_mask` | `(28819, 256)`, mask sum `28819`, failure `0`, NaN `0` |
| Video B0 | `/vePFS-0x0d/DailyEEG_multimodal/embeddings/video/video_B0_2xroi_eeg23win_embeddings.npz` | `video_emb` | `video_mask` | `(28819, 256)`, mask sum `18017` |
| Audio | `/vePFS-0x0d/DailyEEG_multimodal/embeddings/audio/audio_opensmile_eeg23win_embeddings.npz` | `audio_emb` | `audio_mask` | `(28819, 256)`, mask sum `17924` |
| Wear physio | `/vePFS-0x0d/DailyEEG_multimodal/embeddings/wear/wear_physio_preprocessed_eeg23win_embeddings.npz` | `wear_emb` | `wear_mask` | `(28819, 256)`, mask sum `24127` |
| Wear deep | `/vePFS-0x0d/DailyEEG_multimodal/embeddings/wear/wear_deep_sequence_preprocessed_eeg23win_embeddings.npz` | `wear_emb` | `wear_mask` | `(28819, 256)`, mask sum `24127` |

EEGPT generation report:

```text
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/reports/eeg_eegpt_eeg23win_embedding_report.md
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/reports/eeg_eegpt_eeg23win_embedding_report.json
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/checksums/eeg_eegpt_eeg23win_sha256.txt
```

EEGPT checksum:

```text
943046313ae5af275bd6f3b7b134169cdc651ec0a1ec0369151dbe99be75013c
```

Video-route artifacts are present for route comparison:

```text
/vePFS-0x0d/DailyEEG_multimodal/embeddings/video/video_A1_2xroi_eeg23win_embeddings.npz
/vePFS-0x0d/DailyEEG_multimodal/embeddings/video/video_A2_2xroi_eeg23win_embeddings.npz
```

## Required Code/Config Check Before Training

The older B0 runner was originally created with the statistical EEG baseline:

```text
/vePFS-0x0d/DailyEEG_multimodal/embeddings/eeg/eeg_statfft_eeg23win_embeddings.npz
```

Before running the new analysis, make sure the remote runner uses the EEGPT file:

```text
/vePFS-0x0d/DailyEEG_multimodal/embeddings/eeg/eeg_eegpt_eeg23win_embeddings.npz
```

Check:

```bash
cd /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
grep -n "eeg_eegpt_eeg23win_embeddings" scripts/run_b0_fusion_matrix.py
grep -n "eeg_eegpt_eeg23win_embeddings" scripts/validate_b0_alignment.py
```

If either script still points to `eeg_statfft_eeg23win_embeddings.npz`, patch the EEG branch path before running. Keep EEGPT as a distinct artifact and keep the old stat/FFT baseline traceable.

## Cross-Attention Scheme

Use the same B0 EEG-aligned learnable cross-attention scheme as before.

Model input:

```text
tokens shape = (N, modality_count, 256)
token_mask shape = (N, modality_count)
```

For each experiment, stack the enabled modality embeddings as modality tokens. Missing modalities stay in the row with mask `0`; rows are retained. Because EEGPT has full coverage, every default experiment has at least one valid token for all `28819` rows.

Model architecture:

```text
input Linear(256 -> hidden_dim)
learnable modality embedding
1-head torch.nn.MultiheadAttention over modality tokens
learnable query attention pooling over attended modality tokens
LayerNorm + Linear + ReLU + Dropout + Linear regression head
```

Training/evaluation rules:

```text
target_label = fatigue
loss = MSE on train-standardized fatigue target
feature normalization = fit on train tokens only
optimizer = AdamW
default hidden_dim = 128
default dropout = 0.1
default learning_rate = 1e-3
default weight_decay = 1e-4
default epochs = 80
default batch_size = 256
default patience = 15
default seed = 240729
```

Report these metrics for the test split:

```text
RMSE
MAE
pooled raw Pearson r
within-subject centered r
per-subject r mean/std
mask coverage by split and branch
```

## Experiment Matrix

Run the same eight B0 branch combinations:

| Experiment | Modalities |
| --- | --- |
| `B0_Wphysio_full` | EEGPT EEG + wear physio + video B0 + audio |
| `B0_Wphysio_no_audio` | EEGPT EEG + wear physio + video B0 |
| `B0_Wphysio_no_video` | EEGPT EEG + wear physio + audio |
| `B0_Wphysio_bio_only` | EEGPT EEG + wear physio |
| `B0_Wdeep_full` | EEGPT EEG + wear deep + video B0 + audio |
| `B0_Wdeep_no_audio` | EEGPT EEG + wear deep + video B0 |
| `B0_Wdeep_no_video` | EEGPT EEG + wear deep + audio |
| `B0_Wdeep_bio_only` | EEGPT EEG + wear deep |

Run all eight experiments under all three EEG protocols:

```text
cross_subject
cross_day
within_subject_day
```

This gives `24` protocol-experiment runs.

## Suggested Runtime

The GPU environment prepared during EEGPT generation is:

```text
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/runtime/envs/eegpt-gpu-min/bin/python
```

It was verified with CUDA PyTorch:

```text
torch = 2.5.1
cuda = 12.1
device = NVIDIA H20
```

Use `--device cuda` for the cross-attention training unless there is active GPU contention.

## Suggested Commands

First run an alignment preflight after the EEGPT path is wired into validation:

```bash
cd /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
PY=runtime/envs/eegpt-gpu-min/bin/python

$PY scripts/validate_b0_alignment.py \
  --require-eeg-embedding \
  --json-out reports/eegpt_b0_alignment_preflight.json \
  --md-out reports/eegpt_b0_alignment_preflight.md
```

Then run all three protocols:

```bash
cd /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
PY=runtime/envs/eegpt-gpu-min/bin/python

for protocol in cross_subject cross_day within_subject_day; do
  $PY scripts/run_b0_fusion_matrix.py \
    --protocol "$protocol" \
    --target-label fatigue \
    --device cuda \
    --epochs 80 \
    --hidden-dim 128 \
    --batch-size 256 \
    --patience 15 \
    --torch-threads 4 \
    --out-dir "reports/eegpt_b0_fusion_matrix_${protocol}" \
    --intermediate-dir "intermediate/eegpt_b0_fusion_matrix_${protocol}" \
    --summary-json "reports/eegpt_b0_fusion_matrix_${protocol}_summary.json" \
    --summary-md "reports/eegpt_b0_fusion_matrix_${protocol}_summary.md"
done
```

If running through `nohup`, write one log per protocol:

```bash
nohup $PY scripts/run_b0_fusion_matrix.py ... > logs/eegpt_b0_fusion_cross_subject.log 2>&1 &
```

After all three protocol summaries exist, merge them into a single comparison report. The existing summarizer expects the historical filenames; either adapt it to the `eegpt_...` prefix or copy the three summary JSONs to names it expects before running:

```text
scripts/summarize_b0_protocols.py
```

Recommended merged output names:

```text
reports/eegpt_b0_fusion_matrix_all_protocols_summary.json
reports/eegpt_b0_fusion_matrix_all_protocols_summary.md
```

## Validation Checklist

Before training:

```text
all enabled branch sample_id arrays exactly match eeg_aligned_window_index.jsonl
all enabled branch embeddings have shape (28819, 256)
all enabled branch masks have shape (28819,)
EEGPT eeg_sample_index equals 0..28818
EEGPT sample_id order equals the canonical index
EEGPT labels match the canonical labels
splits_new indices are in range
train = pretrain + finetune only
val/test are not modified
```

After training:

```text
3 protocol summaries exist
48 experiment metrics files exist for the completed B0/A1/A2 route-aware matrix
target_label is fatigue in every runtime JSON
branch paths in metrics JSON point to eeg_eegpt_eeg23win_embeddings.npz
test metrics include RMSE, MAE, pooled raw r, centered r, per-subject r mean/std
mask coverage is reported for train/val/test
no split files under /vePFS-0x0d/DailyEEG/splits_new are modified
```

## Guardrails

Use the new server paths in this file. The old `7368`-window within-subject cohort and the older 120s/10s fusion artifacts are historical context.

For this EEG-aligned analysis:

```text
primary EEG branch = eeg_deep_frozen_v1 / braindecode EEGPT
primary video branches = B0/A1/A2 2xROI for route-aware comparison
primary target = fatigue
primary protocols = cross_subject, cross_day, within_subject_day
row policy = keep all 28819 rows and use masks
split policy = consume splits_new exactly
```

Do not report a run as the EEGPT multimodal result unless the saved metrics JSON records the EEG branch path as:

```text
/vePFS-0x0d/DailyEEG_multimodal/embeddings/eeg/eeg_eegpt_eeg23win_embeddings.npz
```
