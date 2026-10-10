# Modality-wise MAE → A1+MT11 Window Route

The consolidated [chat handout dated 2026-10-10](experiments/mae_chat_handout_20261010.md)
records all decisions, the original/v2/final-v4 comparison, complete 11-emotion
tables, verified artifact paths, recovery history, and copy-paste checks. The
final matrix is complete; dated progress notes below remain historical snapshots.

## Main claim under test

This route tests whether a label-free MAE representation can replace one
modality in the strongest completed 11-emotion 0814 reference while every
other input, event construction, downstream model, split, and downstream seed
is held fixed.

The formal reference is:

```text
A1_Wphysio_no_audio__eeg_eegpt_partial_ft_multitask_11label_v1
```

It uses A1 DINO video, Wphysio, the 11-label supervised partial-fine-tuned
EEGPT token, and `window_attention_regression_full_mean` with an event-level
mean over each event's 23 windows. B0 is read from its completed three-seed
artifacts rather than regenerated.

## Fixed evaluation contract

| Item | Contract |
| --- | --- |
| Representation size | one 256D token per modality per canonical window |
| Downstream unit | one EMA event, constructed from its 23 canonical windows |
| Downstream structure | 0814 `window_attention_regression_full_mean` |
| Head | shared two-layer trunk plus 11 label-specific two-layer heads |
| Protocols | `cross_day`; legacy `within_subject_day` same-subject same-day window holdout |
| Seeds | downstream `240800,240801,240802`; fixed upstream token/MAE seed `240800` |
| Checkpoint rule | validation macro standardized RMSE |
| Test reporting | per-emotion raw r, centered r, RMSE; retain all three seeds |

`cross_day` carries the held-out-day comparison. `within_subject_day` remains a
same-day window-level diagnostic split; event-level aggregation is used here
solely to match the established 11-emotion table.

## Legacy v1 route table (artifacts retained)

| Stage | ID | EEG | Wear | Video | Status | Decision role |
| --- | --- | --- | --- | --- | --- | --- |
| SSL encoder | EEG-MAE | label-free temporal MAE | — | — | Completed | fixed EEG candidate token |
| SSL encoder | Wear-MAE | — | label-free temporal MAE | — | Completed | fixed Wear candidate token |
| SSL encoder | VideoMAE | — | — | label-free tubelet MAE | Completed; transferred to new H20 | 18,012 valid video windows, 5 quarantined decode failures |
| Downstream reference | B0 | MT11 EEGPT | Wphysio | A1 DINO | Completed | completed A1+MT11 three-seed table row |
| Single replacement | E1 | EEG-MAE | Wphysio | A1 DINO | Completed | isolates EEG-MAE contribution |
| Single replacement | W1 | MT11 EEGPT | Wear-MAE | A1 DINO | Completed | isolates Wear-MAE contribution |
| Single replacement | V1 | MT11 EEGPT | Wphysio | VideoMAE | Completed; 2 protocols × 3 seeds | video replacement with 9 native-mask differences versus B0 |
| Two-way replacement | M2 | EEG-MAE | Wear-MAE | A1 DINO | Completed | 2 protocols × 3 seeds complete; tests combined EEG+Wear MAE while keeping DINO A1 |
| Two-way replacement | M3 | EEG-MAE | Wphysio | VideoMAE | Completed; 2 protocols × 3 seeds | joint EEG+Video replacement |
| Two-way replacement | M4 | MT11 EEGPT | Wear-MAE | VideoMAE | Completed; 2 protocols × 3 seeds | joint Wear+Video replacement, retaining MT11 EEG |
| Three-way frozen | M5-F | EEG-MAE | Wear-MAE | VideoMAE | Completed; 2 protocols × 3 seeds | measures complete frozen-MAE route |
| Three-way adaptation | M5-FT | MAE last-2 FT | MAE last-2 FT | MAE last-2 FT | Completed; partial_v3, 2 protocols × 3 seeds | supervised adaptation with the same 11-head downstream structure |
| Initialization control | R0 | random-init MAE last-2 FT | matched | matched | Completed; partial_v3, 2 protocols × 3 seeds | matches M5-FT trainable scope to isolate initialization |

## Evidence generated so far

The table-compatible E1/W1 run completed two protocols × two replacements ×
three downstream seeds. Its B0 cells are the saved A1+MT11 baseline metrics.
The raw-r table is stored on H20 at:

```text
outputs/mae_mt11_window_20260928/formal/raw_r_tables.md
```

On cross-day, W1 has higher three-seed mean raw r for inspired, alert,
determined, attentive, nervous, upset, and afraid. E1 remains an informative
label-free EEG representation comparison, with its full per-emotion metrics
retained in the same artifact.

## Remaining-route execution (2026-10-07 authorization)

The user authorized all five remaining routes, superseding the earlier M3/M4
pause and the frozen-result performance gate for M5-FT. Execution retains
smoke, availability, provenance, gradient, and finite-metric checks.

1. `122` completed M3, M4, M5-F: 18 candidate runs, both protocols and all three seeds.
2. `123`/`125` cache frozen prefixes without reading labels. EEG/Wear are
   prepared on H20; video prefixes use the completed ncc decoded cache.
3. `127` completed all three video-prefix transfers and SHA-256 verification
   at 10:21 +08:00, writing `VIDEO_PREFIX_TRANSFER_VERIFIED`. The source cache
   and original videos remain intact on ncc; H20 can now train independently.
4. `126` waits for prefix preparation, verified transfer and frozen completion,
   runs ten full-data epochs per M5-FT/R0 × protocol smoke cell, then starts the
   two-protocol, three-seed, maximum-80-epoch formal adaptation matrix.

M5-FT updates the final two Transformer blocks of each encoder, modality
adapters, fusion, and 11-head regressor. Stems, positions, earlier encoder
blocks, and unused reconstruction decoders remain frozen. The float32 prefix
cache preserves every token: 10 temporal tokens for EEG/Wear and 196 tubelets
for video. This permits video-tail gradients on H20 while raw videos stay on
ncc. Encoder LR is `1e-4`, downstream LR `1e-3`, event batch size 64, with
activation checkpointing over chunks of 128 windows. Normalization is fitted
on each route's initial training-event tokens and fixed during adaptation.
In the current `last2_ft_encoder_eval_math_v3` training version, encoder tails stay
in eval mode to disable their dropout while weights continue receiving
gradients. The downstream head and modality dropout keep their training
behavior. Four-modality fusion attention uses the float32 math backend with
the same attention equation/weights. Numeric guards run on every batch and
every validation pass. No normalization floor or learning-rate change is applied.

R0 independently random-initializes the entire EEG/Wear/Video MAE architecture
without loading SSL weights. It uses the same last-two-block training scope,
learning rates, head, seeds, masks, and split as M5-FT. It is an
initialization control under matched partial adaptation; it does not evaluate
full-encoder supervised training from scratch. Its random prefix initialization
is shared across protocols and fixed per modality (`240800/240801/240802` for
EEG/Wear/Video); downstream seeds remain `240800,240801,240802`.

New output root: `/home/wangzw/outputs/mae_remaining_20261007/`:
`frozen_formal/results.json` and `raw_r_tables.md` contain completed M3/M4/M5-F;
The original `partial_smoke/` and `partial_formal/` are preserved: cross-day
M5-FT/R0 completed three seeds each, and same-day M5-FT stopped on non-finite
training at 10:29 +08:00. Diagnostic replay located non-finite video gradients;
train-mode encoder dropout magnified normalized video values from about 2.6
to 8,900 against a feature std median of `3.3e-5`.

The dropout-only v2 repair was stopped by its gradient guard at same-day
M5-FT epoch 8; it preserved finite weights by blocking optimizer update.
The repaired v3 queue uses `partial_v3/smoke/` for four cells × ten epochs and
`partial_v3/formal/` for the unified two-protocol three-seed rerun of M5-FT/R0.
The original normalization, encoder/downstream LR, labels, splits, masks, and
0814 head remain fixed. M3/M4/M5-F are retained without rerunning. Safety guards
block non-finite gradients before optimizer steps; failures record context
and stop the queue without counting a failed cell as complete.

At 15:16 +08:00, all four v3 smoke cells completed ten epochs each: every
training/validation history value and all 88 val/test per-label raw-r values
were finite, and all three encoder tails had nonzero gradients and parameter
updates. `PARTIAL_FT_SMOKE_COMPLETE` was verified and formal training started
(queue PID 4181338; runner PID 4182925). Real-GPU fusion-attention forward
equivalence against the prior backend passed with max absolute difference
`2.98e-8`. Head, LRs, initial bags, masks, sample order, splits and target
normalization were checked unchanged. Smoke result/table copies are under
`outputs/server_sync/mae_remaining_20261007/partial_v3/smoke/`, hash-matched
to H20. This is a numerical stability gate, not a formal performance result.
An additional M5-FT cross-day one-epoch real-data preflight completed under
`partial_preflight/`: all three tails had nonzero gradients and parameter
updates, all 11 test raw-r values were finite, and initial normalized-token
equivalence error was `9.54e-7`. This preflight does not count as a formal seed.
v3 formal completion requires `partial_v3/PARTIAL_FT_FORMAL_COMPLETE` plus all 12 finite
candidate result rows. V1 remains under `/home/wangzw/outputs/mae_mt11_v1_20261007/`.

## Stage-A repair v2 (2026-10-08)

The repair preserves canonical indices, splits, 1-second preprocessing, model
dimensions, the A1+MT11 B0 reference, 23-window event aggregation, and three
downstream seeds. EEG/Wear now mask signal content before adding position;
their unmasked export path and parameter shapes remain unchanged. Video now
seeds before model construction. All three use fixed-count validation masks
with an independent `seed + 100003` NumPy generator. Wear logs PPG/EDA/ACC
losses separately with zero-predictor baselines. Numeric guards reject
non-finite loss/gradients before optimizer updates.

Every epoch probes 256 representative validation windows. Near-constant
features are rejected when centered feature RMS / feature RMS is below
`0.001`; dimension-wise std and variance effective rank are also recorded.
This gate detects the observed within-day Video failure; passing it does not
establish emotion-prediction improvement. No emotion/test labels participate
in Stage-A selection. The original normalization remains unchanged in v2;
train-only fixed normalization is implemented in the separate second round below.

`128_queue_mae_repair_stage_a.sh h20|ncc all` runs both protocols with 15
epochs and 1,024 representative train/validation rows per smoke cell, skipping
full token export. All six smoke cells and six full-data pretraining cells
completed on both hosts. Each host releases its formal phase only after its
smoke gate; failed numeric/representation checks stop the queue. New roots:

| Host | Staging | Stage-A output |
| --- | --- | --- |
| H20 `wangzw@124.174.8.252:10022` | `/home/wangzw/mae_repair_20261008` | `outputs/stage_a_v2/{smoke,formal}` |
| ncc | `/tmp/wangzw_mae_repair_20261008` | `outputs/stage_a_v2/{smoke,formal}` |

Video reuses the existing `video_8x112.uint8.mmap` in read-only mode; raw MP4
and the decoded cache stay on ncc. Its native valid mask remains 18,012 rows.
`129_continue_mae_repair_pipeline.ps1` waits for both formal gates, copies and
hash-verifies Video checkpoint/token/config, then starts `130` downstream and
`131` ncc pretrained-prefix preparation. It copies both video prefixes and
verifies their hashes before releasing M5-FT. This is a one-time local
controller; keep its background process and network connection alive until
transfer handoff completes. Server training is detached and independent of
the local controller afterward.

After verified Stage-A transfer and server queue launch, `129 -PrefixTransferOnly`
resumes only the video-prefix copy and hash gate. The recovery is hosted by the
one-time Windows task `DailyMultimodal-MAE-v2-PrefixTransfer-20261008`, with no
recurring trigger. Its parent was verified as the Task Scheduler service rather
than the Codex executor. Keep Windows awake, logged in, and online until the
prefix-transfer marker is written. Recovery commands live in
[commands and artifacts](../../../../repo-docs/references/commands-and-artifacts.md#mae-stage-a-repair-v22026-10-08).

`130` executes E1/W1/V1, then M2/M3/M4, then M5-F, with two-protocol smoke
and three-seed formal runs per group; M5-FT uses a two-protocol ten-epoch smoke
before its six formal cells. Outputs are `outputs/downstream_v2/`. B0 remains
the archived reference. MAE branches are label-free upstream; only retained
EEGPT branches retain MT11, and downstream heads share 11-label supervision.

The six completed v3 R0 rows are retained as matched initialization controls:
legacy and repaired random EEG/Wear/Video initial parameters and unmasked
encoder outputs were checked identical (max difference `0`); raw EEG/Wear
preprocessing is identical. `R0_reference.json` cites those original rows.
Changing normalization or architecture in a later ablation requires a new
matched R0. Prefix reuse now checks checkpoint, token and encoder-source
fingerprints; old prefixes cannot silently satisfy a new repair checkpoint.

All 48 v2 formal candidate cells completed, including M5-FT. Both video
prefixes are present and verified on H20; the earlier Windows transfer/recovery
instructions apply only when another copy is required.

## Train-only raw normalization (2026-10-08)

The second round isolates the loss of amplitude and slow variation introduced
by per-second standardization. EEG now supports one fixed mean/scale per each
of 59 channels; Wear supports independent PPG, EDA and three ACC channel
statistics. Moments are fitted only on the selected valid training rows
(`pretrain + finetune`), then remain fixed for validation, test export, and
prefix generation. The affine transform is applied before 1-second patching,
with no per-patch recentering or rescaling. Constant-channel scale floor is
`1e-6`; data are not clipped. The legacy preprocessing stays the CLI default.

`112 --raw-normalization train_channel` records
`preprocessing_version=train_channel_zscore_v1`, the fit row count/index
fingerprint, and all channel moments in checkpoint/config `raw_normalization`.
`123` reads the checkpoint transform for pretrained prefixes and uses
`--normalization-config <Stage-A config.json>` for matched random prefixes.
`124 --random-prefix-per-protocol` keeps the random initialization weights fixed
while allowing each protocol to use its own train-only raw statistics.

The isolated H20 staging is `/home/wangzw/mae_norm_20261008`; entrypoint
`132_queue_mae_train_channel_normalization.sh` runs four 15-epoch,
1,024-row smoke cells before four full-data EEG/Wear Stage-A runs. Its technical
gate checks training-only fit indices, finite losses/gradients, noncollapsed
features, and canonical token order/masks. Emotion/test labels do not select
the Stage-A checkpoint. Reconstruction values use different normalized targets
across preprocessing conditions and must not be compared as the same loss scale.

Downstream order is E1/W1 → M2/M3/M4 → M5-F → M5-FT/R0. Each frozen group has
a two-protocol smoke before three-seed formal runs; partial FT has four
10-epoch smoke cells before twelve formal runs. These are 48 new formal cells.
B0 and V1 inputs/results stay unchanged. Video pretrained and random prefixes
are checked against v2 canonical IDs/masks and referenced on H20; no raw video,
decode cache, or additional large transfer is needed. All other architectures,
labels, protocol roots, supervision boundaries, LRs, selectors and seeds match
v2. R0 is rerun because EEG/Wear input preprocessing has changed.

Outputs are `outputs/stage_a_train_channel/`,
`outputs/prefixes_train_channel/`, and `outputs/downstream_train_channel/`.
Final `results.json`/`raw_r_tables.md` combine the 48 new cells with twelve
retained B0/V1 rows; `NORMALIZATION_COMPLETE` requires all 60 matched cells.
The queue is detached on H20 and needs no local controller. An error writes
`NORMALIZATION_QUEUE_FAILED` and blocks subsequent phases. Commands and markers
are in the [runbook](../../../../repo-docs/references/commands-and-artifacts.md#mae-train-only-raw-normalization2026-10-08).

## Robust fixed scaling and channel-masked EEG recovery (2026-10-09)

The ordinary train-channel run stopped at within-subject-day EEG epoch 22.
Raw-signal inspection found that the highest-energy 100 training windows
account for 94.05% of cross-day energy and 90.59% of same-day energy under
that transform; the corresponding same-day validation share is 99.94%.
The fixed ordinary scales and unweighted waveform MSE therefore give those
windows disproportionate influence. All source rows and previous artifacts
remain intact.

`112 --raw-normalization train_channel_robust` fits a fixed median window mean
and median window RMS about that center per channel, using train rows only.
It applies the resulting affine transform unchanged during training,
validation, full export, and pretrained/random prefix generation. No raw
values are clipped and no windows are removed. The robust `std` metadata field
stores that median RMS; `center_estimator` and `scale_estimator` identify its
meaning. Preprocessing version is `train_channel_robust_zscore_v1`.

EEG masks 41 of 59 channels consistently across the ten 1-second patches and
reconstructs only their values from the 18 observed channels. Masked raw values
are zeroed before the stem and cannot reach predictions or their gradients.
Wear retains whole-second masking; Video keeps its v2 preprocessing, encoder
and caches. All unmasked encoder architectures and downstream trainable scopes
remain unchanged.

EEG/Wear use `window_energy_balanced_mse_v1`: each window's masked MSE is
divided by its full normalized target mean-square energy, with denominator
floor 1, then averaged equally over windows (and independently over Wear
branches). This keeps extreme amplitudes from dominating while avoiding
amplification of flat-window noise. Validation uses the identical loss and
fixed independent masks, including a matched zero-predictor baseline.
Training version is `position_preserved_balanced_channel_eeg_v4`.

The recovery root is `/home/wangzw/mae_norm_channel_20261009`. Entry `132` uses
`MAE_NORM_MODE=train_channel_robust` and the explicit `MAE_NORM_STAGING` root.
It runs four 35-epoch, 1,024-row smoke cells before the full-data Stage-A queue.
Both overall and median-window relative embedding variation must exceed
`0.001`; the selected validation reconstruction must beat the zero predictor.
The subsequent 48-cell downstream order, B0/V1 reuse, protocols and three
paired downstream seeds remain as above. This recovery changes the EEG
masking and EEG/Wear reconstruction objective as well as fixed scaling;
interpret its results as the combined recovery condition.

The failed ordinary root `/home/wangzw/mae_norm_20261008` and the isolated
35-epoch temporal-mask diagnostic `/home/wangzw/mae_norm_robust_20261009`
are retained separately. The latter passed the noncollapse checks but its
EEG reconstruction did not beat the zero predictor, so its gate prevented
formal training. Monitoring commands and completion markers for the active
recovery are in the same [runbook](../../../../repo-docs/references/commands-and-artifacts.md#mae-train-only-raw-normalization2026-10-08).

### Prefix export batch matching (2026-10-09)

All four formal Stage-A cells and the 36 frozen replacement cells completed
before prefix export stopped. The failing EEG cache used batch64 while the
Stage-A token export used batch128. Recomputing the worst batch with the same
checkpoint gave a batch64-vs-batch128 maximum difference of `0.0016806126`;
same-batch full encoding, saved Stage-A tokens, and prefix+tail agreed exactly.
Only two of 28,819 rows exceeded the existing `2e-4` gate.

`123 --match-stage-a-batch-size` now reads the checkpoint's recorded export
batch size, and `132` enables it for pretrained EEG/Wear. The `2e-4` token
gate and `1e-5` same-batch composition gate remain unchanged. H20 passed all
45 regression tests, and all four repaired pretrained prefixes reported
zero maximum error against Stage-A tokens. The resumed queue retained the
completed frozen groups and rechecked their gates, then passed all four
partial-FT smoke cells and completed the remaining 12 formal M5-FT/R0 cells.
Original data, training objectives, checkpoints, B0/V1, protocols and seeds
are retained.

All 48 new formal cells completed by 14:27 +08:00. Final merging stopped on
an argument-unpacking error: two path variables received those paths plus
the preprocessing version. Entry `132` now reads the two paths separately
from the third argument. All 46 regression tests passed, including a
three-argument final-merge test; aggregation resumed without retraining and
preserved the size and modification time of all 180 training artifacts.
`NORMALIZATION_COMPLETE` was written at 17:03 +08:00 after the 60-row,
11-label finite-metric gate passed. The merged JSON and tables are copied to
`outputs/server_sync/mae_norm_channel_20261009/` with matching SHA-256 values.
