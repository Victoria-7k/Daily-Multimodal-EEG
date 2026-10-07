# Modality-wise MAE → A1+MT11 Window Route

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

## Route table

| Stage | ID | EEG | Wear | Video | Status | Decision role |
| --- | --- | --- | --- | --- | --- | --- |
| SSL encoder | EEG-MAE | label-free temporal MAE | — | — | Completed | fixed EEG candidate token |
| SSL encoder | Wear-MAE | — | label-free temporal MAE | — | Completed | fixed Wear candidate token |
| SSL encoder | VideoMAE | — | — | label-free tubelet MAE | Completed; transferred to new H20 | 18,012 valid video windows, 5 quarantined decode failures |
| Downstream reference | B0 | MT11 EEGPT | Wphysio | A1 DINO | Completed | completed A1+MT11 three-seed table row |
| Single replacement | E1 | EEG-MAE | Wphysio | A1 DINO | Completed | isolates EEG-MAE contribution |
| Single replacement | W1 | MT11 EEGPT | Wear-MAE | A1 DINO | Completed | isolates Wear-MAE contribution |
| Single replacement | V1 | MT11 EEGPT | Wphysio | VideoMAE | Running; smoke passed | video replacement with 9 native-mask differences versus B0 |
| Two-way replacement | M2 | EEG-MAE | Wear-MAE | A1 DINO | Completed | 2 protocols × 3 seeds complete; tests combined EEG+Wear MAE while keeping DINO A1 |
| Paused combinations | M3/M4 | MAE/current | current/MAE | MAE | Skipped by decision | excluded from the current queue |
| Three-way frozen | M5-F | EEG-MAE | Wear-MAE | VideoMAE | Waiting for V1 | measures complete frozen-MAE route |
| Three-way adaptation | M5-FT | MAE partial FT | MAE partial FT | MAE partial FT | Gated on M5-F | tests supervised adaptation after frozen evidence |
| Architecture control | R0 | random-init MAE architecture | matched | matched | Optional | separates SSL pretraining value from encoder capacity |

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

## Next execution order

1. Finish V1 using the same A1+MT11 reference contract.
2. Run M5-F only after V1 is valid.
3. Advance M5-FT only when frozen M5 provides a stable multi-seed signal.
4. Add R0 if the frozen-MAE comparison needs an architecture-capacity control.

M3 and M4 stay outside this execution order. M2 is complete. VideoMAE tokens
passed shape, finite-value, canonical sample-order, mask, and zero-invalid-row
checks on ncc and were SHA-256 verified after transfer to new H20. V1 outputs
are under `/home/wangzw/outputs/mae_mt11_v1_20261007/`.
