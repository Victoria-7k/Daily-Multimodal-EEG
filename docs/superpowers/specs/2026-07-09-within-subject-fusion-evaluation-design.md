# Within-Subject Fusion Evaluation Design

## Goal

Evaluate all 12 existing multimodal fusion experiments under a strict
within-subject protocol. Train an independent model for every subject and
report both per-subject and overall results without allowing windows from the
same event to cross train, validation, and test boundaries.

## Scope

The evaluation expands the current fusion matrix without changing its branch
definitions or model architecture:

- EEG: `eeg_deep_frozen_v1_120s10s`
- Wear: `wear_physio_features_preprocessed_v1` and
  `wear_deep_sequence_preprocessed_v1`
- Video: `V4a_upper` and `B1`
- Audio: `audio_opensmile_egemaps_v1_120s10s`
- Experiments: all 12 combinations already produced by
  `matrix_experiment_specs`
- Target: `fatigue`
- Model: the existing learnable multimodal attention model

This work does not alter embedding extraction or the cross-subject results.

## Evaluation Protocol

### Subject isolation

Each subject is evaluated independently. A model trained for one subject must
only see that subject's samples during training, validation, and testing.
Normalization statistics and early-stopping decisions are computed from that
subject's training and validation partitions only.

### Event-grouped five-fold cross-validation

For each subject:

1. Collect the subject's distinct `event_id` values.
2. Shuffle the event IDs deterministically using the configured seed.
3. Split the event IDs into five balanced groups.
4. For fold `i`, use group `i` as test, group `(i + 1) mod 5` as validation,
   and the remaining three groups as training.
5. Expand event groups back to window indices.

All windows from one event remain in exactly one partition in a fold. Split
validation must reject any overlap in event IDs between train, validation, and
test.

A subject with fewer than five distinct events cannot produce five non-empty
groups. Such a subject is recorded with status `insufficient_events`, including
its event and window counts, and is excluded from aggregate metrics.

### Reproducibility

The runner accepts a seed and uses deterministic subject-local event shuffling.
The same dataset, experiment, subject, and seed must produce identical fold
membership. Each fold records its train, validation, and test event IDs.

## Metrics

Each fold reports:

- train, validation, and test window counts
- train, validation, and test event counts
- test RMSE
- test MAE
- test Pearson correlation
- mean final pooling attention weight for each enabled modality

Pearson correlation is `null` when either prediction or target has zero
variance. Such folds remain valid for RMSE and MAE but do not contribute to
Pearson mean or standard deviation.

Each subject reports the mean and population standard deviation across its
five folds for RMSE, MAE, and available Pearson correlations.

Each experiment reports two overall views:

- `macro`: mean and population standard deviation of valid per-subject mean
  metrics, giving every subject equal weight
- `pooled`: metrics recomputed after concatenating every valid fold's held-out
  predictions and targets

The experiment summary also reports total, valid, and skipped subject counts,
plus total held-out prediction count. A fold's test predictions are included
once, so every valid subject window contributes exactly one pooled held-out
prediction across five-fold cross-validation.

## Outputs

Reports are written below:

```text
outputs/reports/fusion_matrix_within_subject_120s10s/
```

The directory contains:

- `fusion_matrix_within_subject_manifest.json`: command configuration and the
  expanded 12-experiment matrix
- `<experiment>_metrics.json`: fold details, per-subject summaries, skipped
  subjects, macro metrics, and pooled metrics
- `<experiment>_table.md`: one row per subject plus aggregate rows
- `fusion_matrix_within_subject_summary.json`: compact results for all 12
  experiments
- `fusion_matrix_within_subject_summary.md`: comparison table for all 12
  experiments

Model checkpoints are written below:

```text
outputs/models/fusion_matrix_within_subject_120s10s/
  <experiment>/<subject>/fold-00.pt
```

No existing cross-subject report or checkpoint is overwritten.

## Implementation Boundaries

The split builder and aggregation logic belong in a focused training module so
they can be tested without invoking PyTorch training. The command-line runner
reuses:

- `load_fusion_matrix_config`
- `matrix_experiment_specs`
- `build_fusion_dataset`
- `fit_learnable_cross_attention`
- `predict_with_learnable_cross_attention`
- `save_learnable_cross_attention_model`

The existing cross-subject runner remains behaviorally unchanged.

## Failure Handling

- Duplicate or missing sample IDs continue to fail through the existing fusion
  dataset contract.
- Event overlap across partitions raises an error before model training.
- Empty train, validation, or test partitions raise an error.
- Subjects with fewer than five events are recorded and skipped.
- A failed experiment exits non-zero after preserving results already written
  for completed experiments.
- Missing PyTorch reports the existing
  `learnable_cross_attention requires torch` error.

## Testing

Automated tests cover:

- deterministic event-grouped five-fold construction
- no event overlap between train, validation, and test
- all windows from an event remain together
- every valid window appears exactly once in test across five folds
- subjects with fewer than five events are reported as skipped
- macro aggregation gives subjects equal weight
- pooled metrics are recomputed from held-out predictions
- dry-run expansion contains all 12 experiments
- report schemas and output paths

A server smoke run executes one experiment for one subject with reduced epochs.
After it succeeds, the server runs all 12 experiments for all valid subjects
using the production training settings.

## Acceptance Criteria

- All 12 experiments complete or explicitly report a failure.
- Every subject has either five completed folds or an explicit skip reason.
- No event ID crosses partitions within any fold.
- Per-subject JSON and Markdown results are available for every experiment.
- Summary files contain both macro and pooled metrics.
- The existing cross-subject test suite remains green.
