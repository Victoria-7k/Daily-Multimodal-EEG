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
- Primary cohort: the exact `sample_id` intersection across all 12 experiments
- Primary protocol: event-grouped five-fold within-subject OOF
- Secondary protocol: within-subject session-held-out OOF
- Baselines: train-target mean and fixed-alpha Ridge on concatenated embeddings

This work does not alter embedding extraction or the cross-subject results.

## Evaluation Protocol

### Subject isolation

Each subject is evaluated independently. A model trained for one subject must
only see that subject's samples during training, validation, and testing.
Normalization statistics and early-stopping decisions are computed from that
subject's training and validation partitions only.

### Frozen split and cohort manifests

Before any model training, a preparation command builds two immutable inputs:

- a cohort manifest containing the ordered `sample_id` intersection across all
  12 experiments
- a split manifest assigning every eligible event or overlap-connected event
  group to a fold for both evaluation protocols

All experiments and baselines load these manifests and fail if their hashes,
source window-index hash, or required sample IDs differ. `split_seed` controls
only manifest construction. `model_seed` controls only model initialization
and minibatch order; changing it cannot change fold membership.

### Temporal-overlap audit

The preparation command reads
`outputs/window_index/real_cache_face_detected_full_v2_mainface.jsonl` to
recover `window_start_time`, `window_end_time`, and session metadata that are
not stored in the B1 embedding bundle. Events from the same subject/session
whose raw time intervals overlap are placed in the same connected component.
The component, rather than the individual event, is the indivisible split
unit. The manifest records every overlap pair and duration. A post-build audit
must show no raw-time overlap between train, validation, and test.

### Event-grouped five-fold cross-validation

For each subject:

1. Collect the subject's overlap-connected event groups.
2. Shuffle those groups deterministically using `split_seed`.
3. Split the groups into five balanced folds by window count.
4. For fold `i`, use group `i` as test, group `(i + 1) mod 5` as validation,
   and the remaining three groups as training.
5. Expand event groups back to window indices.

All windows from one event remain in exactly one partition in a fold. Split
validation must reject any overlap in event IDs between train, validation, and
test.

A subject with fewer than five split units cannot produce five non-empty
groups. Such a subject is recorded with status `insufficient_split_units`,
including event, split-unit, and window counts, and is excluded from aggregate
metrics.

### Session-held-out protocol

The secondary protocol parses `session_id` from the window index, with the
structured `event_id` used only as a verified fallback. For each subject and
fold, one complete session is test, the next complete session is validation,
and all remaining sessions are training. Each session appears exactly once as
test. Subjects with fewer than three sessions are explicitly skipped for this
protocol.

### Reproducibility and normalization

The runner accepts separate `split_seed` and `model_seed` values. A stable hash
of model seed, protocol, experiment, subject, and fold derives each training
seed without depending on execution order or worker count.

The existing attention model's feature and target normalization must be fitted
only on training indices. Every fold records the training sample-ID hash,
normalization-statistics hash, and a successful recomputation audit. Validation
or test outliers must not change fitted normalization statistics.

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

Each subject's primary metrics are recomputed from its complete concatenated
OOF predictions, not averaged from fold metrics. Fold mean and standard
deviation remain diagnostics only.

Metrics are reported at two levels:

- window-level OOF
- event-level OOF after averaging all window predictions for each event

The event target must be constant across its windows within floating-point
tolerance; otherwise evaluation fails.

Each experiment reports two overall views:

- `macro`: mean and population standard deviation of valid per-subject mean
  metrics, giving every subject equal weight
- `pooled`: RMSE and MAE from all held-out predictions; pooled Pearson is
  computed only after centering predictions and targets by their own subject's
  OOF mean (`within_subject_centered_pearson`)

Raw pooled Pearson may be retained as a clearly marked diagnostic but is never
the headline pooled correlation.

### Paired ablation cohort

The primary matrix uses one global paired cohort: the ordered sample-ID
intersection across all 12 experiment datasets after modality availability
filtering. Every experiment trains and evaluates on exactly this cohort.
Native-coverage row counts are descriptive only and are not used for primary
comparisons.

### Baselines

Both protocols run on the same manifests and cohort:

- `train_mean`: predict the training-target mean
- `concat_ridge_alpha10`: concatenate enabled 256-D embeddings, apply
  train-only standardization, and fit Ridge with fixed `alpha=10`

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
- `split_manifest.json` and `cohort_manifest.json`, including source hashes
- `predictions/<protocol>/<experiment>/<model>.npz`, containing independent
  OOF sample IDs, event IDs, subjects, sessions, folds, targets, predictions,
  and attention weights when available
- `run_state/<protocol>/<experiment>/<subject>.json`, atomically written for
  resume

Model checkpoints are written below:

```text
outputs/models/fusion_matrix_within_subject_120s10s/
  <experiment>/<subject>/fold-00.pt
```

No existing cross-subject report or checkpoint is overwritten.

Production mode does not compute train-set predictions. A completed
experiment-subject job is reused only when its state file, checkpoint hashes,
prediction shard, cohort hash, split hash, and model configuration all match.

## Implementation Boundaries

The split/cohort preparation and metric aggregation logic belong in focused
training modules so they can be tested without invoking PyTorch training. The
command-line runner supports experiment-by-subject process parallelism while
preserving deterministic seeds and unique output paths. It reuses:

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
- Subjects with fewer than five overlap-connected split units are recorded and skipped.
- Subjects with too few sessions are skipped only for session-held-out.
- Missing or stale resume artifacts are recomputed rather than trusted.
- Manifest or cohort hash mismatches stop the run before training.
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
- subjects with fewer than five overlap-connected split units are reported as skipped
- macro aggregation gives subjects equal weight
- pooled metrics are recomputed from held-out predictions
- dry-run expansion contains all 12 experiments
- report schemas and output paths
- frozen split reuse across all 12 experiments
- split/model seed independence
- overlap-connected events never cross partitions
- train-only normalization with validation/test outliers
- subject metrics recomputed from complete OOF predictions
- subject-centered pooled Pearson
- global paired cohort equality across all experiments
- window-level and event-level OOF metrics
- session-held-out coverage and skip rules
- train-mean and Ridge baselines
- resume validation and independent prediction artifacts
- deterministic results across worker counts

Server execution has three gates:

1. Cheap screening of all 12 experiments on two subjects with reduced model
   size and epochs; this validates every branch without pruning the matrix.
2. A benchmark comparing CPU worker counts with single-worker CUDA on the same
   jobs, selecting the fastest valid backend.
3. Full production runs for all 12 experiments and both protocols.

## Acceptance Criteria

- All 12 experiments complete or explicitly report a failure.
- Every subject has either five completed folds or an explicit skip reason.
- No event ID crosses partitions within any fold.
- Per-subject JSON and Markdown results are available for every experiment.
- Summary files contain both macro and pooled metrics.
- Headline pooled correlation is within-subject centered.
- Both window-level and event-level OOF results are present.
- Event and session protocols use frozen manifests.
- All primary ablations use the same global paired cohort.
- Production reports confirm train-only normalization and no train prediction.
- The existing cross-subject test suite remains green.
