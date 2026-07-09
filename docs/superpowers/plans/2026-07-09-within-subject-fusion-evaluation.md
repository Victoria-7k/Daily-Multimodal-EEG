# Leakage-Audited Within-Subject Fusion Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate all 12 fusion experiments with frozen, leakage-audited within-subject splits, complete OOF metrics, paired cohorts, baselines, resumable prediction artifacts, and efficient production execution.

**Architecture:** A preparation phase builds immutable cohort and split manifests before training. Focused split, metric, and execution modules then run event-grouped and session-held-out protocols using deterministic experiment-by-subject jobs; every job writes an independent prediction shard and atomic resume state before experiment-level aggregation.

**Tech Stack:** Python 3, NumPy, PyTorch, scikit-learn Ridge, argparse, concurrent.futures, JSON/JSONL/NPZ, pytest

## Global Constraints

- Run all 12 configurations produced by `matrix_experiment_specs`; cheap screening is a gate, not a pruning step.
- Use one ordered global paired cohort: the `sample_id` intersection across all 12 post-mask fusion datasets.
- Build the cohort and split manifests once, hash them, and require every experiment to load them unchanged.
- Separate `split_seed=17` from `model_seed=1701`.
- Treat overlap-connected events from the same subject/session as one split unit.
- Primary protocol: event-grouped five-fold OOF with three groups train, one validation, one test.
- Secondary protocol: within-subject session-held-out OOF with one session test, one validation, and the remainder train.
- Fit feature and target normalization only on training indices and audit it for every attention fold.
- Recompute subject metrics from complete OOF predictions, not fold-metric averages.
- Report window-level and event-level OOF metrics.
- Headline pooled Pearson must be within-subject centered.
- Run `train_mean` and `concat_ridge_alpha10` baselines on the same cohort and splits.
- Production mode must not compute train-set predictions.
- Resume only from artifacts whose cohort, split, configuration, and prediction hashes match.
- Parallelism unit is `protocol x experiment x subject`; CUDA uses one worker per visible device.
- Preserve all existing cross-subject code paths and outputs.

## Review Incorporation

| Priority | Review item | Planned implementation |
| --- | --- | --- |
| P0-1 | Fixed shared split manifest | Task 2 writes one hashed manifest loaded by all 12 experiments |
| P0-2 | Separate split/model seeds | Task 2 owns `split_seed`; Task 5 derives job seeds from `model_seed` |
| P0-3 | Cross-event raw-time overlap | Task 1 builds overlap-connected components from the window index |
| P0-4 | Train-only normalization audit | Task 3 exposes and verifies the exact training-only fit |
| P0-5 | Complete OOF subject metrics | Task 4 concatenates all held-out predictions before recomputing |
| P0-6 | Within-subject-centered pooled r | Task 4 centers prediction and target within subject before pooled r |
| P0-7 | Paired ablation cohort | Task 1 uses one global 12-experiment intersection |
| P1-8 | Window and event OOF | Task 4 reports both levels |
| P1-9 | Session-held-out protocol | Task 2 adds a frozen session protocol |
| P1-10 | Simple baselines | Task 4 adds train mean and fixed-alpha concatenated Ridge |
| P1-11 | Resume | Task 5 validates atomic job state and prediction hashes |
| P1-12 | Independent predictions | Task 4 writes per-job NPZ shards; Task 5 merges them |
| P2-13 | Cheap screening before full matrix | Task 6 runs all branches cheaply before production |
| P2-14 | CPU parallel vs CUDA benchmark | Task 6 benchmarks identical jobs and freezes backend settings |
| P2-15 | No production train prediction | Task 5 predicts validation and test only |
| P2-16 | Experiment-subject parallelism | Task 5 uses deterministic process jobs with isolated paths |

---

### Task 1: Build the global paired cohort and temporal-overlap audit

**Files:**
- Create: `src/daily_multimodal/training/within_subject_splits.py`
- Create: `tests/test_within_subject_splits.py`
- Create: `configs/within_subject_fusion.yaml`

**Interfaces:**
- `build_global_paired_cohort(sample_ids_by_experiment, reference_order) -> np.ndarray`
- `load_window_metadata(path, required_sample_ids) -> list[WindowMetadata]`
- `build_overlap_components(rows) -> tuple[dict[str, str], list[dict]]`
- `write_cohort_manifest(...) -> dict`

- [ ] **Step 1: Write failing tests for global pairing and raw-time overlap**

```python
from datetime import datetime
import numpy as np

from daily_multimodal.training.within_subject_splits import (
    WindowMetadata,
    build_global_paired_cohort,
    build_overlap_components,
)


def test_global_cohort_is_ordered_intersection_across_all_experiments():
    sample_ids = {
        f"exp-{index:02d}": np.asarray(["s3", "s1", "s2"] if index == 0 else ["s1", "s2", "extra"])
        for index in range(12)
    }
    cohort = build_global_paired_cohort(sample_ids, reference_order=sample_ids["exp-00"])
    assert cohort.tolist() == ["s1", "s2"]


def test_overlapping_events_form_one_connected_split_unit():
    rows = [
        WindowMetadata("w1", "e1", "sub-01", "ses-01", datetime.fromisoformat("2026-01-01 10:00:00"), datetime.fromisoformat("2026-01-01 10:02:00")),
        WindowMetadata("w2", "e2", "sub-01", "ses-01", datetime.fromisoformat("2026-01-01 10:01:00"), datetime.fromisoformat("2026-01-01 10:03:00")),
        WindowMetadata("w3", "e3", "sub-01", "ses-01", datetime.fromisoformat("2026-01-01 11:00:00"), datetime.fromisoformat("2026-01-01 11:02:00")),
    ]
    component_by_event, overlaps = build_overlap_components(rows)
    assert component_by_event["e1"] == component_by_event["e2"]
    assert component_by_event["e1"] != component_by_event["e3"]
    assert overlaps == [{
        "subject_id": "sub-01",
        "session_id": "ses-01",
        "event_a": "e1",
        "event_b": "e2",
        "overlap_seconds": 60.0,
    }]
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest tests/test_within_subject_splits.py -q
```

Expected: collection fails because `within_subject_splits` does not exist.

- [ ] **Step 3: Implement cohort and overlap primitives**

Create immutable `WindowMetadata` and implement:

```python
@dataclass(frozen=True)
class WindowMetadata:
    sample_id: str
    event_id: str
    subject_id: str
    session_id: str
    start: datetime
    end: datetime


def build_global_paired_cohort(
    sample_ids_by_experiment: Mapping[str, np.ndarray],
    *,
    reference_order: np.ndarray,
) -> np.ndarray:
    if len(sample_ids_by_experiment) != 12:
        raise ValueError("paired fusion cohort requires exactly 12 experiments")
    common = set(np.asarray(reference_order).astype(str).tolist())
    for name, values in sample_ids_by_experiment.items():
        ids = np.asarray(values).astype(str)
        if len(ids) != len(set(ids.tolist())):
            raise ValueError(f"{name} contains duplicate sample_id values")
        common &= set(ids.tolist())
    ordered = [value for value in np.asarray(reference_order).astype(str) if value in common]
    if not ordered:
        raise ValueError("global paired cohort is empty")
    return np.asarray(ordered, dtype=str)
```

`load_window_metadata` must read the JSONL once, require exactly one row for
each cohort sample, parse `window_start_time`/`window_end_time`, and use the
JSONL `session_id`. `build_overlap_components` must union events when
`max(start_a, start_b) < min(end_a, end_b)` within the same subject/session.

- [ ] **Step 4: Add the evaluation configuration**

Create `configs/within_subject_fusion.yaml` as JSON-compatible YAML:

```json
{
  "fusion_config": "configs/fusion_matrix.yaml",
  "window_index": "outputs/window_index/real_cache_face_detected_full_v2_mainface.jsonl",
  "cohort_manifest": "outputs/splits/fusion_within_subject_120s10s_cohort.json",
  "split_manifest": "outputs/splits/fusion_within_subject_120s10s_splits_seed17.json",
  "split_seed": 17,
  "model_seed": 1701,
  "protocols": ["event_grouped_5fold", "session_held_out"],
  "models": ["train_mean", "concat_ridge_alpha10", "learnable_cross_attention"],
  "production": {
    "epochs": 200,
    "hidden_dim": 128,
    "learning_rate": 0.001
  }
}
```

- [ ] **Step 5: Verify GREEN**

Run:

```powershell
python -m pytest tests/test_within_subject_splits.py -q
```

Expected: both tests pass.

- [ ] **Step 6: Commit**

```powershell
git add -- src/daily_multimodal/training/within_subject_splits.py tests/test_within_subject_splits.py configs/within_subject_fusion.yaml
git commit -m "feat: prepare paired within-subject cohort"
```

---

### Task 2: Freeze event and session split manifests

**Files:**
- Modify: `src/daily_multimodal/training/within_subject_splits.py`
- Modify: `tests/test_within_subject_splits.py`
- Create: `scripts/44_prepare_within_subject_fusion_splits.py`

**Interfaces:**
- `build_split_manifest(cohort, metadata, split_seed=17) -> dict`
- `validate_split_manifest(manifest, cohort_hash, window_index_hash) -> None`
- CLI writes cohort and split manifests before any training

- [ ] **Step 1: Write failing manifest tests**

```python
def test_split_manifest_is_fixed_across_model_seeds_and_blocks_overlap():
    first = build_split_manifest(cohort, metadata, split_seed=17)
    second = build_split_manifest(cohort, metadata, split_seed=17)
    assert first == second
    assert "model_seed" not in first
    for protocol in first["protocols"].values():
        for subject in protocol["subjects"]:
            for fold in subject.get("folds", []):
                train = set(fold["train_event_ids"])
                val = set(fold["val_event_ids"])
                test = set(fold["test_event_ids"])
                assert not train & val
                assert not train & test
                assert not val & test
                assert fold["cross_partition_time_overlap_count"] == 0


def test_session_protocol_holds_out_each_session_once():
    manifest = build_split_manifest(cohort, metadata, split_seed=17)
    row = manifest["protocols"]["session_held_out"]["subjects"][0]
    held_out = [fold["test_session_ids"][0] for fold in row["folds"]]
    assert sorted(held_out) == sorted(row["session_ids"])
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest tests/test_within_subject_splits.py -q
```

Expected: failures report missing `build_split_manifest`.

- [ ] **Step 3: Implement frozen protocol construction**

For `event_grouped_5fold`, balance overlap components by descending window
count into the currently lightest of five fold buckets, using a seeded stable
shuffle only to break equal-size ties. For fold `i`, test bucket `i`,
validation bucket `(i+1) mod 5`, and the other three buckets train.

For `session_held_out`, preserve deterministic session order from the window
index. For fold `i`, test session `i`, validation session `(i+1) mod S`, and
remaining sessions train. Record `insufficient_split_units` or
`insufficient_sessions` rather than silently dropping a subject.

Every fold must store sample IDs, event IDs, split-unit IDs, session IDs,
counts, and `cross_partition_time_overlap_count=0`.

- [ ] **Step 4: Implement the preparation CLI**

The CLI must:

1. Load all 12 fusion datasets without training.
2. Compute native row counts and the global ordered intersection.
3. Rebuild all 12 datasets with strict `base_sample_ids=cohort`.
4. Assert identical sample IDs, subjects, events, and targets.
5. Hash source files and ordered cohort IDs with SHA-256.
6. Write manifests atomically using temporary files plus `Path.replace`.
7. Refuse to overwrite a different existing manifest unless
   `--force-rebuild` is supplied.

Run:

```powershell
python scripts/44_prepare_within_subject_fusion_splits.py --config configs/within_subject_fusion.yaml
```

- [ ] **Step 5: Verify manifest tests and dry preparation**

```powershell
python -m pytest tests/test_within_subject_splits.py -q
python scripts/44_prepare_within_subject_fusion_splits.py --config configs/within_subject_fusion.yaml --dry-run
```

Expected: tests pass; dry-run reports 12 native datasets, one paired cohort,
both protocols, and no files written.

- [ ] **Step 6: Commit**

```powershell
git add -- src/daily_multimodal/training/within_subject_splits.py tests/test_within_subject_splits.py scripts/44_prepare_within_subject_fusion_splits.py
git commit -m "feat: freeze within-subject split manifests"
```

---

### Task 3: Make train-only normalization auditable

**Files:**
- Modify: `src/daily_multimodal/training/cross_attention_fusion.py`
- Modify: `tests/test_cross_attention_fusion.py`
- Create: `tests/test_within_subject_normalization.py`

**Interfaces:**
- `fit_token_normalization(tokens, mask, indices) -> TokenNormalization`
- `audit_model_normalization(model, dataset, train_indices) -> dict`
- Existing `fit_learnable_cross_attention` must call the public fit function

- [ ] **Step 1: Write a failing outlier-isolation test**

```python
def test_normalization_is_unchanged_by_validation_and_test_outliers():
    dataset = make_dataset_with_train_rows_and_outlier_holdout()
    train = np.asarray([0, 1, 2, 3])
    first = fit_token_normalization(dataset.tokens, dataset.token_mask, train)
    changed = dataset.tokens.copy()
    changed[4:] = 1.0e9
    second = fit_token_normalization(changed, dataset.token_mask, train)
    np.testing.assert_allclose(first.x_mean, second.x_mean)
    np.testing.assert_allclose(first.x_std, second.x_std)


def test_attention_fit_records_training_only_normalization_hash():
    model = fit_learnable_cross_attention(dataset, train_indices=train, val_indices=val, config=config)
    audit = audit_model_normalization(model, dataset, train)
    assert audit["fit_scope"] == "train_only"
    assert audit["fit_sample_count"] == len(train)
    assert audit["verified"] is True
```

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest tests/test_within_subject_normalization.py -q
```

Expected: collection fails because the normalization API does not exist.

- [ ] **Step 3: Refactor normalization without changing model behavior**

Add a frozen `TokenNormalization` dataclass containing `x_mean`, `x_std`,
`fit_count`, and `fit_index_sha256`. Move the current
`_token_normalization(dataset.tokens[train], dataset.token_mask[train])`
calculation into `fit_token_normalization(tokens, mask, train_indices)`.
Store its hash and count on `LearnableAttentionModel`.

`audit_model_normalization` must recompute from train indices, compare model
statistics with `np.testing.assert_allclose`, and return:

```python
{
    "fit_scope": "train_only",
    "fit_sample_count": int(len(train_indices)),
    "fit_sample_id_sha256": sha256_lines(dataset.sample_id[train_indices]),
    "statistics_sha256": sha256_arrays(model.x_mean, model.x_std),
    "verified": True,
}
```

- [ ] **Step 4: Verify normalization and existing fusion tests**

```powershell
python -m pytest tests/test_within_subject_normalization.py tests/test_cross_attention_fusion.py -q
```

Expected: all tests pass and existing attention predictions remain valid.

- [ ] **Step 5: Commit**

```powershell
git add -- src/daily_multimodal/training/cross_attention_fusion.py tests/test_cross_attention_fusion.py tests/test_within_subject_normalization.py
git commit -m "feat: audit train-only fusion normalization"
```

---

### Task 4: OOF metrics, event aggregation, centered pooled r, and baselines

**Files:**
- Create: `src/daily_multimodal/training/within_subject_metrics.py`
- Create: `tests/test_within_subject_metrics.py`

**Interfaces:**
- `regression_metrics(prediction, target) -> dict`
- `aggregate_event_predictions(records) -> PredictionRecords`
- `summarize_subject_oof(records) -> dict`
- `summarize_pooled_oof(records) -> dict`
- `fit_predict_train_mean(...)`
- `fit_predict_concat_ridge(...)`
- `save_prediction_shard(path, records, metadata)`

- [ ] **Step 1: Write failing complete-OOF and centered-r tests**

```python
def test_subject_metrics_are_recomputed_from_complete_oof_not_fold_mean():
    records = make_records(
        subjects=["sub-01"] * 4,
        events=["e1", "e2", "e3", "e4"],
        folds=[0, 1, 2, 3],
        target=[0.0, 0.0, 10.0, 10.0],
        prediction=[0.0, 0.0, 0.0, 0.0],
    )
    result = summarize_subject_oof(records)
    assert result["window"]["rmse"] == pytest.approx(np.sqrt(50.0))
    assert "fold_rmse_mean" not in result["window"]


def test_pooled_pearson_is_centered_within_subject():
    records = make_records(
        subjects=["sub-01", "sub-01", "sub-02", "sub-02"],
        events=["a", "b", "c", "d"],
        folds=[0, 1, 0, 1],
        target=[0.0, 1.0, 100.0, 101.0],
        prediction=[0.0, 1.0, 200.0, 201.0],
    )
    result = summarize_pooled_oof(records)
    assert result["within_subject_centered_pearson"] == pytest.approx(1.0)
    assert result["pearson_definition"] == "center prediction and target by subject OOF mean"
```

- [ ] **Step 2: Write failing event and baseline tests**

```python
def test_event_oof_averages_window_predictions():
    records = make_records(
        subjects=["sub-01"] * 4,
        events=["e1", "e1", "e2", "e2"],
        folds=[0, 0, 1, 1],
        target=[1.0, 1.0, 3.0, 3.0],
        prediction=[0.0, 2.0, 2.0, 4.0],
    )
    event_records = aggregate_event_predictions(records)
    assert event_records.prediction.tolist() == [1.0, 3.0]
    assert regression_metrics(event_records.prediction, event_records.target)["rmse"] == 0.0


def test_baselines_fit_only_training_rows():
    mean_prediction = fit_predict_train_mean(target, train, test)
    ridge_prediction, audit = fit_predict_concat_ridge(tokens, mask, target, train, test, alpha=10.0)
    assert np.all(mean_prediction == np.mean(target[train]))
    assert audit["normalization_fit_scope"] == "train_only"
    assert audit["alpha"] == 10.0
```

- [ ] **Step 3: Verify RED**

```powershell
python -m pytest tests/test_within_subject_metrics.py -q
```

Expected: collection fails because `within_subject_metrics` does not exist.

- [ ] **Step 4: Implement metrics and prediction contracts**

Define `PredictionRecords` with aligned arrays:

```python
sample_id, event_id, subject_id, session_id, fold_id,
target, prediction, attention, model_name, experiment, protocol
```

`summarize_subject_oof` must assert each eligible sample occurs exactly once,
then recompute window metrics from all subject records and event metrics from
event-mean records. `summarize_pooled_oof` must report raw pooled RMSE/MAE and
compute Pearson after subtracting each subject's OOF prediction mean and OOF
target mean.

The Ridge baseline concatenates enabled tokens after zero-filling masked
modalities and appending the modality mask, standardizes using training rows
only, and fits `sklearn.linear_model.Ridge(alpha=10.0)`.

`save_prediction_shard` writes compressed NPZ plus a sidecar JSON containing
cohort hash, split hash, job fingerprint, schema version, and NPZ SHA-256.

- [ ] **Step 5: Verify GREEN**

```powershell
python -m pytest tests/test_within_subject_metrics.py -q
```

Expected: all metric and baseline tests pass.

- [ ] **Step 6: Commit**

```powershell
git add -- src/daily_multimodal/training/within_subject_metrics.py tests/test_within_subject_metrics.py
git commit -m "feat: add leakage-safe within-subject OOF metrics"
```

---

### Task 5: Resumable experiment-subject execution

**Files:**
- Create: `src/daily_multimodal/training/within_subject_runner.py`
- Create: `tests/test_within_subject_runner.py`
- Create: `scripts/45_run_within_subject_fusion_matrix.py`

**Interfaces:**
- `JobSpec(protocol, experiment, subject_id, model_name, model_seed, ...)`
- `derive_job_seed(model_seed, protocol, experiment, subject, fold) -> int`
- `run_job(job) -> JobResult`
- `validate_resume_state(job, state_path) -> bool`
- CLI options include `--workers`, `--device`, `--resume`, `--screen`, and `--production`

- [ ] **Step 1: Write failing seed, resume, and no-train-prediction tests**

```python
def test_job_seed_depends_on_model_seed_not_split_seed_or_worker_order():
    first = derive_job_seed(1701, "event_grouped_5fold", "exp", "sub-01", "fold-00")
    second = derive_job_seed(1701, "event_grouped_5fold", "exp", "sub-01", "fold-00")
    changed = derive_job_seed(1702, "event_grouped_5fold", "exp", "sub-01", "fold-00")
    assert first == second
    assert first != changed


def test_resume_requires_matching_prediction_and_manifest_hashes(tmp_path):
    job, state = write_completed_job(tmp_path)
    assert validate_resume_state(job, state) is True
    job.prediction_path.write_bytes(b"changed")
    assert validate_resume_state(job, state) is False


def test_production_attention_job_never_predicts_train_indices(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "predict_with_learnable_cross_attention", lambda model, data, indices: calls.append(indices.copy()) or fake_prediction(indices))
    run_attention_fold(fake_job(), fake_fold(), production=True)
    assert len(calls) == 2
    assert np.array_equal(calls[0], fake_fold().val)
    assert np.array_equal(calls[1], fake_fold().test)
```

- [ ] **Step 2: Write failing parallel determinism test**

```python
def test_job_results_are_identical_for_one_and_two_workers(tmp_path):
    serial = run_jobs(fake_jobs(tmp_path / "serial"), workers=1)
    parallel = run_jobs(fake_jobs(tmp_path / "parallel"), workers=2)
    assert [(row.job_id, row.prediction_sha256) for row in serial] == [
        (row.job_id, row.prediction_sha256) for row in parallel
    ]
```

- [ ] **Step 3: Verify RED**

```powershell
python -m pytest tests/test_within_subject_runner.py -q
```

Expected: collection fails because `within_subject_runner` does not exist.

- [ ] **Step 4: Implement deterministic jobs and atomic resume**

Derive the fold seed from the first eight bytes of:

```python
sha256(f"{model_seed}|{protocol}|{experiment}|{subject}|{fold}".encode()).digest()
```

Each job writes only under:

```text
predictions/<protocol>/<experiment>/<model>/<subject>.npz
run_state/<protocol>/<experiment>/<model>/<subject>.json
models/<protocol>/<experiment>/<subject>/<fold>.pt
```

Write prediction and state files to sibling `.tmp` paths, fsync, then replace.
A resume hit requires matching schema version, cohort SHA-256, split SHA-256,
model config SHA-256, all checkpoint SHA-256 values, and prediction SHA-256.

Production attention jobs call prediction only for validation (early-stopping
audit) and test. They use model training losses already returned by
`fit_learnable_cross_attention` instead of predicting training rows.

- [ ] **Step 5: Implement experiment-subject process parallelism**

Use `concurrent.futures.ProcessPoolExecutor(max_workers=workers)` for CPU.
Sort `JobResult` by immutable `job_id` before aggregation so output does not
depend on completion order. For CUDA, require `workers <= visible_device_count`
and bind one process to one device; default to one CUDA worker on the current
single-GPU server.

- [ ] **Step 6: Implement the matrix CLI**

`scripts/45_run_within_subject_fusion_matrix.py` must:

1. Require and validate frozen cohort/split manifests.
2. Build each of the 12 datasets using the exact global cohort.
3. Expand both protocols, all subjects, and three models.
4. Support `--screen-subjects sub-02,sub-03`, reduced epochs/dimension, and
   all 12 experiments without pruning.
5. Support `--resume`, `--workers`, and `--device`.
6. Merge independent shards into per-experiment protocol NPZ files.
7. Write per-subject window/event OOF JSON rows and aggregate Markdown/JSON.
8. Fail the command if any required job is failed or missing.

- [ ] **Step 7: Verify runner and regression tests**

```powershell
python -m pytest tests/test_within_subject_runner.py tests/test_within_subject_metrics.py tests/test_within_subject_splits.py tests/test_cross_attention_fusion.py -q
python -m compileall -q src scripts tests
```

Expected: all tests pass and compileall exits `0`.

- [ ] **Step 8: Commit**

```powershell
git add -- src/daily_multimodal/training/within_subject_runner.py tests/test_within_subject_runner.py scripts/45_run_within_subject_fusion_matrix.py
git commit -m "feat: run resumable within-subject fusion jobs"
```

---

### Task 6: Cheap screening, backend benchmark, and full matrix

**Files:**
- Generated: `outputs/reports/fusion_matrix_within_subject_screen/`
- Generated: `outputs/reports/fusion_matrix_within_subject_benchmark/`
- Generated: `outputs/reports/fusion_matrix_within_subject_120s10s/`

**Interfaces:**
- Consumes the frozen manifests and production runner
- Produces a backend decision, complete OOF predictions, metrics, checkpoints, and resume state

- [ ] **Step 1: Sync implementation to the server**

Use explicit `scp` commands for the new/modified config, scripts, modules, and
tests. Do not copy the dirty repository wholesale.

- [ ] **Step 2: Build and verify frozen manifests**

```powershell
ssh ncc_serve_4090 'cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && python scripts/44_prepare_within_subject_fusion_splits.py --config configs/within_subject_fusion.yaml'
```

Then run safe piped Python asserting:

- 12 native experiments are represented
- all 12 strict datasets have identical ordered sample IDs
- the cohort is non-empty
- `split_seed == 17` and no `model_seed` appears in the split manifest
- both protocols exist
- every fold has zero event, session, and raw-time cross-partition overlap
- the known server audit reports the `sub-09/ses-01` overlap component rather
  than splitting its two events

- [ ] **Step 3: Run cheap screening across every branch**

```powershell
ssh ncc_serve_4090 'cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && python scripts/45_run_within_subject_fusion_matrix.py --config configs/within_subject_fusion.yaml --out-dir outputs/reports/fusion_matrix_within_subject_screen --model-dir outputs/models/fusion_matrix_within_subject_screen --screen-subjects sub-02,sub-03 --epochs 5 --hidden-dim 32 --workers 1 --device cpu'
```

Gate conditions:

- all 12 experiments complete for both protocols where subjects are eligible
- all three models produce prediction shards
- normalization audits pass
- every eligible OOF sample appears exactly once
- resume rerun reports all jobs as reused

No experiment is dropped based on screening accuracy.

- [ ] **Step 4: Benchmark CPU parallelism versus CUDA**

Run the identical subset `fusion_WphysioPre_B1_full`, subjects `sub-02,sub-03`,
event protocol, 10 epochs, hidden dimension 32 under:

- CPU workers 1
- CPU workers 2
- CPU workers 4
- CUDA workers 1

Each benchmark starts from an empty benchmark output directory and records wall
time, completed fold count, windows/second, peak host memory, peak GPU memory,
and prediction SHA-256. Select the fastest configuration whose metrics and
predictions match the single-worker reference within `atol=1e-6`; if CUDA
kernels are nondeterministic beyond tolerance, select the fastest CPU setting.
Write the decision to
`outputs/reports/fusion_matrix_within_subject_benchmark/backend_decision.json`.

- [ ] **Step 5: Run the full event-grouped matrix with resume enabled**

Use the selected backend and worker count:

```powershell
ssh ncc_serve_4090 'cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && python scripts/45_run_within_subject_fusion_matrix.py --config configs/within_subject_fusion.yaml --protocol event_grouped_5fold --out-dir outputs/reports/fusion_matrix_within_subject_120s10s --model-dir outputs/models/fusion_matrix_within_subject_120s10s --production --resume'
```

- [ ] **Step 6: Run the full session-held-out matrix**

```powershell
ssh ncc_serve_4090 'cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && python scripts/45_run_within_subject_fusion_matrix.py --config configs/within_subject_fusion.yaml --protocol session_held_out --out-dir outputs/reports/fusion_matrix_within_subject_120s10s --model-dir outputs/models/fusion_matrix_within_subject_120s10s --production --resume'
```

- [ ] **Step 7: Verify full-run invariants**

Run a piped server audit that exits non-zero unless:

- all 12 experiments and all three models have complete state
- every valid subject has complete OOF coverage
- prediction artifacts are separate from metric JSON
- event and window metrics are both present
- all attention normalization audits are `verified=true`
- no production job records train predictions
- pooled correlation field is `within_subject_centered_pearson`
- all result rows carry the same cohort and protocol split hashes
- paired experiment comparisons use identical subject/sample cohorts

Print each experiment's window/event macro metrics and centered pooled
correlations for the final report.

---

### Task 7: Update living documentation with methods and actual results

**Files:**
- Modify: `fushion plan.md`
- Modify: `repo-docs/modules/embedding-contract.md`
- Modify: `repo-docs/references/commands-and-artifacts.md`
- Modify: `repo-docs/change-log.md`

- [ ] **Step 1: Update the fusion plan**

Document:

- the global paired cohort count and hash
- both frozen protocols and seeds
- temporal-overlap audit count and handling
- train-only normalization evidence
- baseline, window OOF, event OOF, macro, and centered pooled results
- screening and backend benchmark decision
- resume and prediction artifact locations

- [ ] **Step 2: Update repo-doc ownership pages**

Add the split/cohort/prediction contracts to
`repo-docs/modules/embedding-contract.md`; add exact preparation, screening,
benchmark, event-production, and session-production commands to
`repo-docs/references/commands-and-artifacts.md`.

Prepend a `repo-docs/change-log.md` entry with exact verification commands,
counts, hashes, best configurations, and a clear distinction between
within-subject and LOSO conclusions.

- [ ] **Step 3: Run final verification**

```powershell
python -m pytest tests/test_within_subject_splits.py tests/test_within_subject_normalization.py tests/test_within_subject_metrics.py tests/test_within_subject_runner.py tests/test_cross_attention_fusion.py tests/test_subject_cv.py tests/test_fair_embedding_ablation.py -q
python -m compileall -q src scripts tests
python C:\Users\28303\.codex\skills\repo-docs\scripts\validate_repo_docs.py repo-docs --repo-root .
git diff --check
```

Expected: all tests pass, compileall exits `0`, repo-doc validation reports
zero errors, and `git diff --check` reports no whitespace errors.

- [ ] **Step 4: Commit documentation**

```powershell
git add -- 'fushion plan.md' repo-docs/modules/embedding-contract.md repo-docs/references/commands-and-artifacts.md repo-docs/change-log.md
git commit -m "docs: record leakage-audited within-subject fusion results"
```
