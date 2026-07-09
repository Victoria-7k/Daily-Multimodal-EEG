# Within-Subject Fusion Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run all 12 existing fusion experiments as independent event-grouped five-fold evaluations for every subject, with per-subject, macro, and pooled results.

**Architecture:** Add a focused `within_subject_fusion` module that owns deterministic event-grouped folds and metric aggregation without depending on PyTorch. Add a separate matrix runner that reuses the existing fusion configuration, aligned `FusionDataset`, learnable attention trainer, predictor, and checkpoint writer, leaving the cross-subject runner unchanged.

**Tech Stack:** Python 3, NumPy, PyTorch, argparse, JSON, Markdown, pytest/unittest

## Global Constraints

- Run all 12 experiments produced by `matrix_experiment_specs`.
- Use five folds per subject, with three event groups for training, one for validation, and one for testing.
- Never split windows from the same `event_id` across partitions.
- Train, normalize, and early-stop independently for each subject.
- Record subjects with fewer than five distinct events as `insufficient_events`.
- Report per-fold, per-subject, macro, and pooled RMSE, MAE, and Pearson correlation.
- Pearson is `null` for constant predictions or targets and is excluded only from Pearson aggregation.
- Preserve existing cross-subject reports, checkpoints, runner behavior, and branch definitions.
- Use `apply_patch` for manual edits and safe PowerShell here-strings for nontrivial remote commands.

---

### Task 1: Deterministic within-subject event folds

**Files:**
- Create: `src/daily_multimodal/training/within_subject_fusion.py`
- Create: `tests/test_within_subject_fusion.py`

**Interfaces:**
- Consumes: aligned `subject_id: np.ndarray` and `event_id: np.ndarray` from `FusionDataset`
- Produces: `WithinSubjectFold`, `SkippedSubject`, and `build_within_subject_event_folds(subject_id, event_id, n_splits=5, seed=17)`

- [ ] **Step 1: Write failing fold-construction tests**

```python
from __future__ import annotations

import unittest

import numpy as np

from daily_multimodal.training.within_subject_fusion import (
    build_within_subject_event_folds,
)


class WithinSubjectFoldTests(unittest.TestCase):
    def test_event_grouped_folds_are_deterministic_disjoint_and_exhaustive(self):
        subjects = np.asarray(["sub-01"] * 12 + ["sub-02"] * 10)
        events = np.asarray(
            [f"s1-event-{idx // 2}" for idx in range(12)]
            + [f"s2-event-{idx // 2}" for idx in range(10)]
        )

        first_folds, first_skipped = build_within_subject_event_folds(
            subjects, events, n_splits=5, seed=17
        )
        second_folds, second_skipped = build_within_subject_event_folds(
            subjects, events, n_splits=5, seed=17
        )

        self.assertEqual(first_skipped, second_skipped)
        self.assertEqual(
            [(fold.name, fold.test.tolist()) for fold in first_folds],
            [(fold.name, fold.test.tolist()) for fold in second_folds],
        )
        for subject in ("sub-01", "sub-02"):
            subject_folds = [fold for fold in first_folds if fold.subject_id == subject]
            self.assertEqual(len(subject_folds), 5)
            subject_rows = set(np.flatnonzero(subjects == subject).tolist())
            test_rows = []
            for fold in subject_folds:
                train_events = set(events[fold.train])
                val_events = set(events[fold.val])
                test_events = set(events[fold.test])
                self.assertFalse(train_events & val_events)
                self.assertFalse(train_events & test_events)
                self.assertFalse(val_events & test_events)
                test_rows.extend(fold.test.tolist())
            self.assertEqual(sorted(test_rows), sorted(subject_rows))

    def test_subject_with_fewer_than_five_events_is_skipped(self):
        subjects = np.asarray(["sub-01"] * 8 + ["sub-02"] * 5)
        events = np.asarray(
            [f"s1-event-{idx // 2}" for idx in range(8)]
            + [f"s2-event-{idx}" for idx in range(5)]
        )

        folds, skipped = build_within_subject_event_folds(
            subjects, events, n_splits=5, seed=17
        )

        self.assertFalse(any(fold.subject_id == "sub-01" for fold in folds))
        self.assertEqual(len([fold for fold in folds if fold.subject_id == "sub-02"]), 5)
        self.assertEqual(
            skipped,
            [
                {
                    "subject_id": "sub-01",
                    "status": "insufficient_events",
                    "event_count": 4,
                    "window_count": 8,
                    "required_events": 5,
                }
            ],
        )
```

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run:

```powershell
python -m pytest tests/test_within_subject_fusion.py -q
```

Expected: collection fails with
`ModuleNotFoundError: No module named 'daily_multimodal.training.within_subject_fusion'`.

- [ ] **Step 3: Implement fold construction**

Create `src/daily_multimodal/training/within_subject_fusion.py` with these
public types and behavior:

```python
from __future__ import annotations

from dataclasses import dataclass
import zlib

import numpy as np


@dataclass(frozen=True)
class WithinSubjectFold:
    name: str
    subject_id: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_events: tuple[str, ...]
    val_events: tuple[str, ...]
    test_events: tuple[str, ...]


SkippedSubject = dict[str, str | int]


def build_within_subject_event_folds(
    subject_id: np.ndarray,
    event_id: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 17,
) -> tuple[list[WithinSubjectFold], list[SkippedSubject]]:
    subjects = np.asarray(subject_id).astype(str)
    events = np.asarray(event_id).astype(str)
    if subjects.ndim != 1 or events.ndim != 1 or len(subjects) != len(events):
        raise ValueError("subject_id and event_id must be aligned one-dimensional arrays")
    if int(n_splits) < 3:
        raise ValueError("within-subject CV requires at least three folds")

    folds: list[WithinSubjectFold] = []
    skipped: list[SkippedSubject] = []
    for subject in dict.fromkeys(subjects.tolist()):
        subject_rows = np.flatnonzero(subjects == subject)
        subject_events = list(dict.fromkeys(events[subject_rows].tolist()))
        if len(subject_events) < int(n_splits):
            skipped.append(
                {
                    "subject_id": subject,
                    "status": "insufficient_events",
                    "event_count": len(subject_events),
                    "window_count": len(subject_rows),
                    "required_events": int(n_splits),
                }
            )
            continue

        stable_subject_seed = zlib.crc32(subject.encode("utf-8"))
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), stable_subject_seed]))
        shuffled = np.asarray(subject_events, dtype=str)
        rng.shuffle(shuffled)
        groups = [tuple(group.tolist()) for group in np.array_split(shuffled, int(n_splits))]

        for fold_index in range(int(n_splits)):
            test_events = groups[fold_index]
            val_events = groups[(fold_index + 1) % int(n_splits)]
            train_events = tuple(
                event
                for group_index, group in enumerate(groups)
                if group_index not in {fold_index, (fold_index + 1) % int(n_splits)}
                for event in group
            )
            fold = WithinSubjectFold(
                name=f"fold-{fold_index:02d}",
                subject_id=subject,
                train=subject_rows[np.isin(events[subject_rows], train_events)],
                val=subject_rows[np.isin(events[subject_rows], val_events)],
                test=subject_rows[np.isin(events[subject_rows], test_events)],
                train_events=train_events,
                val_events=val_events,
                test_events=test_events,
            )
            _validate_fold(fold)
            folds.append(fold)
    return folds, skipped


def _validate_fold(fold: WithinSubjectFold) -> None:
    train_events = set(fold.train_events)
    val_events = set(fold.val_events)
    test_events = set(fold.test_events)
    if train_events & val_events or train_events & test_events or val_events & test_events:
        raise ValueError(f"{fold.subject_id} {fold.name} has event leakage")
    if len(fold.train) == 0 or len(fold.val) == 0 or len(fold.test) == 0:
        raise ValueError(f"{fold.subject_id} {fold.name} has an empty partition")
```

- [ ] **Step 4: Run fold tests**

Run:

```powershell
python -m pytest tests/test_within_subject_fusion.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit the fold builder**

```powershell
git add -- src/daily_multimodal/training/within_subject_fusion.py tests/test_within_subject_fusion.py
git commit -m "feat: add within-subject event folds"
```

---

### Task 2: Per-subject, macro, and pooled aggregation

**Files:**
- Modify: `src/daily_multimodal/training/within_subject_fusion.py`
- Modify: `tests/test_within_subject_fusion.py`

**Interfaces:**
- Consumes: serializable fold rows containing test metrics, predictions, and targets
- Produces: `regression_metrics`, `summarize_subject`, and `summarize_experiment`

- [ ] **Step 1: Write failing aggregation tests**

Append to `tests/test_within_subject_fusion.py`:

```python
from daily_multimodal.training.within_subject_fusion import (
    regression_metrics,
    summarize_experiment,
    summarize_subject,
)


class WithinSubjectAggregationTests(unittest.TestCase):
    def test_subject_summary_aggregates_five_fold_metrics(self):
        folds = [
            {
                "test": {"rmse": float(value), "mae": float(value) / 2, "pearson": 0.1 * value},
                "test_prediction": [float(value)],
                "test_target": [0.0],
            }
            for value in range(1, 6)
        ]

        summary = summarize_subject("sub-01", event_count=10, window_count=20, folds=folds)

        self.assertEqual(summary["fold_count"], 5)
        self.assertAlmostEqual(summary["rmse_mean"], 3.0)
        self.assertAlmostEqual(summary["rmse_std"], np.std([1, 2, 3, 4, 5]))
        self.assertAlmostEqual(summary["mae_mean"], 1.5)
        self.assertAlmostEqual(summary["pearson_r_mean"], 0.3)

    def test_experiment_summary_reports_macro_and_recomputed_pooled_metrics(self):
        subjects = [
            {
                "subject_id": "sub-01",
                "status": "completed",
                "rmse_mean": 1.0,
                "mae_mean": 0.8,
                "pearson_r_mean": 0.2,
                "folds": [
                    {"test_prediction": [0.0, 1.0], "test_target": [0.0, 2.0]}
                ],
            },
            {
                "subject_id": "sub-02",
                "status": "completed",
                "rmse_mean": 3.0,
                "mae_mean": 2.0,
                "pearson_r_mean": None,
                "folds": [
                    {"test_prediction": [2.0, 3.0], "test_target": [1.0, 4.0]}
                ],
            },
        ]

        summary = summarize_experiment(
            subjects,
            skipped_subjects=[
                {
                    "subject_id": "sub-03",
                    "status": "insufficient_events",
                    "event_count": 4,
                    "window_count": 8,
                    "required_events": 5,
                }
            ],
        )

        expected = regression_metrics(
            np.asarray([0.0, 1.0, 2.0, 3.0]),
            np.asarray([0.0, 2.0, 1.0, 4.0]),
        )
        self.assertEqual(summary["valid_subject_count"], 2)
        self.assertEqual(summary["skipped_subject_count"], 1)
        self.assertEqual(summary["pooled_prediction_count"], 4)
        self.assertAlmostEqual(summary["macro"]["rmse_mean"], 2.0)
        self.assertAlmostEqual(summary["macro"]["pearson_r_mean"], 0.2)
        self.assertEqual(summary["pooled"], expected)

    def test_regression_metrics_returns_null_pearson_for_constant_target(self):
        metrics = regression_metrics(np.asarray([1.0, 2.0]), np.asarray([3.0, 3.0]))
        self.assertIsNone(metrics["pearson"])
```

- [ ] **Step 2: Run tests and verify missing-function failures**

Run:

```powershell
python -m pytest tests/test_within_subject_fusion.py -q
```

Expected: collection fails because `regression_metrics`,
`summarize_subject`, and `summarize_experiment` are not defined.

- [ ] **Step 3: Implement aggregation**

Append these public functions to
`src/daily_multimodal/training/within_subject_fusion.py`:

```python
from typing import Any, Sequence


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int | None]:
    pred = np.asarray(prediction, dtype=np.float32).reshape(-1)
    truth = np.asarray(target, dtype=np.float32).reshape(-1)
    if len(pred) != len(truth):
        raise ValueError("prediction and target must have the same length")
    if len(truth) == 0:
        return {"count": 0, "rmse": None, "mae": None, "pearson": None}
    error = pred - truth
    pearson = None
    if len(truth) >= 2 and float(np.std(pred)) > 0.0 and float(np.std(truth)) > 0.0:
        pearson = float(np.corrcoef(pred, truth)[0, 1])
    return {
        "count": int(len(truth)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "pearson": pearson,
    }


def summarize_subject(
    subject_id: str,
    *,
    event_count: int,
    window_count: int,
    folds: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "subject_id": subject_id,
        "status": "completed",
        "event_count": int(event_count),
        "window_count": int(window_count),
        "fold_count": len(folds),
        **_metric_distribution(folds, "rmse", "rmse"),
        **_metric_distribution(folds, "mae", "mae"),
        **_metric_distribution(folds, "pearson", "pearson_r"),
        "folds": list(folds),
    }


def summarize_experiment(
    subjects: Sequence[dict[str, Any]],
    *,
    skipped_subjects: Sequence[SkippedSubject],
) -> dict[str, Any]:
    completed = [row for row in subjects if row.get("status") == "completed"]
    predictions = [
        value
        for subject in completed
        for fold in subject["folds"]
        for value in fold["test_prediction"]
    ]
    targets = [
        value
        for subject in completed
        for fold in subject["folds"]
        for value in fold["test_target"]
    ]
    return {
        "subject_count": len(completed) + len(skipped_subjects),
        "valid_subject_count": len(completed),
        "skipped_subject_count": len(skipped_subjects),
        "pooled_prediction_count": len(predictions),
        "macro": {
            **_subject_metric_distribution(completed, "rmse_mean", "rmse"),
            **_subject_metric_distribution(completed, "mae_mean", "mae"),
            **_subject_metric_distribution(completed, "pearson_r_mean", "pearson_r"),
        },
        "pooled": regression_metrics(np.asarray(predictions), np.asarray(targets)),
        "subjects": list(completed),
        "skipped_subjects": list(skipped_subjects),
    }


def _metric_distribution(
    folds: Sequence[dict[str, Any]], metric: str, prefix: str
) -> dict[str, float | None]:
    values = np.asarray(
        [fold["test"][metric] for fold in folds if fold["test"].get(metric) is not None],
        dtype=np.float32,
    )
    return {
        f"{prefix}_mean": None if values.size == 0 else float(np.mean(values)),
        f"{prefix}_std": None if values.size == 0 else float(np.std(values)),
    }


def _subject_metric_distribution(
    subjects: Sequence[dict[str, Any]], metric: str, prefix: str
) -> dict[str, float | None]:
    values = np.asarray(
        [subject[metric] for subject in subjects if subject.get(metric) is not None],
        dtype=np.float32,
    )
    return {
        f"{prefix}_mean": None if values.size == 0 else float(np.mean(values)),
        f"{prefix}_std": None if values.size == 0 else float(np.std(values)),
    }
```

- [ ] **Step 4: Run aggregation and fold tests**

Run:

```powershell
python -m pytest tests/test_within_subject_fusion.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Commit aggregation**

```powershell
git add -- src/daily_multimodal/training/within_subject_fusion.py tests/test_within_subject_fusion.py
git commit -m "feat: aggregate within-subject fusion metrics"
```

---

### Task 3: Add the 12-experiment within-subject runner

**Files:**
- Create: `scripts/44_run_within_subject_fusion_matrix.py`
- Modify: `tests/test_within_subject_fusion.py`

**Interfaces:**
- Consumes: `configs/fusion_matrix.yaml` and the existing branch `.npz` files
- Produces: experiment manifest, per-experiment JSON/Markdown, summary JSON/Markdown, and per-subject fold checkpoints

- [ ] **Step 1: Write a failing CLI dry-run test**

Append to `tests/test_within_subject_fusion.py`:

```python
import json
from pathlib import Path
import subprocess
import sys
import tempfile


class WithinSubjectFusionCliTests(unittest.TestCase):
    def test_dry_run_expands_all_twelve_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "reports"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/44_run_within_subject_fusion_matrix.py",
                    "--config",
                    "configs/fusion_matrix.yaml",
                    "--out-dir",
                    str(out_dir),
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads(
                (out_dir / "fusion_matrix_within_subject_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["experiment_count"], 12)
            self.assertEqual(manifest["strategy"], "within_subject_event_grouped_5fold")
            self.assertEqual(manifest["n_splits"], 5)
```

- [ ] **Step 2: Run the CLI test and verify the missing-script failure**

Run:

```powershell
python -m pytest tests/test_within_subject_fusion.py::WithinSubjectFusionCliTests -q
```

Expected: FAIL because
`scripts/44_run_within_subject_fusion_matrix.py` does not exist.

- [ ] **Step 3: Implement the CLI and training loop**

Create `scripts/44_run_within_subject_fusion_matrix.py` with:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.cross_attention_fusion import (
    LearnableAttentionConfig,
    build_fusion_dataset,
    fit_learnable_cross_attention,
    predict_with_learnable_cross_attention,
    save_learnable_cross_attention_model,
)
from daily_multimodal.training.fusion_matrix import (
    branches_for_experiment,
    load_fusion_matrix_config,
    matrix_experiment_specs,
)
from daily_multimodal.training.within_subject_fusion import (
    build_within_subject_event_folds,
    regression_metrics,
    summarize_experiment,
    summarize_subject,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run within-subject fusion matrix evaluation.")
    parser.add_argument("--config", default="configs/fusion_matrix.yaml")
    parser.add_argument("--out-dir", default="outputs/reports/fusion_matrix_within_subject_120s10s")
    parser.add_argument("--model-dir", default="outputs/models/fusion_matrix_within_subject_120s10s")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device")
    parser.add_argument("--max-experiments", type=int)
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_fusion_matrix_config(args.config)
    specs = matrix_experiment_specs(config)
    if args.max_experiments is not None:
        specs = specs[: max(0, int(args.max_experiments))]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    manifest = {
        "config": str(args.config),
        "target_label": config.target_label,
        "model": "learnable_cross_attention",
        "strategy": "within_subject_event_grouped_5fold",
        "n_splits": 5,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "experiment_count": len(specs),
        "experiments": [
            {
                "name": spec.name,
                "enabled_modalities": list(spec.enabled_modalities),
                "min_available_modalities": spec.min_available_modalities,
            }
            for spec in specs
        ],
    }
    (out_dir / "fusion_matrix_within_subject_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.dry_run:
        print(f"experiment_count={len(specs)}")
        return 0

    model_dir.mkdir(parents=True, exist_ok=True)
    results = []
    paired_ids: dict[str, np.ndarray] = {}
    for experiment_index, spec in enumerate(specs):
        dataset = build_fusion_dataset(
            branches=branches_for_experiment(config, spec.name),
            experiment=spec,
            base_sample_ids=paired_ids.get(spec.name),
            metadata_source=config.metadata_source,
        )
        if spec.name.endswith("_full"):
            paired_ids[spec.name.replace("_full", "_no_audio")] = dataset.sample_id
        result = _run_experiment(
            dataset,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            seed=args.seed + experiment_index,
            device=args.device,
            model_dir=model_dir / spec.name,
            max_subjects=args.max_subjects,
        )
        result.update(
            {
                "experiment": spec.name,
                "modalities": list(dataset.modalities),
                "branch_profiles": dataset.branch_profiles,
                "row_count": int(len(dataset.sample_id)),
                "target_label": config.target_label,
            }
        )
        (out_dir / f"{spec.name}_metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / f"{spec.name}_table.md").write_text(
            _experiment_table(result), encoding="utf-8"
        )
        results.append(result)
        print(
            f"completed={spec.name} valid_subjects={result['valid_subject_count']} "
            f"macro_rmse={_fmt(result['macro']['rmse_mean'])}"
        )

    summary = {
        "config": str(args.config),
        "target_label": config.target_label,
        "model": "learnable_cross_attention",
        "strategy": "within_subject_event_grouped_5fold",
        "experiment_count": len(results),
        "experiments": [
            {
                "experiment": row["experiment"],
                "modalities": row["modalities"],
                "branch_profiles": row["branch_profiles"],
                "row_count": row["row_count"],
                "valid_subject_count": row["valid_subject_count"],
                "skipped_subject_count": row["skipped_subject_count"],
                "macro": row["macro"],
                "pooled": row["pooled"],
            }
            for row in results
        ],
    }
    (out_dir / "fusion_matrix_within_subject_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "fusion_matrix_within_subject_summary.md").write_text(
        _summary_table(summary), encoding="utf-8"
    )
    return 0
```

Continue the same file with the complete per-subject loop and report helpers:

```python
def _run_experiment(
    dataset,
    *,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
    device: str | None,
    model_dir: Path,
    max_subjects: int | None,
) -> dict:
    folds, skipped = build_within_subject_event_folds(
        dataset.subject_id, dataset.event_id, n_splits=5, seed=seed
    )
    subjects = list(dict.fromkeys(dataset.subject_id.astype(str).tolist()))
    valid_subjects = [
        subject for subject in subjects
        if any(fold.subject_id == subject for fold in folds)
    ]
    if max_subjects is not None:
        selected = set(valid_subjects[: max(0, int(max_subjects))])
        folds = [fold for fold in folds if fold.subject_id in selected]
        skipped = [row for row in skipped if row["subject_id"] in selected]
        valid_subjects = [subject for subject in valid_subjects if subject in selected]

    subject_results = []
    for subject_index, subject in enumerate(valid_subjects):
        subject_rows = np.flatnonzero(dataset.subject_id.astype(str) == subject)
        subject_folds = [fold for fold in folds if fold.subject_id == subject]
        fold_results = []
        for fold_index, fold in enumerate(subject_folds):
            model = fit_learnable_cross_attention(
                dataset,
                train_indices=fold.train,
                val_indices=fold.val,
                config=LearnableAttentionConfig(
                    token_dim=int(hidden_dim),
                    epochs=int(epochs),
                    learning_rate=float(learning_rate),
                    seed=int(seed) + subject_index * 100 + fold_index,
                    device=device,
                ),
            )
            checkpoint = model_dir / subject / f"{fold.name}.pt"
            save_learnable_cross_attention_model(model, checkpoint)
            train_pred, _ = predict_with_learnable_cross_attention(
                model, dataset, indices=fold.train
            )
            val_pred, _ = predict_with_learnable_cross_attention(
                model, dataset, indices=fold.val
            )
            test_pred, attention = predict_with_learnable_cross_attention(
                model, dataset, indices=fold.test
            )
            fold_results.append(
                {
                    "fold": fold.name,
                    "train_events": list(fold.train_events),
                    "val_events": list(fold.val_events),
                    "test_events": list(fold.test_events),
                    "sample_counts": {
                        "train": len(fold.train),
                        "val": len(fold.val),
                        "test": len(fold.test),
                    },
                    "event_counts": {
                        "train": len(fold.train_events),
                        "val": len(fold.val_events),
                        "test": len(fold.test_events),
                    },
                    "train": regression_metrics(train_pred, dataset.target[fold.train]),
                    "val": regression_metrics(val_pred, dataset.target[fold.val]),
                    "test": regression_metrics(test_pred, dataset.target[fold.test]),
                    "attention_summary": {
                        modality: float(np.mean(attention[:, index]))
                        for index, modality in enumerate(dataset.modalities)
                    },
                    "checkpoint": str(checkpoint),
                    "test_sample_id": dataset.sample_id[fold.test].astype(str).tolist(),
                    "test_prediction": test_pred.astype(float).tolist(),
                    "test_target": dataset.target[fold.test].astype(float).tolist(),
                }
            )
        subject_results.append(
            summarize_subject(
                subject,
                event_count=len(set(dataset.event_id[subject_rows].astype(str))),
                window_count=len(subject_rows),
                folds=fold_results,
            )
        )
    return summarize_experiment(subject_results, skipped_subjects=skipped)


def _experiment_table(result: dict) -> str:
    rows = [
        "| subject | status | events | windows | folds | rmse_mean | rmse_std | mae_mean | r_mean | r_std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for subject in result["subjects"]:
        rows.append(
            f"| {subject['subject_id']} | completed | {subject['event_count']} | "
            f"{subject['window_count']} | {subject['fold_count']} | "
            f"{_fmt(subject['rmse_mean'])} | {_fmt(subject['rmse_std'])} | "
            f"{_fmt(subject['mae_mean'])} | {_fmt(subject['pearson_r_mean'])} | "
            f"{_fmt(subject['pearson_r_std'])} |"
        )
    for subject in result["skipped_subjects"]:
        rows.append(
            f"| {subject['subject_id']} | {subject['status']} | {subject['event_count']} | "
            f"{subject['window_count']} | 0 | NA | NA | NA | NA | NA |"
        )
    rows.append(
        f"| macro | completed | NA | NA | {result['valid_subject_count'] * 5} | "
        f"{_fmt(result['macro']['rmse_mean'])} | {_fmt(result['macro']['rmse_std'])} | "
        f"{_fmt(result['macro']['mae_mean'])} | "
        f"{_fmt(result['macro']['pearson_r_mean'])} | "
        f"{_fmt(result['macro']['pearson_r_std'])} |"
    )
    rows.append(
        f"| pooled | completed | NA | {result['pooled_prediction_count']} | NA | "
        f"{_fmt(result['pooled']['rmse'])} | NA | {_fmt(result['pooled']['mae'])} | "
        f"{_fmt(result['pooled']['pearson'])} | NA |"
    )
    return "\n".join(rows) + "\n"


def _summary_table(summary: dict) -> str:
    rows = [
        "| experiment | modalities | subjects | skipped | macro_rmse | macro_r | pooled_rmse | pooled_r |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["experiments"]:
        rows.append(
            f"| {row['experiment']} | {','.join(row['modalities'])} | "
            f"{row['valid_subject_count']} | {row['skipped_subject_count']} | "
            f"{_fmt(row['macro']['rmse_mean'])} | "
            f"{_fmt(row['macro']['pearson_r_mean'])} | "
            f"{_fmt(row['pooled']['rmse'])} | {_fmt(row['pooled']['pearson'])} |"
        )
    return "\n".join(rows) + "\n"


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run focused tests and dry-run**

Run:

```powershell
python -m pytest tests/test_within_subject_fusion.py tests/test_cross_attention_fusion.py -q
python scripts/44_run_within_subject_fusion_matrix.py --config configs/fusion_matrix.yaml --dry-run --out-dir outputs/reports/fusion_matrix_within_subject_dry_run
```

Expected: all focused tests pass and dry-run prints `experiment_count=12`.

- [ ] **Step 5: Compile the affected Python tree**

Run:

```powershell
python -m compileall -q src scripts tests
```

Expected: exit code `0` with no syntax errors.

- [ ] **Step 6: Commit the runner**

```powershell
git add -- scripts/44_run_within_subject_fusion_matrix.py tests/test_within_subject_fusion.py
git commit -m "feat: run within-subject fusion matrix"
```

---

### Task 4: Server smoke, full run, verification, and living docs

**Files:**
- Modify: `fushion plan.md`
- Modify: `repo-docs/modules/embedding-contract.md`
- Modify: `repo-docs/references/commands-and-artifacts.md`
- Modify: `repo-docs/change-log.md`

**Interfaces:**
- Consumes: the new runner and server branch embeddings
- Produces: complete reports under `outputs/reports/fusion_matrix_within_subject_120s10s/` and checkpoints under `outputs/models/fusion_matrix_within_subject_120s10s/`

- [ ] **Step 1: Sync only the new implementation files to the server**

Run:

```powershell
scp 'src/daily_multimodal/training/within_subject_fusion.py' 'ncc_serve_4090:/mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding/src/daily_multimodal/training/within_subject_fusion.py'
scp 'scripts/44_run_within_subject_fusion_matrix.py' 'ncc_serve_4090:/mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding/scripts/44_run_within_subject_fusion_matrix.py'
scp 'tests/test_within_subject_fusion.py' 'ncc_serve_4090:/mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding/tests/test_within_subject_fusion.py'
```

Expected: all three transfers exit `0`.

- [ ] **Step 2: Run server tests and one-subject smoke**

Run:

```powershell
ssh ncc_serve_4090 'cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && python -m pytest tests/test_within_subject_fusion.py tests/test_cross_attention_fusion.py -q'
ssh ncc_serve_4090 'cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && python scripts/44_run_within_subject_fusion_matrix.py --config configs/fusion_matrix.yaml --out-dir outputs/reports/fusion_matrix_within_subject_smoke --model-dir outputs/models/fusion_matrix_within_subject_smoke --max-experiments 1 --max-subjects 1 --epochs 2 --hidden-dim 32 --learning-rate 0.001 --device cpu'
```

Expected: tests pass; smoke writes one experiment with one completed subject,
five folds, and five checkpoints.

- [ ] **Step 3: Verify smoke invariants**

Run the following safely piped Python:

```powershell
@'
import json
from pathlib import Path

root = Path("outputs/reports/fusion_matrix_within_subject_smoke")
files = list(root.glob("*_metrics.json"))
assert len(files) == 1, files
data = json.loads(files[0].read_text())
assert data["valid_subject_count"] == 1, data["valid_subject_count"]
assert len(data["subjects"]) == 1
assert data["subjects"][0]["fold_count"] == 5
test_ids = [
    sample
    for fold in data["subjects"][0]["folds"]
    for sample in fold["test_sample_id"]
]
assert len(test_ids) == len(set(test_ids))
for fold in data["subjects"][0]["folds"]:
    train = set(fold["train_events"])
    val = set(fold["val_events"])
    test = set(fold["test_events"])
    assert not train & val
    assert not train & test
    assert not val & test
print("smoke_invariants=ok")
'@ | ssh ncc_serve_4090 'cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && python -'
```

Expected: `smoke_invariants=ok`.

- [ ] **Step 4: Run all 12 experiments for every valid subject**

Run:

```powershell
ssh ncc_serve_4090 'cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && python scripts/44_run_within_subject_fusion_matrix.py --config configs/fusion_matrix.yaml --out-dir outputs/reports/fusion_matrix_within_subject_120s10s --model-dir outputs/models/fusion_matrix_within_subject_120s10s --epochs 200 --hidden-dim 128 --learning-rate 0.001 --seed 17 --device cpu'
```

Expected: the command exits `0`, prints 12 `completed=` lines, and writes
`fusion_matrix_within_subject_summary.json`.

- [ ] **Step 5: Verify full-run coverage and report the actual metrics**

Run:

```powershell
@'
import json
from pathlib import Path

root = Path("outputs/reports/fusion_matrix_within_subject_120s10s")
summary = json.loads((root / "fusion_matrix_within_subject_summary.json").read_text())
assert summary["experiment_count"] == 12
assert len(summary["experiments"]) == 12
for row in summary["experiments"]:
    metrics = json.loads((root / f"{row['experiment']}_metrics.json").read_text())
    assert metrics["valid_subject_count"] + metrics["skipped_subject_count"] == metrics["subject_count"]
    for subject in metrics["subjects"]:
        assert subject["fold_count"] == 5
        test_ids = [
            sample
            for fold in subject["folds"]
            for sample in fold["test_sample_id"]
        ]
        assert len(test_ids) == len(set(test_ids))
    print(
        row["experiment"],
        "subjects=", row["valid_subject_count"],
        "macro_rmse=", row["macro"]["rmse_mean"],
        "macro_r=", row["macro"]["pearson_r_mean"],
        "pooled_rmse=", row["pooled"]["rmse"],
        "pooled_r=", row["pooled"]["pearson"],
    )
'@ | ssh ncc_serve_4090 'cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && python -'
```

Expected: 12 result lines and no assertion failures. Preserve these printed
values for the user-facing result summary and documentation.

- [ ] **Step 6: Update the fusion plan and repo docs with actual behavior and results**

Use `apply_patch` to:

- add the within-subject command, split contract, output paths, and the 12
  actual macro/pooled results to `fushion plan.md`
- add the within-subject event-grouping contract to
  `repo-docs/modules/embedding-contract.md`
- add the production command and artifact locations to
  `repo-docs/references/commands-and-artifacts.md`
- prepend a dated entry to `repo-docs/change-log.md` with exact tests,
  smoke/full-run verification, subject counts, and best experiment

The changelog entry must state that same-event windows never cross partitions
and distinguish within-subject metrics from the existing LOSO metrics.

- [ ] **Step 7: Run final local verification**

Run:

```powershell
python -m pytest tests/test_within_subject_fusion.py tests/test_cross_attention_fusion.py tests/test_subject_cv.py tests/test_fair_embedding_ablation.py -q
python -m compileall -q src scripts tests
python C:\Users\28303\.codex\skills\repo-docs\scripts\validate_repo_docs.py repo-docs --repo-root .
git diff --check
```

Expected: tests pass, compileall exits `0`, repo-doc validation reports `0`
errors, and `git diff --check` reports no whitespace errors.

- [ ] **Step 8: Commit implementation documentation**

```powershell
git add -- 'fushion plan.md' repo-docs/modules/embedding-contract.md repo-docs/references/commands-and-artifacts.md repo-docs/change-log.md
git commit -m "docs: record within-subject fusion results"
```
