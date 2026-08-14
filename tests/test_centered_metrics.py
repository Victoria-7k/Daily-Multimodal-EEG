import unittest
import importlib.util
from pathlib import Path

import numpy as np

from daily_multimodal.training.centered_metrics import (
    evaluate_regression_with_centered,
    predict_subject_train_mean,
    safe_pearsonr,
    within_subject_centered_arrays,
)


_STRICT_SPLIT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "archive_legacy" / "33_build_strict_within_subject_day_split.py"
_STRICT_SPLIT_SPEC = importlib.util.spec_from_file_location("strict_within_subject_day_split", _STRICT_SPLIT_PATH)
assert _STRICT_SPLIT_SPEC is not None and _STRICT_SPLIT_SPEC.loader is not None
strict_split = importlib.util.module_from_spec(_STRICT_SPLIT_SPEC)
_STRICT_SPLIT_SPEC.loader.exec_module(strict_split)


class CenteredMetricsTests(unittest.TestCase):
    def test_safe_pearsonr_handles_constant_and_short_inputs(self):
        self.assertIsNone(safe_pearsonr([1.0], [1.0]))
        self.assertIsNone(safe_pearsonr([1.0, 1.0], [1.0, 2.0]))
        self.assertAlmostEqual(safe_pearsonr([1.0, 2.0], [2.0, 4.0]), 1.0)

    def test_centering_removes_subject_means_without_reordering(self):
        truth = np.array([1.0, 3.0, 10.0, 14.0], dtype=np.float32)
        prediction = np.array([2.0, 4.0, 9.0, 13.0], dtype=np.float32)
        subjects = np.array(["a", "a", "b", "b"])

        centered_truth, centered_prediction = within_subject_centered_arrays(truth, prediction, subjects)

        np.testing.assert_allclose(centered_truth, [-1.0, 1.0, -2.0, 2.0])
        np.testing.assert_allclose(centered_prediction, [-1.0, 1.0, -2.0, 2.0])

    def test_evaluate_reports_raw_and_centered_correlation_separately(self):
        truth = np.array([1.0, 3.0, 10.0, 14.0], dtype=np.float32)
        prediction = np.array([1.5, 3.5, 9.0, 13.0], dtype=np.float32)
        subjects = np.array(["a", "a", "b", "b"])

        result = evaluate_regression_with_centered(truth, prediction, subjects)

        self.assertAlmostEqual(result["raw_r"], 0.9988, places=3)
        self.assertAlmostEqual(result["within_subject_centered_r"], 1.0, places=6)
        self.assertEqual(result["per_subject_r"]["subject_count"], 2)

    def test_subject_mean_uses_train_only_and_global_fallback(self):
        prediction = predict_subject_train_mean(
            train_y=[1.0, 3.0, 10.0],
            train_subjects=["a", "a", "b"],
            test_subjects=["a", "b", "c"],
        )
        np.testing.assert_allclose(prediction, [2.0, 10.0, 14.0 / 3.0])

    def test_strict_within_subject_split_keeps_subject_days_disjoint(self):
        rows = []
        for subject in ("sub-01", "sub-02"):
            for day in range(5):
                for window in range(2):
                    rows.append({"subject_id": subject, "day_id": day, "sample_id": f"{subject}-{day}-{window}"})

        split = strict_split.build_strict_within_subject_day_split(
            rows,
            train_ratio=0.6,
            val_ratio=0.2,
            pretrain_ratio_of_train=0.5,
        )

        def pairs(indices):
            return {f"{rows[index]['subject_id']}::{rows[index]['day_id']}" for index in indices}

        train_pairs = pairs(split["pretrain"] + split["finetune"])
        val_pairs = pairs(split["val"])
        test_pairs = pairs(split["test"])
        self.assertFalse(train_pairs & val_pairs)
        self.assertFalse(train_pairs & test_pairs)
        self.assertFalse(val_pairs & test_pairs)
        self.assertEqual(len(train_pairs | val_pairs | test_pairs), 10)


if __name__ == "__main__":
    unittest.main()
