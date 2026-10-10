"""Guard val-only disclosure and common-resampling interaction semantics."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES
from daily_multimodal.training.multihead_regression import evaluate_event_level


class Round2ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/multilabel/140_report_round2_experiment_a.py"
        spec = importlib.util.spec_from_file_location("report_a_tests", path)
        cls.report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.report)

    def data(self):
        rng = np.random.default_rng(9)
        y = rng.normal(size=(40, 11)).astype(np.float32)
        p = (0.5*y + rng.normal(size=y.shape)).astype(np.float32)
        return {"target": y, "prediction": p, "event_id": np.asarray([f"event_{i}" for i in range(40)]),
                "subject_id": np.asarray([f"s{i//10}" for i in range(40)]),
                "day_id": np.asarray([f"d{i//5}" for i in range(40)])}

    def test_val_reader_never_loads_pickled_test_arrays(self):
        data = self.data()
        scale = np.ones((1, 11), np.float32)
        val = evaluate_event_level(data, scale)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "metrics.json").write_text(json.dumps({"status": "ok", "protocol": "cross_day",
                                                            "train_target_std": scale.ravel().tolist(), "val": val,
                                                            "test": {"invalid": "sealed"}}))
            np.savez_compressed(root / "event_predictions.npz", label_names=np.asarray(LABEL_NAMES),
                                **{"val_" + k: v for k, v in data.items()},
                                test_prediction=np.asarray([None], dtype=object))
            _, _, stored = self.report.read_val(root, expected_count=40)
            self.assertEqual(stored.shape, (11, 5))

    def test_shared_bootstrap_zero_interaction_for_equal_input_pairs(self):
        data = self.data()
        adapted = {**data, "prediction": data["prediction"] * .9}
        result, blocks = self.report.bootstraps([data, data, adapted, adapted], np.ones(11),
                                                self.report.COEFFICIENTS, 20, 17)
        np.testing.assert_allclose(result["input_adaptation_interaction"]["per_label"], 0, atol=1e-12)
        self.assertEqual(blocks, 8)
        changed = {**adapted, "event_id": adapted["event_id"][::-1]}
        with self.assertRaises(ValueError):
            self.report.bootstraps([data, changed], np.ones(11), {"pair": (-1, 1)}, 20, 17)


if __name__ == "__main__":
    unittest.main()
