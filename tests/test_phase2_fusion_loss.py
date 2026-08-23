import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local runtime.
    torch = None


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "32_run_eegpt_centered_loss.py"
    spec = importlib.util.spec_from_file_location("run_eegpt_centered_loss", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Phase2FusionLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None:
            raise unittest.SkipTest("torch is not installed in this runtime")
        cls.mod = _load_script_module()

    def test_weighted_mse_label_bins_uses_train_label_frequency(self):
        raw_y = np.asarray([1, 1, 1, 2, 3, 4, 5], dtype=np.float32)
        config = self.mod._phase2_loss_config("weighted_mse_label_bins", raw_y, y_mean=2.0, y_std=1.0)

        self.assertEqual(config["label_counts"]["1"], 3)
        self.assertGreater(config["label_weights"]["4"], config["label_weights"]["1"])
        self.assertLessEqual(config["label_weights"]["4"], 3.0)

    def test_huber_extreme_weight_marks_low_and_high_fatigue(self):
        raw_y = np.asarray([1, 2, 3, 4, 5], dtype=np.float32)
        config = self.mod._phase2_loss_config("huber_extreme_weight", raw_y, y_mean=3.0, y_std=1.0)
        weights = self.mod._sample_weights_for_loss(raw_y, config)

        np.testing.assert_allclose(weights, np.asarray([2, 2, 1, 2, 2], dtype=np.float32))

    def test_variance_regularization_penalizes_collapsed_prediction(self):
        prediction = torch.ones(4)
        target = torch.tensor([0.0, 1.0, 0.0, 1.0])
        subjects = np.asarray(["s1", "s1", "s2", "s2"])

        raw, aux, eligible = self.mod._loss_components(
            prediction,
            target,
            subjects,
            "mse_variance_reg",
            sample_weights=None,
            phase2_config={"kind": "mse_variance_reg"},
        )

        self.assertGreater(float(raw), 0.0)
        self.assertGreater(float(aux), 0.0)
        self.assertEqual(eligible, 2)

    def test_label_subject_sampler_keeps_all_indices_once(self):
        subjects = np.asarray(["s1", "s1", "s2", "s2", "s2"])
        labels = np.asarray([1, 5, 1, 3, 5], dtype=np.float32)
        rng = np.random.default_rng(7)

        batches = self.mod._make_batches(subjects, labels, batch_size=2, rng=rng, sampler="label_subject_balanced")
        observed = sorted(int(index) for batch in batches for index in batch.tolist())

        self.assertEqual(observed, [0, 1, 2, 3, 4])

    def test_ordinal_cumulative_loss_accepts_raw_fatigue_labels(self):
        logits = torch.zeros((5, 4), dtype=torch.float32)
        labels = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)

        loss = self.mod._ordinal_cumulative_loss(logits, labels)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

    def test_phase3_heads_return_scalar_predictions(self):
        tokens = torch.randn(3, 4, 256)
        mask = torch.ones(3, 4, dtype=torch.bool)
        label_values = np.linspace(-1.0, 1.0, 5, dtype=np.float32)

        for head in ("ordinal_cumulative", "classification_expectation", "hybrid_regression_ordinal"):
            module = self.mod.AttentionRegressor(
                modality_count=4,
                hidden_dim=16,
                dropout=0.0,
                head=head,
                normalized_label_values=label_values,
            )
            prediction = module(tokens, mask)
            self.assertEqual(tuple(prediction.shape), (3,))


if __name__ == "__main__":
    unittest.main()
