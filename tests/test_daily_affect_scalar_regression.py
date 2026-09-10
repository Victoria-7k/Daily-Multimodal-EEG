from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from daily_multimodal.daily_affect.regression import DailyAffectRegressionConfig, DailyAffectRegressionModel
from daily_multimodal.daily_affect.regression_training import event_balanced_window_mse, run_daily_affect_regression_run
from daily_multimodal.daily_affect.training import DailyAffectBagDataset


@unittest.skipIf(torch is None, "torch is unavailable")
class DailyAffectScalarRegressionTest(unittest.TestCase):
    def dataset(self, root: Path) -> DailyAffectBagDataset:
        rng, count = np.random.default_rng(7), 10
        labels = np.asarray([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=np.float32)
        return DailyAffectBagDataset(
            bag_path=root / "synthetic_bags.npz", tokens=rng.normal(size=(count, 23, 4, 256)).astype(np.float32),
            modality_mask=np.ones((count, 23, 4), dtype=bool), label=labels, label_zero_based=labels.astype(np.int64) - 1,
            event_id=np.asarray([f"event_{index}" for index in range(count)]), subject_id=np.asarray([f"s{index // 2}" for index in range(count)]),
            day_id=np.asarray([f"d{index // 2}" for index in range(count)]), sample_id_matrix=np.full((count, 23), "sample", dtype=str),
            split=np.asarray(["train"] * 6 + ["val"] * 2 + ["test"] * 2), route_id="synthetic", target_label="fatigue",
            source_npz_json="{}", supervision_boundary="synthetic", pretrain_index=np.asarray([], dtype=np.int64),
            finetune_index=np.arange(6, dtype=np.int64), train_index=np.arange(6, dtype=np.int64), val_index=np.asarray([6, 7], dtype=np.int64),
            test_index=np.asarray([8, 9], dtype=np.int64),
        )

    def test_supported_models_produce_event_scalar(self) -> None:
        for model_id in ("window_replicated", "bag_static", "prior_ordD_uniform", "dynamic_kernel"):
            model = DailyAffectRegressionModel(DailyAffectRegressionConfig(model_id=model_id, hidden_dim=16, dropout=0.0))
            output = model(torch.randn(3, 23, 4, 256), torch.ones(3, 23, 4, dtype=torch.bool))
            self.assertEqual(output["prediction"].shape, (3,))
            if model_id == "window_replicated":
                self.assertEqual(output["window_prediction"].shape, (3, 23))

    def test_window_mse_is_event_balanced(self) -> None:
        prediction, target = torch.tensor([[0.0, 2.0], [0.0, 0.0]]), torch.tensor([1.0, 2.0])
        mask = torch.tensor([[True, True], [True, False]])
        self.assertAlmostEqual(event_balanced_window_mse(prediction, target, mask).item(), 2.5)

    def test_scalar_runs_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for model_id in ("window_replicated", "bag_static"):
                result = run_daily_affect_regression_run(
                    dataset=self.dataset(root), protocol="synthetic", condition_id=model_id, model_id=model_id, normalization="shared",
                    seed=7, run_dir=root / model_id, epochs=1, batch_size=6, learning_rate=1e-3, weight_decay=0.0,
                    dropout=0.0, hidden_dim=16, patience=1, modality_dropout_prob=0.0, device="cpu",
                )
                self.assertEqual(result["test"]["count"], 2)
                with np.load(root / model_id / "predictions.npz") as loaded:
                    self.assertEqual(loaded["test_event_prediction"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
