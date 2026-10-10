"""Basic representation pooling: masked aggregation, paired initialization and saved inference."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from daily_multimodal.daily_affect.regression import DailyAffectRegressionConfig
from daily_multimodal.daily_affect.training import _seed_everything
from daily_multimodal.training.mae_emotion_adaptation import state_hash
from daily_multimodal.training.structure_emotion import MultiEmotionStructureModel, _predict, run_condition


class BasicEventPoolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.backends.mha.set_fastpath_enabled(True)

    def model(self, kind):
        _seed_everything(240800)
        return MultiEmotionStructureModel(DailyAffectRegressionConfig(model_id=kind, adapter_mode="per_modality", temporal_policy="uniform", hidden_dim=128, dropout=.1))

    def test_same_parameters_and_pool_before_nonlinear_head(self):
        bag, window = self.model("bag_static"), self.model("window_replicated")
        self.assertEqual(state_hash(bag.state_dict()), state_hash(window.state_dict()))
        bag.eval(); window.eval()
        x = torch.randn(3, 23, 4, 256)
        mask = torch.ones(3, 23, 4, dtype=torch.bool)
        mask[:, :, 3] = False
        mask[0, 1:] = False
        mask[1, 7:] = False
        with torch.no_grad():
            output = bag(x, mask)
            expected = bag.head((output["states"] * output["temporal_weights"][:, :, None]).sum(dim=1))
            self.assertTrue(torch.equal(output["prediction"], expected))
            self.assertNotIn("window_prediction", output)
            w = window(x, mask)
            self.assertTrue(torch.allclose(output["states"], w["states"], atol=0, rtol=0))
            self.assertTrue(torch.allclose(output["prediction"][0], w["prediction"][0], atol=1e-7, rtol=1e-6))
            self.assertGreater(float((output["prediction"][1:] - w["prediction"][1:]).abs().max()), 1e-5)
            changed = x.clone(); changed[~mask] = 10000
            self.assertTrue(torch.equal(output["prediction"], bag(changed, mask)["prediction"]))

    def test_event_mse_gradients_and_masked_windows(self):
        model = self.model("bag_static").train()
        x = torch.randn(2, 23, 4, 256, requires_grad=True)
        mask = torch.ones(2, 23, 4, dtype=torch.bool)
        mask[:, :, 3] = False
        mask[0, 3:] = False
        target = torch.randn(2, 11)
        loss = (model(x, mask)["prediction"] - target).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.equal(x.grad[~mask], torch.zeros_like(x.grad[~mask])))
        self.assertGreater(float(x.grad[mask].abs().sum()), 0.)
        for value in model.parameters():
            if value.grad is not None:
                self.assertTrue(torch.isfinite(value.grad).all())

    def test_smoke_selected_checkpoint_replays_validation(self):
        rng = np.random.default_rng(87)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = np.ones((8, 23, 4), bool); mask[:, :, 3] = False
            ds = SimpleNamespace(tokens=rng.normal(size=(8, 23, 4, 256)).astype(np.float32), modality_mask=mask,
                row_count=8, subject_id=np.asarray(["a"]*4 + ["b"]*4), event_id=np.arange(8).astype(str), day_id=np.arange(8).astype(str),
                bag_path=root/'fixture.npz', supervision_boundary="fixture_frozen_tokens",
                split_indices=lambda: {"train":np.arange(4), "val":np.arange(4, 8), "test":np.arange(4, 8)})
            targets = rng.normal(size=(8, 11)).astype(np.float32)
            result = run_condition(dataset=ds, targets=targets, protocol="cross_day", condition_id="C_BAG_UNIFORM",
                model_id="bag_static", temporal_policy="uniform", seed=240800, out_dir=root, epochs=3, patience=3, device="cpu")
            cp = torch.load(root/'best_checkpoint.pt', weights_only=False)
            model = self.model("bag_static"); model.load_state_dict(cp["state_dict"])
            pred = _predict(model, ds, np.arange(4, 8), cp["x_mean"], cp["x_std"], cp["y_mean"], cp["y_std"], torch.device("cpu"))
            with np.load(root/'event_predictions.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(pred, z['val_prediction'])
            self.assertEqual(result["best_val_macro_standardized_rmse"], min(row["val_macro_standardized_rmse"] for row in result["history"]))
            np.testing.assert_array_equal(cp["y_mean"], targets[:4].mean(axis=0, keepdims=True))


if __name__ == "__main__":
    unittest.main()
