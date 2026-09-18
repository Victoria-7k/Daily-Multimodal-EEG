from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
    from daily_multimodal.daily_affect.regression import DailyAffectRegressionConfig
    from daily_multimodal.training.structure_emotion import MultiEmotionStructureModel, conditions
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class StructureEmotionTests(unittest.TestCase):
    def test_all_conditions_produce_eleven_finite_predictions(self) -> None:
        torch.manual_seed(4)
        tokens = torch.randn(2, 23, 4, 256)
        mask = torch.ones(2, 23, 4, dtype=torch.bool)
        mask[:, :, 3] = False
        mask[0, 0, 2] = False
        for condition_id, (model_id, policy) in conditions().items():
            with self.subTest(condition_id=condition_id):
                model = MultiEmotionStructureModel(DailyAffectRegressionConfig(
                    model_id=model_id, temporal_policy=policy, adapter_mode="per_modality",
                ))
                result = model(tokens, mask)
                self.assertEqual(tuple(result["prediction"].shape), (2, 11))
                self.assertTrue(torch.isfinite(result["prediction"]).all())
                if model.uses_window_regression:
                    self.assertEqual(tuple(result["window_prediction"].shape), (2, 23, 11))
                    expected = (result["window_prediction"] * result["temporal_weights"][:, :, None]).sum(dim=1)
                    self.assertTrue(torch.allclose(result["prediction"], expected, atol=1e-6))

    def test_eleven_heads_are_independent_and_receive_gradients(self) -> None:
        model = MultiEmotionStructureModel(DailyAffectRegressionConfig(model_id="bag_static"))
        tokens = torch.randn(3, 23, 4, 256)
        mask = torch.ones(3, 23, 4, dtype=torch.bool)
        model(tokens, mask)["prediction"].sum().backward()
        heads = model.head.heads
        self.assertEqual(len(heads), 11)
        ids = [id(heads[label][0].weight) for label in heads]
        self.assertEqual(len(set(ids)), 11)
        for label in heads:
            self.assertIsNotNone(heads[label][0].weight.grad)


if __name__ == "__main__":
    unittest.main()
