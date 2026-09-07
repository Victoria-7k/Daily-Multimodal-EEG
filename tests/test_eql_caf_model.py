import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local runtime.
    torch = None

from daily_multimodal.training.eql_caf import EQLCAFConfig, EQLCAFRegressor


class EQLCAFModelTests(unittest.TestCase):
    def setUp(self):
        if torch is None:
            raise unittest.SkipTest("torch is not installed in this runtime")

    def test_forward_returns_prediction_and_traces(self):
        torch.manual_seed(7)
        model = EQLCAFRegressor(EQLCAFConfig(hidden_dim=32, num_heads=4, dropout=0.0, quality_feature_dim=2))
        tokens = torch.randn(3, 4, 5, 256)
        mask = torch.ones(3, 4, 5, dtype=torch.bool)
        quality = torch.ones(3, 4, 5, 2)
        output = model(tokens, mask, quality)
        self.assertEqual(tuple(output["prediction"].shape), (3,))
        self.assertEqual(tuple(output["modality_gates"].shape), (3, 3))
        self.assertEqual(tuple(output["attention"].shape), (3, 3, 4, 5, 5))

    def test_all_auxiliary_missing_falls_back_to_eeg_prediction(self):
        torch.manual_seed(9)
        model = EQLCAFRegressor(EQLCAFConfig(hidden_dim=32, num_heads=4, dropout=0.0, quality_feature_dim=1))
        tokens = torch.randn(2, 4, 5, 256)
        mask = torch.zeros(2, 4, 5, dtype=torch.bool)
        mask[:, 0, :] = True
        output = model(tokens, mask, torch.ones(2, 4, 5, 1))
        torch.testing.assert_close(output["prediction"], output["eeg_prediction"], rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.all(output["modality_gates"] == 0))

    def test_invalid_hidden_dim_rejected(self):
        with self.assertRaises(ValueError):
            EQLCAFRegressor(EQLCAFConfig(hidden_dim=30, num_heads=4))


if __name__ == "__main__":
    unittest.main()
