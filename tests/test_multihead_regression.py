import importlib.util
import unittest

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from daily_multimodal.training.multihead_regression import (
        LABEL_NAMES,
        EventAwareMultiLabelRegressor,
        EventAwareScalarRegressor,
        event_equal_window_weights,
        event_level_arrays,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class MultiheadRegressionTests(unittest.TestCase):
    def test_h1_has_independent_heads_and_expected_shape(self) -> None:
        model = EventAwareMultiLabelRegressor(modality_count=4, hidden_dim=16, dropout=0.0, head_variant="H1_shared2_11xhead2")
        self.assertEqual(tuple(model.heads.keys()), LABEL_NAMES)
        tokens = torch.randn(3, 4, 256)
        mask = torch.ones(3, 4, dtype=torch.bool)
        output = model(tokens, mask)
        self.assertEqual(tuple(output.shape), (3, 11))
        output.sum().backward()
        for label in LABEL_NAMES:
            self.assertTrue(any(parameter.grad is not None for parameter in model.heads[label].parameters()))

    def test_changing_one_head_changes_only_one_output(self) -> None:
        model = EventAwareMultiLabelRegressor(modality_count=2, hidden_dim=8, dropout=0.0, head_variant="H1_shared2_11xhead2")
        model.eval()
        tokens = torch.randn(2, 2, 256)
        mask = torch.ones(2, 2, dtype=torch.bool)
        before = model(tokens, mask).detach().numpy()
        with torch.no_grad():
            model.heads["fatigue"][-1].bias.add_(1.0)
        after = model(tokens, mask).detach().numpy()
        self.assertTrue(np.allclose(before[:, :-1], after[:, :-1]))
        self.assertTrue(np.allclose(after[:, -1] - before[:, -1], 1.0))

    def test_event_weights_and_aggregation(self) -> None:
        events = np.asarray(["a", "a", "b"])
        weights = event_equal_window_weights(events)
        self.assertTrue(np.allclose(weights, [0.5, 0.5, 1.0]))
        target = np.tile(np.arange(11, dtype=np.float32), (3, 1))
        prediction = target.copy()
        prediction[0] += 2.0
        arrays = event_level_arrays(
            target,
            prediction,
            events,
            np.asarray(["s1", "s1", "s2"]),
            np.asarray(["d1", "d1", "d2"]),
        )
        self.assertEqual(arrays["target"].shape, (2, 11))
        self.assertTrue(np.allclose(arrays["prediction"][0], target[0] + 1.0))

    def test_scalar_st_head_shape_and_gradient(self) -> None:
        model = EventAwareScalarRegressor(modality_count=3, hidden_dim=16, dropout=0.0)
        tokens = torch.randn(4, 3, 256)
        mask = torch.ones(4, 3, dtype=torch.bool)
        output = model(tokens, mask)
        self.assertEqual(tuple(output.shape), (4, 1))
        output.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.head.parameters()))


if __name__ == "__main__":
    unittest.main()
