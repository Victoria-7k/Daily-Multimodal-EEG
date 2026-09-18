from __future__ import annotations

import unittest

import numpy as np

from daily_multimodal.training.eeg_multitask import (
    aggregate_events,
    configure_strict_last_blocks,
    event_equal_window_weights,
)


class _Parameter:
    def __init__(self) -> None:
        self.requires_grad = True


class _Encoder:
    def __init__(self) -> None:
        self.parameters = {
            "target_encoder.patch_embed.proj.weight": _Parameter(),
            "target_encoder.blocks.0.norm1.weight": _Parameter(),
            "target_encoder.blocks.0.attn.proj.weight": _Parameter(),
            "target_encoder.blocks.6.norm1.weight": _Parameter(),
            "target_encoder.blocks.6.attn.proj.weight": _Parameter(),
            "target_encoder.blocks.7.norm1.weight": _Parameter(),
            "target_encoder.blocks.7.attn.proj.weight": _Parameter(),
            "target_encoder.norm.weight": _Parameter(),
        }

    def named_parameters(self):
        return list(self.parameters.items())


class EEGMultitaskTests(unittest.TestCase):
    def test_strict_partial_ft_does_not_match_early_block_proj_or_norm(self) -> None:
        encoder = _Encoder()
        report = configure_strict_last_blocks(encoder, last_n_blocks=2)
        self.assertEqual(report["kept_block_indices"], [6, 7])
        self.assertFalse(encoder.parameters["target_encoder.blocks.0.norm1.weight"].requires_grad)
        self.assertFalse(encoder.parameters["target_encoder.blocks.0.attn.proj.weight"].requires_grad)
        self.assertFalse(encoder.parameters["target_encoder.patch_embed.proj.weight"].requires_grad)
        self.assertTrue(encoder.parameters["target_encoder.blocks.6.norm1.weight"].requires_grad)
        self.assertTrue(encoder.parameters["target_encoder.blocks.7.attn.proj.weight"].requires_grad)
        self.assertTrue(encoder.parameters["target_encoder.norm.weight"].requires_grad)

    def test_event_weights_and_aggregation_keep_events_equal(self) -> None:
        event_id = np.asarray(["a", "a", "b"])
        weights = event_equal_window_weights(event_id)
        self.assertTrue(np.allclose(weights, [0.5, 0.5, 1.0]))
        arrays = aggregate_events(
            np.asarray([[2.0], [2.0], [4.0]], dtype=np.float32),
            np.asarray([[1.0], [3.0], [5.0]], dtype=np.float32),
            event_id,
            np.asarray(["s1", "s1", "s2"]),
            np.asarray(["d1", "d1", "d2"]),
        )
        self.assertEqual(arrays["event_id"].tolist(), ["a", "b"])
        self.assertTrue(np.allclose(arrays["prediction"].reshape(-1), [2.0, 5.0]))
        self.assertTrue(np.allclose(arrays["target"].reshape(-1), [2.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
