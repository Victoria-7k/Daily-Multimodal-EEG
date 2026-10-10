"""Scientific gates for EEG-only head matching, tail freezing and recovery."""
from __future__ import annotations

import copy
import unittest

import numpy as np
import torch
from torch import nn

from daily_multimodal.training.mae_emotion_adaptation import (
    EEGEmotionModel, restore_rng, rng_state, state_hash, window_loss,
)


class AdaptationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)

    def setup_pair(self):
        torch.manual_seed(7)
        layer = nn.TransformerEncoderLayer(256, 8, 1024, batch_first=True, activation="gelu")
        tail = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False).eval()
        prefix = np.random.default_rng(42).normal(size=(46, 10, 256)).astype(np.float32)
        with torch.no_grad():
            tokens = tail(torch.from_numpy(prefix)).mean(dim=1).numpy()
        payload = {"heads": 8, "state_dict": tail.state_dict()}
        rows = np.arange(46).reshape(2, 23)
        pair = []
        for adapt in (False, True):
            torch.manual_seed(23)
            pair.append(EEGEmotionModel(prefix, payload, tokens, rows,
                                        np.zeros(256, np.float32), np.ones(256, np.float32),
                                        adapt=adapt, device=torch.device("cpu")))
        return pair

    def test_matched_heads_tail_equivalence_and_dropout_stream(self):
        probe, adapt = self.setup_pair()
        self.assertEqual(probe.head_initial_sha256, adapt.head_initial_sha256)
        for model in (probe, adapt):
            model.train()
            self.assertFalse(model.tail.training)
            self.assertTrue(model.head.training)
        torch.manual_seed(3)
        a = probe(np.arange(2))
        torch.manual_seed(3)
        b = adapt(np.arange(2))
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)

    def test_only_adapt_tail_updates_and_prefix_remains_exact(self):
        for model in self.setup_pair():
            initial = state_hash(model.tail.state_dict())
            prefix = model.prefix.copy()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            model.train()
            output = model(np.arange(2))
            window_loss(output, torch.ones(2, 11), torch.ones(2, 23)).backward()
            grads = [p.grad for p in model.tail.parameters() if p.grad is not None]
            self.assertEqual(bool(grads), model.adapt)
            if grads:
                self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
            optimizer.step()
            self.assertEqual(initial == state_hash(model.tail.state_dict()), not model.adapt)
            np.testing.assert_array_equal(prefix, model.prefix)

    def test_event_equal_masked_loss_and_empty_event_stop(self):
        pred = torch.tensor([[[1.], [3.]], [[5.], [100.]]])
        target = torch.zeros(2, 1)
        valid = torch.tensor([[1., 1.], [1., 0.]])
        self.assertEqual(float(window_loss(pred, target, valid)), 15.)
        with self.assertRaises(ValueError):
            window_loss(pred, target, torch.tensor([[1., 1.], [0., 0.]]))

    def test_interrupted_optimizer_and_all_rng_recover_exactly(self):
        model = self.setup_pair()[1]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        rng = np.random.default_rng(20)
        def step():
            model.train()
            batch = rng.permutation(2)
            loss = window_loss(model(batch), torch.zeros(2, 11), torch.ones(2, 23))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        step()
        state, opt = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        saved_rng = rng_state(rng)
        step()
        expected = state_hash(model.state_dict())
        model.load_state_dict(state)
        optimizer.load_state_dict(opt)
        restore_rng(saved_rng, rng)
        step()
        self.assertEqual(expected, state_hash(model.state_dict()))


if __name__ == "__main__":
    unittest.main()
