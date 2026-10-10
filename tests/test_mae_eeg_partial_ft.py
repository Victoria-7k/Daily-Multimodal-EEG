"""Check B's actual cut, random-stream matching, frozen inputs and recovery."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from daily_multimodal.training.mae_eeg_partial_ft import EEGPartialFTModel, array_hash, train_ft
from daily_multimodal.training.mae_emotion_adaptation import state_hash


class PartialFTTests(unittest.TestCase):
    def fixture(self, root):
        torch.set_num_threads(1)
        torch.backends.mha.set_fastpath_enabled(True)
        torch.manual_seed(4)
        layer = torch.nn.TransformerEncoderLayer(256, 8, 1024, batch_first=True, activation="gelu")
        full = torch.nn.TransformerEncoder(layer, 2, enable_nested_tensor=False).eval()
        prefix = np.random.default_rng(5).normal(size=(184, 10, 256)).astype(np.float32)
        with torch.no_grad():
            hidden = full.layers[0](torch.from_numpy(prefix))
            tokens = full.layers[1](hidden).mean(dim=1).numpy()
        dirs = []
        for blocks, values in ((2, prefix), (1, hidden.numpy())):
            d = root / str(blocks)
            d.mkdir()
            tail = copy.deepcopy(full) if blocks == 2 else torch.nn.TransformerEncoder(copy.deepcopy(full.layers[1]), 1, enable_nested_tensor=False)
            torch.save({"blocks": blocks, "heads": 8, "state_dict": tail.state_dict()}, d / "tail.pt")
            np.save(d / "prefix.npy", values)
            np.savez(d / "window_embeddings.npz", sample_id=np.arange(184).astype(str), valid_mask=np.ones(184, bool))
            (d / "manifest.json").write_text(json.dumps({"trainable_tail_blocks": blocks, "prefix_shape": list(values.shape)}))
            (d / "PREFIX_COMPLETE").touch()
            dirs.append(d)
        bag = np.random.default_rng(8).normal(size=(8, 23, 4, 256)).astype(np.float32)
        bag[:, :, 0] = tokens.reshape(8, 23, 256)
        mask = np.ones((8, 23, 4), bool)
        mask[:, :, 3] = False
        ds = SimpleNamespace(tokens=bag, modality_mask=mask, sample_id_matrix=np.arange(184).astype(str).reshape(8, 23),
            subject_id=np.asarray(["a"] * 4 + ["b"] * 4), event_id=np.arange(8).astype(str), day_id=np.arange(8).astype(str),
            bag_path=root / "bag.npz", split_indices=lambda: {"train": np.arange(4), "val": np.arange(4, 6), "test": np.arange(6, 8)})
        cp = root / "source.pt"
        torch.save({"state_dict": {"position": torch.zeros(1, 10, 256)}, "manifest": {"raw_normalization": {}}}, cp)
        return ds, dirs, cp

    def test_cut_equivalence_head_rng_and_dropout_match(self):
        with tempfile.TemporaryDirectory() as t:
            ds, dirs, _ = self.fixture(Path(t))
            outputs, audits, configs = [], [], []
            for d in dirs:
                torch.manual_seed(29)
                model = EEGPartialFTModel(ds, d, np.zeros((1, 1, 4, 256), np.float32), np.ones((1, 1, 4, 256), np.float32), "cpu")
                model.eval()
                with torch.no_grad():
                    np.testing.assert_allclose(model.encode_rows(model.rows[:2].reshape(-1)).numpy(), ds.tokens[:2, :, 0].reshape(-1, 256), atol=2e-4)
                model.train()
                self.assertFalse(model.tail.training)
                self.assertTrue(model.head.training)
                outputs.append(model(np.arange(2), rng=np.random.default_rng(29))["prediction"].detach())
                audits.append(model.first_dropout_audit)
                configs.append((model.head_initial_hash, model.post_head_rng_hash))
            self.assertEqual(configs[0], configs[1])
            self.assertEqual(audits[0], audits[1])
            torch.testing.assert_close(outputs[0], outputs[1], atol=1e-6, rtol=1e-6)

    def test_tail_updates_fixed_inputs_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            ds, dirs, _ = self.fixture(Path(t))
            model = EEGPartialFTModel(ds, dirs[1], np.zeros((1, 1, 4, 256), np.float32), np.ones((1, 1, 4, 256), np.float32), "cpu")
            hashes = [array_hash(x.cpu().numpy()) for x in (model.prefix, model.fixed_tokens, model.native_mask)]
            initial = state_hash(model.tail.state_dict())
            model.train()
            model(np.arange(2))["prediction"].square().mean().backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.tail.parameters() if p.grad is not None), 0)
            torch.optim.AdamW(model.tail.parameters(), lr=1e-5).step()
            self.assertNotEqual(initial, state_hash(model.tail.state_dict()))
            self.assertEqual(hashes, [array_hash(x.cpu().numpy()) for x in (model.prefix, model.fixed_tokens, model.native_mask)])

    def test_batch_recovery_reproduces_uninterrupted_selection(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            ds, dirs, cp = self.fixture(root)
            targets = np.random.default_rng(90).normal(size=(8, 11)).astype(np.float32)
            args = dict(dataset=ds, targets=targets, prefix_dir=dirs[1], source_checkpoint=cp,
                variant="B_T1_LOW", seed=240800, smoke=True, device="cpu")
            with self.assertRaises(InterruptedError):
                train_ft(**args, out_dir=root / "resumed", interrupt_after_batches=1)
            train_ft(**args, out_dir=root / "resumed")
            train_ft(**args, out_dir=root / "direct")
            a = torch.load(root / "resumed/best_checkpoint.pt", weights_only=False)
            b = torch.load(root / "direct/best_checkpoint.pt", weights_only=False)
            self.assertEqual(state_hash(a["state_dict"]), state_hash(b["state_dict"]))
            for k in ("history", "best_epoch", "audit"):
                self.assertEqual(json.loads((root / "resumed/metrics.json").read_text())[k], json.loads((root / "direct/metrics.json").read_text())[k])

if __name__ == "__main__":
    unittest.main()
