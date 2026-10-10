"""Train-only fixed raw-signal normalization and export/prefix consistency."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from daily_multimodal.training import modality_mae as mae


class MAENormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_eeg_statistics_streaming_and_train_only(self):
        raw = np.random.default_rng(7).normal(size=(7, 2000, 59)).astype(np.float32)
        indices = np.array([0, 2, 4])
        norm = mae.fit_signal_normalization("eeg", (raw,), indices, batch_size=1)
        changed = raw.copy(); changed[[1, 3, 5, 6]] = np.nan
        self.assertEqual(norm, mae.fit_signal_normalization("eeg", (changed,), indices, batch_size=1))
        reference = raw[indices].astype(np.float64).reshape(-1, 59)
        stats = norm["branches"]["eeg"]
        np.testing.assert_allclose(stats["mean"], reference.mean(axis=0), atol=1e-13)
        np.testing.assert_allclose(stats["std"], reference.std(axis=0), atol=1e-13)
        other = mae.fit_signal_normalization("eeg", (raw,), indices, batch_size=2)
        np.testing.assert_allclose(other["branches"]["eeg"]["std"], stats["std"], atol=1e-13)
        self.assertEqual(stats["sample_count_per_channel"], 6000)

    def test_eeg_preserves_level_and_amplitude(self):
        rng = np.random.default_rng(8)
        train = rng.normal(size=(3, 2000, 59)).astype(np.float32)
        norm = mae.fit_signal_normalization("eeg", (train,), np.arange(3))
        first = train[:1]; shifted = first * 3 + 10
        legacy = mae.eeg_to_patches(first)
        np.testing.assert_allclose(legacy, mae.eeg_to_patches(shifted), atol=4e-6)
        fixed = mae.eeg_to_patches(first, norm)
        changed = mae.eeg_to_patches(shifted, norm)
        self.assertGreater(float(np.abs(fixed - changed).mean()), 5)
        expected = (first - np.asarray(norm["branches"]["eeg"]["mean"], np.float32)) / np.asarray(norm["branches"]["eeg"]["scale"], np.float32)
        np.testing.assert_array_equal(fixed, expected.reshape(1, 10, 200, 59).transpose(0, 1, 3, 2).reshape(1, 10, -1))

    def test_wear_normalizes_independent_acc_axes_and_ignores_holdout(self):
        rng = np.random.default_rng(9)
        p = rng.normal(size=(5, 1250)).astype(np.float32)
        e = rng.normal(size=(5, 1, 400)).astype(np.float32) * 2 + 5
        a = rng.normal(size=(5, 3, 300)).astype(np.float32) * np.array([1, 4, 20], np.float32)[None, :, None] + np.array([0, 10, 200], np.float32)[None, :, None]
        idx = np.array([0, 2, 3]); norm = mae.fit_signal_normalization("wear", (p, e, a), idx, batch_size=2)
        for name, value, axes in (("ppg", p, (0, 1)), ("eda", e, (0, 2)), ("acc", a, (0, 2))):
            stats = norm["branches"][name]
            np.testing.assert_allclose(stats["mean"], value[idx].astype(np.float64).mean(axis=axes).reshape(-1), atol=1e-12)
            np.testing.assert_allclose(stats["std"], value[idx].astype(np.float64).std(axis=axes).reshape(-1), atol=1e-12)
        altered = [x.copy() for x in (p, e, a)]
        for x in altered: x[[1, 4]] = np.nan
        self.assertEqual(norm, mae.fit_signal_normalization("wear", tuple(altered), idx, batch_size=2))
        pp, ee, aa = mae.wear_to_patches(p[idx], e[idx], a[idx], norm)
        self.assertEqual((pp.shape, ee.shape, aa.shape), ((3, 10, 125), (3, 10, 40), (3, 10, 90)))
        np.testing.assert_allclose(aa.reshape(3, 10, 3, 30).mean(axis=(0, 1, 3)), np.zeros(3), atol=1e-6)
        shifted = mae.wear_to_patches(p[idx] + 10, e[idx] + 10, a[idx] + 10, norm)
        for before, after in zip((pp, ee, aa), shifted): self.assertGreater(float(np.abs(before - after).mean()), .2)

    def test_constant_channels_are_finite_and_invalid_fit_rows_rejected(self):
        raw = np.ones((4, 2000, 59), np.float32) * 5
        norm = mae.fit_signal_normalization("eeg", (raw,), np.array([0, 1]))
        self.assertTrue(np.isfinite(mae.eeg_to_patches(raw, norm)).all())
        self.assertTrue((mae.eeg_to_patches(raw, norm) == 0).all())
        for idx in (np.array([], dtype=int), np.array([0, 0]), np.array([-1]), np.array([4])):
            with self.assertRaises(ValueError): mae.fit_signal_normalization("eeg", (raw,), idx)
        raw[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite raw signal"):
            mae.fit_signal_normalization("eeg", (raw,), np.array([0, 1]))

    def test_eeg_legacy_path_is_exactly_preserved(self):
        raw = np.random.default_rng(10).normal(size=(2, 2000, 59)).astype(np.float32)
        x = raw.reshape(2, 10, 200, 59).transpose(0, 1, 3, 2)
        expected = ((x - x.mean(axis=-1, keepdims=True)) / np.maximum(x.std(axis=-1, keepdims=True), 1e-6)).reshape(2, 10, -1)
        np.testing.assert_array_equal(mae.eeg_to_patches(raw), expected)

    def test_fixed_wear_export_preserves_mask_and_matches_encode(self):
        rng = np.random.default_rng(11)
        p = rng.normal(size=(5, 1250)).astype(np.float32); e = rng.normal(size=(5, 1, 400)).astype(np.float32); a = rng.normal(size=(5, 3, 300)).astype(np.float32)
        valid = np.array([True, False, True, True, False])
        norm = mae.fit_signal_normalization("wear", (p, e, a), np.array([0, 2]))
        model = mae.WearMaskedAutoencoder(embedding_dim=16, encoder_layers=1, decoder_layers=1, heads=4).eval()
        out = mae.export_wear_embeddings(model, p, e, a, valid, batch_size=2, device="cpu", normalization=norm)
        with torch.no_grad(): direct = model.encode(*(torch.from_numpy(x) for x in mae.wear_to_patches(p[valid], e[valid], a[valid], norm))).mean(dim=1).numpy()
        self.assertTrue((out[~valid] == 0).all())
        np.testing.assert_allclose(out[valid], direct, atol=1e-6)

    def test_fixed_validation_is_repeatable(self):
        raw = np.random.default_rng(12).normal(size=(6, 2000, 59)).astype(np.float32)
        norm = mae.fit_signal_normalization("eeg", (raw,), np.array([0, 1, 2]))
        model = mae.TemporalMaskedAutoencoder(patch_dim=11800, embedding_dim=16, encoder_layers=1, decoder_layers=1, heads=4)
        runtime = mae.MAERuntime(batch_size=2, seed=5, device="cpu")
        first = mae._evaluate_eeg(model, raw, np.array([3, 4, 5]), runtime, torch.device("cpu"), normalization=norm)
        second = mae._evaluate_eeg(model, raw, np.array([3, 4, 5]), runtime, torch.device("cpu"), normalization=norm)
        self.assertEqual(first, second)

    def test_fixed_normalization_checkpoint_export_and_prefix_agree(self):
        self._check_checkpoint_export_and_prefix(robust=False)

    def test_robust_normalization_checkpoint_export_and_prefix_agree(self):
        self._check_checkpoint_export_and_prefix(robust=True)

    def _check_checkpoint_export_and_prefix(self, *, robust):
        scripts = Path(os.environ.get("MAE_SCRIPT_ROOT", Path(__file__).resolve().parents[1] / "scripts/multilabel"))
        spec = importlib.util.spec_from_file_location("fixed_norm_prefix_test", scripts / "123_export_mae_frozen_prefix.py")
        prefix = importlib.util.module_from_spec(spec); sys.modules[spec.name] = prefix; spec.loader.exec_module(prefix)
        rng = np.random.default_rng(13)
        raw = rng.normal(size=(4, 2000, 59)).astype(np.float32)
        norm = mae.fit_signal_normalization("eeg", (raw,), np.array([0, 2]), robust=robust)
        torch.manual_seed(31)
        model = mae.TemporalMaskedAutoencoder(patch_dim=11800, embedding_dim=256, encoder_layers=6, decoder_layers=2, heads=8).eval()
        expected = mae.export_eeg_embeddings(model, raw, batch_size=2, device="cpu", normalization=norm)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); checkpoint = root / "checkpoint.pt"; token = root / "token.npz"; source = root / "raw.npy"
            np.save(source, raw); (root / "modality_mae.py").write_text(Path(mae.__file__).read_text(encoding="utf-8"), encoding="utf-8")
            torch.save({"state_dict": model.state_dict(), "manifest": {"raw_normalization": norm, "preprocessing_version": norm['version'], "runtime": {"batch_size": 2}}}, checkpoint)
            np.savez(token, sample_id=np.array([f"w{i}" for i in range(4)]), embedding=expected, valid_mask=np.ones(4, bool))
            argv = ["123", "--modality", "eeg", "--initialization", "pretrained", "--checkpoint", str(checkpoint), "--token-file", str(token), "--out-dir", str(root / "prefix"), "--eeg-source", str(source), "--batch-size", "1", "--match-stage-a-batch-size", "--device", "cpu"]
            with patch.object(sys, "argv", argv), patch.object(prefix, "__file__", str(root / "123.py")):
                prefix.main()
            manifest = json.loads((root / "prefix/manifest.json").read_text())
            self.assertEqual(manifest["raw_normalization"], norm)
            self.assertEqual(manifest["prefix_export_batch_size"], 2)
            self.assertEqual(manifest["requested_batch_size"], 1)
            self.assertTrue(manifest["batch_size_matched_to_stage_a"])
            self.assertLess(manifest["stage_a_embedding_max_abs_error"], 1e-5)
            with np.load(root / "prefix/window_embeddings.npz") as result:
                np.testing.assert_allclose(result["embedding"], expected, atol=1e-5)


if __name__ == "__main__": unittest.main()
