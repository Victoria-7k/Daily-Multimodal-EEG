"""Behavioral regression gates for label-free Stage-A MAE repairs."""
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

SCRIPTS = Path(os.environ.get("MAE_SCRIPT_ROOT", Path(__file__).resolve().parents[1] / "scripts/multilabel"))


def load_script(filename, name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MAERepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.video = load_script("114_run_video_mae.py", "repaired_video_test")
        cls.prefix = load_script("123_export_mae_frozen_prefix.py", "repaired_prefix_test")

    def setUp(self):
        torch.manual_seed(23)
        self.mask = torch.tensor([[True] * 6 + [False] * 4] * 2)

    def eeg_model(self):
        return mae.TemporalMaskedAutoencoder(patch_dim=12, embedding_dim=16, encoder_layers=3, decoder_layers=1, heads=4).eval()

    def wear_model(self):
        return mae.WearMaskedAutoencoder(embedding_dim=16, encoder_layers=3, decoder_layers=1, heads=4).eval()

    def video_model(self):
        return self.video.VideoMaskedAutoencoder(frames=4, size=32, embedding_dim=16, encoder_layers=3, decoder_layers=1, heads=4).eval()

    def test_eeg_masked_positions_keep_position_and_differ(self):
        model = self.eeg_model(); captured = []
        hook = model.encoder.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
        pred, _ = model(torch.randn(2, 10, 12), self.mask); hook.remove()
        torch.testing.assert_close(captured[0][:, :6], (model.mask_token + model.position[:, :6]).expand(2, -1, -1))
        self.assertGreater(float((pred[:, :6] - pred[:, :1]).abs().max()), 1e-5)

    def test_wear_masked_positions_keep_position_and_differ(self):
        model = self.wear_model(); captured = []
        hook = model.encoder.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
        preds, _ = model(torch.randn(2, 10, 125), torch.randn(2, 10, 40), torch.randn(2, 10, 90), self.mask); hook.remove()
        torch.testing.assert_close(captured[0][:, :6], (model.mask_token + model.position[:, :6]).expand(2, -1, -1))
        for pred in preds:
            self.assertGreater(float((pred[:, :6] - pred[:, :1]).abs().max()), 1e-5)

    def test_eeg_masked_contents_cannot_reach_prediction(self):
        model = self.eeg_model(); x = torch.randn(2, 10, 12, requires_grad=True)
        pred, _ = model(x, self.mask)
        altered = x.detach().clone(); altered[:, :6] += 100
        other, _ = model(altered, self.mask)
        torch.testing.assert_close(pred.detach(), other.detach(), atol=0, rtol=0)
        pred.square().sum().backward()
        self.assertEqual(float(x.grad[:, :6].abs().max()), 0)
        self.assertGreater(float(x.grad[:, 6:].abs().sum()), 0)

    def test_wear_masked_contents_cannot_reach_prediction(self):
        model = self.wear_model(); inputs = [torch.randn(2, 10, size, requires_grad=True) for size in (125, 40, 90)]
        preds, _ = model(*inputs, self.mask)
        altered = [x.detach().clone() for x in inputs]
        for x in altered: x[:, :6] += 100
        others, _ = model(*altered, self.mask)
        for pred, other in zip(preds, others): torch.testing.assert_close(pred.detach(), other.detach(), atol=0, rtol=0)
        sum(p.square().sum() for p in preds).backward()
        for x in inputs:
            self.assertEqual(float(x.grad[:, :6].abs().max()), 0)
            self.assertGreater(float(x.grad[:, 6:].abs().sum()), 0)

    def test_fixed_validation_mask_counts_and_rng_isolation(self):
        before = torch.random.get_rng_state().clone()
        for tokens, ratio, expected in ((10, .7, 7), (10, .6, 6), (196, .9, 176)):
            a = mae.fixed_validation_mask(19, tokens, ratio, np.random.default_rng(8), torch.device("cpu"))
            b = mae.fixed_validation_mask(19, tokens, ratio, np.random.default_rng(8), torch.device("cpu"))
            self.assertTrue(torch.equal(a, b)); self.assertTrue((a.sum(dim=1) == expected).all())
            self.assertFalse(a.all(dim=1).any())
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_validation_mask_independent_of_batch_partition(self):
        whole = mae.fixed_validation_mask(11, 10, .6, np.random.default_rng(8), torch.device("cpu"))
        gen = np.random.default_rng(8)
        split = torch.cat([mae.fixed_validation_mask(n, 10, .6, gen, torch.device("cpu")) for n in (4, 4, 3)])
        self.assertTrue(torch.equal(whole, split))

    def test_representative_subset_preserves_membership_and_full_order(self):
        ids = np.array([90, 80, 10, 60, 70, 30, 40, 20, 50])
        np.testing.assert_array_equal(mae.representative_indices(ids, 0), ids)
        np.testing.assert_array_equal(mae.representative_indices(ids, 3), [10, 50, 90])

    def test_health_rejects_old_video_style_near_constant_features(self):
        noise = np.random.default_rng(4).normal(size=(32, 16))
        self.assertTrue(mae.embedding_health(np.ones((32, 16)) + noise * 1.6e-5)["collapsed"])
        self.assertTrue(mae.embedding_health(np.zeros((32, 16)))["collapsed"])
        healthy = mae.embedding_health(noise)
        self.assertFalse(healthy["collapsed"]); self.assertGreater(healthy["effective_rank"], 2)
        with self.assertRaisesRegex(RuntimeError, "collapsed embeddings"):
            mae.require_healthy(mae.embedding_health(np.ones((32, 16))), "unit test")

    def test_nonfinite_gradient_blocks_optimizer_update(self):
        model = torch.nn.Linear(2, 1); before = model.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        hook = model.weight.register_hook(lambda grad: grad * float("nan"))
        with self.assertRaisesRegex(RuntimeError, "non-finite gradients"):
            mae.guarded_optimizer_step(model, optimizer, model(torch.ones(2, 2)).square().mean(), "test")
        hook.remove(); torch.testing.assert_close(model.weight, before, atol=0, rtol=0)
        self.assertFalse(optimizer.state)

    def test_nonfinite_loss_blocks_backward(self):
        model = torch.nn.Linear(2, 1); optimizer = torch.optim.AdamW(model.parameters())
        with self.assertRaisesRegex(RuntimeError, "non-finite tensor"):
            mae.guarded_optimizer_step(model, optimizer, model(torch.ones(2, 2)).sum() * float("inf"), "test")
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_wear_export_keeps_invalid_rows_zero(self):
        rng = np.random.default_rng(4)
        p = rng.normal(size=(6, 1250)).astype(np.float32); e = rng.normal(size=(6, 1, 400)).astype(np.float32); a = rng.normal(size=(6, 3, 300)).astype(np.float32)
        valid = np.array([True, False, True, True, False, True]); model = self.wear_model()
        out = mae.export_wear_embeddings(model, p, e, a, valid, batch_size=2, device="cpu")
        self.assertEqual(out.shape, (6, 16)); self.assertTrue((out[~valid] == 0).all()); self.assertTrue(np.isfinite(out).all())
        with torch.no_grad(): direct = model.encode(*(torch.from_numpy(v) for v in mae.wear_to_patches(p[valid], e[valid], a[valid]))).mean(dim=1).numpy()
        np.testing.assert_allclose(out[valid], direct, atol=1e-6, rtol=1e-6)

    def test_wear_validation_repeatable_with_branch_baselines(self):
        rng = np.random.default_rng(6); p = rng.normal(size=(6, 1250)).astype(np.float32); e = rng.normal(size=(6, 1, 400)).astype(np.float32); a = rng.normal(size=(6, 3, 300)).astype(np.float32)
        model = self.wear_model(); runtime = mae.MAERuntime(batch_size=2, seed=9, mask_ratio=.6, device="cpu")
        before = torch.random.get_rng_state().clone()
        first = mae._evaluate_wear(model, p, e, a, np.arange(6), runtime, torch.device("cpu"), details=True)
        second = mae._evaluate_wear(model, p, e, a, np.arange(6), runtime, torch.device("cpu"), details=True)
        self.assertEqual(first, second); self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(set(first["branch_nmse"]), {"ppg", "eda", "acc"})
        for baseline in first["zero_predictor_nmse"].values(): self.assertAlmostEqual(baseline, 1, places=5)

    def test_eeg_validation_repeatable_and_rng_isolated(self):
        data = np.random.default_rng(6).normal(size=(6, 2000, 59)).astype(np.float32)
        model = mae.TemporalMaskedAutoencoder(patch_dim=11800, embedding_dim=16, encoder_layers=1, decoder_layers=1, heads=4).eval()
        runtime = mae.MAERuntime(batch_size=2, seed=9, mask_ratio=.7, device="cpu")
        before = torch.random.get_rng_state().clone()
        first = mae._evaluate_eeg(model, data, np.arange(6), runtime, torch.device("cpu"))
        second = mae._evaluate_eeg(model, data, np.arange(6), runtime, torch.device("cpu"))
        self.assertEqual(first, second); self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_completed_prefix_rejects_changed_checkpoint_or_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); token = root / "token.npz"; checkpoint = root / "checkpoint.pt"
            np.savez(token, sample_id=np.array([f"w{i}" for i in range(4)]), embedding=np.ones((4, 256)), valid_mask=np.ones(4, bool))
            torch.save({"state_dict": {}}, checkpoint)
            (root / "modality_mae.py").write_text("# encoder source fixture\n")
            out = root / "prefix"; out.mkdir(); (out / "PREFIX_COMPLETE").touch()
            (out / "manifest.json").write_text(json.dumps({"initialization": "pretrained", "modality": "eeg", "checkpoint_sha256": "stale"}))
            argv = ["123", "--modality", "eeg", "--initialization", "pretrained", "--checkpoint", str(checkpoint), "--token-file", str(token), "--out-dir", str(out), "--device", "cpu"]
            with patch.object(sys, "argv", argv), patch.object(self.prefix, "__file__", str(root / "123.py")):
                with self.assertRaisesRegex(ValueError, "provenance mismatch"): self.prefix.main()

    def test_video_validation_repeatable_and_rng_isolated(self):
        model = self.video_model(); rng = np.random.default_rng(9)
        cache = rng.integers(0, 256, size=(6, 3, 4, 32, 32), dtype=np.uint8)
        runtime = self.video.Runtime(1, 2, 1e-4, 1, .9, 11, "cpu")
        before = torch.random.get_rng_state().clone()
        first = self.video._loss(model, {}, np.arange(6), runtime, torch.device("cpu"), frames=4, size=32, cache=cache)
        second = self.video._loss(model, {}, np.arange(6), runtime, torch.device("cpu"), frames=4, size=32, cache=cache)
        self.assertEqual(first, second); self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_video_main_seeds_before_model_initialization(self):
        captured = []
        def fake_train(model, *args, **kwargs):
            captured.append(model.patch_embed.weight.detach().clone())
            return model, {"best_epoch": 1, "best_val_masked_nmse": 1.0}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / "raw.mp4"; raw.touch(); meta = root / "metadata.npz"
            np.savez(meta, sample_id=np.array([f"w{i}" for i in range(6)]), source_video_file=np.array([str(raw)] * 6), clip_start_seconds=np.zeros(6), clip_end_seconds=np.ones(6), video_mask=np.ones(6, bool))
            splits = root / "splits/cross_day"; splits.mkdir(parents=True)
            for name, idx in (("pretrain", [0, 1]), ("finetune", [2, 3]), ("val", [4]), ("test", [5])): (splits / f"{name}.json").write_text(json.dumps(idx))
            for run in range(2):
                torch.manual_seed(50 + run)
                argv = ["114", "--video-metadata", str(meta), "--splits-root", str(root / "splits"), "--protocol", "cross_day", "--out-root", str(root / str(run)), "--device", "cpu", "--seed", "99", "--frames", "4", "--size", "32", "--embedding-dim", "16", "--encoder-layers", "1", "--decoder-layers", "1", "--heads", "4", "--skip-full-export"]
                with patch.object(sys, "argv", argv), patch.object(self.video, "_train", fake_train): self.video.main()
        torch.testing.assert_close(captured[0], captured[1], atol=0, rtol=0)

    def test_video_masked_pixels_do_not_leak(self):
        model = self.video_model(); video = torch.rand(2, 3, 4, 32, 32)
        mask = self.video._mask(2, model.token_count, .75, torch.device("cpu"))
        pixels = mask.reshape(2, 1, 2, 1, 2, 1, 2, 1).expand(-1, 3, -1, 2, -1, 16, -1, 16).reshape_as(video)
        with torch.no_grad():
            pred, target = model(video, mask); other, other_target = model(torch.where(pixels, 1 - video, video), mask)
        torch.testing.assert_close(pred, other, atol=0, rtol=0)
        self.assertGreater(float((target - other_target).abs().max()), 1)
        self.assertGreater(float((pred[0, mask[0]] - pred[0, mask[0]][0]).abs().max()), 1e-5)

    def test_complete_cache_reuse_is_read_only_and_checks_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mmap"; shape = (6, 3, 4, 32, 32)
            np.zeros(shape, np.uint8).tofile(path)
            manifest = path.with_suffix(".mmap.json")
            status = {"complete": True, "shape": list(shape), "dtype": "uint8", "completed_indices": [0, 1, 2, 3, 4], "decode_failures": [{"index": 5, "sample_id": "w5"}]}
            manifest.write_text(json.dumps(status)); before = (path.stat().st_mtime_ns, manifest.stat().st_mtime_ns, manifest.read_bytes())
            valid, audit = self.video._reuse_complete_cache({"sample_id": np.array([f"w{i}" for i in range(6)])}, np.ones(6, bool), frames=4, size=32, path=path)
            self.assertEqual(int(valid.sum()), 5); self.assertTrue(audit["cache_reused_read_only"])
            self.assertEqual(before, (path.stat().st_mtime_ns, manifest.stat().st_mtime_ns, manifest.read_bytes()))
            status["completed_indices"].remove(2); manifest.write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError, "undecoded valid row"):
                self.video._reuse_complete_cache({"sample_id": np.array([f"w{i}" for i in range(6)])}, np.ones(6, bool), frames=4, size=32, path=path)

    def test_all_modalities_prefix_tail_compose_after_repair(self):
        models = [self.eeg_model(), self.wear_model(), self.video_model()]
        hidden = [models[0].patch_embed(torch.randn(2, 10, 12)) + models[0].position,
                  models[1]._embed(torch.randn(2, 10, 125), torch.randn(2, 10, 40), torch.randn(2, 10, 90)),
                  models[2]._tokens(torch.rand(2, 3, 4, 32, 32)) + models[2].position]
        with torch.no_grad():
            for model, value in zip(models, hidden):
                tail = self.prefix.make_tail(model).eval(); prefix = value
                for block in model.encoder.layers[:-2]: prefix = block(prefix)
                torch.testing.assert_close(tail(prefix), model.encoder(value), atol=1e-6, rtol=1e-6)


if __name__ == "__main__": unittest.main()
