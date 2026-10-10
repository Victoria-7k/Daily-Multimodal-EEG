"""CPU contract checks for frozen combinations and true encoder-tail gradients."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import torch
except ImportError:
    torch = None

SCRIPTS = Path(os.environ.get("MAE_SCRIPT_ROOT", Path(__file__).resolve().parents[1] / "scripts/multilabel"))


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@unittest.skipIf(torch is None, "torch unavailable")
class RemainingMAEContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.frozen = module("118_run_mae_mt11_event_ablation.py", "remaining_frozen")
        cls.ft = module("124_run_mae_partial_ft.py", "remaining_ft")
        cls.prefix = module("123_export_mae_frozen_prefix.py", "remaining_prefix")

    def test_replacements_preserve_unselected_slots(self):
        ids = np.asarray([f"window_{i}" for i in range(46)])
        rng = np.random.default_rng(7)
        tokens = rng.normal(size=(2, 23, 4, 256)).astype(np.float32)
        masks = np.ones((2, 23, 4), dtype=np.int8)
        masks[:, :, 3] = 0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "reference.npz"
            sources = {"eeg_eegpt_partial_ft_multitask_11label_v1": "old_eeg",
                       "wear_physio": "old_wear", "video_A1": "old_video"}
            np.savez(source, tokens=tokens, modality_mask=masks, sample_id_matrix=ids.reshape(2, 23),
                     source_npz_json=np.asarray(json.dumps(sources)))
            replacements = {}
            for slot, name in enumerate(("eeg", "wear", "video")):
                replacements[name] = root / f"{name}.npz"
                embedding = np.full((46, 256), slot + 10, dtype=np.float32)
                valid = np.ones(46, dtype=bool)
                valid[0] = False
                embedding[0] = 0
                np.savez(replacements[name], sample_id=ids, embedding=embedding, valid_mask=valid)
            for condition, slots in (("M3", {0, 2}), ("M4", {1, 2}), ("M5-F", {0, 1, 2})):
                target = root / f"{condition}.npz"
                self.frozen._write_replaced_bag(reference=source, mae_paths=replacements,
                                                destination=target, condition=condition)
                self.assertEqual({r[1] for r in self.frozen._replacements(condition)}, slots)
                with np.load(target) as result:
                    for slot in range(4):
                        if slot not in slots:
                            np.testing.assert_array_equal(result["tokens"][:, :, slot], tokens[:, :, slot])
                            np.testing.assert_array_equal(result["modality_mask"][:, :, slot], masks[:, :, slot])
                        else:
                            self.assertEqual(result["modality_mask"][0, 0, slot], 0)
                            self.assertTrue((result["tokens"][0, 0, slot] == 0).all())
                    self.assertFalse(result["modality_mask"][:, :, 3].any())

    def test_prefix_tail_exact_composition(self):
        torch.manual_seed(7)
        base = torch.nn.Module()
        base.encoder = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(
            8, 2, 32, batch_first=True, activation="gelu"), 6, enable_nested_tensor=False)
        base.eval()
        tail = self.prefix.make_tail(base).eval()
        x = torch.randn(3, 10, 8)
        with torch.no_grad():
            prefix = x
            for block in base.encoder.layers[:-2]:
                prefix = block(prefix)
            torch.testing.assert_close(tail(prefix), base.encoder(x), atol=1e-6, rtol=1e-6)

    def test_prefix_export_matches_recorded_stage_a_batch_size(self):
        value = self.prefix.export_batch_size(64, match_stage_a=True, modality="eeg", initialization="pretrained",
                                             checkpoint_payload={"manifest": {"runtime": {"batch_size": 128}}})
        self.assertEqual(value, 128)
        self.assertEqual(self.prefix.export_batch_size(64, match_stage_a=False, modality="eeg",
                         initialization="random", checkpoint_payload=None), 64)

    def test_prefix_batch_matching_rejects_missing_or_invalid_metadata(self):
        for payload in ({}, {"manifest": {"runtime": {"batch_size": 0}}}, {"manifest": {"runtime": {"batch_size": True}}}):
            with self.assertRaisesRegex(ValueError, "positive runtime batch_size"):
                self.prefix.export_batch_size(64, match_stage_a=True, modality="wear", initialization="pretrained", checkpoint_payload=payload)
        with self.assertRaisesRegex(ValueError, "pretrained EEG/Wear"):
            self.prefix.export_batch_size(64, match_stage_a=True, modality="video", initialization="pretrained", checkpoint_payload={})

    def test_normalization_final_merge_accepts_preprocessing_argument(self):
        queue = (SCRIPTS / "132_queue_mae_train_channel_normalization.sh").read_text()
        entry = '"$PY" - "$OUT" "$PREVIOUS/downstream_v2" "$PREPROCESSING_VERSION" <<\'PY\'\n'
        program = queue.split(entry, 1)[1].split("\nPY\n", 1)[0]
        protocols = ("cross_day", "within_subject_day")
        seeds = (240800, 240801, 240802)

        def rows(conditions):
            result = []
            for protocol in protocols:
                for condition in conditions:
                    for index, seed in enumerate(seeds):
                        per_label = {label: {"raw_r": (index + 1) / 10}
                                     for label in self.frozen.LABEL_NAMES}
                        metric = {"status": "ok", "val": {"per_label": per_label},
                                  "test": {"per_label": per_label}}
                        result.append({"protocol": protocol, "condition": condition,
                                       "seed": seed, "metrics": metric})
            return result

        with tempfile.TemporaryDirectory() as tmp:
            out, old = Path(tmp) / "out", Path(tmp) / "old"
            groups = {"single": ("B0", "E1", "W1"),
                      "pairs": ("B0", "M2", "M3", "M4"),
                      "all": ("B0", "M5-F"), "partial": ("B0", "M5-FT", "R0")}
            for group, conditions in groups.items():
                directory = out / group / "formal"
                directory.mkdir(parents=True)
                (directory / "results.json").write_text(json.dumps({"results": rows(conditions)}))
            previous = old / "single/formal"
            previous.mkdir(parents=True)
            (previous / "results.json").write_text(json.dumps({"results": rows(("B0", "V1"))}))
            version = "train_channel_robust_zscore_v1"
            run = subprocess.run([sys.executable, "-", str(out), str(old), version],
                                 input=program, text=True, capture_output=True,
                                 cwd=SCRIPTS.parents[1], check=False)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("new_rows=48 reused_B0_V1_rows=12 total=60", run.stdout)
            merged = json.loads((out / "results.json").read_text())
            self.assertEqual(len(merged["results"]), 60)
            self.assertEqual(merged["raw_normalization"], f"EEG_Wear_{version}")
            self.assertAlmostEqual(merged["macro_summary"]["cross_day"]["M5-FT"]["mean"], 0.2)
            self.assertAlmostEqual(merged["macro_summary"]["cross_day"]["M5-FT"]["sample_sd"], 0.1)
            self.assertIn("0.2000 ± 0.1000", (out / "raw_r_tables.md").read_text(encoding="utf-8"))

    def test_all_three_tails_get_supervised_gradients(self):
        ids = np.asarray([f"window_{i}" for i in range(46)])
        mask = np.ones((2, 23, 4), dtype=bool)
        mask[:, :, 3] = False
        with tempfile.TemporaryDirectory() as tmp:
            directories = {}
            for name, heads in (("eeg", 8), ("wear", 4), ("video", 8)):
                directory = Path(tmp) / name
                directory.mkdir()
                directories[name] = directory
                layer = torch.nn.TransformerEncoderLayer(256, heads, 1024, batch_first=True, activation="gelu")
                tail = torch.nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
                torch.save({"heads": heads, "state_dict": tail.state_dict()}, directory / "tail.pt")
                prefix = np.random.default_rng(8).normal(size=(46, 2, 256)).astype(np.float32)
                np.save(directory / "prefix.npy", prefix)
                np.savez(directory / "window_embeddings.npz", sample_id=ids, valid_mask=np.ones(46, bool),
                         valid_indices=np.arange(46))
                (directory / "manifest.json").write_text(json.dumps({"prefix_shape": list(prefix.shape), "initialization": "random"}))
                (directory / "PREFIX_COMPLETE").touch()
            dataset = SimpleNamespace(sample_id_matrix=ids.reshape(2, 23), modality_mask=mask)
            model = self.ft.PartialFTModel(dataset, directories, np.zeros((1, 1, 4, 256), np.float32),
                                           np.ones((1, 1, 4, 256), np.float32), torch.device("cpu"), chunk_size=8)
            model.train()
            self.assertTrue(model.training)
            self.assertTrue(model.head.training)
            for tail in model.tails.values():
                self.assertFalse(tail.training)
                self.assertTrue(all(p.requires_grad for p in tail.parameters()))
                prefix_values = model.prefixes['video'][:3]
                with torch.no_grad():
                    torch.testing.assert_close(tail(prefix_values), tail(prefix_values), atol=0, rtol=0)
            model.eval()
            with torch.no_grad():
                tokens, mask_values = model.encode_events(np.asarray([0, 1]))
                direct = model.head(tokens, mask_values)['prediction']
                stable = model(np.asarray([0, 1]))['prediction']
                torch.testing.assert_close(stable, direct, atol=1e-5, rtol=1e-5)
            model.train()
            before = {m: next(t.parameters()).detach().clone() for m, t in model.tails.items()}
            output = model(np.asarray([0, 1]))
            target = torch.randn(2, 1, 11)
            (output["window_prediction"] - target).square().mean().backward()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            optimizer.step()
            for name, tail in model.tails.items():
                gradient = sum(float(p.grad.abs().sum()) for p in tail.parameters() if p.grad is not None)
                self.assertGreater(gradient, 0)
                self.assertFalse(torch.equal(before[name], next(tail.parameters())))
                self.assertFalse(model.prefixes[name].requires_grad)
                self.assertNotIn(f"prefixes.{name}", model.state_dict())

    def test_nonfinite_gradients_rejected_before_optimizer_step(self):
        model = torch.nn.Linear(2, 1)
        before = model.weight.detach().clone()
        model.weight.grad = torch.full_like(model.weight, float('nan'))
        with self.assertRaisesRegex(RuntimeError, 'non-finite gradients'):
            self.ft.require_finite_parameters(model, gradients=True, context='test bad gradient')
        torch.testing.assert_close(model.weight, before)
        with self.assertRaisesRegex(RuntimeError, 'non-finite tensor'):
            self.ft.require_finite_tensor(torch.tensor(float('inf')), 'test bad loss')
        model.weight.grad = torch.ones_like(model.weight)
        self.ft.require_finite_parameters(model, gradients=True, context='test finite gradient')


if __name__ == "__main__":
    unittest.main()
