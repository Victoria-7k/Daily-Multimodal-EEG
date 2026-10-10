"""C's masked objective, legacy equivalence and batch recovery contracts."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from daily_multimodal.training.event_loss_adaptation import event_consistency_loss, train_event_loss
from daily_multimodal.training.mae_emotion_adaptation import state_hash
from daily_multimodal.training.structure_emotion import run_condition

class EventLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def output(self, p, mask):
        valid = mask.to(p.dtype)
        mean = (p * valid[:, :, None]).sum(dim=1) / valid.sum(dim=1)[:, None]
        return {"window_prediction": p, "window_mask": mask, "prediction": mean}

    def test_lambda1_loss_and_gradient_exactly_legacy_masked_formula(self):
        torch.manual_seed(33)
        for mask in [torch.ones(3, 4, dtype=torch.bool), torch.tensor([[1,0,0,0],[1,1,0,0],[1,1,1,1]], dtype=torch.bool)]:
            p = torch.randn(3, 4, 11, requires_grad=True)
            target = torch.randn(3, 11)
            output = self.output(p, mask)
            a = event_consistency_loss(output, target, 1.)
            valid = mask.to(target.dtype)
            error = (p - target[:, None, :]).square()
            b = ((error * valid[:, :, None]).sum(dim=1) / valid.sum(dim=1).clamp_min(1)[:, None]).mean()
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(torch.autograd.grad(a, p, retain_graph=True)[0], torch.autograd.grad(b, p)[0]))

    def test_event_cancellation_weak_weight_single_window_and_constant(self):
        p = torch.tensor([[[2.] * 11, [-2.] * 11, [999.] * 11], [[3.] * 11, [999.] * 11, [999.] * 11]], requires_grad=True)
        mask = torch.tensor([[1,1,0],[1,0,0]], dtype=torch.bool)
        output = self.output(p, mask)
        target = torch.zeros(2, 11)
        self.assertAlmostEqual(float(event_consistency_loss(output, target, 0.).detach()), 4.5)
        self.assertAlmostEqual(float(event_consistency_loss(output, target, .1).detach()), 4.7, places=6)
        self.assertAlmostEqual(float(event_consistency_loss(output, target, 1.).detach()), 6.5)
        gradient = torch.autograd.grad(event_consistency_loss(output, target, .1), p)[0]
        self.assertTrue(torch.equal(gradient[~mask], torch.zeros_like(gradient[~mask])))
        constant = torch.ones(2, 3, 11)
        o = self.output(constant, mask)
        self.assertEqual(float(event_consistency_loss(o, target, 0.)), float(event_consistency_loss(o, target, .1)))
        with self.assertRaises(ValueError):
            event_consistency_loss({**o, "window_mask": torch.zeros_like(mask)}, target, .1)

    def fixture(self, root):
        rng = np.random.default_rng(77)
        mask = np.ones((8, 23, 4), bool)
        mask[:, :, 3] = False
        ds = SimpleNamespace(tokens=rng.normal(size=(8, 23, 4, 256)).astype(np.float32), modality_mask=mask,
            row_count=8, subject_id=np.asarray(["a"]*4 + ["b"]*4), event_id=np.arange(8).astype(str), day_id=np.arange(8).astype(str),
            bag_path=root / "fixture.npz", source_npz_json='{}', supervision_boundary="test_frozen_tokens",
            split_indices=lambda: {"train":np.arange(4), "val":np.arange(4,8), "test":np.arange(4,8)})
        return ds, rng.normal(size=(8,11)).astype(np.float32)

    def test_lambda1_training_checkpoint_and_history_match_original(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);ds,targets=self.fixture(r)
            train_event_loss(dataset=ds,targets=targets,variant="C_EVENT",seed=240800,out_dir=r/'new',smoke=True,device="cpu",weight_override=1.)
            run_condition(dataset=ds,targets=targets,protocol="cross_day",condition_id="window_attention_regression_full_mean",
                model_id="window_replicated",temporal_policy="uniform",seed=240800,out_dir=r/'old',epochs=3,patience=3,device="cpu")
            a=torch.load(r/'new/best_checkpoint.pt',weights_only=False);b=torch.load(r/'old/best_checkpoint.pt',weights_only=False)
            self.assertEqual(state_hash(a['state_dict']),state_hash(b['state_dict']))
            self.assertEqual(json.loads((r/'new/metrics.json').read_text())['history'],json.loads((r/'old/metrics.json').read_text())['history'])

    def test_batch_recovery_matches_uninterrupted_weak_loss(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);ds,targets=self.fixture(r)
            args=dict(dataset=ds,targets=targets,variant="C_EVENT_WEAK",seed=240801,smoke=True,device="cpu")
            with self.assertRaises(InterruptedError):train_event_loss(**args,out_dir=r/'resumed',interrupt_after_batches=1)
            train_event_loss(**args,out_dir=r/'resumed')
            train_event_loss(**args,out_dir=r/'direct')
            a=torch.load(r/'resumed/best_checkpoint.pt',weights_only=False);b=torch.load(r/'direct/best_checkpoint.pt',weights_only=False)
            self.assertEqual(state_hash(a['state_dict']),state_hash(b['state_dict']))
            self.assertEqual(json.loads((r/'resumed/metrics.json').read_text())['history'],json.loads((r/'direct/metrics.json').read_text())['history'])

if __name__ == "__main__":
    unittest.main()
