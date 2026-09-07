import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local runtime.
    torch = None


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "window_fatigue" / "32_run_eegpt_centered_loss.py"
    spec = importlib.util.spec_from_file_location("run_eegpt_centered_loss", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VARIANTS = ("attention", "concat", "attention_multihead_pma", "eeg_anchor")


class FusionVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None:
            raise unittest.SkipTest("torch is not installed in this runtime")
        cls.mod = _load_script_module()

    def _make_module(self, variant, modality_count=4, hidden_dim=16, head="regression", **kwargs):
        self.mod._seed_everything(7)
        return self.mod.AttentionRegressor(
            modality_count=modality_count,
            hidden_dim=hidden_dim,
            dropout=0.0,
            head=head,
            normalized_label_values=np.linspace(-1.0, 1.0, 5, dtype=np.float32),
            variant=variant,
            **kwargs,
        )

    def test_all_variants_return_scalar_predictions(self):
        tokens = torch.randn(3, 4, 256)
        mask = torch.ones(3, 4, dtype=torch.bool)

        for variant in VARIANTS:
            with self.subTest(variant=variant):
                module = self._make_module(variant)
                prediction = module(tokens, mask)
                self.assertEqual(tuple(prediction.shape), (3,))

    def test_all_variants_work_with_all_heads(self):
        tokens = torch.randn(2, 4, 256)
        mask = torch.ones(2, 4, dtype=torch.bool)

        for variant in VARIANTS:
            for head in ("regression", "ordinal_cumulative", "classification_expectation", "hybrid_regression_ordinal"):
                with self.subTest(variant=variant, head=head):
                    module = self._make_module(variant, head=head)
                    prediction = module(tokens, mask)
                    self.assertEqual(tuple(prediction.shape), (2,))

    def test_concat_handles_variable_modality_count(self):
        for modality_count in (3, 4):
            with self.subTest(modality_count=modality_count):
                module = self._make_module("concat", modality_count=modality_count)
                tokens = torch.randn(2, modality_count, 256)
                mask = torch.ones(2, modality_count, dtype=torch.bool)
                self.assertEqual(tuple(module(tokens, mask).shape), (2,))

    def test_masked_token_values_do_not_affect_output(self):
        rng = np.random.default_rng(3)
        tokens_a = torch.as_tensor(rng.standard_normal((2, 4, 256)), dtype=torch.float32)
        tokens_b = tokens_a.clone()
        tokens_b[:, 2, :] = 12345.0  # masked token index 2 set to arbitrary large values
        mask = torch.ones(2, 4, dtype=torch.bool)
        mask[:, 2] = False

        for variant in VARIANTS:
            with self.subTest(variant=variant):
                module = self._make_module(variant)
                out_a = module(tokens_a, mask)
                out_b = module(tokens_b, mask)
                torch.testing.assert_close(out_a, out_b, rtol=1e-5, atol=1e-6)

    def test_mask_actually_changes_output_when_token_missing(self):
        rng = np.random.default_rng(5)
        tokens = torch.as_tensor(rng.standard_normal((2, 4, 256)), dtype=torch.float32)
        mask_all = torch.ones(2, 4, dtype=torch.bool)
        mask_missing = mask_all.clone()
        mask_missing[:, 2] = False

        for variant in VARIANTS:
            with self.subTest(variant=variant):
                module = self._make_module(variant)
                out_all = module(tokens, mask_all)
                out_missing = module(tokens, mask_missing)
                self.assertFalse(torch.allclose(out_all, out_missing, atol=1e-5),
                                 f"{variant} output unchanged when a modality is masked")

    def test_determinism_same_seed(self):
        tokens = torch.randn(3, 4, 256)
        mask = torch.ones(3, 4, dtype=torch.bool)

        for variant in VARIANTS:
            with self.subTest(variant=variant):
                first = self._make_module(variant)(tokens, mask)
                second = self._make_module(variant)(tokens, mask)
                torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_invalid_variant_rejected(self):
        with self.assertRaises(ValueError):
            self._make_module("bogus")

    def test_token_normalization_shared_pools_modalities(self):
        tokens = np.asarray(
            [
                [[1.0, 10.0], [3.0, 30.0]],
                [[5.0, 50.0], [7.0, 70.0]],
            ],
            dtype=np.float32,
        )
        mask = np.asarray([[True, True], [True, False]])

        mean, std = self.mod._fit_token_normalization(tokens, mask, np.asarray([0, 1]), scope="shared")

        self.assertEqual(mean.shape, (1, 1, 2))
        np.testing.assert_allclose(mean.reshape(-1), [3.0, 30.0])
        np.testing.assert_allclose(std.reshape(-1), [np.std([1.0, 3.0, 5.0]), np.std([10.0, 30.0, 50.0])])

    def test_token_normalization_per_modality_keeps_modality_slots(self):
        tokens = np.asarray(
            [
                [[1.0, 10.0], [3.0, 30.0]],
                [[5.0, 50.0], [7.0, 70.0]],
            ],
            dtype=np.float32,
        )
        mask = np.asarray([[True, True], [True, False]])

        mean, std = self.mod._fit_token_normalization(tokens, mask, np.asarray([0, 1]), scope="per_modality")

        self.assertEqual(mean.shape, (1, 2, 2))
        np.testing.assert_allclose(mean[0], [[3.0, 30.0], [3.0, 30.0]])
        np.testing.assert_allclose(std[0, 0], [2.0, 20.0])
        np.testing.assert_allclose(std[0, 1], [1.0, 1.0])

    def test_pma_requires_divisible_hidden_dim(self):
        with self.assertRaises(ValueError):
            self._make_module("attention_multihead_pma", hidden_dim=17, num_heads=4)

    def test_eeg_anchor_requires_at_least_one_modality(self):
        with self.assertRaises(ValueError):
            self._make_module("eeg_anchor", modality_count=0)

    def test_fit_model_builds_requested_variant(self):
        rng = np.random.default_rng(11)
        n = 24
        tokens = rng.standard_normal((n, 4, 256)).astype(np.float32)
        token_mask = np.ones((n, 4), dtype=bool)
        target = rng.integers(1, 6, size=n).astype(np.float32)
        subjects = np.asarray([f"s{i % 3}" for i in range(n)], dtype=str)
        train_idx = np.asarray(range(0, 16), dtype=np.int64)
        val_idx = np.asarray(range(16, 24), dtype=np.int64)

        model, audit = self.mod._fit_model(
            tokens=tokens,
            token_mask=token_mask,
            target=target,
            subjects=subjects,
            train_idx=train_idx,
            val_idx=val_idx,
            loss_mode="raw",
            centered_lambda=0.0,
            head="regression",
            head_lambda=0.0,
            hidden_dim=16,
            epochs=1,
            batch_size=8,
            learning_rate=1e-3,
            weight_decay=0.0,
            dropout=0.0,
            patience=5,
            seed=7,
            device="cpu",
            subject_balanced_batches=False,
            sampler="default",
            fusion_variant="concat",
        )

        self.assertEqual(model["module"].variant, "concat")
        self.assertIn("best_epoch", audit)


if __name__ == "__main__":
    unittest.main()
