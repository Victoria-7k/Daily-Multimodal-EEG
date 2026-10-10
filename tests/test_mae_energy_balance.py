"""Robust fixed channel statistics and equal-window reconstruction gates."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from daily_multimodal.training import modality_mae as mae


class MAEEnergyBalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_robust_fit_keeps_typical_scale_and_excludes_holdout(self):
        pattern = np.tile(np.array([-1., 1.], dtype=np.float32), 1000)
        raw = np.broadcast_to(pattern[None, :, None], (7, 2000, 59)).copy()
        raw[4] *= 10000
        train = np.arange(5)
        norm = mae.fit_signal_normalization('eeg', (raw,), train, batch_size=2, robust=True)
        np.testing.assert_array_equal(norm['branches']['eeg']['mean'], np.zeros(59))
        np.testing.assert_array_equal(norm['branches']['eeg']['scale'], np.ones(59))
        changed = raw.copy(); changed[5:] = np.nan
        self.assertEqual(norm, mae.fit_signal_normalization('eeg', (changed,), train, batch_size=1, robust=True))
        legacy = mae.fit_signal_normalization('eeg', (raw,), train)
        self.assertGreater(min(legacy['branches']['eeg']['scale']), 4000)
        self.assertEqual(norm['mode'], 'train_channel_robust')
        self.assertEqual(norm['version'], mae.MAE_ROBUST_PREPROCESSING_VERSION)
        self.assertEqual(norm['branches']['eeg']['sample_count_per_channel'], 10000)
        self.assertGreater(float(np.abs(mae.eeg_to_patches(raw[4:5], norm)).max()), 9000)

    def test_robust_wear_axes_keep_levels_and_scale_independently(self):
        pattern = np.tile(np.array([-1., 1.], np.float32), 625)
        p = np.tile(pattern, (5, 1))
        e = np.tile(pattern[:400], (5, 1))[:, None, :] * 2 + 5
        a = np.tile(pattern[:300], (5, 3, 1)) * np.array([1, 4, 20])[None, :, None] + np.array([10, 50, 200])[None, :, None]
        for value in (p, e, a): value[4] *= 1000
        norm = mae.fit_signal_normalization('wear', (p, e, a), np.arange(5), robust=True)
        np.testing.assert_allclose(norm['branches']['acc']['scale'], [1, 4, 20])
        np.testing.assert_allclose(norm['branches']['acc']['mean'], [10, 50, 200])
        pp, ee, aa = mae.wear_to_patches(p[:1], e[:1], a[:1], norm)
        for value in (pp, ee, aa): np.testing.assert_allclose(np.abs(value), 1)

    def test_balanced_loss_limits_outlier_energy_without_clipping_targets(self):
        target = torch.ones(2, 10, 4); target[1] *= 10000
        prediction = torch.zeros_like(target, requires_grad=True)
        mask = torch.tensor([[True] * 7 + [False] * 3] * 2)
        runtime = mae.MAERuntime(reconstruction_loss='window_energy_balanced_mse_v1')
        loss = mae.reconstruction_loss(prediction, target, mask, runtime)
        self.assertAlmostEqual(float(loss), 1.)
        self.assertGreater(float(mae.reconstruction_loss(prediction, target, mask, mae.MAERuntime())), 1e7)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(float(prediction.grad[:, 7:].abs().max()), 0.)
        self.assertGreater(float(prediction.grad[1, :7].abs().sum()), 0.)
        self.assertEqual(float(target[1].max()), 10000.)

    def test_balanced_loss_floor_handles_flat_and_low_energy_windows(self):
        target = torch.zeros(3, 10, 4); target[1] = 0.1
        prediction = torch.ones_like(target, requires_grad=True)
        mask = torch.tensor([[True] * 7 + [False] * 3] * 3)
        runtime = mae.MAERuntime(reconstruction_loss='window_energy_balanced_mse_v1')
        loss = mae.reconstruction_loss(prediction, target, mask, runtime)
        self.assertAlmostEqual(float(loss), (1 + .81 + 1) / 3, places=6)
        loss.backward(); self.assertTrue(torch.isfinite(prediction.grad).all())
        mask[0] = False
        with self.assertRaisesRegex(ValueError, 'each MAE window'):
            mae.reconstruction_loss(prediction, target, mask, runtime)

    def test_robust_health_rejects_outlier_only_variation(self):
        values = np.ones((100, 16)); values[-1] *= 1000
        self.assertFalse(mae.embedding_health(values)['collapsed'])
        result = mae.embedding_health(values, robust=True)
        self.assertTrue(result['collapsed']); self.assertEqual(result['median_relative_variation'], 0.)
        healthy = np.random.default_rng(10).normal(size=(100, 16))
        self.assertFalse(mae.embedding_health(healthy, robust=True)['collapsed'])

    def test_balanced_validation_reports_matched_zero_baseline_repeatably(self):
        raw = np.random.default_rng(12).normal(size=(6, 2000, 59)).astype(np.float32)
        norm = mae.fit_signal_normalization('eeg', (raw,), np.array([0, 1, 2]), robust=True)
        model = mae.TemporalMaskedAutoencoder(patch_dim=11800, embedding_dim=16, encoder_layers=1, decoder_layers=1, heads=4)
        runtime = mae.MAERuntime(batch_size=2, seed=5, device='cpu', reconstruction_loss='window_energy_balanced_mse_v1', robust_health=True, eeg_masking='channel')
        args = (model, raw, np.array([3, 4, 5]), runtime, torch.device('cpu'))
        first = mae._evaluate_eeg(*args, normalization=norm, details=True)
        self.assertEqual(first, mae._evaluate_eeg(*args, normalization=norm, details=True))
        self.assertAlmostEqual(first['relative_to_zero'], first['masked_nmse'] / first['zero_predictor_nmse'])
        self.assertGreater(first['zero_predictor_nmse'], 0.)

    def test_channel_mask_counts_time_alignment_and_batch_independence(self):
        before = torch.random.get_rng_state().clone()
        mask = mae.eeg_reconstruction_mask(7, .7, torch.device('cpu'), kind='channel', generator=np.random.default_rng(7))
        self.assertEqual(mask.shape, (7, 10, 11800))
        self.assertTrue((mask.sum(dim=(1, 2)) == 41 * 10 * 200).all())
        channels = mask.reshape(7, 10, 59, 200)
        self.assertTrue(torch.equal(channels[:, 0], channels[:, -1]))
        self.assertTrue(torch.equal(channels[..., 0], channels[..., -1]))
        gen = np.random.default_rng(7)
        partitioned = torch.cat([mae.eeg_reconstruction_mask(n, .7, torch.device('cpu'), kind='channel', generator=gen) for n in (3, 4)])
        self.assertTrue(torch.equal(mask, partitioned)); self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_channel_mask_blocks_masked_input_values_and_gradients(self):
        model = mae.TemporalMaskedAutoencoder(patch_dim=12, embedding_dim=16, encoder_layers=1, decoder_layers=1, heads=4).eval()
        values = torch.randn(2, 10, 12, requires_grad=True)
        mask = torch.zeros_like(values, dtype=torch.bool); mask[..., :8] = True
        prediction, _ = model(values, mask)
        changed = values.detach().clone(); changed[mask] += 1000
        other, _ = model(changed, mask)
        torch.testing.assert_close(prediction.detach(), other.detach(), atol=0, rtol=0)
        prediction.square().sum().backward()
        self.assertEqual(float(values.grad[mask].abs().max()), 0.)
        self.assertGreater(float(values.grad[~mask].abs().sum()), 0.)

    def test_element_mask_loss_excludes_observed_targets_and_balances_windows(self):
        target = torch.ones(2, 10, 12); target[1] *= 10000
        mask = torch.zeros_like(target, dtype=torch.bool); mask[..., :8] = True
        prediction = torch.zeros_like(target); prediction[~mask] = -1e6
        runtime = mae.MAERuntime(reconstruction_loss='window_energy_balanced_mse_v1')
        self.assertAlmostEqual(float(mae.reconstruction_loss(prediction, target, mask, runtime)), 1.)


if __name__ == '__main__': unittest.main()
