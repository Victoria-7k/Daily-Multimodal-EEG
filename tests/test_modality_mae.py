import unittest

import numpy as np
import torch

from daily_multimodal.training.modality_mae import TemporalMaskedAutoencoder, WearMaskedAutoencoder, eeg_to_patches, wear_to_patches


class ModalityMAETest(unittest.TestCase):
    def test_eeg_mae_patch_contract_and_forward(self) -> None:
        patches = eeg_to_patches(np.random.default_rng(1).normal(size=(2, 2000, 59)).astype(np.float32))
        self.assertEqual(patches.shape, (2, 10, 11800))
        model = TemporalMaskedAutoencoder(patch_dim=11800, embedding_dim=16, encoder_layers=1, decoder_layers=1, heads=4)
        reconstruction, encoded = model(torch.from_numpy(patches), torch.tensor([[True] * 7 + [False] * 3] * 2))
        self.assertEqual(reconstruction.shape, patches.shape)
        self.assertEqual(encoded.shape, (2, 10, 16))


    def test_wear_mae_patch_contract_and_forward(self) -> None:
        rng = np.random.default_rng(2)
        ppg, eda, acc = wear_to_patches(rng.normal(size=(2, 1250)).astype(np.float32), rng.normal(size=(2, 1, 400)).astype(np.float32), rng.normal(size=(2, 3, 300)).astype(np.float32))
        self.assertEqual((ppg.shape, eda.shape, acc.shape), ((2, 10, 125), (2, 10, 40), (2, 10, 90)))
        model = WearMaskedAutoencoder(embedding_dim=16, encoder_layers=1, decoder_layers=1, heads=4)
        reconstructed, encoded = model(torch.from_numpy(ppg), torch.from_numpy(eda), torch.from_numpy(acc), torch.tensor([[True] * 6 + [False] * 4] * 2))
        self.assertEqual([item.shape for item in reconstructed], [ppg.shape, eda.shape, acc.shape])
        self.assertEqual(encoded.shape, (2, 10, 16))
