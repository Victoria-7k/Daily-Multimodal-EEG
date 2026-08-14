import importlib.util
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np


if importlib.util.find_spec("torch") is None:
    fake_torch = types.ModuleType("torch")
    fake_torch.nn = types.SimpleNamespace(Module=object)
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = fake_torch

_ROOT = Path(__file__).resolve().parents[1]
_TOKEN_SCRIPT = _ROOT / "scripts" / "archive_legacy" / "35_prepare_eeg_encoder_prediction_tokens.py"
_FUSION_SCRIPT = _ROOT / "scripts" / "32_run_eegpt_centered_loss.py"
_TOKEN_SPEC = importlib.util.spec_from_file_location("eeg_encoder_prediction_tokens", _TOKEN_SCRIPT)
_FUSION_SPEC = importlib.util.spec_from_file_location("eegpt_centered_loss", _FUSION_SCRIPT)
tokens_script = importlib.util.module_from_spec(_TOKEN_SPEC)
fusion_script = importlib.util.module_from_spec(_FUSION_SPEC)
assert _TOKEN_SPEC.loader is not None
assert _FUSION_SPEC.loader is not None
sys.modules[_TOKEN_SPEC.name] = tokens_script
sys.modules[_FUSION_SPEC.name] = fusion_script
_TOKEN_SPEC.loader.exec_module(tokens_script)
_FUSION_SPEC.loader.exec_module(fusion_script)


class EEGEncoderFusionTokenTests(unittest.TestCase):
    def test_prediction_npz_converts_to_fusion_eeg_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "seed_1.npz"
            target = root / "token.npz"
            np.savez_compressed(
                source,
                train_index=np.asarray([0, 1], dtype=np.int64),
                val_index=np.asarray([2], dtype=np.int64),
                test_index=np.asarray([3], dtype=np.int64),
                train_prediction=np.asarray([0.1, 0.2], dtype=np.float32),
                val_prediction=np.asarray([0.3], dtype=np.float32),
                test_prediction=np.asarray([0.4], dtype=np.float32),
                sample_id=np.asarray(["s0", "s1", "s2", "s3"], dtype=object),
                subject_id=np.asarray(["sub-01"] * 4, dtype=object),
                day_id=np.asarray(["d0"] * 4, dtype=object),
            )

            report = tokens_script.convert_prediction_npz(source, target, embedding_dim=256)

            self.assertEqual(report["mask_sum"], 4)
            with np.load(target, allow_pickle=True) as loaded:
                self.assertEqual(loaded["eeg_emb"].shape, (4, 256))
                np.testing.assert_allclose(loaded["eeg_emb"][:, 0], [0.1, 0.2, 0.3, 0.4], rtol=1e-6)
                self.assertTrue(np.all(loaded["eeg_emb"][:, 1:] == 0.0))
                self.assertEqual(loaded["eeg_mask"].tolist(), [1, 1, 1, 1])
                self.assertEqual(loaded["modality_mask"].tolist(), [[1, 0, 0, 0]] * 4)

    def test_video_only_experiment_set_drops_no_video_and_bio_only(self):
        args = Namespace(
            experiment_set="video_only",
            protocols="cross_day,within_subject_day",
            experiments="cross_day:B0_Wphysio_bio_only",
        )

        requested = fusion_script._requested_experiments(args)

        self.assertTrue(requested)
        self.assertTrue(all("no_video" not in experiment and "bio_only" not in experiment for _, experiment in requested))
        self.assertIn(("cross_day", "A1_Wphysio_no_audio"), requested)
        replaced = fusion_script._replace_eeg_branch(
            fusion_script.EXPERIMENT_BRANCHES["A1_Wphysio_no_audio"],
            "eeg_eegpt_partial_ft_v1",
        )
        self.assertEqual(replaced[0], "eeg_eegpt_partial_ft_v1")
        self.assertIn("video_A1", replaced)

    def test_eeg_branch_filename_can_use_256d_token_root(self):
        branch = fusion_script.BRANCHES["eeg_eegpt_frozen_v1"]

        filename = branch.filename.format(
            eeg_token_root="eeg_encoder_256d_tokens",
            protocol="cross_day",
            eeg_seed=240800,
        )

        self.assertEqual(
            filename,
            "eeg_encoder_256d_tokens/cross_day/eegpt_frozen_v1/seed_240800.npz",
        )


if __name__ == "__main__":
    unittest.main()
