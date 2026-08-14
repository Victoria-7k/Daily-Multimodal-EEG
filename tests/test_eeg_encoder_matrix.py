import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_multimodal.training.eeg_encoder_matrix import (
    DEFAULT_PROTOCOLS,
    MatrixRuntime,
    cbramod_acquisition_plan,
    compute_de_features,
    configure_encoder_trainability,
    load_or_compute_de_features,
    load_split_protocols,
    run_eeg_encoder_matrix,
    run_preflight,
)


class EEGEncoderMatrixTests(unittest.TestCase):
    def test_de_features_are_channel_band_matrix_and_finite(self):
        sfreq = 200
        time = np.arange(10 * sfreq, dtype=np.float32) / sfreq
        x = np.zeros((2, 10 * sfreq, 2), dtype=np.float32)
        x[:, :, 0] = np.sin(2 * np.pi * 10.0 * time)
        x[:, :, 1] = np.sin(2 * np.pi * 35.0 * time)

        features = compute_de_features(x, sample_rate_hz=sfreq, seconds_per_window=10, batch_size=1)

        self.assertEqual(features.shape, (2, 10))
        self.assertTrue(np.isfinite(features).all())
        # Layout is five bands per channel group: delta/theta/alpha/beta/gamma for all channels.
        alpha_channel0 = features[0, 2 * 2 + 0]
        gamma_channel0 = features[0, 4 * 2 + 0]
        gamma_channel1 = features[0, 4 * 2 + 1]
        alpha_channel1 = features[0, 2 * 2 + 1]
        self.assertGreater(alpha_channel0, gamma_channel0)
        self.assertGreater(gamma_channel1, alpha_channel1)

    def test_split_loader_uses_all_three_existing_protocols_without_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for protocol in DEFAULT_PROTOCOLS:
                _write_split(root / protocol)

            splits = load_split_protocols(root, DEFAULT_PROTOCOLS, row_count=30)

            self.assertEqual(tuple(splits), DEFAULT_PROTOCOLS)
            for split in splits.values():
                self.assertEqual(split.train.tolist(), [0, 1, 2, 3, 4, 5])
                self.assertEqual(split.val.tolist(), [6, 7])
                self.assertEqual(split.test.tolist(), [8, 9])

    def test_trainability_depth_sets_expected_requires_grad(self):
        encoder = _FakeEncoder(
            [
                "patch_embed.weight",
                "blocks.0.attn.weight",
                "blocks.1.attn.weight",
                "blocks.2.attn.weight",
                "blocks.3.attn.weight",
                "norm.weight",
            ]
        )

        partial = configure_encoder_trainability(encoder, "partial", last_n_blocks=2)

        self.assertEqual(partial["trainable_count"], 3)
        self.assertFalse(encoder.params["blocks.1.attn.weight"].requires_grad)
        self.assertTrue(encoder.params["blocks.2.attn.weight"].requires_grad)
        self.assertTrue(encoder.params["blocks.3.attn.weight"].requires_grad)
        self.assertTrue(encoder.params["norm.weight"].requires_grad)

        frozen = configure_encoder_trainability(encoder, "frozen")
        self.assertEqual(frozen["trainable_count"], 0)

        full = configure_encoder_trainability(encoder, "full")
        self.assertEqual(full["trainable_count"], len(encoder.params))

    def test_preflight_reports_cbramod_acquisition_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root, index_path, splits_root = _write_mock_eeg_project(root)

            result = run_preflight(
                data_root=data_root,
                index_path=index_path,
                splits_root=splits_root,
                protocols=DEFAULT_PROTOCOLS,
                eegpt_frozen_embeddings=None,
            )

            self.assertTrue(result["ok"])
            self.assertIn("CBraMod.from_pretrained", result["cbramod_acquisition"]["primary_source"])
            self.assertIn("braindecode[hub]", result["cbramod_acquisition"]["install_command"])

    def test_de_profile_matrix_smoke_writes_metrics_for_three_protocols(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root, index_path, splits_root = _write_mock_eeg_project(root, row_count=30, channel_count=3)
            out_json = root / "metrics.json"
            out_md = root / "metrics.md"
            embeddings_dir = root / "tokens"

            result = run_eeg_encoder_matrix(
                data_root=data_root,
                index_path=index_path,
                splits_root=splits_root,
                profiles=("eeg_de_5band_1s_avg_v1",),
                protocols=DEFAULT_PROTOCOLS,
                seeds=(123,),
                runtime=MatrixRuntime(epochs=3, hidden_dim=8, batch_size=4, patience=2, device="cpu"),
                out_json=out_json,
                out_md=out_md,
                embeddings_dir=embeddings_dir,
            )

            self.assertEqual(result["run_count"], 3)
            self.assertEqual({row["protocol"] for row in result["results"]}, set(DEFAULT_PROTOCOLS))
            self.assertTrue(all(row["status"] == "ok" for row in result["results"]))
            self.assertTrue(all(row["train_audit"]["history"] for row in result["results"]))
            self.assertIn("train_loss", result["results"][0]["train_audit"]["history"][0])
            self.assertIn("val_loss", result["results"][0]["train_audit"]["history"][0])
            self.assertTrue(out_json.is_file())
            self.assertTrue(out_md.is_file())
            text = out_md.read_text(encoding="utf-8")
            self.assertIn("## Profile Means", text)
            self.assertIn("## CBraMod Acquisition", text)
            token_path = embeddings_dir / "cross_day" / "eeg_de_5band_1s_avg_v1" / "seed_123.npz"
            self.assertTrue(token_path.is_file())
            with np.load(token_path, allow_pickle=True) as loaded:
                self.assertEqual(loaded["eeg_emb"].shape, (30, 256))
                self.assertTrue(np.isfinite(loaded["eeg_emb"]).all())
                self.assertEqual(loaded["eeg_mask"].tolist(), [1] * 30)
                self.assertEqual(loaded["modality_mask"][:, 0].tolist(), [1] * 30)
                self.assertEqual(str(loaded["train_supervision"][0]), "fatigue_supervised_train_val_selected")

    def test_eegpt_frozen_smoke_filters_embedding_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root, index_path, splits_root = _write_mock_eeg_project(root, row_count=30, channel_count=3)
            embedding_path = root / "eegpt_embeddings.npz"
            sample_id = np.asarray([f"eeg_{idx:06d}" for idx in range(30)], dtype=object)
            eeg_emb = np.arange(30 * 5, dtype=np.float32).reshape(30, 5)
            np.savez_compressed(embedding_path, eeg_emb=eeg_emb, sample_id=sample_id)

            result = run_eeg_encoder_matrix(
                data_root=data_root,
                index_path=index_path,
                splits_root=splits_root,
                profiles=("eegpt_frozen_v1",),
                protocols=DEFAULT_PROTOCOLS,
                seeds=(123,),
                eegpt_frozen_embeddings=embedding_path,
                max_rows=12,
                runtime=MatrixRuntime(epochs=3, hidden_dim=8, batch_size=4, patience=2, device="cpu"),
            )

            self.assertFalse(result["profile_errors"])
            self.assertTrue(all(row["status"] == "ok" for row in result["results"]))

    def test_eegpt_frozen_embedding_export_reuses_existing_256d(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root, index_path, splits_root = _write_mock_eeg_project(root, row_count=30, channel_count=3)
            embedding_path = root / "eegpt_embeddings.npz"
            embeddings_dir = root / "tokens"
            sample_id = np.asarray([f"eeg_{idx:06d}" for idx in range(30)], dtype=object)
            eeg_emb = np.arange(30 * 256, dtype=np.float32).reshape(30, 256)
            np.savez_compressed(embedding_path, eeg_emb=eeg_emb, sample_id=sample_id)

            run_eeg_encoder_matrix(
                data_root=data_root,
                index_path=index_path,
                splits_root=splits_root,
                profiles=("eegpt_frozen_v1",),
                protocols=("cross_day",),
                seeds=(123,),
                eegpt_frozen_embeddings=embedding_path,
                runtime=MatrixRuntime(epochs=2, hidden_dim=8, batch_size=4, patience=1, device="cpu"),
                embeddings_dir=embeddings_dir,
            )

            token_path = embeddings_dir / "cross_day" / "eegpt_frozen_v1" / "seed_123.npz"
            with np.load(token_path, allow_pickle=True) as loaded:
                np.testing.assert_allclose(loaded["eeg_emb"], eeg_emb)
                self.assertEqual(str(loaded["train_supervision"][0]), "label_free_frozen_eegpt_existing_256d")

    def test_de_cache_recomputes_when_cached_row_count_mismatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "de_cache.npz"
            np.savez_compressed(cache_path, features=np.ones((1, 10), dtype=np.float32))
            x = np.zeros((2, 2000, 2), dtype=np.float32)

            features = load_or_compute_de_features(
                x,
                cache_path=cache_path,
                sample_rate_hz=200.0,
                seconds_per_window=10,
                batch_size=1,
            )

            self.assertEqual(features.shape, (2, 10))
            with np.load(cache_path) as loaded:
                self.assertEqual(loaded["features"].shape, (2, 10))

    def test_cbramod_plan_prefers_explicit_local_cache(self):
        plan = cbramod_acquisition_plan("outputs/checkpoints/cbramod")
        self.assertIn("huggingface-cli download braindecode/cbramod-pretrained", plan.download_command)
        self.assertTrue(plan.explicit_download_required_by_default)


class _FakeParam:
    def __init__(self) -> None:
        self.requires_grad = True


class _FakeEncoder:
    def __init__(self, names: list[str]) -> None:
        self.params = {name: _FakeParam() for name in names}

    def named_parameters(self):
        return list(self.params.items())


def _write_mock_eeg_project(root: Path, *, row_count: int = 30, channel_count: int = 4):
    data_root = root / "data"
    data_root.mkdir()
    rng = np.random.default_rng(7)
    x = rng.normal(size=(row_count, 2000, channel_count)).astype(np.float32)
    y = rng.normal(size=(row_count, 11)).astype(np.float32)
    y[:, 10] = np.linspace(-1.0, 1.0, row_count, dtype=np.float32)
    sub = np.asarray([(idx % 3) + 1 for idx in range(row_count)], dtype=np.int64)
    day = np.asarray([idx // 3 for idx in range(row_count)], dtype=np.int64)
    np.save(data_root / "X.npy", x)
    np.save(data_root / "y.npy", y)
    np.save(data_root / "sub.npy", sub)
    np.save(data_root / "d.npy", day)
    index_path = root / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for idx in range(row_count):
            handle.write(json.dumps({"sample_id": f"eeg_{idx:06d}", "subject_id": f"sub-{sub[idx]:02d}", "day_id": str(day[idx])}) + "\n")
    splits_root = root / "splits"
    for protocol in DEFAULT_PROTOCOLS:
        _write_split(splits_root / protocol)
    return data_root, index_path, splits_root


def _write_split(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "pretrain": [0, 1, 2],
        "finetune": [3, 4, 5],
        "val": [6, 7],
        "test": [8, 9],
    }
    for name, values in payloads.items():
        (root / f"{name}.json").write_text(json.dumps(values), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
