import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_multimodal.training.cross_attention_fusion import (
    FusionBranchSpec,
    FusionDataset,
    FusionExperimentSpec,
    LearnableAttentionConfig,
    build_fusion_dataset,
    fit_learnable_cross_attention,
    predict_with_learnable_cross_attention,
    require_torch_for_learnable_cross_attention,
)
from daily_multimodal.training.fusion_matrix import load_fusion_matrix_config, matrix_experiment_specs


class CrossAttentionFusionDatasetTests(unittest.TestCase):
    def test_build_fusion_dataset_aligns_branches_and_maps_video_from_face_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eeg = root / "eeg.npz"
            wear = root / "wear_physio.npz"
            video = root / "video_v4a_upper.npz"
            audio = root / "audio.npz"
            sample_ids = ["sample-0", "sample-1", "sample-2"]
            _write_branch(eeg, sample_ids=sample_ids, modality="eeg", offset=10.0)
            _write_branch(wear, sample_ids=sample_ids, modality="wear", offset=20.0)
            _write_branch(video, sample_ids=sample_ids, modality="video", offset=30.0)
            _write_branch(audio, sample_ids=sample_ids, modality="audio", offset=40.0)

            dataset = build_fusion_dataset(
                branches={
                    "eeg": FusionBranchSpec(path=eeg, modality="eeg", profile="eeg_current"),
                    "wear": FusionBranchSpec(path=wear, modality="wear", profile="wear_physio_features_preprocessed_v1"),
                    "video": FusionBranchSpec(path=video, modality="video", profile="V4a_upper"),
                    "audio": FusionBranchSpec(path=audio, modality="audio", profile="audio_current"),
                },
                experiment=FusionExperimentSpec(
                    name="fusion_WphysioPre_V4aUpper_full",
                    enabled_modalities=("eeg", "wear", "video", "audio"),
                    target_label="fatigue",
                    min_available_modalities=2,
                ),
            )

        self.assertEqual(dataset.name, "fusion_WphysioPre_V4aUpper_full")
        self.assertEqual(dataset.modalities, ("eeg", "wear", "video", "audio"))
        self.assertEqual(dataset.sample_id.tolist(), sample_ids)
        self.assertEqual(dataset.tokens.shape, (3, 4, 256))
        self.assertEqual(dataset.token_mask.tolist(), [[True, True, True, True]] * 3)
        np.testing.assert_allclose(dataset.tokens[:, 2, 0], np.array([30.0, 31.0, 32.0], dtype=np.float32))
        self.assertEqual(dataset.branch_profiles["video"], "V4a_upper")

    def test_no_audio_and_no_video_keep_the_same_base_rows_when_masks_allow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_ids = ["sample-0", "sample-1", "sample-2", "sample-3"]
            eeg = root / "eeg.npz"
            wear = root / "wear_deep.npz"
            video = root / "video_b1.npz"
            audio = root / "audio.npz"
            _write_branch(eeg, sample_ids=sample_ids, modality="eeg", offset=1.0)
            _write_branch(wear, sample_ids=sample_ids, modality="wear", offset=2.0)
            _write_branch(video, sample_ids=sample_ids, modality="video", offset=3.0)
            _write_branch(audio, sample_ids=sample_ids, modality="audio", offset=4.0)
            branches = {
                "eeg": FusionBranchSpec(path=eeg, modality="eeg", profile="eeg_current"),
                "wear": FusionBranchSpec(path=wear, modality="wear", profile="wear_deep_sequence_preprocessed_v1"),
                "video": FusionBranchSpec(path=video, modality="video", profile="B1"),
                "audio": FusionBranchSpec(path=audio, modality="audio", profile="audio_current"),
            }

            full = build_fusion_dataset(
                branches=branches,
                experiment=FusionExperimentSpec("full", ("eeg", "wear", "video", "audio"), "fatigue"),
            )
            no_audio = build_fusion_dataset(
                branches=branches,
                experiment=FusionExperimentSpec("no_audio", ("eeg", "wear", "video"), "fatigue"),
                base_sample_ids=full.sample_id,
            )
            no_video = build_fusion_dataset(
                branches=branches,
                experiment=FusionExperimentSpec("no_video", ("eeg", "wear", "audio"), "fatigue"),
                base_sample_ids=full.sample_id,
            )

        self.assertEqual(no_audio.sample_id.tolist(), full.sample_id.tolist())
        self.assertEqual(no_video.sample_id.tolist(), full.sample_id.tolist())
        self.assertEqual(no_audio.modalities, ("eeg", "wear", "video"))
        self.assertEqual(no_video.modalities, ("eeg", "wear", "audio"))

    def test_duplicate_sample_ids_are_rejected_before_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eeg = root / "eeg.npz"
            wear = root / "wear.npz"
            _write_branch(eeg, sample_ids=["sample-0", "sample-0"], modality="eeg", offset=1.0)
            _write_branch(wear, sample_ids=["sample-0", "sample-1"], modality="wear", offset=2.0)

            with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                build_fusion_dataset(
                    branches={
                        "eeg": FusionBranchSpec(path=eeg, modality="eeg", profile="eeg_current"),
                        "wear": FusionBranchSpec(path=wear, modality="wear", profile="wear_current"),
                    },
                    experiment=FusionExperimentSpec("bio_only", ("eeg", "wear"), "fatigue"),
                )

    def test_non_target_branches_do_not_need_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eeg = root / "eeg.npz"
            wear = root / "wear.npz"
            sample_ids = ["sample-0", "sample-1", "sample-2"]
            _write_branch(eeg, sample_ids=sample_ids, modality="eeg", offset=1.0)
            _write_branch(wear, sample_ids=sample_ids, modality="wear", offset=2.0, include_labels=False)

            dataset = build_fusion_dataset(
                branches={
                    "eeg": FusionBranchSpec(path=eeg, modality="eeg", profile="eeg_current"),
                    "wear": FusionBranchSpec(path=wear, modality="wear", profile="wear_current"),
                },
                experiment=FusionExperimentSpec("bio_only", ("eeg", "wear"), "fatigue"),
            )

        self.assertEqual(dataset.target.tolist(), [0.0, 1.0, 2.0])

    def test_labels_can_come_from_later_branch_when_eeg_branch_is_embedding_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eeg = root / "eeg.npz"
            wear = root / "wear.npz"
            sample_ids = ["sample-0", "sample-1", "sample-2"]
            _write_branch(eeg, sample_ids=sample_ids, modality="eeg", offset=1.0, include_labels=False)
            _write_branch(wear, sample_ids=sample_ids, modality="wear", offset=2.0)

            dataset = build_fusion_dataset(
                branches={
                    "eeg": FusionBranchSpec(path=eeg, modality="eeg", profile="eeg_current"),
                    "wear": FusionBranchSpec(path=wear, modality="wear", profile="wear_current"),
                },
                experiment=FusionExperimentSpec("bio_only", ("eeg", "wear"), "fatigue"),
            )

        self.assertEqual(dataset.sample_id.tolist(), sample_ids)
        self.assertEqual(dataset.target.tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(dataset.subject_id.tolist(), ["sub-01", "sub-02", "sub-03"])

    def test_labels_can_come_from_metadata_source_outside_enabled_modalities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eeg = root / "eeg.npz"
            wear = root / "wear.npz"
            metadata = root / "metadata_video.npz"
            sample_ids = ["sample-0", "sample-1", "sample-2"]
            _write_branch(eeg, sample_ids=sample_ids, modality="eeg", offset=1.0, include_labels=False)
            _write_branch(wear, sample_ids=sample_ids, modality="wear", offset=2.0, include_labels=False)
            _write_branch(metadata, sample_ids=sample_ids, modality="video", offset=3.0)

            dataset = build_fusion_dataset(
                branches={
                    "eeg": FusionBranchSpec(path=eeg, modality="eeg", profile="eeg_current"),
                    "wear": FusionBranchSpec(path=wear, modality="wear", profile="wear_current"),
                },
                experiment=FusionExperimentSpec("bio_only", ("eeg", "wear"), "fatigue"),
                metadata_source=FusionBranchSpec(path=metadata, modality="video", profile="labels_only"),
            )

        self.assertEqual(dataset.modalities, ("eeg", "wear"))
        self.assertEqual(dataset.tokens.shape, (3, 2, 256))
        self.assertEqual(dataset.target.tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(dataset.subject_id.tolist(), ["sub-01", "sub-02", "sub-03"])

    def test_default_alignment_uses_common_sample_ids_in_target_branch_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eeg = root / "eeg.npz"
            wear = root / "wear.npz"
            _write_branch(eeg, sample_ids=["sample-0", "sample-1", "sample-2"], modality="eeg", offset=1.0)
            _write_branch(wear, sample_ids=["sample-2", "sample-0"], modality="wear", offset=2.0, include_labels=False)

            dataset = build_fusion_dataset(
                branches={
                    "eeg": FusionBranchSpec(path=eeg, modality="eeg", profile="eeg_current"),
                    "wear": FusionBranchSpec(path=wear, modality="wear", profile="wear_current"),
                },
                experiment=FusionExperimentSpec("bio_only", ("eeg", "wear"), "fatigue"),
            )

        self.assertEqual(dataset.sample_id.tolist(), ["sample-0", "sample-2"])
        self.assertEqual(dataset.target.tolist(), [0.0, 2.0])


class LearnableCrossAttentionTests(unittest.TestCase):
    def test_require_torch_reports_clear_error_when_dependency_is_missing(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(ImportError, "learnable_cross_attention requires torch"):
                require_torch_for_learnable_cross_attention()
        else:
            self.assertIsNotNone(require_torch_for_learnable_cross_attention())

    def test_learnable_attention_forward_returns_predictions_and_attention(self):
        pytest = __import__("pytest")
        torch = pytest.importorskip("torch")
        dataset = _fusion_dataset_for_torch()
        model = fit_learnable_cross_attention(
            dataset,
            train_indices=np.arange(4),
            val_indices=np.arange(4, 6),
            config=LearnableAttentionConfig(
                token_dim=16,
                epochs=40,
                batch_size=2,
                learning_rate=0.01,
                patience=12,
                seed=7,
            ),
            torch_module=torch,
        )

        pred, attention = predict_with_learnable_cross_attention(
            model,
            dataset,
            indices=np.arange(6),
            torch_module=torch,
        )

        self.assertEqual(pred.shape, (6,))
        self.assertEqual(attention.shape, (6, 4))
        np.testing.assert_allclose(attention[0, 3], 0.0, atol=1e-6)
        np.testing.assert_allclose(attention.sum(axis=1), np.ones(6), atol=1e-5)
        self.assertLess(model.final_train_loss, model.initial_train_loss)


class FusionMatrixConfigTests(unittest.TestCase):
    def test_matrix_config_expands_fixed_wear_video_audio_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "fusion_matrix.json"
            payload = {
                "metadata_source": {"path": "metadata.npz", "modality": "video", "profile": "labels_only"},
                "branches": {
                    "eeg": {"path": "eeg.npz", "modality": "eeg", "profile": "eeg_current"},
                    "wear": {
                        "WphysioPre": {
                            "path": "wear_physio.npz",
                            "modality": "wear",
                            "profile": "wear_physio_features_preprocessed_v1",
                        },
                        "WdeepPre": {
                            "path": "wear_deep.npz",
                            "modality": "wear",
                            "profile": "wear_deep_sequence_preprocessed_v1",
                        },
                    },
                    "video": {
                        "FullSweepB0": {"path": "video_b0.npz", "modality": "video", "profile": "full_sweep/B0"},
                        "FullSweepB3Lam005": {"path": "video_b3.npz", "modality": "video", "profile": "full_sweep/B3_lam0.05"},
                        "A1A2TrainOnlyA2": {"path": "video_a2.npz", "modality": "video", "profile": "a1_a2_train_only/A2"},
                        "B5A1Lam0001": {"path": "video_b5.npz", "modality": "video", "profile": "b5_a1/B5_A1_lam0.001"},
                    },
                    "audio": {"path": "audio.npz", "modality": "audio", "profile": "audio_current"},
                },
                "target_label": "fatigue",
            }
            config.write_text(json.dumps(payload), encoding="utf-8")

            matrix = load_fusion_matrix_config(config)
            specs = matrix_experiment_specs(matrix)
            names = {spec.name for spec in specs}

        self.assertEqual(len(specs), 20)
        self.assertIn("fusion_WphysioPre_FullSweepB3Lam005_no_audio", names)
        self.assertIn("fusion_WdeepPre_B5A1Lam0001_full", names)
        self.assertIn("fusion_WphysioPre_no_video", names)
        self.assertIn("fusion_WdeepPre_bio_only", names)
        by_name = {spec.name: spec for spec in specs}
        self.assertEqual(by_name["fusion_WphysioPre_no_video"].enabled_modalities, ("eeg", "wear", "audio"))
        self.assertEqual(by_name["fusion_WdeepPre_bio_only"].enabled_modalities, ("eeg", "wear"))
        self.assertIsNotNone(matrix.metadata_source)
        self.assertEqual(str(matrix.metadata_source.path), "metadata.npz")

    def test_fusion_matrix_cli_dry_run_writes_comparison_families_and_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "fusion_matrix.json"
            out_dir = root / "reports"
            payload = {
                "branches": {
                    "eeg": {"path": "eeg.npz", "modality": "eeg", "profile": "eeg_current"},
                    "wear": {
                        "WphysioPre": {
                            "path": "wear_physio.npz",
                            "modality": "wear",
                            "profile": "wear_physio_features_preprocessed_v1",
                        },
                        "WdeepPre": {
                            "path": "wear_deep.npz",
                            "modality": "wear",
                            "profile": "wear_deep_sequence_preprocessed_v1",
                        },
                    },
                    "video": {
                        "FullSweepB0": {"path": "video_b0.npz", "modality": "video", "profile": "full_sweep/B0"},
                        "FullSweepB3Lam005": {"path": "video_b3.npz", "modality": "video", "profile": "full_sweep/B3_lam0.05"},
                        "A1A2TrainOnlyA2": {"path": "video_a2.npz", "modality": "video", "profile": "a1_a2_train_only/A2"},
                        "B5A1Lam0001": {"path": "video_b5.npz", "modality": "video", "profile": "b5_a1/B5_A1_lam0.001"},
                    },
                    "audio": {"path": "audio.npz", "modality": "audio", "profile": "audio_current"},
                },
                "target_label": "fatigue",
            }
            config.write_text(json.dumps(payload), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/43_run_fusion_matrix.py",
                    "--config",
                    str(config),
                    "--dry-run",
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )
            manifest = json.loads((out_dir / "fusion_matrix_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(manifest["experiment_count"], 20)
        by_name = {row["name"]: row for row in manifest["experiments"]}
        self.assertEqual(
            by_name["fusion_WphysioPre_FullSweepB3Lam005_no_audio"]["comparison_family"],
            "WphysioPre_FullSweepB3Lam005",
        )
        self.assertEqual(by_name["fusion_WphysioPre_no_video"]["comparison_family"], "WphysioPre_no_video")
        self.assertIn("decision_rule", manifest)


def _write_branch(path: Path, *, sample_ids: list[str], modality: str, offset: float, include_labels: bool = True) -> None:
    count = len(sample_ids)
    emb = np.zeros((count, 256), dtype=np.float32)
    emb[:, 0] = np.arange(count, dtype=np.float32) + offset
    mask = np.zeros((count, 4), dtype=np.int8)
    key = {
        "eeg": "eeg_emb",
        "wear": "wear_emb",
        "video": "face_emb",
        "audio": "audio_emb",
    }[modality]
    mask_index = {"eeg": 0, "wear": 1, "video": 2, "audio": 3}[modality]
    mask[:, mask_index] = 1
    payload = dict(
        sample_id=np.asarray(sample_ids, dtype=object),
        event_id=np.asarray([f"event-{idx}" for idx in range(count)], dtype=object),
        subject_id=np.asarray([f"sub-{idx % 3 + 1:02d}" for idx in range(count)], dtype=object),
        session_id=np.asarray(["ses-01"] * count, dtype=object),
        **{key: emb},
        modality_mask=mask,
        encoder_version=np.asarray([f"{modality}_encoder"] * count, dtype=object),
    )
    if include_labels:
        payload["labels"] = np.asarray([json.dumps({"fatigue": float(idx)}) for idx in range(count)], dtype=object)
    np.savez_compressed(path, **payload)


def _fusion_dataset_for_torch() -> FusionDataset:
    tokens = np.zeros((6, 4, 256), dtype=np.float32)
    base = np.arange(6, dtype=np.float32)
    tokens[:, 0, 0] = base
    tokens[:, 1, 0] = base + 0.5
    tokens[:, 2, 0] = base + 1.0
    tokens[:, 3, 0] = base + 1.5
    mask = np.ones((6, 4), dtype=bool)
    mask[0, 3] = False
    return FusionDataset(
        name="torch_smoke",
        modalities=("eeg", "wear", "video", "audio"),
        sample_id=np.asarray([f"sample-{idx}" for idx in range(6)], dtype=str),
        event_id=np.asarray([f"event-{idx}" for idx in range(6)], dtype=str),
        subject_id=np.asarray(["sub-01", "sub-01", "sub-02", "sub-02", "sub-03", "sub-03"], dtype=str),
        target=base.astype(np.float32),
        tokens=tokens,
        token_mask=mask,
        branch_profiles={"eeg": "eeg", "wear": "wear", "video": "video", "audio": "audio"},
        target_label="fatigue",
    )


if __name__ == "__main__":
    unittest.main()
