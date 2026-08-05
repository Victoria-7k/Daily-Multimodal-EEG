import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_multimodal.training.fair_embedding_ablation import run_fair_embedding_ablation, run_fusion_spec_fair_ablation


class FairEmbeddingAblationTests(unittest.TestCase):
    def test_run_fair_embedding_ablation_builds_aligned_leakage_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            basic_npz = root / "basic.npz"
            real_npz = root / "real.npz"
            metrics_json = root / "fair_metrics.json"
            table_md = root / "fair_table.md"
            _write_embeddings(basic_npz, offset=0.0)
            _write_embeddings(real_npz, offset=10.0)

            result = run_fair_embedding_ablation(
                basic_embeddings=basic_npz,
                real_embeddings=real_npz,
                target_label="alert",
                out_json=metrics_json,
                out_table=table_md,
                epochs=10,
                overfit_limit=4,
                hidden_dim=8,
            )

            metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
            table = table_md.read_text(encoding="utf-8")

        self.assertEqual(result["row_count"], 6)
        self.assertTrue(result["sample_id_aligned"])
        self.assertLessEqual(
            {"basic_aligned", "basic_no_path", "path_only", "real"},
            set(result["experiments"]),
        )
        self.assertEqual(metrics["row_count"], 6)
        self.assertEqual(metrics["modalities"], ["eeg", "wear", "face", "audio"])
        self.assertIn("test_pearson_r", metrics["experiments"]["real"])
        self.assertIn("| experiment |", table)
        self.assertIn("test_r", table)
        self.assertIn("basic_no_path", table)

    def test_fair_ablation_rejects_misaligned_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            basic_npz = root / "basic.npz"
            real_npz = root / "real.npz"
            _write_embeddings(basic_npz, offset=0.0)
            _write_embeddings(real_npz, offset=1.0, reverse_sample_ids=True)

            result = run_fair_embedding_ablation(
                basic_embeddings=basic_npz,
                real_embeddings=real_npz,
                target_label="alert",
                out_json=root / "metrics.json",
                out_table=root / "table.md",
                epochs=5,
            )

        self.assertFalse(result["sample_id_aligned"])
        self.assertEqual(result["failure_count"], 1)
        self.assertEqual(result["failures"][0]["error_type"], "sample_id_mismatch")

    def test_cli_runs_fair_embedding_ablation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            basic_npz = root / "basic.npz"
            real_npz = root / "real.npz"
            metrics_json = root / "metrics.json"
            table_md = root / "table.md"
            _write_embeddings(basic_npz, offset=0.0)
            _write_embeddings(real_npz, offset=2.0)

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/18_run_fair_embedding_ablation.py",
                    "--basic-embeddings",
                    str(basic_npz),
                    "--real-embeddings",
                    str(real_npz),
                    "--target-label",
                    "alert",
                    "--out-json",
                    str(metrics_json),
                    "--out-table",
                    str(table_md),
                    "--epochs",
                    "10",
                    "--overfit-limit",
                    "4",
                    "--hidden-dim",
                    "8",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("row_count=6", completed.stdout)
            self.assertTrue(metrics_json.is_file())
            self.assertTrue(table_md.is_file())

    def test_learnable_cross_attention_reports_missing_torch_dependency(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                basic_npz = root / "basic.npz"
                real_npz = root / "real.npz"
                _write_embeddings(basic_npz, offset=0.0)
                _write_embeddings(real_npz, offset=2.0)

                with self.assertRaisesRegex(ImportError, "learnable_cross_attention requires torch"):
                    run_fair_embedding_ablation(
                        basic_embeddings=basic_npz,
                        real_embeddings=real_npz,
                        target_label="alert",
                        out_json=root / "metrics.json",
                        out_table=root / "table.md",
                        epochs=2,
                        model="learnable_cross_attention",
                    )
        else:
            self.skipTest("PyTorch is installed; missing-dependency path is not active")

    def test_fusion_spec_fair_ablation_reports_missing_torch_dependency(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sample_ids = [f"sample-{idx}" for idx in range(6)]
                eeg = root / "eeg.npz"
                wear = root / "wear.npz"
                video = root / "video.npz"
                _write_branch(eeg, sample_ids=sample_ids, modality="eeg", offset=0.1, include_labels=True)
                _write_branch(wear, sample_ids=sample_ids, modality="wear", offset=0.2, include_labels=False)
                _write_branch(video, sample_ids=sample_ids, modality="video", offset=0.3, include_labels=False)
                config = root / "fusion_matrix.json"
                config.write_text(
                    json.dumps(
                        {
                            "target_label": "alert",
                            "branches": {
                                "eeg": {"path": str(eeg), "modality": "eeg", "profile": "eeg_current"},
                                "wear": {"WphysioPre": {"path": str(wear), "modality": "wear", "profile": "wear_physio_features_preprocessed_v1"}},
                                "video": {"V4aUpper": {"path": str(video), "modality": "video", "profile": "V4a_upper"}},
                                "audio": {"path": str(eeg), "modality": "audio", "profile": "audio_current"},
                            },
                        }
                    ),
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(ImportError, "learnable_cross_attention requires torch"):
                    run_fusion_spec_fair_ablation(
                        fusion_spec=config,
                        fusion_experiment="fusion_WphysioPre_V4aUpper_no_audio",
                        target_label="alert",
                        out_json=root / "metrics.json",
                        out_table=root / "table.md",
                        epochs=2,
                    )
        else:
            self.skipTest("PyTorch is installed; missing-dependency path is not active")


def _write_embeddings(path: Path, *, offset: float, reverse_sample_ids: bool = False) -> None:
    subjects = np.array(["sub-02", "sub-02", "sub-11", "sub-11", "sub-13", "sub-13"], dtype=object)
    sample_ids = np.array([f"sample-{idx}" for idx in range(6)], dtype=object)
    if reverse_sample_ids:
        sample_ids = sample_ids[::-1]
    labels = np.array([json.dumps({"alert": float(idx)}) for idx in range(6)], dtype=object)
    base = np.zeros((6, 256), dtype=np.float32)
    base[:, 0] = np.arange(6, dtype=np.float32) + offset
    np.savez_compressed(
        path,
        sample_id=sample_ids,
        event_id=np.array([f"event-{idx}" for idx in range(6)], dtype=object),
        subject_id=subjects,
        session_id=np.array(["ses-01", "ses-02", "ses-01", "ses-02", "ses-01", "ses-02"], dtype=object),
        eeg_emb=base + 0.1,
        wear_emb=base + 0.2,
        face_emb=base + 0.3,
        audio_emb=base + 0.4,
        modality_mask=np.ones((6, 4), dtype=np.int8),
        labels=labels,
        source_paths=np.array(
            [
                json.dumps({"video": f"/raw/sub-{idx % 3}/clip-{idx}.mp4", "audio": f"clip-{idx}.wav"})
                for idx in range(6)
            ],
            dtype=object,
        ),
    )


def _write_branch(
    path: Path,
    *,
    sample_ids: list[str],
    modality: str,
    offset: float,
    include_labels: bool,
) -> None:
    count = len(sample_ids)
    emb = np.zeros((count, 256), dtype=np.float32)
    emb[:, 0] = np.arange(count, dtype=np.float32) + offset
    key = {"eeg": "eeg_emb", "wear": "wear_emb", "video": "face_emb", "audio": "audio_emb"}[modality]
    mask_index = {"eeg": 0, "wear": 1, "video": 2, "audio": 3}[modality]
    mask = np.zeros((count, 4), dtype=np.int8)
    mask[:, mask_index] = 1
    payload = {
        "sample_id": np.asarray(sample_ids, dtype=object),
        "event_id": np.asarray([f"event-{idx}" for idx in range(count)], dtype=object),
        "subject_id": np.asarray(["sub-02", "sub-02", "sub-11", "sub-11", "sub-13", "sub-13"], dtype=object),
        key: emb,
        "modality_mask": mask,
    }
    if include_labels:
        payload["labels"] = np.asarray([json.dumps({"alert": float(idx)}) for idx in range(count)], dtype=object)
    np.savez_compressed(path, **payload)


if __name__ == "__main__":
    unittest.main()
