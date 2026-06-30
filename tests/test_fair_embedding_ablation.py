import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_multimodal.training.fair_embedding_ablation import run_fair_embedding_ablation


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
        self.assertIn("| experiment |", table)
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


if __name__ == "__main__":
    unittest.main()
