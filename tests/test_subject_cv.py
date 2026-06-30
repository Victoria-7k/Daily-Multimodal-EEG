import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_multimodal.training.subject_cv import (
    build_subject_folds,
    run_subject_cv,
)


class SubjectCvTests(unittest.TestCase):
    def test_leave_one_subject_out_folds_do_not_leak_subjects(self):
        subjects = np.array(["sub-01", "sub-01", "sub-02", "sub-02", "sub-03", "sub-03"], dtype=object)

        folds = build_subject_folds(subjects, strategy="leave_one_subject_out")

        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertTrue(set(subjects[fold.train]).isdisjoint(set(subjects[fold.test])))
            self.assertGreater(len(fold.train), 0)
            self.assertGreater(len(fold.test), 0)

    def test_grouped_k_fold_is_deterministic_and_subject_disjoint(self):
        subjects = np.array([f"sub-{idx:02d}" for idx in range(1, 7) for _ in range(2)], dtype=object)

        first = build_subject_folds(subjects, strategy="grouped_k_fold", n_splits=3, seed=7)
        second = build_subject_folds(subjects, strategy="grouped_k_fold", n_splits=3, seed=7)

        self.assertEqual([fold.test.tolist() for fold in first], [fold.test.tolist() for fold in second])
        for fold in first:
            self.assertTrue(set(subjects[fold.train]).isdisjoint(set(subjects[fold.test])))
            self.assertTrue(set(subjects[fold.val]).isdisjoint(set(subjects[fold.test])))

    def test_run_subject_cv_writes_metrics_and_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings = root / "embeddings.npz"
            out_json = root / "subject_cv.json"
            out_table = root / "subject_cv.md"
            _write_embeddings(embeddings)

            result = run_subject_cv(
                embeddings=embeddings,
                target_label="alert",
                out_json=out_json,
                out_table=out_table,
                strategy="leave_one_subject_out",
                epochs=15,
                hidden_dim=8,
            )

            metrics = json.loads(out_json.read_text(encoding="utf-8"))
            table = out_table.read_text(encoding="utf-8")

        self.assertGreaterEqual(result["fold_count"], 4)
        self.assertFalse(result["subject_leakage"])
        self.assertIn("rmse_mean", metrics)
        self.assertIn("| fold |", table)

    def test_cli_runs_subject_cv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings = root / "embeddings.npz"
            out_json = root / "subject_cv.json"
            out_table = root / "subject_cv.md"
            _write_embeddings(embeddings)

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/20_run_subject_cv.py",
                    "--embeddings",
                    str(embeddings),
                    "--target-label",
                    "alert",
                    "--out-json",
                    str(out_json),
                    "--out-table",
                    str(out_table),
                    "--epochs",
                    "15",
                    "--hidden-dim",
                    "8",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("fold_count=", completed.stdout)


def _write_embeddings(path: Path) -> None:
    subjects = np.array([f"sub-{idx:02d}" for idx in range(1, 7) for _ in range(3)], dtype=object)
    base = np.linspace(0.0, 1.0, len(subjects), dtype=np.float32)
    emb = np.zeros((len(subjects), 256), dtype=np.float32)
    emb[:, 0] = base
    labels = np.array([json.dumps({"alert": float(value)}) for value in base], dtype=object)
    np.savez_compressed(
        path,
        sample_id=np.array([f"sample-{idx}" for idx in range(len(subjects))], dtype=object),
        event_id=np.array([f"event-{idx}" for idx in range(len(subjects))], dtype=object),
        subject_id=subjects,
        session_id=np.array(["ses-01"] * len(subjects), dtype=object),
        eeg_emb=emb + 0.1,
        wear_emb=emb + 0.2,
        face_emb=emb + 0.3,
        audio_emb=emb + 0.4,
        modality_mask=np.ones((len(subjects), 4), dtype=np.int8),
        labels=labels,
        source_paths=np.array(["{}"] * len(subjects), dtype=object),
    )


if __name__ == "__main__":
    unittest.main()
